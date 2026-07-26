# Shared Quantum TDA advantage core (lokalno Aer / Statevector)
# Dicke state prep → block-encode Hermitskog operatora → qubitization walk (QSP)
# Qubit budget: 1 ancilla + NQ_B (bez QPE) — isti obrazac kao data/Q30_qreg5_QSP.
# Bez RNG. SEED=39 samo kao fiksna konstanta.

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation, UnitaryGate
from qiskit.quantum_info import Statevector

SEED = 39
N_BALL = 39
K_DRAW = 7
NQ_B = 6
BE_EIG_MARGIN = 0.9
EPS0 = 1e-10

DATA = Path(__file__).resolve().parents[1] / "data"
CSV_LOTO = DATA / "loto7_4656_k59_loto_2950.csv"
CSV_PLUS = DATA / "loto7_4656_k59_loto_plus_1706.csv"
CSV_PATHS = (("LOTO", CSV_LOTO), ("PLUS", CSV_PLUS))


def load_draws(path: Path) -> np.ndarray:
    draws = np.loadtxt(path, delimiter=",", dtype=int)
    if draws.ndim == 1:
        draws = draws.reshape(1, -1)
    assert draws.shape[1] == K_DRAW, draws.shape
    return draws


def cooccurrence(draws: np.ndarray) -> np.ndarray:
    C = np.zeros((N_BALL, N_BALL), dtype=np.float64)
    for row in draws:
        idx = row - 1
        for a in range(K_DRAW):
            for b in range(a + 1, K_DRAW):
                i, j = int(idx[a]), int(idx[b])
                C[i, j] += 1.0
                C[j, i] += 1.0
    return C


def freq_vector(draws: np.ndarray) -> np.ndarray:
    f = np.zeros(N_BALL, dtype=np.float64)
    for row in draws:
        for x in row:
            f[int(x) - 1] += 1.0
    return f


def combinatorial_laplacian(draws: np.ndarray) -> np.ndarray:
    C = cooccurrence(draws)
    deg = C.sum(axis=1)
    return np.diag(deg) - C


def hodge_normalized_laplacian(draws: np.ndarray) -> np.ndarray:
    L = combinatorial_laplacian(draws)
    deg = np.diag(L).copy()  # for L=D-A, diag is degree
    # safer: from cooccurrence
    C = cooccurrence(draws)
    deg = C.sum(axis=1)
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    Dmh = np.diag(dinv)
    return Dmh @ L @ Dmh


def dirac_sqrt_laplacian(draws: np.ndarray) -> np.ndarray:
    """Hermitski '√L' na prostoru brojeva (Dirac-inspired, lokalno)."""
    L = combinatorial_laplacian(draws)
    evals, evecs = np.linalg.eigh(L)
    s = np.sign(evals) * np.sqrt(np.maximum(np.abs(evals), 0.0))
    return (evecs * s) @ evecs.T


def build_operator(draws: np.ndarray, kind: str) -> np.ndarray:
    if kind == "laplacian":
        return combinatorial_laplacian(draws)
    if kind == "hodge":
        return hodge_normalized_laplacian(draws)
    if kind == "dirac_sqrt":
        return dirac_sqrt_laplacian(draws)
    raise ValueError(kind)


def pad_scale_hermitian(Op: np.ndarray, nq: int, alpha: float) -> Tuple[np.ndarray, float]:
    dim = 2 ** nq
    A = float(alpha) * np.eye(dim, dtype=np.float64)
    n = min(Op.shape[0], dim)
    A[:n, :n] = Op[:n, :n] + float(alpha) * np.eye(n)
    eigs = np.linalg.eigvalsh(A)
    max_abs = float(max(abs(eigs.max()), abs(eigs.min()), 1e-18))
    scale = max_abs / BE_EIG_MARGIN
    return A / scale, scale


def dicke_amplitudes(nq: int, k: int) -> np.ndarray:
    """Jednaka amplituda na baznim stanjima težine k (Dicke |D_k^n⟩)."""
    k = int(max(0, min(k, nq)))
    dim = 2 ** nq
    amp = np.zeros(dim, dtype=np.float64)
    for i in range(dim):
        if bin(i).count("1") == k:
            amp[i] = 1.0
    nrm = float(np.linalg.norm(amp))
    if nrm < 1e-18:
        amp[:] = 1.0 / np.sqrt(dim)
    else:
        amp /= nrm
    return amp


def block_encode_hermitian(A: np.ndarray) -> np.ndarray:
    eigs, V = np.linalg.eigh(A)
    one_minus_sq = np.clip(1.0 - eigs ** 2, 0.0, None)
    B = V @ np.diag(np.sqrt(one_minus_sq)) @ V.conj().T
    dim = A.shape[0]
    U = np.zeros((2 * dim, 2 * dim), dtype=np.complex128)
    U[:dim, :dim] = A
    U[:dim, dim:] = B
    U[dim:, :dim] = B
    U[dim:, dim:] = -A
    return U


def phase_sequence(profile: str, d: int) -> List[float]:
    if profile == "zero":
        return [0.0] * max(0, d - 1)
    if profile == "pi8_alt":
        base = float(np.pi / 8.0)
        return [base * ((-1) ** k) for k in range(max(0, d - 1))]
    return [0.0] * max(0, d - 1)


def build_qsp_circuit(
    A_scaled: np.ndarray, b_amp: np.ndarray, n_b: int, d: int, phases: List[float]
) -> QuantumCircuit:
    anc = QuantumRegister(1, name="a")
    b_reg = QuantumRegister(n_b, name="b")
    qc = QuantumCircuit(anc, b_reg)
    qc.append(StatePreparation(b_amp.tolist()), b_reg)
    UA_gate = UnitaryGate(block_encode_hermitian(A_scaled), label="U_A")
    for k in range(d):
        qc.append(UA_gate, list(anc) + list(b_reg))
        if k < d - 1:
            phi_k = float(phases[k]) if k < len(phases) else 0.0
            if abs(phi_k) > 1e-14:
                qc.rz(2.0 * phi_k, anc[0])
            qc.z(anc[0])
    return qc


def qsp_postselect(
    A_scaled: np.ndarray, b_amp: np.ndarray, n_b: int, d: int, profile: str
) -> Tuple[np.ndarray, float]:
    phases = phase_sequence(profile, d)
    qc = build_qsp_circuit(A_scaled, b_amp, n_b, d, phases)
    p = np.abs(Statevector(qc).data) ** 2
    dim_b = 2 ** n_b
    mat = p.reshape(dim_b, 2)
    p_b = mat[:, 0]
    p_post = float(p_b.sum())
    if p_post < 1e-18:
        return np.zeros(dim_b, dtype=np.float64), 0.0
    return p_b / p_post, p_post


def bias_39(probs: np.ndarray) -> np.ndarray:
    b = np.zeros(N_BALL, dtype=np.float64)
    for idx, p in enumerate(probs):
        b[idx % N_BALL] += float(p)
    s = float(b.sum())
    return b / s if s > 0 else b


def pick_next(scores: np.ndarray, draws: np.ndarray) -> List[int]:
    """Opsezi [1..33]..[7..39]; tie-break skor pa veći broj. Bez seen."""
    chosen, used = [], set()
    for pos in range(K_DRAW):
        lo, hi = pos + 1, 33 + pos
        cands = [n for n in range(lo, hi + 1) if n not in used]
        cands.sort(key=lambda n: (scores[n - 1], n), reverse=True)
        pick = cands[0]
        chosen.append(pick)
        used.add(pick)
    return [int(x) for x in sorted(chosen)]


def quantum_tda_predict(
    draws: np.ndarray,
    *,
    kind: str,
    d: int,
    alpha: float,
    profile: str,
    dicke_k: int,
    score_mode: str = "last_times_q",
) -> Tuple[List[int], float, np.ndarray]:
    Op = build_operator(draws, kind)
    A_scaled, _ = pad_scale_hermitian(Op, NQ_B, alpha)
    b_amp = dicke_amplitudes(NQ_B, dicke_k)
    probs, p_post = qsp_postselect(A_scaled, b_amp, NQ_B, d, profile)
    q = bias_39(probs)
    # CSV-specifičan frekvencijski tilt (LOTO ≠ PLUS), bez RNG
    f = freq_vector(draws)
    f = f / max(float(f.max()), 1.0)
    q = q + (SEED % 10) / 50.0 * f
    qmax = float(q.max()) if q.size else 0.0
    q_n = q / qmax if qmax > 0 else q

    C = cooccurrence(draws)
    score = np.zeros(N_BALL, dtype=np.float64)
    for n in draws[-1]:
        score += C[int(n) - 1]
    if score_mode == "last_times_one_minus_q":
        # jači last-draw uticaj kad q skor previše stabilan (npr. v9)
        score = score * (1.0 + (1.0 - q_n))
    else:
        # default: A[poslednje] × (1 + q̂)
        score = score * (1.0 + q_n)
    ticket = pick_next(score, draws)
    return ticket, p_post, score


def run_quantum_version(
    *,
    version: int,
    part_name: str,
    kind: str,
    d: int,
    alpha: float,
    profile: str,
    dicke_k: int,
    score_mode: str = "last_times_q",
) -> None:
    print()
    print("--------------------------------")
    print(f"topology_v{version}: Quantum TDA advantage — {part_name}")
    print("pipeline: Dicke → block-encode → qubitization walk (QSP Chebyshev)")
    print(f"SEED={SEED}  NQ_B={NQ_B}  qubits={1 + NQ_B}  (lokalni Statevector)")
    print(f"kind={kind}  d={d}  alpha={alpha}  profile={profile}  dicke_k={dicke_k}")
    print(f"score_mode={score_mode}")
    print("--------------------------------")

    for label, csv_path in CSV_PATHS:
        print()
        print("================================")
        print(f"{label}: {csv_path.name}")
        print("================================")
        draws = load_draws(csv_path)
        print(f"kola: {draws.shape[0]}")

        # klasični β₀ na L (referenca)
        L = combinatorial_laplacian(draws)
        evals = np.linalg.eigvalsh(L)
        beta0 = int(np.sum(evals <= EPS0))
        print(f"β₀ (klasični ker L brojeva): {beta0}")

        ticket, p_post, _ = quantum_tda_predict(
            draws,
            kind=kind,
            d=d,
            alpha=alpha,
            profile=profile,
            dicke_k=dicke_k,
            score_mode=score_mode,
        )
        print(f"P(anc=0) postselect ≈ {p_post:.6f}")
        print(f"next_{label.lower()}: {ticket}")
