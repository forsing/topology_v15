from topology_qtda_core import run_quantum_version

if __name__ == "__main__":
    run_quantum_version(
        version=15,
        part_name="Quantum Advantage (Dicke/BE/qubitization/Chebyshev)",
        kind="laplacian",
        d=4,
        alpha=1.0,
        profile="pi8_alt",
        dicke_k=5,
        score_mode="last_times_one_minus_q",
    )

 

"""
--------------------------------
topology_v15: Quantum TDA advantage — Quantum Advantage (Dicke/BE/qubitization/Chebyshev)
pipeline: Dicke → block-encode → qubitization walk (QSP Chebyshev)
SEED=39  NQ_B=6  qubits=7  (lokalni Statevector)
kind=laplacian  d=4  alpha=1.0  profile=pi8_alt  dicke_k=5
score_mode=last_times_one_minus_q
--------------------------------

================================
LOTO: loto7_4656_k59_loto_2950.csv
================================
kola: 2950
β₀ (klasični ker L brojeva): 1
P(anc=0) postselect ≈ 0.992296
next_loto: [8, x, 25, y, 29, z, 34]

================================
PLUS: loto7_4656_k59_loto_plus_1706.csv
================================
kola: 1706
β₀ (klasični ker L brojeva): 1
P(anc=0) postselect ≈ 0.992332
next_plus: [10, x, 18, y, 31, z, 37]
"""
