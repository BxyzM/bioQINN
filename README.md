# bioQINN: Regressing protein stability with Quantum Machine Learning (1 Pixel - 1 Qubit)


**Authors:** [Aritra Bal](https://etpwww.etp.kit.edu/~abal/), Benedikt Maier, Michael Spannowsky
**Contact:** [Email](mailto:aritra.bal@do-not-spam.kit.edu)

[![arXiv](https://img.shields.io/badge/arXiv-2502.17301-b31b1b.svg)](https://arxiv.org/abs/2510.17984) 

Theory/bio paper from [Nat. Mach. Intel](https://www.nature.com/articles/s42256-023-00715-4).

Work in Progress. Readme gets updated over time.

------

### Summary

- We use up to $32\times32\times7$ image representations of protein-protein interactions, from the original PISToN implementation (see NMI link above). 
- For now, model barnase-barstar complex (`1BRS_A_D`) with up to $100$ docking configurations + $1$ native configuration. 
- Regress stability of interaction to $Z$ spin of control qubit, direct mapping to PISToN scores. 