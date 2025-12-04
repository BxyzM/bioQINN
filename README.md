# bioQINN: Regressing protein stability with Quantum Machine Learning (1 Pixel - 1 Qubit)


**Authors:** [Aritra Bal](https://etpwww.etp.kit.edu/~abal/), Benedikt Maier, Michael Spannowsky

**Contact:** [Email](mailto:aritra.bal@do-not-spam.kit.edu)

[![arXiv](https://img.shields.io/badge/arXiv-2502.17301-b31b1b.svg)](https://arxiv.org/abs/2510.17984) 

Theory/bio paper from [Nat. Mach. Intel](https://www.nature.com/articles/s42256-023-00715-4).

Work in Progress. Readme gets updated over time.

------

### Summary

- We use up to `(32,32,7)` image representations of protein-protein interactions, these being the so-called interaction maps, from the original PISToN implementation (see NMI link above). 
- For now, model barnase-barstar complex (`1BRS_A_D`) with up to 100 docking configurations + 1 native configuration. 
- Regress stability of interaction to Z spin of control qubit, direct mapping to PISToN scores. 


------

### Important Scripts

- `src/architectures.py` : contains the quantum circuit architecture, adapted from the original 1P1Q implementation. Major change: the encoding, entanglement and trainable Clifford gate operations are now user-supplied instead of being hardcoded. 
- `src/trainer.py` : contains a general framework for training quantum circuits.
- `models/operations.py`: contains the modular definitions of the encoding, entanglement and rotation operations. A copy of this script is made for each run and preserved as part of the model, since without this, one cannot reconstruct the correct model for inference. 
- `data_handlers/dataloader.py` : dataloader that fetches the data (expects `.npy` files) given a path to a YAML config file and batches the resultant data efficiently. This can be tested by calling the script, also with the optional argument `--path`, without which it uses dummy data for a demonstration. 
- `train.py`: single script for training the QML classifier/regressor. 