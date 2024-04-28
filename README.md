# Alchemical MLIP
[![arXiv](https://img.shields.io/badge/arXiv-2404.10746-84cc16)](https://arxiv.org/abs/2404.10746)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281/zenodo.11081395-14b8a6.svg)](https://zenodo.org/doi/10.5281/zenodo.11081395)
[![MIT](https://img.shields.io/badge/License-MIT-3b82f6.svg)](https://opensource.org/license/mit)

This repository contains the code to modify machine learning interatomic potentials (MLIPs) to enable continuous and differentiable alchemical transformations.
Currently, we provide the alchemical modification for the [MACE](https://github.com/ACEsuit/mace) model.
The details of the method are described in the paper: [Interpolation and differentiation of alchemical degrees of freedom in machine learning interatomic potentials](https://arxiv.org/abs/2404.10746).

## Installation
We tested the code with Python 3.10 and the packages in `requirements.txt`.
For example, you can create a conda environment and install the required packages as follows (assuming CUDA 11.8):
```bash
conda create -n alchemical-mlip python=3.10
conda activate alchemical-mlip
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e .
```

## Static calculations
We provide the jupyter notebooks for the lattice parameter calculations (Fig. 2 in the paper) and the compositional optimization (Fig. 3) in the `notebook` directory.
```
notebook/
├── 1_solid_solution.ipynb
└── 2_compositional_optimization.ipynb
```

## Free energy calculations
We provide the scripts for the free energy calculations for the vacancy (Fig. 4) and perovskites (Fig. 5) in the `scripts` directory.
```
scripts/
├── vacancy_frenkel_ladd.py
├── perovskite_frenkel_ladd.py
└── perovskite_alchemy.py
```

The arguments for the scripts are as follows:
```bash
# Vacancy Frenkel-Ladd calculation
python vacancy_frenkel_ladd.py \
    --structure-file data/structures/Fe.cif \
    --supercell 5 5 5 \
    --temperature 100 \
    --output-dir data/results/vacancy/Fe_5x5x5_100K/0

# Perovskite Frenkel-Ladd calculation (alpha phase)
python perovskite_frenkel_ladd.py \
    --structure-file data/structures/CsPbI3_alpha.cif \
    --supercell 6 6 6 \
    --temperature 400 \
    --output-dir data/results/perovskite/frenkel_ladd/CsPbI3_alpha_6x6x6_400K/0

# Perovskite Frenkel-Ladd calculation (delta phase)
python perovskite_frenkel_ladd.py \
    --structure-file data/structures/CsPbI3_delta.cif \
    --supercell 6 3 3 \
    --temperature 400 \
    --output-dir data/results/perovskite/frenkel_ladd/CsPbI3_delta_6x3x3_400K/0

# Perovskite alchemy calculation (alpha phase)
python -u perovskite_alchemy.py \
    --structure-file data/structures/CsPbI3_alpha.cif \
    --supercell 6 6 6 \
    --switch-pair Pb Sn \
    --temperature 400 \
    --output-dir data/results/perovskite/alchemy/CsPbI3_CsSnI3_alpha_400K/0

# Perovskite alchemy calculation (delta phase)
python -u perovskite_alchemy.py \
    --structure-file data/structures/CsPbI3_delta.cif \
    --supercell 6 3 3 \
    --switch-pair Pb Sn \
    --temperature 400 \
    --output-dir data/results/perovskite/alchemy/CsPbI3_CsSnI3_delta_400K/0
```

The result files are large and not included in the repository.
If you want to reproduce the results without running the calculations, the result files are uploaded in the [Zenodo repository](https://zenodo.org/doi/10.5281/zenodo.11081395).
Please download the files and place them in the `data/results` directory.

The post-processing scripts for the free energy calculations are provided in the `notebook` directory.
```
notebook/
├── 3_vacancy_analysis.ipynb
└── 4_perovskite_analysis.ipynb
```

## Citation
```
@misc{nam2024interpolation,
    title={Interpolation and differentiation of alchemical degrees of freedom in machine learning interatomic potentials},
    author={Juno Nam and Rafael G{\'o}mez-Bombarelli},
    year={2024},
    eprint={2404.10746},
    archivePrefix={arXiv},
    primaryClass={cond-mat.mtrl-sci}
}
```
