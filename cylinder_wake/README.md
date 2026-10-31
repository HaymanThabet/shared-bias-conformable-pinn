# Cylinder-wake benchmark

This folder contains the two-dimensional observation-driven cylinder-wake implementation used in the manuscript.

The script `cylinder_wake_q07.py` compares:

- Standard PINN + Adam
- R3-PINN + Adam
- Shared-bias PINN + Adam–vCFGD

Clean, limited-observation, and 1% noise regimes are evaluated under a common training and held-out testing protocol.

## Data

The benchmark uses the public `cylinder_nektar_wake.mat` CFD dataset associated with Raissi et al.

The script first looks for the dataset at:

`cylinder_wake/cylinder_nektar_wake.mat`

If the file is unavailable, the script attempts to download it automatically. It may also be downloaded manually and placed beside `cylinder_wake_q07.py`. The dataset is not redistributed in this repository.

## Run

```bash
python cylinder_wake_q07.py
