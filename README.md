# Shared-Bias PINN with Conformable-Gradient Training

This repository contains the Python implementations used in the manuscript:

**“Shared-Bias Physics-Informed Neural Network with Conformable Gradient Training for Time-Conformable Differential Equations.”**

The repository includes the main forward benchmarks, component analysis, statistical robustness experiments, sensitivity analysis, inverse identification, adaptive-sampling comparison, and the observation-driven cylinder-wake benchmark.

## Main experiments

- Time-conformable Burgers equation
- Time-conformable Burgers–Fisher equation
- Time-conformable reaction–diffusion equation
- Discontinuous-coefficient advection–diffusion benchmark
- Two-dimensional reaction–diffusion benchmark
- Four-configuration architecture/optimizer component analysis
- Five-run statistical robustness study
- R3-PINN adaptive-sampling comparison
- Sensitivity analysis
- Inverse identification under noisy observations
- Two-dimensional cylinder-wake benchmark at q = 0.7

## Software

The experiments were implemented in Python using PyTorch and automatic differentiation.

Detailed dependencies and execution instructions are provided in `requirements.txt` and in the benchmark-specific files.

## Reproducibility

The codes are intended to reproduce the numerical experiments reported in the manuscript. Random seeds, training settings, network sizes, and benchmark parameters are specified in the corresponding scripts.

The cylinder-wake experiment uses the public `cylinder_nektar_wake.mat` CFD dataset associated with the PINN benchmark of Raissi et al.

## License

This repository is released under the MIT License.
