# Generalized and Optimal Straight-Through Estimators

Official experimental code for the paper **"Generalized and Optimal Straight-Through Estimators"**
(Spotlight, AISTATS 2026).

The repository contains the gradient estimators introduced in the paper together with the
scripts used to produce the experimental results.

## Repository structure

| Path | Description |
|------|-------------|
| `Gradient_estimators.py` | Core implementations of the straight-through and baseline estimators (`ST`, `ZGR`, `gumbel_rao`, `reinmax`, `MVE`, …) and the temperature schedules. |
| `Bias_Variance/` | Bias–variance comparison of the estimators (`Bias_Variance.py` + SLURM submit script). |
| `VAE/` | Discrete-VAE experiments on MNIST. `Data.py` loads MNIST; `VAE.py` defines the model; `VAE/`, `Greedy_VAE/`, and `Reinmax/` contain the training entry points and submit scripts for each variant. |
| `poly_p.py`, `poly_p_int.py` | Synthetic polynomial-objective experiments (`run_poly_p*.sh` to launch). |
| `plot_result.py`, `plot_poly_p.py`, `summarize_greedy.py` | Plotting and summary utilities for the saved results. |
| `Results/` | Saved experiment outputs (`.pkl`) used to regenerate the figures in the paper. |
| `test.py` | Standalone numerical check verifying the estimator implementations against their closed-form expectations. |

## Requirements

- Python 3.12
- See [`requirements.txt`](requirements.txt):

```bash
pip install -r requirements.txt
```

MNIST is downloaded automatically by `torchvision` on first run (into `_data/`), so no data
needs to be committed.

## Running the experiments

Each experiment has a SLURM submit script (`submit_*.sh`) that shows the exact invocation and
parameter grid; the underlying Python scripts can also be run directly. For example, the
bias–variance experiment:

```bash
python Bias_Variance/Bias_Variance.py --dim 8 --seed 0
```

On a SLURM cluster the full grids are launched with, e.g.:

```bash
sbatch Bias_Variance/submit_bias_variance.sh
sbatch VAE/VAE/submit_vae.sh
```

Results are written to `Results/` and turned into figures with the `plot_*.py` scripts.

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{hooper2026straightthrough,
  title     = {Generalized and Optimal Straight-Through Estimators},
  author    = {Hooper, James and Shekhovtsov, Alexander},
  booktitle = {Proceedings of the 29th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year      = {2026}
}
```
