# CMB Power Spectrum Emulator Competition

## The Task

The goal is to train a fast, accurate neural-network emulator of the four CMB angular power spectra — **TT**, **TE**, **EE**, and lensing-potential **φφ** — as a function of six ΛCDM cosmological parameters.

The emulator will be used inside MCMC sampling chains, where it must replace the full Boltzmann solver (CLASS/CAMB) at each likelihood evaluation. This places two competing demands on the design:

1. **Precision** — the emulated spectra must reproduce the likelihood function accurately enough that parameter inference is not biased.
2. **Speed** — inference must be fast enough for MCMC use; the target is sub-millisecond CPU `predict()` calls.

## Inputs and Outputs

### Input

A Python dict with six cosmological parameters:

```python
{
    "omega_b":       float,   # Ω_b h²      ∈ [0.020, 0.025]
    "omega_cdm":     float,   # Ω_cdm h²    ∈ [0.090, 0.150]
    "H0":            float,   # km/s/Mpc    ∈ [55.0,  85.0]
    "tau_reio":      float,   #             ∈ [0.030, 0.100]
    "ln10^{10}A_s":  float,   #             ∈ [2.700, 3.300]
    "n_s":           float,   #             ∈ [0.920, 1.020]
}
```

### Output

A Python dict with four spectra:

```python
{
    "tt": np.ndarray (6001,),   # C_ℓ^TT, ℓ = 0…6000
    "te": np.ndarray (6001,),   # C_ℓ^TE
    "ee": np.ndarray (6001,),   # C_ℓ^EE
    "pp": np.ndarray (3001,),   # C_ℓ^φφ, ℓ = 0…3000
}
```

ℓ = 0 and ℓ = 1 entries must be present but are ignored by the scorer.

## Scoring

Scoring has two components.

### Precision: MAE on Δχ²

For each pair of test cosmologies (i, j), the scorer computes the Wishart log-likelihood χ²(i,j) using the true spectra as mock data and the emulated spectra as theory. The precision score is the mean absolute error between the emulated and true χ² values, averaged over all N(N−1) = 24,995,000 off-diagonal pairs from the 5,000-point test set:

```
S_prec = mean_{i≠j} |χ²_emu(i,j) − χ²_true(i,j)|
```

TT/TE/EE are evaluated jointly via a 2×2 Wishart covariance (`mae_cmb`). φφ is evaluated separately as a scalar Wishart (`mae_pp`). The primary score is `mae_total = mae_cmb + mae_pp`.

### Speed: Mean CPU inference time

The timing score is the mean wall time per `predict()` call, measured over 1,000 single-threaded CPU calls with fresh inputs each time (to prevent caching). The soft floor is 1 ms — sub-millisecond speed provides no further benefit.

### Combined score

```
S = log10(mae_total) + log10(max(t_cpu_ms, 1.0))
```

Lower is better. One decade of precision improvement trades equally against one decade of speed improvement (above the 1 ms floor).

### Baseline

`cec.ConstantPlanck()` always returns the Planck fiducial spectrum regardless of input. It achieves `mae_total ≈ 1.13 × 10⁷` (enormous precision error) and near-zero inference time (hits the floor). A real emulator must beat this by many orders of magnitude on precision.

## Interface Requirement

Any object with a `predict(params: dict) -> dict` method. No base class or registration required. Example:

```python
class MyEmulator:
    def predict(self, params: dict) -> dict[str, np.ndarray]:
        ...
        return {"tt": tt, "te": te, "ee": ee, "pp": pp}
```

## Key Design Constraints

- **Positive definiteness**: At each ℓ, the 2×2 matrix [[C_ℓ^TT, C_ℓ^TE], [C_ℓ^TE, C_ℓ^EE]] must be positive definite (i.e., |C_ℓ^TE|² < C_ℓ^TT · C_ℓ^EE). Violations cause the Wishart likelihood to be undefined.
- **CPU timing**: The `predict()` call is timed on a single CPU thread. GPU inference is not valid for the speed benchmark. The model must be JIT-compiled for fast CPU execution.
- **No test set leakage**: The 5,000-point test set is held out for final scoring only. Do not use it for training or hyperparameter selection.

## Scoring API

```python
import cmbemu as cec

emu = MyEmulator(...)

# Use during training (fast, deterministic):
acc = cec.get_accuracy_score(emu)
print(acc['mae_total']['mae'])   # target: << 10^6

# Use after training is complete, with JAX pinned to CPU:
# os.environ['JAX_PLATFORMS'] = 'cpu'  (set before import jax)
full = cec.get_score(emu)
print(full['combined_S'])
```
