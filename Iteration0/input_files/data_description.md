# CMB Power Spectrum Emulator — Data Description

## Competition Goal

Train a neural-network emulator that maps 6 ΛCDM cosmological parameters to four CMB angular power spectra (TT, TE, EE, φφ). The emulator is scored on two criteria: precision at the likelihood level, and CPU inference speed. The combined score is:

```
S = log10(mae_total) + log10(max(t_cpu_ms, 1.0))
```

Lower is better. The scoring function is `cmbemu.get_score(emulator)`.

---

## Package Access

```python
import sys
sys.path.insert(0, '/home/node/work/cmbemu/src/')
import cmbemu as cec
```

---

## Training Data

50,000 cosmologies, pre-cached locally (no download needed):

```python
train = cec.load_train()
```

Or load directly:
```python
import numpy as np
train = np.load(
    '/home/node/.cache/cmbemu/datasets--borisbolliet--cmbemu-competition-v1'
    '/snapshots/e45fd9a3f451038e3ce677a56f7f6cb81c8c47c9/train.npz',
    allow_pickle=False
)
```

### Shapes and units

| Key | Shape | Description |
|-----|-------|-------------|
| `params` | (50000, 6) | Cosmological parameters, float32, physical units |
| `tt` | (50000, 6001) | C_ℓ^TT, ℓ = 0…6000, units K² |
| `te` | (50000, 6001) | C_ℓ^TE, ℓ = 0…6000 |
| `ee` | (50000, 6001) | C_ℓ^EE, ℓ = 0…6000 |
| `pp` | (50000, 3001) | C_ℓ^φφ, ℓ = 0…3000 |
| `box_lo` | (6,) | Parameter box lower bounds |
| `box_hi` | (6,) | Parameter box upper bounds |
| `param_names` | (6,) | Parameter name strings |

ℓ = 0 and ℓ = 1 entries are present but ignored by the scorer. Only ℓ ≥ 2 is evaluated.

### Parameter ranges

| Parameter | Low | High |
|-----------|-----|------|
| omega_b | 0.020 | 0.025 |
| omega_cdm | 0.090 | 0.150 |
| H0 | 55.0 | 85.0 |
| tau_reio | 0.030 | 0.100 |
| ln10^{10}A_s | 2.700 | 3.300 |
| n_s | 0.920 | 1.020 |

---

## Emulator Interface

Any Python object with a `predict(params: dict) -> dict` method:

```python
class MyEmulator:
    def predict(self, params: dict) -> dict:
        # Input: dict with keys matching PARAM_NAMES, float values
        # Output: dict with keys 'tt', 'te', 'ee', 'pp'
        #   tt, te, ee: np.ndarray shape (6001,)
        #   pp:         np.ndarray shape (3001,)
        ...
```

Parameter names in the input dict (canonical order):
```python
PARAM_ORDER = ('omega_b', 'omega_cdm', 'H0', 'tau_reio', 'ln10^{10}A_s', 'n_s')
```

---

## Scoring API

```python
# Precision only (~2 s, deterministic — use this during training)
acc = cec.get_accuracy_score(emu)
# Returns: {'mae_total': {'mae': ..., ...}, 'mae_cmb': {...}, 'mae_pp': {...}, ...}

# CPU timing only (1000 single-threaded calls)
tim = cec.get_time_score(emu)
# Returns: {'t_cpu_ms_mean': ..., 't_cpu_ms_median': ..., 't_cpu_ms_std': ...}

# Combined score (runs both above)
full = cec.get_score(emu)
# Returns: {'combined_S': ..., 'mae_total': {...}, 'timing': {...}, ...}
```

Baseline reference: `cec.ConstantPlanck()` gives `mae_total ≈ 1.13 × 10⁷`.

**`mae_total` is the primary validation metric.** Do not use MSE loss on spectra as a proxy — it does not correlate reliably with the Wishart likelihood score. Instead, call `cec.get_accuracy_score(emu)` every N epochs (e.g. every 10–20 epochs) and use `acc['mae_total']['mae']` to monitor and guide training. A well-trained emulator should reach `mae_total << 10⁶` (orders of magnitude below the ConstantPlanck baseline of 1.13×10⁷). Target: `mae_total < 10⁴`.

**`get_accuracy_score` uses the held-out 5,000-point test set.** To avoid test set leakage during hyperparameter tuning, reserve a separate validation split (e.g. 5,000 cosmologies) from the training data for intermediate monitoring. Only call `get_accuracy_score` at the end of each training run or sparingly during training.

**Important:** `cec.get_time_score` and `cec.get_score` require JAX pinned to CPU (scorer uses single-threaded CPU timing). Set `os.environ['JAX_PLATFORMS'] = 'cpu'` at the top of any benchmarking script.

### Additional data generation

```python
extra = cec.generate_data(n=50000, seed=99)
# Returns same dict format as load_train()
```

---

## MANDATORY: GPU Setup for Training

Every training script must begin with the following lines **before any other imports**:

```python
import os
os.environ['JAX_PLATFORMS'] = 'cuda'   # must come before import jax

import jax
assert len([d for d in jax.devices() if 'cuda' in str(d).lower()]) > 0, \
    f"GPU not found — devices = {jax.devices()}"
print("Training on:", jax.devices())
# Expected output: Training on: [CudaDevice(id=0)]
```

This snippet is tested and confirmed working in this environment:
- 1× NVIDIA RTX PRO 6000 Blackwell, 96 GB VRAM, CUDA 13.0
- JAX 0.10.0 + jax-cuda12-plugin 0.10.0

The env var **must** be set before `import jax`. Setting it after has no effect.

### MANDATORY: float64 in predict()

The CMB spectra span 13 orders of magnitude (C_ℓ^TT ~ 10⁻¹⁰ at the first acoustic peak, dropping to ~10⁻²¹ at ℓ=6000; C_ℓ^EE drops to ~10⁻²³). The Wishart likelihood evaluates the log-determinant of the 2×2 covariance matrix, which involves products like C_ℓ^TT × C_ℓ^EE ~ 10⁻⁴⁴. This is below the float32 minimum normal (~1.18×10⁻³⁸), causing underflow to zero and log(0) = −∞ → NaN in the scorer.

**The predict() method MUST cast all spectral outputs to float64 before returning.** The neural network forward pass can use float32 (for speed), but the final exponentiation and output arrays must be float64:

```python
def predict(self, params_dict):
    # ... float32 NN forward pass ...
    log_tt_f64 = log_tt.astype(np.float64)
    log_ee_f64 = log_ee.astype(np.float64)
    log_pp_f64 = log_pp.astype(np.float64)
    C_tt = np.exp(np.clip(log_tt_f64, -700, 700))  # float64
    C_ee = np.exp(np.clip(log_ee_f64, -700, 700))  # float64
    C_pp = np.exp(np.clip(log_pp_f64, -700, 700))  # float64
    rho  = np.tanh(rho_output.astype(np.float64))  # float64
    C_te = rho * np.sqrt(C_tt * C_ee)              # float64
    # Return float64 arrays — the scorer requires this
    return {'tt': C_tt, 'te': C_te, 'ee': C_ee, 'pp': C_pp}
```

Returning float32 arrays will produce NaN in `get_accuracy_score` for all models. This has been verified empirically.

---

## Hardware

- GPU: 1× NVIDIA RTX PRO 6000 Blackwell, 96 GB VRAM
- CPU: 64 cores, 128 GB RAM
- CUDA 13.0, JAX 0.10.0, Flax 0.12.6, Optax 0.2.8
