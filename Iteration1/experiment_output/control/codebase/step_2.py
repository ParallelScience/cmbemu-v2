# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

def transform_targets(tt, te, ee, pp):
    log_tt = np.log(tt)
    log_ee = np.log(ee)
    log_pp = np.log(pp)
    rho = te / np.sqrt(tt * ee)
    rho_clipped = np.clip(rho, -0.999999, 0.999999)
    atanh_rho = np.arctanh(rho_clipped)
    return np.concatenate([log_tt, atanh_rho, log_ee, log_pp], axis=-1)

def inverse_transform_targets(y):
    log_tt = y[..., 0:5999]
    atanh_rho = y[..., 5999:11998]
    log_ee = y[..., 11998:17997]
    log_pp = y[..., 17997:20996]
    C_tt = np.exp(np.clip(log_tt, -700, 700))
    C_ee = np.exp(np.clip(log_ee, -700, 700))
    C_pp = np.exp(np.clip(log_pp, -700, 700))
    rho = np.tanh(atanh_rho)
    C_te = rho * np.sqrt(C_tt * C_ee)
    return C_tt, C_te, C_ee, C_pp

class MLP(nn.Module):
    hidden_dims: list
    output_dim: int
    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x

class Emulator:
    def __init__(self, model_params=None, norm_stats_path='data/norm_stats.npz'):
        stats = np.load(norm_stats_path, allow_pickle=True)
        self.box_lo = stats['box_lo']
        self.box_hi = stats['box_hi']
        self.targets_mean = stats['targets_mean']
        self.targets_std = stats['targets_std']
        self.param_names = [name.decode('utf-8') if isinstance(name, bytes) else name for name in stats['param_names']]
        self.model = MLP(hidden_dims=[1024, 1024, 1024, 1024, 1024], output_dim=20996)
        if model_params is None:
            key = jax.random.PRNGKey(0)
            self.model_params = self.model.init(key, jnp.ones((6,)))
        else:
            self.model_params = model_params
        @jax.jit
        def forward_pass(params, x):
            return self.model.apply(params, x)
        self._forward_pass = forward_pass

    def predict(self, params_dict: dict) -> dict:
        x = np.array([params_dict[name] for name in self.param_names], dtype=np.float32)
        x_norm = (x - self.box_lo) / (self.box_hi - self.box_lo)
        y_norm = self._forward_pass(self.model_params, jnp.array(x_norm))
        y_norm = np.array(y_norm)
        y = y_norm * self.targets_std + self.targets_mean
        y_f64 = y.astype(np.float64)
        C_tt_l2, C_te_l2, C_ee_l2, C_pp_l2 = inverse_transform_targets(y_f64)
        C_tt = np.zeros(6001, dtype=np.float64)
        C_te = np.zeros(6001, dtype=np.float64)
        C_ee = np.zeros(6001, dtype=np.float64)
        C_pp = np.zeros(3001, dtype=np.float64)
        C_tt[2:] = C_tt_l2
        C_te[2:] = C_te_l2
        C_ee[2:] = C_ee_l2
        C_pp[2:] = C_pp_l2
        return {'tt': C_tt, 'te': C_te, 'ee': C_ee, 'pp': C_pp}

if __name__ == '__main__':
    emu = Emulator()
    dummy_params = {'omega_b': 0.022, 'omega_cdm': 0.12, 'H0': 67.5, 'tau_reio': 0.06, 'ln10^{10}A_s': 3.05, 'n_s': 0.965}
    preds = emu.predict(dummy_params)
    print('Prediction successful.')
    print('TT shape: ' + str(preds['tt'].shape) + ', dtype: ' + str(preds['tt'].dtype))