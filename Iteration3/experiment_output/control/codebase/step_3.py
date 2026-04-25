# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
os.environ['JAX_PLATFORMS'] = 'cuda'
import sys
sys.path.insert(0, '/home/node/work/cmbemu/src/')
sys.path.insert(0, os.path.abspath('codebase'))
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import flax.serialization
import optax
import json
from flax.training import train_state
from functools import partial
import cmbemu as cec
DATA_DIR = 'data/'
PARAM_ORDER = ('omega_b', 'omega_cdm', 'H0', 'tau_reio', 'ln10^{10}A_s', 'n_s')
BOX_LO = np.array([0.020, 0.090, 55.0, 0.030, 2.700, 0.920], dtype=np.float32)
BOX_HI = np.array([0.025, 0.150, 85.0, 0.100, 3.300, 1.020], dtype=np.float32)
class CMBEmulatorModel(nn.Module):
    trunk_depth: int
    trunk_width: int
    head_width: int = 256
    @nn.compact
    def __call__(self, x):
        for _ in range(self.trunk_depth):
            x = nn.Dense(self.trunk_width)(x)
            x = nn.LayerNorm()(x)
            x = nn.silu(x)
        trunk_out = x
        tt_h = nn.silu(nn.Dense(self.head_width)(trunk_out))
        tt_out = nn.Dense(6001)(tt_h)
        ee_h = nn.silu(nn.Dense(self.head_width)(trunk_out))
        ee_out = nn.Dense(6001)(ee_h)
        pp_h = nn.silu(nn.Dense(self.head_width)(trunk_out))
        pp_out = nn.Dense(3001)(pp_h)
        rho_h = nn.silu(nn.Dense(self.head_width)(trunk_out))
        rho_out = nn.Dense(6001)(rho_h)
        return tt_out, ee_out, pp_out, rho_out
class CMBEmulator:
    def __init__(self, model, params, baselines):
        self.model = model
        self.params = params
        self.baselines = {'log_tt': np.array(baselines['log_tt'], dtype=np.float32), 'log_ee': np.array(baselines['log_ee'], dtype=np.float32), 'log_pp': np.array(baselines['log_pp'], dtype=np.float32), 'rho': np.array(baselines['rho'], dtype=np.float32)}
        self._jit_predict = jax.jit(self._forward)
    def _forward(self, params_jax, x):
        return self.model.apply(params_jax, x)
    def predict(self, params_dict):
        x = np.array([[params_dict[k] for k in PARAM_ORDER]], dtype=np.float32)
        x_norm = (2.0 * (x - BOX_LO) / (BOX_HI - BOX_LO) - 1.0)
        x_jax = jnp.array(x_norm)
        tt_res, ee_res, pp_res, rho_res = self._jit_predict(self.params, x_jax)
        log_tt = (np.array(tt_res[0]) + self.baselines['log_tt']).astype(np.float64)
        log_ee = (np.array(ee_res[0]) + self.baselines['log_ee']).astype(np.float64)
        log_pp = (np.array(pp_res[0]) + self.baselines['log_pp']).astype(np.float64)
        rho_raw = (np.array(rho_res[0]) + self.baselines['rho']).astype(np.float64)
        C_tt = np.exp(np.clip(log_tt, -700, 700))
        C_ee = np.exp(np.clip(log_ee, -700, 700))
        C_pp = np.exp(np.clip(log_pp, -700, 700))
        rho = np.tanh(rho_raw)
        C_te = rho * np.sqrt(C_tt * C_ee)
        return {'tt': C_tt, 'te': C_te, 'ee': C_ee, 'pp': C_pp}
if __name__ == '__main__':
    model = CMBEmulatorModel(trunk_depth=5, trunk_width=1024)
    baselines = np.load(os.path.join(DATA_DIR, 'baselines.npz'))
    key = jax.random.PRNGKey(0)
    dummy = jnp.ones((1, 6), dtype=jnp.float32)
    init_params = model.init(key, dummy)
    emu = CMBEmulator(model, init_params, baselines)
    acc = cec.get_accuracy_score(emu)
    print('Initial Accuracy (Fresh Model):', acc['mae_total']['mae'])
    print('Step 3 complete.')