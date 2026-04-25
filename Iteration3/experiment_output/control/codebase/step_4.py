# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
os.environ['JAX_PLATFORMS'] = 'cuda'
sys.path.insert(0, '/home/node/work/cmbemu/src/')
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import flax.serialization
import optax
import time
from flax.training import train_state
from functools import partial
import cmbemu as cec
DATA_DIR = 'data/'
TRAIN_PATH = '/home/node/.cache/cmbemu/datasets--borisbolliet--cmbemu-competition-v1/snapshots/e45fd9a3f451038e3ce677a56f7f6cb81c8c47c9/train.npz'
PARAM_ORDER = ('omega_b', 'omega_cdm', 'H0', 'tau_reio', 'ln10^{10}A_s', 'n_s')
BOX_LO = np.array([0.020, 0.090, 55.0, 0.030, 2.700, 0.920], dtype=np.float32)
BOX_HI = np.array([0.025, 0.150, 85.0, 0.100, 3.300, 1.020], dtype=np.float32)
TRUNK_DEPTH = 5
TRUNK_WIDTH = 1024
HEAD_WIDTH = 256
BATCH_SIZE = 512
EPOCHS_PHASE1 = 500
MAE_EVERY = 20
VAL_SIZE = 5000
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
        t = x
        tt_out = nn.Dense(6001)(nn.silu(nn.Dense(self.head_width)(t)))
        ee_out = nn.Dense(6001)(nn.silu(nn.Dense(self.head_width)(t)))
        pp_out = nn.Dense(3001)(nn.silu(nn.Dense(self.head_width)(t)))
        rho_out = nn.Dense(6001)(nn.silu(nn.Dense(self.head_width)(t)))
        return tt_out, ee_out, pp_out, rho_out
class CMBEmulator:
    def __init__(self, model, params, baselines):
        self.model = model
        self.params = params
        self.bl_log_tt = np.array(baselines['log_tt'], dtype=np.float32)
        self.bl_log_ee = np.array(baselines['log_ee'], dtype=np.float32)
        self.bl_log_pp = np.array(baselines['log_pp'], dtype=np.float32)
        self.bl_rho = np.array(baselines['rho'], dtype=np.float32)
        self._fwd = jax.jit(lambda p, x: self.model.apply(p, x))
    def predict(self, params_dict):
        x = np.array([[params_dict[k] for k in PARAM_ORDER]], dtype=np.float32)
        x_norm = 2.0 * (x - BOX_LO) / (BOX_HI - BOX_LO) - 1.0
        tt_r, ee_r, pp_r, rho_r = self._fwd(self.params, jnp.array(x_norm))
        log_tt = (np.array(tt_r[0]) + self.bl_log_tt).astype(np.float64)
        log_ee = (np.array(ee_r[0]) + self.bl_log_ee).astype(np.float64)
        log_pp = (np.array(pp_r[0]) + self.bl_log_pp).astype(np.float64)
        rho_raw = (np.array(rho_r[0]) + self.bl_rho).astype(np.float64)
        C_tt = np.exp(np.clip(log_tt, -700, 700))
        C_ee = np.exp(np.clip(log_ee, -700, 700))
        C_pp = np.exp(np.clip(log_pp, -700, 700))
        C_te = np.tanh(rho_raw) * np.sqrt(C_tt * C_ee)
        return {'tt': C_tt, 'te': C_te, 'ee': C_ee, 'pp': C_pp}
def prep(data, baselines):
    x = (2.0 * (data['params'].astype(np.float32) - BOX_LO) / (BOX_HI - BOX_LO) - 1.0)
    tt, ee, pp = data['tt'], data['ee'], data['pp']
    f_tt = float(np.min(tt[:, 2:][tt[:, 2:] > 0])) * 1e-3
    f_ee = float(np.min(ee[:, 2:][ee[:, 2:] > 0])) * 1e-3
    f_pp = float(np.min(pp[:, 2:][pp[:, 2:] > 0])) * 1e-3
    log_tt = np.log(np.where(tt > 0, tt, f_tt).astype(np.float64))
    log_ee = np.log(np.where(ee > 0, ee, f_ee).astype(np.float64))
    log_pp = np.log(np.where(pp > 0, pp, f_pp).astype(np.float64))
    denom = np.sqrt(np.maximum(tt.astype(np.float64) * ee.astype(np.float64), 0.0))
    safe = np.where(denom > 0, denom, 1.0)
    rho = np.clip(np.where(denom > 0, data['te'].astype(np.float64) / safe, 0.0), -1.0, 1.0)
    tt_r = (log_tt - baselines['log_tt']).astype(np.float32)
    ee_r = (log_ee - baselines['log_ee']).astype(np.float32)
    pp_r = (log_pp - baselines['log_pp']).astype(np.float32)
    rho_r = (rho - baselines['rho']).astype(np.float32)
    vt, ve, vp, vr = float(np.var(tt_r[:, 2:])), float(np.var(ee_r[:, 2:])), float(np.var(pp_r[:, 2:])), float(np.var(rho_r[:, 2:]))
    s = 1/vt + 1/ve + 1/vp + 1/vr
    w = {'tt': (1/vt)/s, 'ee': (1/ve)/s, 'pp': (1/vp)/s, 'rho': (1/vr)/s}
    return x, tt_r, ee_r, pp_r, rho_r, w
@partial(jax.jit, static_argnums=(0,))
def tstep(model, state, xb, ttb, eeb, ppb, rhob, wt, we, wp, wr):
    def lf(p):
        tp, ep, pp_, rp = model.apply(p, xb)
        return wt*jnp.mean((tp-ttb)**2) + we*jnp.mean((ep-eeb)**2) + wp*jnp.mean((pp_-ppb)**2) + wr*jnp.mean((rp-rhob)**2)
    loss, g = jax.value_and_grad(lf)(state.params)
    return state.apply_gradients(grads=g), loss
def save_w(params, path):
    with open(path, 'wb') as f: f.write(flax.serialization.to_bytes(params))
if __name__ == '__main__':
    baselines = dict(np.load(os.path.join(DATA_DIR, 'baselines.npz')))
    raw = np.load(TRAIN_PATH, allow_pickle=False)
    data_tr = {k: raw[k][np.arange(raw['params'].shape[0])[VAL_SIZE:]] for k in ['params', 'tt', 'te', 'ee', 'pp']}
    x_tr, tt_tr, ee_tr, pp_tr, rho_tr, weights = prep(data_tr, baselines)
    model = CMBEmulatorModel(trunk_depth=TRUNK_DEPTH, trunk_width=TRUNK_WIDTH, head_width=HEAD_WIDTH)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 6), dtype=jnp.float32))
    print('Training initialized.')