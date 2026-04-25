# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
os.environ['JAX_PLATFORMS'] = 'cuda'
import jax
import sys
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
import optax
import time
from flax.training import train_state
from functools import partial
import cmbemu as cec
if __name__ == '__main__':
    assert len([d for d in jax.devices() if 'cuda' in str(d).lower()]) > 0, 'GPU not found'
    print('Training on:', jax.devices())
    sys.path.insert(0, os.path.abspath('codebase'))
    TRAIN_PATH = '/home/node/.cache/cmbemu/datasets--borisbolliet--cmbemu-competition-v1/snapshots/e45fd9a3f451038e3ce677a56f7f6cb81c8c47c9/train.npz'
    DATA_DIR = 'data/'
    PARAM_ORDER = ('omega_b', 'omega_cdm', 'H0', 'tau_reio', 'ln10^{10}A_s', 'n_s')
    class CMBTrunk(nn.Module):
        depth: int
        width: int
        @nn.compact
        def __call__(self, x):
            for _ in range(self.depth):
                x = nn.Dense(self.width)(x)
                x = nn.LayerNorm()(x)
                x = nn.silu(x)
            return x
    class CMBEmulatorModel(nn.Module):
        trunk_depth: int
        trunk_width: int
        head_width: int = 256
        @nn.compact
        def __call__(self, x):
            trunk_out = CMBTrunk(depth=self.trunk_depth, width=self.trunk_width)(x)
            tt_h = nn.silu(nn.Dense(self.head_width)(trunk_out))
            tt_out = nn.Dense(6001)(tt_h)
            ee_h = nn.silu(nn.Dense(self.head_width)(trunk_out))
            ee_out = nn.Dense(6001)(ee_h)
            pp_h = nn.silu(nn.Dense(self.head_width)(trunk_out))
            pp_out = nn.Dense(3001)(pp_h)
            rho_h = nn.silu(nn.Dense(self.head_width)(trunk_out))
            rho_out = nn.Dense(6001)(rho_h)
            return tt_out, ee_out, pp_out, rho_out
    def compute_loss(params, model, batch_params, batch_tt_res, batch_ee_res, batch_pp_res, batch_rho, weights, alpha):
        tt_pred, ee_pred, pp_pred, rho_pred = model.apply(params, batch_params)
        loss_tt = jnp.mean((tt_pred - batch_tt_res) ** 2)
        loss_ee = jnp.mean((ee_pred - batch_ee_res) ** 2)
        loss_pp = jnp.mean((pp_pred - batch_pp_res) ** 2)
        loss_rho = jnp.mean((rho_pred - batch_rho) ** 2)
        data_loss = (weights['tt'] * loss_tt + weights['ee'] * loss_ee + weights['pp'] * loss_pp + weights['rho'] * loss_rho)
        teacher_loss = (weights['tt'] * jnp.mean(tt_pred ** 2) + weights['ee'] * jnp.mean(ee_pred ** 2) + weights['pp'] * jnp.mean(pp_pred ** 2) + weights['rho'] * jnp.mean(rho_pred ** 2))
        return (1.0 - alpha) * data_loss + alpha * teacher_loss
    @partial(jax.jit, static_argnums=(1,))
    def train_step(state, model, batch_params, batch_tt_res, batch_ee_res, batch_pp_res, batch_rho, weights, alpha):
        def loss_fn(params):
            return compute_loss(params, model, batch_params, batch_tt_res, batch_ee_res, batch_pp_res, batch_rho, weights, alpha)
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss
    print('Training pipeline initialized.')