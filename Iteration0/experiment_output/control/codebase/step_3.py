# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
os.environ['JAX_PLATFORMS'] = 'cpu'
import sys
sys.path.insert(0, os.path.abspath('codebase'))
sys.path.insert(0, '/home/node/data/compsep_data/')
sys.path.insert(0, '/home/node/work/cmbemu/src/')
import time
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.serialization
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmbemu as cec
mpl.rcParams['text.usetex'] = False
class EmulatorMLP(nn.Module):
    hidden_dim: int = 512
    num_layers: int = 4
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        for _ in range(self.num_layers - 1):
            y = nn.Dense(self.hidden_dim)(x)
            y = nn.gelu(y)
            x = x + y
        out_tt = nn.Dense(6001)(x)
        out_ee = nn.Dense(6001)(x)
        out_pp = nn.Dense(3001)(x)
        out_rho = nn.Dense(6001)(x)
        return out_tt, out_ee, out_pp, out_rho
class MyEmulator:
    def __init__(self, weights_path='data/model_weights.msgpack', stats_path='data/normalization_stats.npz'):
        stats = np.load(stats_path)
        self.box_lo = jnp.array(stats['box_lo'])
        self.box_hi = jnp.array(stats['box_hi'])
        self.mean_log_tt = jnp.array(stats['mean_log_tt'])
        self.std_log_tt = jnp.array(stats['std_log_tt'])
        self.mean_log_ee = jnp.array(stats['mean_log_ee'])
        self.std_log_ee = jnp.array(stats['std_log_ee'])
        self.mean_log_pp = jnp.array(stats['mean_log_pp'])
        self.std_log_pp = jnp.array(stats['std_log_pp'])
        self.param_order = ('omega_b', 'omega_cdm', 'H0', 'tau_reio', 'ln10^{10}A_s', 'n_s')
        self.model = EmulatorMLP()
        key = jax.random.PRNGKey(0)
        dummy_x = jnp.ones((1, 6))
        variables = self.model.init(key, dummy_x)
        with open(weights_path, 'rb') as f:
            weights_bytes = f.read()
        self.params = flax.serialization.from_bytes(variables['params'], weights_bytes)
        self._predict_core_jit = jax.jit(self._predict_core)
    def _predict_core(self, x, params):
        x_norm = (x - self.box_lo) / (self.box_hi - self.box_lo)
        out_tt, out_ee, out_pp, out_rho = self.model.apply({'params': params}, x_norm)
        pred_log_tt = out_tt * self.std_log_tt + self.mean_log_tt
        pred_log_ee = out_ee * self.std_log_ee + self.mean_log_ee
        pred_log_pp = out_pp * self.std_log_pp + self.mean_log_pp
        pred_log_tt = jnp.clip(pred_log_tt, -80.0, 80.0)
        pred_log_ee = jnp.clip(pred_log_ee, -80.0, 80.0)
        pred_log_pp = jnp.clip(pred_log_pp, -80.0, 80.0)
        pred_tt = jnp.exp(pred_log_tt)
        pred_ee = jnp.exp(pred_log_ee)
        pred_pp = jnp.exp(pred_log_pp)
        pred_rho = jnp.tanh(out_rho)
        pred_te = pred_rho * jnp.sqrt(pred_tt * pred_ee + 1e-30)
        return pred_tt, pred_te, pred_ee, pred_pp
    def predict(self, params_dict: dict) -> dict:
        x = jnp.array([[params_dict[p] for p in self.param_order]])
        pred_tt, pred_te, pred_ee, pred_pp = self._predict_core_jit(x, self.params)
        return {'tt': np.array(pred_tt[0]), 'te': np.array(pred_te[0]), 'ee': np.array(pred_ee[0]), 'pp': np.array(pred_pp[0])}
if __name__ == '__main__':
    emu = MyEmulator()
    dummy_params = {p: 0.5 * (emu.box_lo[i] + emu.box_hi[i]) for i, p in enumerate(emu.param_order)}
    _ = emu.predict(dummy_params)
    acc = cec.get_accuracy_score(emu)
    full = cec.get_score(emu)
    metrics_path = os.path.join('data', 'evaluation_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write('Precision Score (mae_total): ' + str(acc['mae_total']['mae']) + '\n')
        f.write('Combined Score (S): ' + str(full['combined_S']) + '\n')
        f.write('Timing (mean ms): ' + str(full['timing']['t_cpu_ms_mean']) + '\n')
    test_data = cec.generate_data(n=20, seed=42)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    spectra_keys = ['tt', 'te', 'ee', 'pp']
    for i, key in enumerate(spectra_keys):
        ax = axes[i // 2, i % 2]
        for j in range(20):
            params = {p: test_data['params'][j, k] for k, p in enumerate(emu.param_order)}
            pred = emu.predict(params)[key]
            true = test_data[key][j]
            if key == 'te':
                true_tt = test_data['tt'][j]
                true_ee = test_data['ee'][j]
                norm = np.sqrt(true_tt * true_ee + 1e-30)
                rel_error = (pred[2:] - true[2:]) / norm[2:]
            else:
                rel_error = (pred[2:] - true[2:]) / (true[2:] + 1e-30)
            ell = np.arange(2, len(pred))
            ax.plot(ell, rel_error, alpha=0.5)
        ax.set_title('Relative Residuals: ' + key.upper())
        ax.set_xlabel('Multipole ell')
        ax.set_ylabel('(Pred - True) / True')
        ax.axhline(0, color='black', linestyle='--', alpha=0.8)
        ax.set_ylim(-0.1, 0.1)
    plt.tight_layout()
    timestamp = str(int(time.time()))
    plot_path = os.path.join('data', 'residuals_1_' + timestamp + '.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()