# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
os.environ['JAX_PLATFORMS'] = 'cpu'
import sys
sys.path.insert(0, '/home/node/work/cmbemu/src/')
import cmbemu as cec
import time
import json
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.serialization
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['text.usetex'] = False
class EmulatorMLP(nn.Module):
    hidden_dim: int
    num_layers: int
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
    def __init__(self, hidden_dim, num_layers, weights_path, stats_path):
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
        self.model = EmulatorMLP(hidden_dim=hidden_dim, num_layers=num_layers)
        key = jax.random.PRNGKey(0)
        dummy_x = jnp.ones((1, 6))
        variables = self.model.init(key, dummy_x)
        with open(weights_path, 'rb') as f:
            weights_bytes = f.read()
        self.params = flax.serialization.from_bytes(variables['params'], weights_bytes)
        self._predict_core_jit = jax.jit(self._predict_core)
        dummy_params = {p: 0.5 * (float(self.box_lo[i]) + float(self.box_hi[i])) for i, p in enumerate(self.param_order)}
        self.predict(dummy_params)
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
    models_info = [{'name': 'Original (512x4)', 'hidden_dim': 512, 'num_layers': 4, 'weights': os.path.join('data', 'model_weights.msgpack'), 'stats': os.path.join('data', 'normalization_stats.npz')}, {'name': 'Refined 256x3', 'hidden_dim': 256, 'num_layers': 3, 'weights': os.path.join('data', 'model_weights_256_3.msgpack'), 'stats': os.path.join('data', 'normalization_stats_256_3.npz')}, {'name': 'Refined 512x2', 'hidden_dim': 512, 'num_layers': 2, 'weights': os.path.join('data', 'model_weights_512_2.msgpack'), 'stats': os.path.join('data', 'normalization_stats_512_2.npz')}]
    results = []
    for info in models_info:
        print('Evaluating ' + info['name'] + '...')
        emu = MyEmulator(info['hidden_dim'], info['num_layers'], info['weights'], info['stats'])
        full_score = cec.get_score(emu)
        mae_total = full_score['mae_total']['mae']
        t_cpu_ms = full_score['timing']['t_cpu_ms_mean']
        combined_S = full_score['combined_S']
        print('  MAE Total: ' + str(mae_total))
        print('  Inference Time: ' + str(t_cpu_ms) + ' ms')
        print('  Combined Score: ' + str(combined_S) + '\n')
        results.append({'name': info['name'], 'mae_total': mae_total, 't_cpu_ms': t_cpu_ms, 'combined_S': combined_S})
    best_model = min(results, key=lambda x: x['combined_S'])
    print('Best Model: ' + best_model['name'] + ' with Combined Score: ' + str(best_model['combined_S']))
    results_path = os.path.join('data', 'benchmark_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print('Benchmark results saved to ' + results_path)
    names = [r['name'] for r in results]
    mae_totals = [r['mae_total'] for r in results]
    times = [r['t_cpu_ms'] for r in results]
    scores = [r['combined_S'] for r in results]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].bar(names, mae_totals, color='skyblue')
    axes[0].set_title('Precision Score (MAE Total)')
    axes[0].set_ylabel('MAE Total (Lower is better)')
    axes[0].set_yscale('log')
    axes[1].bar(names, times, color='lightgreen')
    axes[1].set_title('Inference Time')
    axes[1].set_ylabel('Time (ms) (Lower is better)')
    axes[1].axhline(1.0, color='red', linestyle='--', label='1 ms floor')
    axes[1].legend()
    axes[2].bar(names, scores, color='salmon')
    axes[2].set_title('Combined Score (S)')
    axes[2].set_ylabel('Score (Lower is better)')
    for ax in axes:
        ax.tick_params(axis='x', rotation=15)
    plt.tight_layout()
    timestamp = str(int(time.time()))
    plot_path = os.path.join('data', 'model_comparison_1_' + timestamp + '.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print('Comparison plot saved to ' + plot_path)
    print('Step 5 completed successfully.')