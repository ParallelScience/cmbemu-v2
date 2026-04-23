# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import json
sys.path.insert(0, '/home/node/work/cmbemu/src/')
import cmbemu as cec
class FastStudentEmulator:
    def __init__(self, model_path, data_path):
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        self.params = model_data['params']
        self.hidden_dim = model_data['hidden_dim']
        self.num_layers = model_data['num_layers']
        data = np.load(data_path)
        self.box_lo = data['box_lo'].astype(np.float32)
        self.box_hi = data['box_hi'].astype(np.float32)
        self.param_names = data['param_names']
        self.weights = []
        self.biases = []
        for i in range(self.num_layers + 1):
            layer_name = 'Dense_' + str(i)
            self.weights.append(jnp.array(self.params[layer_name]['kernel']))
            self.biases.append(jnp.array(self.params[layer_name]['bias']))
        @jax.jit
        def forward_pass(x):
            for i in range(self.num_layers):
                x = jnp.dot(x, self.weights[i]) + self.biases[i]
                x = jax.nn.gelu(x)
            x = jnp.dot(x, self.weights[-1]) + self.biases[-1]
            return x
        self.forward_pass = forward_pass
    def predict(self, params_dict):
        p_array = np.array([params_dict[str(k)] for k in self.param_names], dtype=np.float32)
        p_norm = (p_array - self.box_lo) / (self.box_hi - self.box_lo)
        preds = np.asarray(self.forward_pass(p_norm))
        log_tt_f64 = preds[:6001].astype(np.float64)
        log_ee_f64 = preds[6001:12002].astype(np.float64)
        log_pp_f64 = preds[12002:15003].astype(np.float64)
        arctanh_rho_f64 = preds[15003:].astype(np.float64)
        C_tt = np.exp(np.clip(log_tt_f64, -700.0, 700.0))
        C_ee = np.exp(np.clip(log_ee_f64, -700.0, 700.0))
        C_pp = np.exp(np.clip(log_pp_f64, -700.0, 700.0))
        rho = np.tanh(arctanh_rho_f64)
        C_te = rho * np.sqrt(np.maximum(C_tt * C_ee, 1e-300))
        return {'tt': C_tt, 'te': C_te, 'ee': C_ee, 'pp': C_pp}
if __name__ == '__main__':
    emu = FastStudentEmulator('data/student_model.pkl', 'data/distillation_data.npz')
    dummy_params = {str(k): float((emu.box_lo[i] + emu.box_hi[i])/2) for i, k in enumerate(emu.param_names)}
    _ = emu.predict(dummy_params)
    acc = cec.get_accuracy_score(emu)
    tim = cec.get_time_score(emu)
    full = cec.get_score(emu)
    metrics = {'mae_total': float(acc['mae_total']['mae']), 'mae_cmb': float(acc['mae_cmb']['mae']), 'mae_pp': float(acc['mae_pp']['mae']), 't_cpu_ms_mean': float(tim['t_cpu_ms_mean']), 't_cpu_ms_median': float(tim['t_cpu_ms_median']), 't_cpu_ms_std': float(tim['t_cpu_ms_std']), 'combined_S': float(full['combined_S'])}
    with open('data/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    train_data = cec.load_train()
    val_params = train_data['params'][45000:]
    val_tt = train_data['tt'][45000:]
    val_te = train_data['te'][45000:]
    val_ee = train_data['ee'][45000:]
    val_pp = train_data['pp'][45000:]
    res_tt, res_te, res_ee, res_pp = [], [], [], []
    for i in range(len(val_params)):
        p_dict = {str(k): float(val_params[i, j]) for j, k in enumerate(emu.param_names)}
        preds = emu.predict(p_dict)
        with np.errstate(divide='ignore', invalid='ignore'):
            r_tt = (preds['tt'] - val_tt[i]) / val_tt[i]
            r_te = (preds['te'] - val_te[i]) / val_te[i]
            r_ee = (preds['ee'] - val_ee[i]) / val_ee[i]
            r_pp = (preds['pp'] - val_pp[i]) / val_pp[i]
        res_tt.append(np.nan_to_num(r_tt, nan=0.0, posinf=0.0, neginf=0.0))
        res_te.append(np.nan_to_num(r_te, nan=0.0, posinf=0.0, neginf=0.0))
        res_ee.append(np.nan_to_num(r_ee, nan=0.0, posinf=0.0, neginf=0.0))
        res_pp.append(np.nan_to_num(r_pp, nan=0.0, posinf=0.0, neginf=0.0))
    np.savez('data/residuals.npz', res_tt=np.array(res_tt), res_te=np.array(res_te), res_ee=np.array(res_ee), res_pp=np.array(res_pp))