# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
os.environ['JAX_PLATFORMS'] = 'cuda'
import jax
assert len([d for d in jax.devices() if 'cuda' in str(d).lower()]) > 0, 'GPU not found'
import sys
import pickle
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from flax.core import unfreeze
import optax
import jax.tree_util
sys.path.insert(0, '/home/node/work/cmbemu/src/')
import cmbemu as cec
class StudentMLP(nn.Module):
    hidden_dim: int
    num_layers: int
    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.gelu(x)
        x = nn.Dense(21004)(x)
        return x
class StudentEmulator:
    def __init__(self, state, box_lo, box_hi, param_names, force_cpu=False):
        self.box_lo = box_lo
        self.box_hi = box_hi
        self.param_names = param_names
        self.force_cpu = force_cpu
        self.apply_fn = state.apply_fn
        if force_cpu:
            cpu_device = jax.devices('cpu')[0]
            self.params = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_device), state.params)
            def predict_fn(params_array):
                return self.apply_fn({'params': self.params}, params_array)
            self.predict_fn = jax.jit(predict_fn, backend='cpu')
        else:
            self.params = state.params
            def predict_fn(params_array):
                return self.apply_fn({'params': self.params}, params_array)
            self.predict_fn = jax.jit(predict_fn)
    def predict(self, params_dict):
        p_array = np.array([params_dict[str(k)] for k in self.param_names], dtype=np.float32)
        p_norm = (p_array - self.box_lo) / (self.box_hi - self.box_lo)
        p_norm = p_norm.reshape(1, -1)
        preds = self.predict_fn(p_norm)
        preds = np.array(preds)[0]
        log_tt = preds[:6001]
        log_ee = preds[6001:12002]
        log_pp = preds[12002:15003]
        arctanh_rho = preds[15003:]
        log_tt_f64 = log_tt.astype(np.float64)
        log_ee_f64 = log_ee.astype(np.float64)
        log_pp_f64 = log_pp.astype(np.float64)
        arctanh_rho_f64 = arctanh_rho.astype(np.float64)
        C_tt = np.exp(np.clip(log_tt_f64, -700, 700))
        C_ee = np.exp(np.clip(log_ee_f64, -700, 700))
        C_pp = np.exp(np.clip(log_pp_f64, -700, 700))
        rho = np.tanh(arctanh_rho_f64)
        C_te = rho * np.sqrt(np.maximum(C_tt * C_ee, 1e-300))
        return {'tt': C_tt, 'te': C_te, 'ee': C_ee, 'pp': C_pp}
def make_cosine_restarts_schedule(init_value, decay_steps, num_restarts):
    steps_per_cycle = decay_steps // num_restarts
    schedules = []
    boundaries = []
    for i in range(num_restarts):
        schedules.append(optax.cosine_decay_schedule(init_value, steps_per_cycle))
        if i < num_restarts - 1:
            boundaries.append(steps_per_cycle * (i + 1))
    return optax.join_schedules(schedules, boundaries)
class DummyState:
    def __init__(self, params, apply_fn):
        self.params = params
        self.apply_fn = apply_fn
if __name__ == '__main__':
    data = np.load('data/distillation_data.npz')
    train_params_norm = data['train_params_norm']
    train_targets = data['train_targets']
    synth_params_norm = data['synth_params_norm']
    synth_targets = data['synth_targets']
    box_lo = data['box_lo']
    box_hi = data['box_hi']
    param_names = data['param_names']
    target_variances = np.var(train_targets, axis=0)
    weights = 1.0 / (target_variances + 1e-8)
    weights = weights / np.mean(weights)
    weights = jnp.array(weights, dtype=jnp.float32)
    def check_timing(hidden_dim, num_layers):
        model = StudentMLP(hidden_dim=hidden_dim, num_layers=num_layers)
        key = jax.random.PRNGKey(0)
        variables = model.init(key, jnp.ones((1, 6)))
        state = DummyState(variables['params'], model.apply)
        emu = StudentEmulator(state, box_lo, box_hi, param_names, force_cpu=True)
        dummy_params = {k: float((box_lo[i] + box_hi[i])/2) for i, k in enumerate(param_names)}
        emu.predict(dummy_params)
        tim = cec.get_time_score(emu)
        return tim['t_cpu_ms_mean']
    architectures = [(128, 3), (256, 3), (512, 3), (512, 4), (1024, 3)]
    best_hidden_dim, best_num_layers, best_t_ms = 128, 3, 0.0
    for hd, nl in architectures:
        t_ms = check_timing(hd, nl)
        if t_ms < 0.85:
            best_hidden_dim, best_num_layers, best_t_ms = hd, nl, t_ms
        else:
            break
    model = StudentMLP(hidden_dim=best_hidden_dim, num_layers=best_num_layers)
    key = jax.random.PRNGKey(42)
    variables = model.init(key, jnp.ones((1, 6)))
    epochs, batch_size = 300, 256
    steps_per_epoch = len(train_params_norm) // batch_size
    total_steps = epochs * steps_per_epoch
    schedule = make_cosine_restarts_schedule(1e-3, total_steps, 4)
    optimizer = optax.adamw(learning_rate=schedule)
    state = train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=optimizer)
    @jax.jit
    def train_step(state, batch_x, batch_y):
        def loss_fn(params):
            preds = state.apply_fn({'params': params}, batch_x)
            return jnp.mean(weights * (preds - batch_y) ** 2)
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss
    @jax.jit
    def eval_step(state, batch_x, batch_y):
        preds = state.apply_fn({'params': state.params}, batch_x)
        return jnp.mean(weights * (preds - batch_y) ** 2)
    best_val_loss, best_state = float('inf'), state
    for epoch in range(epochs):
        perm = np.random.permutation(len(train_params_norm))
        train_params_shuffled, train_targets_shuffled = train_params_norm[perm], train_targets[perm]
        epoch_loss = 0.0
        for i in range(steps_per_epoch):
            state, loss = train_step(state, train_params_shuffled[i*batch_size:(i+1)*batch_size], train_targets_shuffled[i*batch_size:(i+1)*batch_size])
            epoch_loss += loss
        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            val_loss = eval_step(state, synth_params_norm, synth_targets)
            if val_loss < best_val_loss:
                best_val_loss, best_state = val_loss, state
    emu = StudentEmulator(best_state, box_lo, box_hi, param_names, force_cpu=False)
    with open('data/student_model.pkl', 'wb') as f:
        pickle.dump({'params': unfreeze(best_state.params), 'hidden_dim': best_hidden_dim, 'num_layers': best_num_layers}, f)