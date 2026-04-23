# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
os.environ['JAX_PLATFORMS'] = 'cuda'
import jax
assert len([d for d in jax.devices() if 'cuda' in str(d).lower()]) > 0, 'GPU not found — devices = ' + str(jax.devices())
print('Training on: ' + str(jax.devices()))
import sys
import pickle
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from flax.core import unfreeze
import optax
sys.path.insert(0, '/home/node/work/cmbemu/src/')
import cmbemu as cec
class TeacherMLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        for _ in range(6):
            y = nn.Dense(1024)(x)
            y = nn.gelu(y)
            if x.shape[-1] == 1024:
                x = x + y
            else:
                x = y
        x = nn.Dense(21004)(x)
        return x
class TeacherEmulator:
    def __init__(self, state, box_lo, box_hi, param_names):
        self.state = state
        self.box_lo = box_lo
        self.box_hi = box_hi
        self.param_names = param_names
        @jax.jit
        def predict_fn(params_array):
            return self.state.apply_fn({'params': self.state.params}, params_array)
        self.predict_fn = predict_fn
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
if __name__ == '__main__':
    train_data = cec.load_train()
    params = train_data['params']
    tt = train_data['tt']
    te = train_data['te']
    ee = train_data['ee']
    pp = train_data['pp']
    box_lo = train_data['box_lo']
    box_hi = train_data['box_hi']
    param_names = train_data['param_names']
    params_norm = (params - box_lo) / (box_hi - box_lo)
    tt_f64 = tt.astype(np.float64)
    te_f64 = te.astype(np.float64)
    ee_f64 = ee.astype(np.float64)
    pp_f64 = pp.astype(np.float64)
    log_tt = np.log(np.maximum(tt_f64, 1e-300)).astype(np.float32)
    log_ee = np.log(np.maximum(ee_f64, 1e-300)).astype(np.float32)
    log_pp = np.log(np.maximum(pp_f64, 1e-300)).astype(np.float32)
    rho = te_f64 / np.sqrt(np.maximum(tt_f64 * ee_f64, 1e-300))
    rho = np.clip(rho, -1 + 1e-7, 1 - 1e-7)
    arctanh_rho = np.arctanh(rho).astype(np.float32)
    targets = np.concatenate([log_tt, log_ee, log_pp, arctanh_rho], axis=1)
    train_params = params_norm[:45000]
    train_targets = targets[:45000]
    val_params = params_norm[45000:]
    val_targets = targets[45000:]
    epochs = 200
    batch_size = 256
    steps_per_epoch = len(train_params) // batch_size
    total_steps = epochs * steps_per_epoch
    schedule = make_cosine_restarts_schedule(1e-3, total_steps, 4)
    optimizer = optax.adamw(learning_rate=schedule)
    model = TeacherMLP()
    key = jax.random.PRNGKey(42)
    variables = model.init(key, jnp.ones((1, 6)))
    state = train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=optimizer)
    @jax.jit
    def train_step(state, batch_x, batch_y):
        def loss_fn(params):
            preds = state.apply_fn({'params': params}, batch_x)
            loss = jnp.mean((preds - batch_y) ** 2)
            return loss
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss
    @jax.jit
    def eval_step(state, batch_x, batch_y):
        preds = state.apply_fn({'params': state.params}, batch_x)
        loss = jnp.mean((preds - batch_y) ** 2)
        return loss
    best_mae = float('inf')
    best_state = state
    for epoch in range(epochs):
        perm = np.random.permutation(len(train_params))
        train_params_shuffled = train_params[perm]
        train_targets_shuffled = train_targets[perm]
        epoch_loss = 0.0
        for i in range(steps_per_epoch):
            batch_x = train_params_shuffled[i*batch_size:(i+1)*batch_size]
            batch_y = train_targets_shuffled[i*batch_size:(i+1)*batch_size]
            state, loss = train_step(state, batch_x, batch_y)
            epoch_loss += loss
        epoch_loss /= steps_per_epoch
        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            val_loss = eval_step(state, val_params, val_targets)
            emu = TeacherEmulator(state, box_lo, box_hi, param_names)
            acc = cec.get_accuracy_score(emu)
            mae_total = acc['mae_total']['mae']
            print('Epoch ' + str(epoch+1) + ' - Train Loss: ' + str(epoch_loss) + ' - Val Loss: ' + str(val_loss) + ' - MAE Total: ' + str(mae_total))
            if mae_total < best_mae:
                best_mae = mae_total
                best_state = state
    @jax.jit
    def predict_batch(state, x):
        return state.apply_fn({'params': state.params}, x)
    distill_targets = []
    for i in range(0, len(params_norm), 1000):
        batch = params_norm[i:i+1000]
        preds = predict_batch(best_state, batch)
        distill_targets.append(np.array(preds))
    distill_targets = np.concatenate(distill_targets, axis=0)
    np.random.seed(99)
    synth_params = np.random.uniform(box_lo, box_hi, size=(10000, 6)).astype(np.float32)
    synth_params_norm = (synth_params - box_lo) / (box_hi - box_lo)
    synth_targets = []
    for i in range(0, len(synth_params_norm), 1000):
        batch = synth_params_norm[i:i+1000]
        preds = predict_batch(best_state, batch)
        synth_targets.append(np.array(preds))
    synth_targets = np.concatenate(synth_targets, axis=0)
    np.savez('data/distillation_data.npz', train_params_norm=params_norm, train_targets=distill_targets, synth_params_norm=synth_params_norm, synth_targets=synth_targets, box_lo=box_lo, box_hi=box_hi, param_names=param_names)
    with open('data/teacher_model.pkl', 'wb') as f:
        pickle.dump(unfreeze(best_state.params), f)
    print('Best MAE Total achieved: ' + str(best_mae))