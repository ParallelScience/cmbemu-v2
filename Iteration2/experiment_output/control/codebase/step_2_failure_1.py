# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
os.environ['JAX_PLATFORMS'] = 'cuda'
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from flax.core import unfreeze
import optax
import numpy as np
import sys
import pickle
if __name__ == '__main__':
    assert len([d for d in jax.devices() if 'cuda' in str(d).lower()]) > 0, 'GPU not found — devices = ' + str(jax.devices())
    print('Training on: ' + str(jax.devices()))
    sys.path.insert(0, '/home/node/work/cmbemu/src/')
    import cmbemu as cec
    print('Loading processed data...')
    data = np.load('data/processed_data.npz')
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_val = data['X_val']
    Y_val = data['Y_val']
    box_lo = data['box_lo']
    box_hi = data['box_hi']
    print('Normalizing data...')
    def normalize_params(params, box_lo, box_hi):
        return (params - box_lo) / (box_hi - box_lo)
    X_train_norm = normalize_params(X_train, box_lo, box_hi)
    X_val_norm = normalize_params(X_val, box_lo, box_hi)
    Y_mean = np.mean(Y_train, axis=0)
    Y_std = np.std(Y_train, axis=0)
    Y_std[Y_std == 0] = 1.0
    Y_train_norm = (Y_train - Y_mean) / Y_std
    Y_val_norm = (Y_val - Y_mean) / Y_std
    class TeacherMLP(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(512)(x)
            x = nn.gelu(x)
            x = nn.Dense(512)(x)
            x = nn.gelu(x)
            x = nn.Dense(512)(x)
            x = nn.gelu(x)
            x = nn.Dense(512)(x)
            x = nn.gelu(x)
            x = nn.Dense(21004)(x)
            return x
    batch_size = 256
    num_epochs = 1000
    steps_per_epoch = len(X_train_norm) // batch_size
    total_steps = steps_per_epoch * num_epochs
    lr_schedule = optax.warmup_cosine_decay_schedule(init_value=1e-5, peak_value=1e-3, warmup_steps=10 * steps_per_epoch, decay_steps=total_steps, end_value=1e-6)
    optimizer = optax.adamw(learning_rate=lr_schedule, weight_decay=1e-4)
    model = TeacherMLP()
    rng = jax.random.PRNGKey(42)
    variables = model.init(rng, jnp.ones((1, 6)))
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
    print('Starting training...')
    best_val_loss = float('inf')
    best_params = None
    patience = 50
    patience_counter = 0
    for epoch in range(num_epochs):
        perm = np.random.permutation(len(X_train_norm))
        X_train_shuffled = X_train_norm[perm]
        Y_train_shuffled = Y_train_norm[perm]
        epoch_loss = 0.0
        for i in range(steps_per_epoch):
            batch_x = X_train_shuffled[i*batch_size:(i+1)*batch_size]
            batch_y = Y_train_shuffled[i*batch_size:(i+1)*batch_size]
            state, loss = train_step(state, batch_x, batch_y)
            epoch_loss += float(loss)
        epoch_loss /= steps_per_epoch
        val_loss = float(eval_step(state, X_val_norm, Y_val_norm))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = state.params
            patience_counter = 0
        else:
            patience_counter += 1
        if (epoch + 1) % 50 == 0:
            print('Epoch ' + str(epoch+1) + ' - Train Loss: ' + str(epoch_loss) + ' - Val Loss: ' + str(val_loss))
        if patience_counter >= patience:
            print('Early stopping at epoch ' + str(epoch+1))
            break
    print('Training complete. Best validation loss: ' + str(best_val_loss))
    class TeacherEmulator:
        def __init__(self, model, params, box_lo, box_hi, Y_mean, Y_std):
            self.model = model
            self.params = params
            self.box_lo = box_lo
            self.box_hi = box_hi
            self.Y_mean = Y_mean
            self.Y_std = Y_std
            self.param_names = ('omega_b', 'omega_cdm', 'H0', 'tau_reio', 'ln10^{10}A_s', 'n_s')
            @jax.jit
            def forward(x):
                x_norm = (x - self.box_lo) / (self.box_hi - self.box_lo)
                preds_norm = self.model.apply({'params': self.params}, x_norm)
                preds = preds_norm * self.Y_std + self.Y_mean
                return preds
            self.forward = forward
        def predict(self, params_dict):
            x = np.array([[params_dict[k] for k in self.param_names]], dtype=np.float32)
            preds = self.forward(x)
            preds = np.array(preds)[0]
            log_tt = preds[:6001]
            arctanh_rho = preds[6001:12002]
            log_ee = preds[12002:18003]
            log_pp = preds[18003:21004]
            log_tt_f64 = log_tt.astype(np.float64)
            log_ee_f64 = log_ee.astype(np.float64)
            log_pp_f64 = log_pp.astype(np.float64)
            arctanh_rho_f64 = arctanh_rho.astype(np.float64)
            C_tt = np.exp(np.clip(log_tt_f64, -700, 700))
            C_ee = np.exp(np.clip(log_ee_f64, -700, 700))
            C_pp = np.exp(np.clip(log_pp_f64, -700, 700))
            rho = np.tanh(arctanh_rho_f64)
            C_te = rho * np.sqrt(C_tt * C_ee)
            return {'tt': C_tt, 'te': C_te, 'ee': C_ee, 'pp': C_pp}
    print('Evaluating Teacher model...')
    emu = TeacherEmulator(model, best_params, box_lo, box_hi, Y_mean, Y_std)
    acc = cec.get_accuracy_score(emu)
    mae_total = acc['mae_total']['mae']
    print('Teacher Accuracy Score (mae_total): ' + str(mae_total))
    if mae_total < 1e5:
        print('Teacher model achieved target accuracy.')
    else:
        print('Warning: Teacher model did not achieve target accuracy.')
    print('Saving Teacher model...')
    save_path = 'data/teacher_model.pkl'
    def jax_to_numpy(params):
        if isinstance(params, dict):
            return {k: jax_to_numpy(v) for k, v in params.items()}
        else:
            return np.array(params)
    params_np = jax_to_numpy(unfreeze(best_params))
    with open(save_path, 'wb') as f:
        pickle.dump({'params': params_np, 'box_lo': box_lo, 'box_hi': box_hi, 'Y_mean': Y_mean, 'Y_std': Y_std}, f)
    print('Teacher model saved to ' + save_path)