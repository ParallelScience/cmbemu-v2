# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
print('Setting up JAX platforms...')
try:
    import jax
    jax.devices('cuda')
    os.environ['JAX_PLATFORMS'] = 'cuda'
    print('Training on:', jax.devices('cuda'))
except Exception:
    os.environ['JAX_PLATFORMS'] = 'cpu'
    import jax
    print('CUDA not found. Training on:', jax.devices('cpu'))
sys.path.insert(0, '/home/node/data/compsep_data/')
import time
import numpy as np
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax import serialization
sys.path.insert(0, '/home/node/work/cmbemu/src/')
import cmbemu as cec
print('Importing from step_2...')
from step_2 import MLP, Emulator
def create_train_state(model, params, learning_rate_fn):
    tx = optax.adam(learning_rate_fn)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
def main():
    print('Loading data...')
    train_data = np.load('data/train_data.npz')
    val_data = np.load('data/val_data.npz')
    norm_stats = np.load('data/norm_stats.npz')
    X_train = train_data['params']
    Y_train = train_data['targets']
    X_val = val_data['params']
    Y_val = val_data['targets']
    targets_mean = jnp.array(norm_stats['targets_mean'])
    targets_std = jnp.array(norm_stats['targets_std'])
    batch_size = 1024
    num_epochs = 1000
    steps_per_epoch = len(X_train) // batch_size
    total_steps = num_epochs * steps_per_epoch
    lr_schedule = optax.cosine_decay_schedule(init_value=5e-5, decay_steps=total_steps, alpha=0.1)
    print('Initializing model...')
    model = MLP(hidden_dims=[1024, 1024, 1024, 1024, 1024], output_dim=20996)
    print('Loading Stage 1 best model weights...')
    with open('data/best_model_stage1.msgpack', 'rb') as f:
        model_bytes = f.read()
    rng = jax.random.PRNGKey(0)
    dummy_params = model.init(rng, jnp.ones((1, 6)))
    params = serialization.from_bytes(dummy_params, model_bytes)
    state = create_train_state(model, params, lr_schedule)
    @jax.jit
    def train_step(state, batch):
        def loss_fn(params):
            preds = state.apply_fn(params, batch['x'])
            base_loss = jnp.mean((preds - batch['y']) ** 2)
            preds_unnorm = preds * targets_std + targets_mean
            true_unnorm = batch['y'] * targets_std + targets_mean
            log_tt_pred = preds_unnorm[:, 0:5999]
            atanh_rho_pred = preds_unnorm[:, 5999:11998]
            log_ee_pred = preds_unnorm[:, 11998:17997]
            log_tt_true = true_unnorm[:, 0:5999]
            atanh_rho_true = true_unnorm[:, 5999:11998]
            log_ee_true = true_unnorm[:, 11998:17997]
            rho_pred = jnp.tanh(atanh_rho_pred)
            rho_true = jnp.tanh(atanh_rho_true)
            rho_pred = jnp.clip(rho_pred, -0.99999, 0.99999)
            rho_true = jnp.clip(rho_true, -0.99999, 0.99999)
            log_det_pred = log_tt_pred + log_ee_pred + jnp.log(1.0 - rho_pred**2)
            log_det_true = log_tt_true + log_ee_true + jnp.log(1.0 - rho_true**2)
            log_trace_pred = jnp.logaddexp(log_tt_pred, log_ee_pred)
            log_trace_true = jnp.logaddexp(log_tt_true, log_ee_true)
            wishart_loss = jnp.mean((log_det_pred - log_det_true)**2) + jnp.mean((log_trace_pred - log_trace_true)**2)
            return base_loss + wishart_loss
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss
    @jax.jit
    def eval_step(state, batch):
        preds = state.apply_fn(state.params, batch['x'])
        base_loss = jnp.mean((preds - batch['y']) ** 2)
        preds_unnorm = preds * targets_std + targets_mean
        true_unnorm = batch['y'] * targets_std + targets_mean
        log_tt_pred = preds_unnorm[:, 0:5999]
        atanh_rho_pred = preds_unnorm[:, 5999:11998]
        log_ee_pred = preds_unnorm[:, 11998:17997]
        log_tt_true = true_unnorm[:, 0:5999]
        atanh_rho_true = true_unnorm[:, 5999:11998]
        log_ee_true = true_unnorm[:, 11998:17997]
        rho_pred = jnp.tanh(atanh_rho_pred)
        rho_true = jnp.tanh(atanh_rho_true)
        rho_pred = jnp.clip(rho_pred, -0.99999, 0.99999)
        rho_true = jnp.clip(rho_true, -0.99999, 0.99999)
        log_det_pred = log_tt_pred + log_ee_pred + jnp.log(1.0 - rho_pred**2)
        log_det_true = log_tt_true + log_ee_true + jnp.log(1.0 - rho_true**2)
        log_trace_pred = jnp.logaddexp(log_tt_pred, log_ee_pred)
        log_trace_true = jnp.logaddexp(log_tt_true, log_ee_true)
        wishart_loss = jnp.mean((log_det_pred - log_det_true)**2) + jnp.mean((log_trace_pred - log_trace_true)**2)
        return base_loss + wishart_loss
    best_val_loss = float('inf')
    best_params = None
    patience = 50
    patience_counter = 0
    train_losses = []
    val_losses = []
    start_time = time.time()
    max_time = 6000
    rng = jax.random.PRNGKey(100)
    print('Starting Stage 2 fine-tuning...')
    for epoch in range(num_epochs):
        rng, shuffle_rng = jax.random.split(rng)
        perms = jax.random.permutation(shuffle_rng, len(X_train))
        perms = np.array(perms)
        epoch_train_loss = []
        for i in range(0, len(X_train), batch_size):
            batch_idx = perms[i:i+batch_size]
            if len(batch_idx) < batch_size: continue
            batch = {'x': X_train[batch_idx], 'y': Y_train[batch_idx]}
            state, loss = train_step(state, batch)
            epoch_train_loss.append(loss)
        train_loss = np.mean(epoch_train_loss)
        train_losses.append(train_loss)
        epoch_val_loss = []
        for i in range(0, len(X_val), batch_size):
            batch_idx = np.arange(i, min(i+batch_size, len(X_val)))
            batch = {'x': X_val[batch_idx], 'y': Y_val[batch_idx]}
            loss = eval_step(state, batch)
            epoch_val_loss.append(loss)
        val_loss = np.mean(epoch_val_loss)
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = state.params
            patience_counter = 0
        else:
            patience_counter += 1
        if epoch % 10 == 0:
            elapsed = time.time() - start_time
            print('Epoch ' + str(epoch).zfill(4) + ' | Train Loss: ' + str(np.round(train_loss, 6)) + ' | Val Loss: ' + str(np.round(val_loss, 6)) + ' | Best Val: ' + str(np.round(best_val_loss, 6)) + ' | Time: ' + str(np.round(elapsed, 1)) + 's')
        if patience_counter >= patience:
            print('Early stopping at epoch ' + str(epoch))
            break
        if time.time() - start_time > max_time:
            print('Time limit reached at epoch ' + str(epoch))
            break
    print('Fine-tuning finished.')
    with open('data/best_model_stage2.msgpack', 'wb') as f:
        f.write(serialization.to_bytes(best_params))
    np.savez('data/loss_history_stage2.npz', train_losses=train_losses, val_losses=val_losses)
    print('Model and loss history saved to data/')
    print('Evaluating accuracy score on test set with fine-tuned model...')
    emu = Emulator(model_params=best_params, norm_stats_path='data/norm_stats.npz')
    acc = cec.get_accuracy_score(emu)
    print('Accuracy Score: mae_total = ' + str(acc['mae_total']['mae']))
    print('mae_cmb = ' + str(acc['mae_cmb']['mae']))
    print('mae_pp = ' + str(acc['mae_pp']['mae']))
if __name__ == '__main__':
    main()