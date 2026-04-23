# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
try:
    import jax
    jax.devices('cuda')
    os.environ['JAX_PLATFORMS'] = 'cuda'
    print('Training on:', jax.devices('cuda'))
except Exception:
    os.environ['JAX_PLATFORMS'] = 'cpu'
    import jax
    print('CUDA not found. Training on:', jax.devices('cpu'))
import time
import numpy as np
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax import serialization
sys.path.insert(0, '/home/node/work/cmbemu/src/')
import cmbemu as cec
from step_2 import MLP, Emulator
def create_train_state(rng, learning_rate_fn, model):
    params = model.init(rng, jnp.ones((1, 6)))
    tx = optax.adam(learning_rate_fn)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        preds = state.apply_fn(params, batch['x'])
        loss = jnp.mean((preds - batch['y']) ** 2)
        return loss
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss
@jax.jit
def eval_step(state, batch):
    preds = state.apply_fn(state.params, batch['x'])
    loss = jnp.mean((preds - batch['y']) ** 2)
    return loss
def main():
    print('Loading data...')
    train_data = np.load('data/train_data.npz')
    val_data = np.load('data/val_data.npz')
    X_train = train_data['params']
    Y_train = train_data['targets']
    X_val = val_data['params']
    Y_val = val_data['targets']
    batch_size = 1024
    num_epochs = 2000
    warmup_epochs = 10
    steps_per_epoch = len(X_train) // batch_size
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = num_epochs * steps_per_epoch
    lr_schedule = optax.warmup_cosine_decay_schedule(init_value=1e-5, peak_value=1e-3, warmup_steps=warmup_steps, decay_steps=total_steps, end_value=1e-6)
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    model = MLP(hidden_dims=[1024, 1024, 1024, 1024, 1024], output_dim=20996)
    state = create_train_state(init_rng, lr_schedule, model)
    best_val_loss = float('inf')
    best_params = None
    patience = 100
    patience_counter = 0
    train_losses = []
    val_losses = []
    start_time = time.time()
    max_time = 6000
    print('Starting training...')
    for epoch in range(num_epochs):
        rng, shuffle_rng = jax.random.split(rng)
        perms = jax.random.permutation(shuffle_rng, len(X_train))
        perms = np.array(perms)
        epoch_train_loss = []
        for i in range(0, len(X_train), batch_size):
            batch_idx = perms[i:i+batch_size]
            if len(batch_idx) < batch_size:
                continue
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
    print('Training finished.')
    with open('data/best_model_stage1.msgpack', 'wb') as f:
        f.write(serialization.to_bytes(best_params))
    np.savez('data/loss_history_stage1.npz', train_losses=train_losses, val_losses=val_losses)
    print('Model and loss history saved to data/')
    print('Evaluating accuracy score on test set...')
    emu = Emulator(model_params=best_params, norm_stats_path='data/norm_stats.npz')
    acc = cec.get_accuracy_score(emu)
    print('Accuracy Score: mae_total = ' + str(acc['mae_total']['mae']))
    print('mae_cmb = ' + str(acc['mae_cmb']['mae']))
    print('mae_pp = ' + str(acc['mae_pp']['mae']))
if __name__ == '__main__':
    main()