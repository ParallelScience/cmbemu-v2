# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
os.environ['JAX_PLATFORMS'] = 'cuda'
import jax
assert len([d for d in jax.devices() if 'cuda' in str(d).lower()]) > 0, 'GPU not found — devices = ' + str(jax.devices())
print('Training on: ' + str(jax.devices()))
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import flax.serialization
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
def get_batches(data_dict, batch_size, shuffle=True, key=None):
    num_samples = data_dict['params'].shape[0]
    if shuffle:
        perms = jax.random.permutation(key, num_samples)
    else:
        perms = jnp.arange(num_samples)
    for i in range(0, num_samples, batch_size):
        batch_idx = perms[i:i+batch_size]
        yield {k: v[batch_idx] for k, v in data_dict.items()}
if __name__ == '__main__':
    print('Loading data...')
    data_path = os.path.join('data', 'train_combined.npz')
    data = np.load(data_path)
    params = data['params']
    tt = data['tt']
    te = data['te']
    ee = data['ee']
    pp = data['pp']
    box_lo = data['box_lo']
    box_hi = data['box_hi']
    print('Normalizing parameters...')
    params_norm = (params - box_lo) / (box_hi - box_lo)
    print('Computing normalization statistics...')
    tt_safe = np.where(tt > 0, tt, 1e-30)
    ee_safe = np.where(ee > 0, ee, 1e-30)
    pp_safe = np.where(pp > 0, pp, 1e-30)
    log_tt = np.log(tt_safe)
    log_ee = np.log(ee_safe)
    log_pp = np.log(pp_safe)
    mean_log_tt = np.mean(log_tt, axis=0)
    std_log_tt = np.std(log_tt, axis=0) + 1e-7
    mean_log_ee = np.mean(log_ee, axis=0)
    std_log_ee = np.std(log_ee, axis=0) + 1e-7
    mean_log_pp = np.mean(log_pp, axis=0)
    std_log_pp = np.std(log_pp, axis=0) + 1e-7
    print('Loading fiducial spectra...')
    fid_data_path = os.path.join('data', 'fiducial_spectra.npz')
    fid_data = np.load(fid_data_path)
    fid_tt = fid_data['tt']
    fid_ee = fid_data['ee']
    fid_pp = fid_data['pp']
    weight_tt = 1.0 / (fid_tt**2 + 1e-30)
    weight_ee = 1.0 / (fid_ee**2 + 1e-30)
    weight_pp = 1.0 / (fid_pp**2 + 1e-30)
    weight_te = 1.0 / (fid_tt * fid_ee + 1e-30)
    stat_dict = {'mean_log_tt': jnp.array(mean_log_tt), 'std_log_tt': jnp.array(std_log_tt), 'mean_log_ee': jnp.array(mean_log_ee), 'std_log_ee': jnp.array(std_log_ee), 'mean_log_pp': jnp.array(mean_log_pp), 'std_log_pp': jnp.array(std_log_pp), 'weight_tt': jnp.array(weight_tt), 'weight_ee': jnp.array(weight_ee), 'weight_pp': jnp.array(weight_pp), 'weight_te': jnp.array(weight_te)}
    print('Splitting data into train and validation sets...')
    num_samples = params_norm.shape[0]
    indices = np.random.RandomState(42).permutation(num_samples)
    split = int(0.9 * num_samples)
    train_idx, val_idx = indices[:split], indices[split:]
    print('Transferring data to GPU...')
    train_data = {'params': jnp.array(params_norm[train_idx]), 'tt': jnp.array(tt[train_idx]), 'te': jnp.array(te[train_idx]), 'ee': jnp.array(ee[train_idx]), 'pp': jnp.array(pp[train_idx])}
    val_data = {'params': jnp.array(params_norm[val_idx]), 'tt': jnp.array(tt[val_idx]), 'te': jnp.array(te[val_idx]), 'ee': jnp.array(ee[val_idx]), 'pp': jnp.array(pp[val_idx])}
    print('Initializing model and optimizer...')
    model = EmulatorMLP()
    key = jax.random.PRNGKey(0)
    dummy_x = jnp.ones((1, 6))
    variables = model.init(key, dummy_x)
    total_epochs = 200
    batch_size = 512
    steps_per_epoch = len(train_idx) // batch_size + (1 if len(train_idx) % batch_size != 0 else 0)
    total_steps = total_epochs * steps_per_epoch
    lr_schedule = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=total_steps, alpha=1e-2)
    optimizer = optax.adam(learning_rate=lr_schedule)
    class TrainState(train_state.TrainState):
        pass
    state = TrainState.create(apply_fn=model.apply, params=variables['params'], tx=optimizer)
    def compute_loss(params, batch, stat_dict):
        x = batch['params']
        out_tt, out_ee, out_pp, out_rho = model.apply({'params': params}, x)
        pred_log_tt = out_tt * stat_dict['std_log_tt'] + stat_dict['mean_log_tt']
        pred_log_ee = out_ee * stat_dict['std_log_ee'] + stat_dict['mean_log_ee']
        pred_log_pp = out_pp * stat_dict['std_log_pp'] + stat_dict['mean_log_pp']
        pred_log_tt = jnp.clip(pred_log_tt, -80.0, 80.0)
        pred_log_ee = jnp.clip(pred_log_ee, -80.0, 80.0)
        pred_log_pp = jnp.clip(pred_log_pp, -80.0, 80.0)
        pred_tt = jnp.exp(pred_log_tt)
        pred_ee = jnp.exp(pred_log_ee)
        pred_pp = jnp.exp(pred_log_pp)
        pred_rho = jnp.tanh(out_rho)
        pred_te = pred_rho * jnp.sqrt(pred_tt * pred_ee + 1e-30)
        w_tt = stat_dict['weight_tt']
        w_ee = stat_dict['weight_ee']
        w_pp = stat_dict['weight_pp']
        w_te = stat_dict['weight_te']
        loss_tt = jnp.mean(((pred_tt - batch['tt'])[:, 2:]**2) * w_tt[2:])
        loss_ee = jnp.mean(((pred_ee - batch['ee'])[:, 2:]**2) * w_ee[2:])
        loss_pp = jnp.mean(((pred_pp - batch['pp'])[:, 2:]**2) * w_pp[2:])
        loss_te = jnp.mean(((pred_te - batch['te'])[:, 2:]**2) * w_te[2:])
        total_loss = loss_tt + loss_ee + loss_pp + loss_te
        return total_loss, {'loss_tt': loss_tt, 'loss_ee': loss_ee, 'loss_pp': loss_pp, 'loss_te': loss_te}
    @jax.jit
    def train_step(state, batch, stat_dict):
        def loss_fn(params):
            loss, metrics = compute_loss(params, batch, stat_dict)
            return loss, metrics
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss, metrics
    @jax.jit
    def val_step(state, batch, stat_dict):
        loss, metrics = compute_loss(state.params, batch, stat_dict)
        return loss, metrics
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    best_params = None
    train_losses = []
    val_losses = []
    rng = jax.random.PRNGKey(42)
    print('Starting training...')
    for epoch in range(total_epochs):
        rng, key = jax.random.split(rng)
        epoch_train_loss = 0.0
        batches = 0
        for batch in get_batches(train_data, batch_size, shuffle=True, key=key):
            state, loss, metrics = train_step(state, batch, stat_dict)
            epoch_train_loss += loss.item()
            batches += 1
        epoch_train_loss /= batches
        train_losses.append(epoch_train_loss)
        epoch_val_loss = 0.0
        val_batches = 0
        for batch in get_batches(val_data, batch_size, shuffle=False):
            loss, metrics = val_step(state, batch, stat_dict)
            epoch_val_loss += loss.item()
            val_batches += 1
        epoch_val_loss /= val_batches
        val_losses.append(epoch_val_loss)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print('Epoch ' + str(epoch+1) + ' | Train Loss: ' + str(epoch_train_loss) + ' | Val Loss: ' + str(epoch_val_loss))
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_params = state.params
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping at epoch ' + str(epoch+1))
                break
    print('Training completed. Best Val Loss: ' + str(best_val_loss))
    print('Saving model weights and normalization statistics...')
    stats_save_path = os.path.join('data', 'normalization_stats.npz')
    np.savez_compressed(stats_save_path, mean_log_tt=mean_log_tt, std_log_tt=std_log_tt, mean_log_ee=mean_log_ee, std_log_ee=std_log_ee, mean_log_pp=mean_log_pp, std_log_pp=std_log_pp, box_lo=box_lo, box_hi=box_hi, fid_tt=fid_tt, fid_ee=fid_ee, fid_pp=fid_pp)
    print('Normalization statistics saved to ' + stats_save_path)
    weights_save_path = os.path.join('data', 'model_weights.msgpack')
    with open(weights_save_path, 'wb') as f:
        f.write(flax.serialization.to_bytes(best_params))
    print('Model weights saved to ' + weights_save_path)
    print('Generating training curve plot...')
    timestamp = str(int(time.time()))
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Weighted MSE Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True, which='both', ls='-', alpha=0.2)
    plt.tight_layout()
    plot_path = os.path.join('data', 'training_curve_1_' + timestamp + '.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print('Training curve saved to ' + plot_path)
    print('Step 2 completed successfully.')