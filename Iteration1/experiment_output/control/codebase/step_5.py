# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
import time
import sys
from flax import serialization
sys.path.insert(0, '/home/node/work/cmbemu/src/')
import cmbemu as cec
from step_2 import MLP, Emulator, inverse_transform_targets
def main():
    print('Loading model and evaluating on CPU...')
    model = MLP(hidden_dims=[1024, 1024, 1024, 1024, 1024], output_dim=20996)
    with open('data/best_model_stage2.msgpack', 'rb') as f:
        model_bytes = f.read()
    rng = jax.random.PRNGKey(0)
    dummy_params = model.init(rng, jnp.ones((1, 6)))
    best_params = serialization.from_bytes(dummy_params, model_bytes)
    emu = Emulator(model_params=best_params, norm_stats_path='data/norm_stats.npz')
    print('Running cec.get_score(emu)...')
    score = cec.get_score(emu)
    print('\n--- Evaluation Results ---')
    print('Combined Score S: ' + str(score['combined_S']))
    print('CPU Time (ms): mean=' + str(score['timing']['t_cpu_ms_mean']) + ', median=' + str(score['timing']['t_cpu_ms_median']) + ', std=' + str(score['timing']['t_cpu_ms_std']))
    print('Precision mae_total: ' + str(score['mae_total']['mae']))
    print('Precision mae_cmb: ' + str(score['mae_cmb']['mae']))
    print('Precision mae_pp: ' + str(score['mae_pp']['mae']))
    print('--------------------------\n')
    print('Full score dictionary:')
    print(str(score))
    val_data = np.load('data/val_data.npz')
    norm_stats = np.load('data/norm_stats.npz')
    val_params_norm = val_data['params']
    val_targets_norm = val_data['targets']
    box_lo = norm_stats['box_lo']
    box_hi = norm_stats['box_hi']
    targets_mean = norm_stats['targets_mean']
    targets_std = norm_stats['targets_std']
    param_names = [name.decode('utf-8') if isinstance(name, bytes) else name for name in norm_stats['param_names']]
    np.random.seed(42)
    sample_indices = np.random.choice(len(val_params_norm), 3, replace=False)
    fig_spectra, axes_spectra = plt.subplots(2, 2, figsize=(12, 10))
    fig_resid, axes_resid = plt.subplots(2, 2, figsize=(12, 10))
    spectra_names = ['TT', 'TE', 'EE', 'PP']
    for i, idx in enumerate(sample_indices):
        p_norm = val_params_norm[idx]
        p_orig = p_norm * (box_hi - box_lo) + box_lo
        p_dict = {name: p_orig[j] for j, name in enumerate(param_names)}
        preds = emu.predict(p_dict)
        y_norm = val_targets_norm[idx]
        y_unnorm = y_norm * targets_std + targets_mean
        C_tt_l2, C_te_l2, C_ee_l2, C_pp_l2 = inverse_transform_targets(y_unnorm)
        true_spectra = {'TT': np.concatenate([np.zeros(2), C_tt_l2]), 'TE': np.concatenate([np.zeros(2), C_te_l2]), 'EE': np.concatenate([np.zeros(2), C_ee_l2]), 'PP': np.concatenate([np.zeros(2), C_pp_l2])}
        pred_spectra = {'TT': preds['tt'], 'TE': preds['te'], 'EE': preds['ee'], 'PP': preds['pp']}
        ell_cmb = np.arange(6001)
        ell_pp = np.arange(3001)
        for j, spec in enumerate(spectra_names):
            ax_s = axes_spectra.flatten()[j]
            ax_r = axes_resid.flatten()[j]
            ell = ell_pp if spec == 'PP' else ell_cmb
            true_s = true_spectra[spec][2:]
            pred_s = pred_spectra[spec][2:]
            ell_plot = ell[2:]
            if spec in ['TT', 'EE', 'PP']:
                ax_s.plot(ell_plot, ell_plot*(ell_plot+1)*true_s, color='C' + str(i), linestyle='-', label='True ' + str(i) if j==0 else '')
                ax_s.plot(ell_plot, ell_plot*(ell_plot+1)*pred_s, color='C' + str(i), linestyle='--', label='Pred ' + str(i) if j==0 else '')
                ax_s.set_yscale('log')
            else:
                ax_s.plot(ell_plot, ell_plot*(ell_plot+1)*true_s, color='C' + str(i), linestyle='-', label='True ' + str(i) if j==0 else '')
                ax_s.plot(ell_plot, ell_plot*(ell_plot+1)*pred_s, color='C' + str(i), linestyle='--', label='Pred ' + str(i) if j==0 else '')
            if spec == 'TE':
                resid = (pred_s - true_s) / np.where(np.abs(true_s) > 1e-15, np.abs(true_s), 1e-15)
            else:
                resid = (pred_s - true_s) / true_s
            ax_r.plot(ell_plot, resid, color='C' + str(i), alpha=0.7)
    for j, spec in enumerate(spectra_names):
        ax_s = axes_spectra.flatten()[j]
        ax_s.set_title(spec + ' Spectrum')
        ax_s.set_xlabel('Multipole l')
        ax_s.set_ylabel('l(l+1)C_l')
        if j == 0:
            ax_s.legend()
        ax_r = axes_resid.flatten()[j]
        ax_r.set_title(spec + ' Fractional Residuals')
        ax_r.set_xlabel('Multipole l')
        ax_r.set_ylabel('Delta C_l / C_l')
        ax_r.axhline(0, color='k', linestyle='--', alpha=0.5)
        if spec in ['TT', 'EE', 'PP']:
            ax_r.set_ylim(-0.05, 0.05)
        else:
            ax_r.set_ylim(-0.1, 0.1)
    fig_spectra.tight_layout()
    fig_resid.tight_layout()
    timestamp = int(time.time())
    spectra_path = 'data/spectra_comparison_1_' + str(timestamp) + '.png'
    resid_path = 'data/fractional_residuals_2_' + str(timestamp) + '.png'
    fig_spectra.savefig(spectra_path, dpi=300)
    fig_resid.savefig(resid_path, dpi=300)
    print('Spectra comparison plot saved to ' + spectra_path)
    print('Fractional residuals plot saved to ' + resid_path)
    loss_stage1 = np.load('data/loss_history_stage1.npz')
    loss_stage2 = np.load('data/loss_history_stage2.npz')
    fig_loss1, ax_loss1 = plt.subplots(figsize=(8, 6))
    ax_loss1.plot(loss_stage1['train_losses'], label='Train Loss')
    ax_loss1.plot(loss_stage1['val_losses'], label='Val Loss')
    ax_loss1.set_yscale('log')
    ax_loss1.set_xlabel('Epoch')
    ax_loss1.set_ylabel('MSE Loss')
    ax_loss1.set_title('Stage 1 Training History')
    ax_loss1.legend()
    fig_loss1.tight_layout()
    loss1_path = 'data/loss_stage1_3_' + str(timestamp) + '.png'
    fig_loss1.savefig(loss1_path, dpi=300)
    print('Stage 1 loss plot saved to ' + loss1_path)
    fig_loss2, ax_loss2 = plt.subplots(figsize=(8, 6))
    ax_loss2.plot(loss_stage2['train_losses'], label='Train Loss')
    ax_loss2.plot(loss_stage2['val_losses'], label='Val Loss')
    ax_loss2.set_yscale('log')
    ax_loss2.set_xlabel('Epoch')
    ax_loss2.set_ylabel('Wishart-approximate Loss')
    ax_loss2.set_title('Stage 2 Training History')
    ax_loss2.legend()
    fig_loss2.tight_layout()
    loss2_path = 'data/loss_stage2_4_' + str(timestamp) + '.png'
    fig_loss2.savefig(loss2_path, dpi=300)
    print('Stage 2 loss plot saved to ' + loss2_path)
if __name__ == '__main__':
    main()