# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os

if __name__ == '__main__':
    data_dir = 'data/'
    data = np.load(os.path.join(data_dir, 'residuals.npz'))
    res_tt = data['res_tt']
    res_te = data['res_te']
    res_ee = data['res_ee']
    res_pp = data['res_pp']
    ell_cmb = np.arange(res_tt.shape[1])
    ell_pp = np.arange(res_pp.shape[1])
    mask_cmb = ell_cmb >= 2
    mask_pp = ell_pp >= 2
    ell_cmb = ell_cmb[mask_cmb]
    res_tt = res_tt[:, mask_cmb]
    res_te = res_te[:, mask_cmb]
    res_ee = res_ee[:, mask_cmb]
    ell_pp = ell_pp[mask_pp]
    res_pp = res_pp[:, mask_pp]
    med_tt = np.percentile(res_tt, 50, axis=0)
    p16_tt = np.percentile(res_tt, 16, axis=0)
    p84_tt = np.percentile(res_tt, 84, axis=0)
    med_te = np.percentile(res_te, 50, axis=0)
    p16_te = np.percentile(res_te, 16, axis=0)
    p84_te = np.percentile(res_te, 84, axis=0)
    med_ee = np.percentile(res_ee, 50, axis=0)
    p16_ee = np.percentile(res_ee, 16, axis=0)
    p84_ee = np.percentile(res_ee, 84, axis=0)
    med_pp = np.percentile(res_pp, 50, axis=0)
    p16_pp = np.percentile(res_pp, 16, axis=0)
    p84_pp = np.percentile(res_pp, 84, axis=0)
    plt.rcParams['text.usetex'] = False
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs[0, 0].plot(ell_cmb, med_tt, label='Median', color='blue')
    axs[0, 0].fill_between(ell_cmb, p16_tt, p84_tt, color='blue', alpha=0.3, label='68% Interval')
    axs[0, 0].set_title('TT Relative Residuals')
    axs[0, 0].set_xlabel('Multipole ell')
    axs[0, 0].set_ylabel('Relative Residual (dC/C)')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    axs[0, 1].plot(ell_cmb, med_te, label='Median', color='orange')
    axs[0, 1].fill_between(ell_cmb, p16_te, p84_te, color='orange', alpha=0.3, label='68% Interval')
    axs[0, 1].set_title('TE Relative Residuals')
    axs[0, 1].set_xlabel('Multipole ell')
    axs[0, 1].set_ylabel('Relative Residual (dC/C)')
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    axs[1, 0].plot(ell_cmb, med_ee, label='Median', color='green')
    axs[1, 0].fill_between(ell_cmb, p16_ee, p84_ee, color='green', alpha=0.3, label='68% Interval')
    axs[1, 0].set_title('EE Relative Residuals')
    axs[1, 0].set_xlabel('Multipole ell')
    axs[1, 0].set_ylabel('Relative Residual (dC/C)')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    axs[1, 1].plot(ell_pp, med_pp, label='Median', color='red')
    axs[1, 1].fill_between(ell_pp, p16_pp, p84_pp, color='red', alpha=0.3, label='68% Interval')
    axs[1, 1].set_title('PP Relative Residuals')
    axs[1, 1].set_xlabel('Multipole ell')
    axs[1, 1].set_ylabel('Relative Residual (dC/C)')
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    limits = [(p16_tt, p84_tt, 0.05), (p16_te, p84_te, 0.5), (p16_ee, p84_ee, 0.05), (p16_pp, p84_pp, 0.05)]
    for ax, (p16, p84, max_range) in zip([axs[0,0], axs[0,1], axs[1,0], axs[1,1]], limits):
        y_min = np.percentile(p16, 5)
        y_max = np.percentile(p84, 95)
        if y_max - y_min < 1e-4:
            y_min, y_max = -1e-4, 1e-4
        if y_max > max_range: y_max = max_range
        if y_min < -max_range: y_min = -max_range
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    fig.tight_layout()
    timestamp = int(time.time())
    plot_filename = os.path.join(data_dir, 'residuals_plot_1_' + str(timestamp) + '.png')
    fig.savefig(plot_filename, dpi=300)
    print('Plot saved to ' + plot_filename)
    print('\n--- Residual Statistics (ell >= 2) ---')
    print('TT Relative Residuals: Median = ' + str(np.median(med_tt)) + ', 68% Interval Width = ' + str(np.median(p84_tt - p16_tt)))
    print('TE Relative Residuals: Median = ' + str(np.median(med_te)) + ', 68% Interval Width = ' + str(np.median(p84_te - p16_te)))
    print('EE Relative Residuals: Median = ' + str(np.median(med_ee)) + ', 68% Interval Width = ' + str(np.median(p84_ee - p16_ee)))
    print('PP Relative Residuals: Median = ' + str(np.median(med_pp)) + ', 68% Interval Width = ' + str(np.median(p84_pp - p16_pp)))
    with open(os.path.join(data_dir, 'metrics.json'), 'r') as f:
        metrics = json.load(f)
    print('\n--- Final Emulator Metrics ---')
    for k, v in metrics.items():
        print(k + ': ' + str(v))