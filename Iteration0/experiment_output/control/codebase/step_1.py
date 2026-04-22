# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
sys.path.insert(0, '/home/node/data/compsep_data/')
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['text.usetex'] = False

sys.path.insert(0, '/home/node/work/cmbemu/src/')
import cmbemu as cec

if __name__ == '__main__':
    print('Loading initial 50,000 samples...')
    train_initial = cec.load_train()
    print('Generating additional 50,000 samples...')
    train_extra = cec.generate_data(n=50000, seed=99)
    print('Combining datasets...')
    train_combined = {'params': np.concatenate([train_initial['params'], train_extra['params']], axis=0), 'tt': np.concatenate([train_initial['tt'], train_extra['tt']], axis=0), 'te': np.concatenate([train_initial['te'], train_extra['te']], axis=0), 'ee': np.concatenate([train_initial['ee'], train_extra['ee']], axis=0), 'pp': np.concatenate([train_initial['pp'], train_extra['pp']], axis=0), 'box_lo': train_initial['box_lo'], 'box_hi': train_initial['box_hi'], 'param_names': train_initial['param_names']}
    print('\n--- Exploratory Analysis ---')
    print('Total samples: ' + str(train_combined['params'].shape[0]))
    print('\nCosmological Parameters Summary:')
    for i, name in enumerate(train_combined['param_names']):
        p_data = train_combined['params'][:, i]
        print('  ' + str(name) + ':')
        print('    Min: ' + str(np.min(p_data)) + ', Max: ' + str(np.max(p_data)))
        print('    Mean: ' + str(np.mean(p_data)) + ', Std: ' + str(np.std(p_data)))
        print('    Box bounds: [' + str(train_combined['box_lo'][i]) + ', ' + str(train_combined['box_hi'][i]) + ']')
    print('\nSpectra Summary (ignoring l=0,1):')
    for spec in ['tt', 'te', 'ee', 'pp']:
        s_data = train_combined[spec][:, 2:]
        print('  ' + spec.upper() + ':')
        print('    Shape: ' + str(train_combined[spec].shape))
        print('    Global Min: ' + str(np.min(s_data)) + ', Global Max: ' + str(np.max(s_data)))
        print('    Mean amplitude: ' + str(np.mean(s_data)))
        print('    Std amplitude: ' + str(np.std(s_data)))
        print('    Amplitude at specific multipoles:')
        for l in [2, 100, 1000, 3000]:
            if l < train_combined[spec].shape[1]:
                l_data = train_combined[spec][:, l]
                print('      l=' + str(l) + ': Min=' + str(np.min(l_data)) + ', Max=' + str(np.max(l_data)) + ', Mean=' + str(np.mean(l_data)) + ', Std=' + str(np.std(l_data)))
    print('\nGenerating diagnostic plots...')
    timestamp = str(int(time.time()))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, name in enumerate(train_combined['param_names']):
        ax = axes[0, i % 3] if i < 3 else axes[1, i % 3]
        if i < 6:
            ax.hist(train_combined['params'][:, i], bins=50, color='skyblue', edgecolor='black')
            ax.set_title('Distribution of ' + str(name))
            ax.set_xlabel(str(name))
            ax.set_ylabel('Count')
    plt.tight_layout()
    plot_path_params = os.path.join('data', 'param_distributions_1_' + timestamp + '.png')
    plt.savefig(plot_path_params, dpi=300)
    plt.close()
    print('Parameter distributions plot saved to ' + plot_path_params)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ell_tt_te_ee = np.arange(train_combined['tt'].shape[1])
    ell_pp = np.arange(train_combined['pp'].shape[1])
    factor_tt_te_ee = ell_tt_te_ee * (ell_tt_te_ee + 1) / (2 * np.pi)
    factor_pp = (ell_pp * (ell_pp + 1))**2 / (2 * np.pi)
    for i in range(5):
        axes[0, 0].plot(ell_tt_te_ee[2:], train_combined['tt'][i, 2:] * factor_tt_te_ee[2:], alpha=0.7)
        axes[0, 1].plot(ell_tt_te_ee[2:], train_combined['te'][i, 2:] * factor_tt_te_ee[2:], alpha=0.7)
        axes[1, 0].plot(ell_tt_te_ee[2:], train_combined['ee'][i, 2:] * factor_tt_te_ee[2:], alpha=0.7)
        axes[1, 1].plot(ell_pp[2:], train_combined['pp'][i, 2:] * factor_pp[2:], alpha=0.7)
    axes[0, 0].set_title('TT Spectrum')
    axes[0, 0].set_xlabel('Multipole ell')
    axes[0, 0].set_ylabel('D_ell TT [K^2]')
    axes[0, 0].set_yscale('log')
    axes[0, 1].set_title('TE Spectrum')
    axes[0, 1].set_xlabel('Multipole ell')
    axes[0, 1].set_ylabel('D_ell TE [K^2]')
    axes[0, 1].set_yscale('symlog', linthresh=1e-10)
    axes[1, 0].set_title('EE Spectrum')
    axes[1, 0].set_xlabel('Multipole ell')
    axes[1, 0].set_ylabel('D_ell EE [K^2]')
    axes[1, 0].set_yscale('log')
    axes[1, 1].set_title('PP Spectrum')
    axes[1, 1].set_xlabel('Multipole ell')
    axes[1, 1].set_ylabel('[l(l+1)]^2 C_ell PP / 2pi')
    axes[1, 1].set_yscale('log')
    plt.tight_layout()
    plot_path_spectra = os.path.join('data', 'example_spectra_2_' + timestamp + '.png')
    plt.savefig(plot_path_spectra, dpi=300)
    plt.close()
    print('Example spectra plot saved to ' + plot_path_spectra)
    print('\nComputing normalization parameters...')
    print('Computing fiducial spectra (mean of training set)...')
    fiducial_spectra = {'tt': np.mean(train_combined['tt'], axis=0), 'te': np.mean(train_combined['te'], axis=0), 'ee': np.mean(train_combined['ee'], axis=0), 'pp': np.mean(train_combined['pp'], axis=0)}
    print('Saving data to data/ directory...')
    np.savez_compressed(os.path.join('data', 'train_combined.npz'), params=train_combined['params'], tt=train_combined['tt'], te=train_combined['te'], ee=train_combined['ee'], pp=train_combined['pp'], box_lo=train_combined['box_lo'], box_hi=train_combined['box_hi'], param_names=train_combined['param_names'])
    print('Combined dataset saved to data/train_combined.npz')
    np.savez_compressed(os.path.join('data', 'fiducial_spectra.npz'), tt=fiducial_spectra['tt'], te=fiducial_spectra['te'], ee=fiducial_spectra['ee'], pp=fiducial_spectra['pp'])
    print('Fiducial spectra saved to data/fiducial_spectra.npz')
    print('Step 1 completed successfully.')