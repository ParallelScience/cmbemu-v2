# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
sys.path.insert(0, '/home/node/work/cmbemu/src/')
import cmbemu as cec
def main():
    print('Loading training data...')
    train_data = cec.load_train()
    print('Generating additional data...')
    extra_data = cec.generate_data(n=50000, seed=99)
    print('Combining datasets...')
    combined_data = {}
    for key in ['params', 'tt', 'te', 'ee', 'pp']:
        combined_data[key] = np.concatenate([train_data[key], extra_data[key]], axis=0)
    box_lo = train_data['box_lo']
    box_hi = train_data['box_hi']
    param_names = train_data['param_names']
    print('Normalizing parameters...')
    params_norm = (combined_data['params'] - box_lo) / (box_hi - box_lo)
    print('Transforming target spectra...')
    tt_l2 = combined_data['tt'][:, 2:]
    te_l2 = combined_data['te'][:, 2:]
    ee_l2 = combined_data['ee'][:, 2:]
    pp_l2 = combined_data['pp'][:, 2:]
    log_tt = np.log(tt_l2)
    log_ee = np.log(ee_l2)
    log_pp = np.log(pp_l2)
    rho = te_l2 / np.sqrt(tt_l2 * ee_l2)
    rho_clipped = np.clip(rho, -0.999999, 0.999999)
    atanh_rho = np.arctanh(rho_clipped)
    targets = np.concatenate([log_tt, atanh_rho, log_ee, log_pp], axis=1)
    print('Computing normalization statistics...')
    targets_mean = np.mean(targets, axis=0)
    targets_std = np.std(targets, axis=0)
    targets_norm = (targets - targets_mean) / targets_std
    print('Splitting dataset...')
    np.random.seed(42)
    indices = np.random.permutation(len(params_norm))
    train_indices = indices[:95000]
    val_indices = indices[95000:]
    train_params = params_norm[train_indices]
    train_targets = targets_norm[train_indices]
    val_params = params_norm[val_indices]
    val_targets = targets_norm[val_indices]
    print('Saving processed data...')
    data_dir = 'data/'
    np.savez(os.path.join(data_dir, 'train_data.npz'), params=train_params, targets=train_targets)
    np.savez(os.path.join(data_dir, 'val_data.npz'), params=val_params, targets=val_targets)
    np.savez(os.path.join(data_dir, 'norm_stats.npz'), box_lo=box_lo, box_hi=box_hi, targets_mean=targets_mean, targets_std=targets_std, param_names=param_names)
    print('Data processing complete.')
    print('Training set shape: params ' + str(train_params.shape) + ', targets ' + str(train_targets.shape))
    print('Validation set shape: params ' + str(val_params.shape) + ', targets ' + str(val_targets.shape))
    print('Normalization stats saved to data/norm_stats.npz')
if __name__ == '__main__':
    main()