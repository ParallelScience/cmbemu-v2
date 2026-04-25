# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
import os
import time

DATA_DIR = "data/"
TRAIN_PATH = '/home/node/.cache/cmbemu/datasets--borisbolliet--cmbemu-competition-v1/snapshots/e45fd9a3f451038e3ce677a56f7f6cb81c8c47c9/train.npz'
PARAM_NAMES_DISPLAY = ['omega_b', 'omega_cdm', 'H0', 'tau_reio', 'ln10^10 A_s', 'n_s']
SEED = 42

def load_training_data(path):
    data = np.load(path, allow_pickle=False)
    return dict(data)

def compute_log_spectra(tt, ee, pp):
    floor_tt = np.min(tt[:, 2:][tt[:, 2:] > 0]) * 1e-3
    floor_ee = np.min(ee[:, 2:][ee[:, 2:] > 0]) * 1e-3
    floor_pp = np.min(pp[:, 2:][pp[:, 2:] > 0]) * 1e-3
    tt_safe = np.where(tt > 0, tt, floor_tt)
    ee_safe = np.where(ee > 0, ee, floor_ee)
    pp_safe = np.where(pp > 0, pp, floor_pp)
    log_tt = np.log(tt_safe.astype(np.float64))
    log_ee = np.log(ee_safe.astype(np.float64))
    log_pp = np.log(pp_safe.astype(np.float64))
    return log_tt, log_ee, log_pp

def compute_rho(tt, te, ee):
    denom = np.sqrt(np.maximum(tt.astype(np.float64) * ee.astype(np.float64), 0.0))
    safe_denom = np.where(denom > 0, denom, 1.0)
    rho = te.astype(np.float64) / safe_denom
    rho = np.where(denom > 0, rho, 0.0)
    rho = np.clip(rho, -1.0, 1.0)
    return rho

if __name__ == '__main__':
    data = load_training_data(TRAIN_PATH)
    log_tt, log_ee, log_pp = compute_log_spectra(data['tt'], data['ee'], data['pp'])
    rho = compute_rho(data['tt'], data['te'], data['ee'])
    baseline_log_tt = np.mean(log_tt, axis=0)
    baseline_log_ee = np.mean(log_ee, axis=0)
    baseline_log_pp = np.mean(log_pp, axis=0)
    baseline_rho = np.mean(rho, axis=0)
    np.savez(os.path.join(DATA_DIR, 'baselines.npz'), log_tt=baseline_log_tt, log_ee=baseline_log_ee, log_pp=baseline_log_pp, rho=baseline_rho)
    print('Baselines saved to data/baselines.npz')