# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
os.environ['JAX_PLATFORMS'] = 'cpu'
import sys
sys.path.insert(0, '/home/node/work/cmbemu/src/')
import cmbemu as cec
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import matplotlib.pyplot as plt
class MyEmulator:
    def __init__(self, model_path='model_weights.pkl'):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        self.params_mean = jnp.array(data['params_mean'])
        self.params_std = jnp.array(data['params_std'])
        self.weights = data['weights']
        self.biases = data['biases']
        self.param_order = ('omega_b', 'omega_cdm', 'H0', 'tau_reio', 'ln10^{10}A_s', 'n_s')
    @jax.jit
    def predict_jit(self, x, weights, biases):
        for w, b in zip(weights[:-1], biases[:-1]):
            x = jax.nn.relu(jnp.dot(x, w) + b)
        return jnp.dot(x, weights[-1]) + biases[-1]
    def predict(self, params: dict) -> dict:
        x = jnp.array([params[p] for p in self.param_order])
        x = (x - self.params_mean) / self.params_std
        out = self.predict_jit(x, self.weights, self.biases)
        return {'tt': np.array(out[:6001]), 'te': np.array(out[6001:12002]), 'ee': np.array(out[12002:18003]), 'pp': np.array(out[18003:])}
if __name__ == '__main__':
    emu = MyEmulator()
    acc = cec.get_accuracy_score(emu)
    full = cec.get_score(emu)
    print('Accuracy:', acc)
    print('Full Score:', full)
    with open('data/results.txt', 'w') as f:
        f.write(str(full))
    test_data = cec.generate_data(n=10, seed=42)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    keys = ['tt', 'te', 'ee', 'pp']
    for i, key in enumerate(keys):
        ax = axes.flat[i]
        for j in range(10):
            params = {p: test_data['params'][j, k] for k, p in enumerate(emu.param_order)}
            pred = emu.predict(params)[key]
            true = test_data[key][j]
            ax.plot((pred - true) / (true + 1e-20), alpha=0.5)
        ax.set_title(key.upper())
    plt.tight_layout()
    plt.savefig('data/residuals.png')
    print('Saved to data/residuals.png')