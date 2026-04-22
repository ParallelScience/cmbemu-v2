1. **Data Preparation and Augmentation**: Load the initial 50,000 cosmologies and generate an additional 50,000 samples. Create a three-way split: 89,000 for training, 10,000 for validation/early stopping, and 1,000 as a strictly held-out test set for final evaluation.

2. **Preprocessing Pipeline**: Implement the transformation logic: convert $C_\ell$ to $D_\ell$ for $\ell \ge 2$. Apply natural logarithm to $D_\ell^{TT, EE, PP}$. For $D_\ell^{TE}$, apply an `asinh` transform to handle negative values and dynamic range. Normalize input parameters to $[0, 1]$ using box bounds. Standardize each of the four spectral segments independently to zero mean and unit variance before concatenating into the final 20,996-dimensional target vector. Store all scaling factors and means as class attributes.

3. **Neural Network Architecture**: Construct a residual MLP with a 6→512→1024→1024→512→20996 topology. Incorporate LayerNorm and GELU activations. Augment the input vector with derived physical features (e.g., $\omega_b + \omega_{cdm}$) to assist the MLP in capturing non-linear cosmological relationships. Ensure the model is initialized for JAX on the GPU using the mandatory environment configuration.

4. **Hybrid Loss Function Implementation**: Define a weighted loss function: $L = w_{tt}MSE_{tt} + w_{te}Huber_{te} + w_{ee}MSE_{ee} + w_{pp}MSE_{pp}$. Use the validation set to tune these weights to ensure balanced precision across all four spectra.

5. **Training Protocol**: Train using the AdamW optimizer with a 10-epoch linear warmup phase followed by a cosine learning rate schedule. Utilize a batch size of 8,192. Implement early stopping based on the `cec.get_accuracy_score` evaluated on the validation set.

6. **Inference Pipeline Optimization**: Implement the `MyEmulator` class. Pre-allocate all constant arrays (factors, padding, means, stds) in the `__init__` method. Define the forward pass as a JIT-compiled function using `@jax.jit` to ensure no memory allocation occurs during inference. Ensure all internal calculations use `float32` to prevent `float64` promotion.

7. **Validation and Sanity Checking**: Verify the `predict()` method by comparing emulator output against ground truth for a subset of the training data. Calculate the relative error for each spectrum type to ensure the inverse transformations are mathematically exact and that the model is not defaulting to baseline values.

8. **Performance Benchmarking**: Once the accuracy threshold is met, pin JAX to the CPU using `os.environ['JAX_PLATFORMS'] = 'cpu'`. Execute `cec.get_score(emulator)` on the held-out test set to evaluate the combined precision and CPU inference speed, ensuring the final model satisfies the requirement of `mae_total << 10^6`.