1. **Data Preparation and Normalization**: Load the 50,000 training samples. Define the normalization constants (`box_lo`, `box_hi`) as fixed values to be used for mapping input parameters to the [0, 1] range. Reserve a 5,000-point validation set from the training pool to monitor distillation progress.

2. **Teacher Model Finalization**: Ensure the high-precision Teacher model is fully trained and validated. Use this Teacher to generate a large synthetic validation set (e.g., 10,000 points) to serve as a high-fidelity, noise-reduced target for the Student model during training.

3. **Student Architecture Design**: Initialize a compact MLP architecture (e.g., 3x128). This size is chosen to prioritize the 1 ms CPU inference threshold. Avoid high-level library abstractions (like Flax/Haiku) for the final inference implementation to minimize overhead.

4. **Knowledge Distillation Training**: Train the Student model using the Teacher's outputs as the primary target. Use the Mean Squared Error (MSE) between the Student's predictions and the Teacher's outputs as the sole loss function. Monitor the Student's performance on the synthetic validation set generated in Step 2 to guide training.

5. **Functional Inference Implementation**: Implement the `predict` method as a pure, static JAX function. Extract weights and biases from the trained Student model and hardcode them as constants. Use `jax.lax.dot_general` or standard matrix multiplication operations to define the forward pass, ensuring the entire sequence is JIT-compiled into a single, efficient graph.

6. **Numerical Stability and Casting**: Within the JIT-compiled `predict` function, perform the forward pass in `float32`. Immediately cast the output to `float64` and apply the required transformations: `exp()` with `np.clip` for spectra and `tanh()` for the correlation coefficient $\rho_\ell$. Ensure the final dictionary construction is part of the compiled function to avoid Python-level overhead.

7. **Performance Tuning**: Benchmark the model using `cec.get_time_score`. If the inference time is significantly below 1 ms, increase the network width (e.g., to 3x256) to improve precision. If the inference time exceeds 1 ms, reduce the network width (e.g., to 2x128). Once the 1 ms threshold is reliably met, focus exclusively on improving `mae_total` through longer training or learning rate scheduling.

8. **Final Benchmarking**: Conduct the final evaluation using `cec.get_score(emu)` with `os.environ['JAX_PLATFORMS'] = 'cpu'` set before any JAX imports. Verify that the combined score $S$ is minimized, confirming the model achieves the required precision while remaining within the 1 ms inference budget.