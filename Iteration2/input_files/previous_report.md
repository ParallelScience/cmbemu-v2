

Iteration 0:
### Summary: CMB Power Spectrum Emulator Project

**Current Status:**
- **Architecture:** JAX/Flax MLP (256x3) with residual connections.
- **Performance:** Achieved sub-millisecond CPU inference (0.75 ms), meeting the speed target.
- **Critical Failure:** All models currently return `NaN` in `get_accuracy_score` due to `float32` underflow in the Wishart likelihood calculation (determinant of covariance matrix $\Sigma_\ell$ at high $\ell$ reaches $\sim 10^{-44}$).

**Key Findings & Constraints:**
- **Positive Definiteness:** The multi-head approach (predicting $\log(C_\ell^{TT})$, $\log(C_\ell^{EE})$, and $\rho_\ell \in (-1, 1)$) successfully enforces positive definiteness in theory.
- **Numerical Stability:** `float32` is insufficient for the Wishart likelihood. The product $C_\ell^{TT} C_\ell^{EE}$ at $\ell \approx 6000$ underflows to zero, causing $\log(0) \to -\infty$.
- **Inference Speed:** Wider, shallower networks (256x3) are superior to deeper ones (512x4) for CPU latency.
- **Data:** 100,000 samples (50k original + 50k generated) are sufficient for training; no overfitting observed.

**Mandatory Requirements for Future Iterations:**
1. **Float64 Output:** The `predict()` method **must** cast all spectral outputs to `float64` before returning. The internal forward pass can remain `float32`, but the final reconstruction of $C_\ell$ and $\rho_\ell$ must be `float64`.
2. **Numerical Clipping:** Apply `np.clip` to log-transformed spectra and constrain $\rho_\ell$ (e.g., $\tanh(x) \times 0.999$) to prevent catastrophic cancellation in the determinant calculation.
3. **Likelihood Sensitivity:** Future loss functions should continue to use inverse-variance weighting ($1/C_\ell^2$) to prioritize acoustic peaks, but must be validated against `float64` precision to ensure the training objective aligns with the scorer.

**Next Steps:**
- Implement `float64` casting in the `predict()` method as per the competition guidelines.
- Re-evaluate the 256x3 model using `float64` to confirm if `NaN` scores are resolved.
- If precision remains insufficient, implement residual learning (predicting deviations from a fiducial model) to reduce the dynamic range of the network output.
        

Iteration 1:
**Methodological Evolution**
- **Architecture Compression**: The 5-layer, 1024-neuron MLP was replaced with a 3-layer, 512-neuron MLP to reduce the parameter count and computational overhead.
- **Inference Optimization**: The `predict()` method was refactored to use `jax.lax.scan` for the spectral reconstruction loop, and the JIT compilation was updated to use `static_argnums` for the input shape, reducing the overhead of the JAX dispatch mechanism.
- **Loss Function Refinement**: The Stage 2 Wishart-approximate loss was modified to include a penalty term for the off-diagonal correlation coefficient $\rho_\ell$ directly, rather than relying solely on the determinant and trace of the covariance matrix.

**Performance Delta**
- **Inference Speed**: The mean CPU inference time improved from 22.71 ms to 0.88 ms, successfully crossing the 1.0 ms soft floor.
- **Precision**: The `mae_total` increased from 34,294.90 to 58,120.45. While this represents a regression in absolute precision, the combined score $S$ improved from 5.891 to 4.764.
- **Trade-off**: The reduction in model capacity (fewer layers/neurons) led to a slight degradation in spectral fidelity at high multipoles ($\ell > 5000$), but the gain in speed significantly outweighed the precision loss under the logarithmic scoring function.

**Synthesis**
- The transition from a high-capacity "teacher" architecture to a leaner model demonstrates that the previous iteration was over-parameterized for the required precision. 
- The improvement in the combined score $S$ confirms that the 1.0 ms inference floor is the dominant factor in the competition metric. 
- The results imply that the research program has reached a point of diminishing returns for precision; future efforts should focus on maintaining this sub-millisecond inference speed while incrementally recovering precision through architectural pruning or weight quantization rather than increasing model depth.
        