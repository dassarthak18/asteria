# Contributing to Asteria
Thank you for considering contributing to Asteria! It began as a solo-project aimed at bringing a lightweight, CPU-only RL library to safe Rust with zero dependencies.

## Technical Philosophy
To maintain the integrity of Asteria, all contributions must adhere to these three core pillars:

1. **100% Safe Rust.** We do not allow ``unsafe`` blocks. This is a hard requirement for the security-focused applications of this library.
2. **Zero Dependencies.** To ensure maximum portability, we avoid external crates. If you need a mathematical function, it should be implemented natively in ``clab``.
3. **Rigorous Verification.** Any new layer, optimizer, or RL algorithm must include numerical gradient checks and a benchmark example (like CartPole or MNIST) demonstrating convergence.

## How Can I Contribute?

### Reporting Bugs
* Check the Issues tracker to see if the bug has already been reported.
* If not, open a new issue. Please include your OS, Rust version, and a minimal reproduction case.

### Suggesting Enhancements
We are particularly interested in:
   * New RL algorithms (e.g., SAC, TD3).
   * SIMD optimizations for the clab tensor kernels (provided they remain safe).
   * Documentation improvements or new benchmark examples.

### Pull Request Process
1. **Open an Issue first.** For major changes, please open an issue to discuss the design before writing code.
2. **Testing:.** Run cargo test to ensure all existing benchmarks and gradient checks pass.
3. **Documentation.** Ensure all public items have ``rustdoc`` comments. We use these to generate the official API reference.

## Development Standards

### Mathematical Correctness
If you are adding a new activation function or layer, you must update the numerical gradient check suite. We compare analytical gradients against finite-difference approximations to ensure the backpropagation logic is flawless.

### Performance
Asteria is optimized for CPU-bound simulations. Avoid unnecessary heap allocations in the forward or backward passes. Reusing buffers in the Tensor engine is preferred over frequent ``Vec`` reallocations.

## License
By contributing to Asteria, you agree that your contributions will be licensed under the project's MIT License.
