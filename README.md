# Asteria: Pure Rust Reinforcement Learning and Neural Networks

![](https://raw.githubusercontent.com/dassarthak18/asteria/main/logo/logo.jpg)

*Jupiter and Asteria* by Marco Liberi. ([Wikimedia Commons](https://upload.wikimedia.org/wikipedia/commons/3/3f/Marco_Liberi_-_Jupiter_and_Asteria_-_WGA12975.jpg))

## Table of Contents

- [Introduction](#introduction)
- [Why Asteria?](#why-asteria)
- [What Asteria Adds over Coeus](#what-asteria-adds-over-coeus)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Your First Network (XOR)](#your-first-network-xor)
  - [Your First RL Agent (Maze)](#your-first-rl-agent-maze)
- [External Dependencies](#external-dependencies)
- [Tensor Engine (CLAB)](#tensor-engine-clab)
- [Neural Network Framework](#neural-network-framework)
  - [Weight Initialization](#weight-initialization)
  - [Activation Functions](#activation-functions)
  - [Loss Functions](#loss-functions)
  - [Optimizers](#optimizers)
  - [Learning Rate Schedulers](#learning-rate-schedulers)
  - [Model Serialization](#model-serialization)
- [Reinforcement Learning Algorithms](#reinforcement-learning-algorithms)
  - [Environment Interface](#environment-interface)
  - [Discrete Action Space](#discrete-action-space)
  - [Continuous Action Space](#continuous-action-space)
  - [Motivation Models](#motivation-models)
- [Test Suite](#test-suite)
- [Examples and Benchmarks](#examples-and-benchmarks)
  - [Supervised Learning](#supervised-learning)
  - [Discrete Reinforcement Learning](#discrete-reinforcement-learning)
  - [Continuous Reinforcement Learning](#continuous-reinforcement-learning)
- [Planned Features](#planned-features)
- [Acknowledgements](#acknowledgements)

## Introduction

Asteria is a 100% safe Rust library for reinforcement learning and neural network training. It accepts environments modelled as discrete-time Markov decision processes $M = (S, A, P, R, \gamma)$, where $S \subseteq \mathbb{R}^m$ is the state space, $A \subseteq \mathbb{R}^n$ is the action space, $P: S \times A \times S \rightarrow [0, 1]$ is the transition kernel, $R: S \times A \rightarrow \mathbb{R}$ is the reward function, and $\gamma \in [0, 1)$ is the discount factor. Policies are represented as neural networks $\pi_\theta: S \rightarrow A$ parameterized by weights $\theta$, trained end-to-end via gradient descent directly against collected experience.

Asteria began as a port of [Coeus](https://github.com/Iskandor/Coeus), a C++ reinforcement learning library by Matej Pechac (Iskandor). The name continues the mythological lineage: Asteria is the daughter of Coeus in Greek mythology. Asteria has grown substantially beyond the original: it adds three new RL algorithms (A2C, A3C, PPO), a new optimizer (ADOPT), a full LR scheduler suite, a supervised learning benchmark suite, model serialization, and corrected forward/backward implementations — all with no dependency on any external ML framework.

The library ships:
- A custom N-dimensional tensor engine (**CLAB**) with broadcast, matmul, gather, concat, and argmax.
- A graph-based neural network framework with mini-batch support and topological-sort execution order.
- Eleven RL algorithms across discrete and continuous action spaces, plus two intrinsic motivation models.
- A supervised learning suite (XOR, Iris, Wine, MNIST) reaching up to 98.18% accuracy on MNIST in 10 epochs.

## Why Asteria?

If you come from Python-based deep learning (PyTorch, JAX) and OpenAI Gym, you probably know that most RL research tooling is Python-first. So why bother with a Rust library?

**The case for Rust RL:**
- **Zero-overhead abstraction.** No GIL, no garbage collector, no runtime. Inference and training run at full CPU speed without Python interpreter overhead.
- **Memory safety without sacrificing control.** The borrow checker eliminates whole classes of bugs (dangling pointers, data races) that are common in RL simulation code. All code here is `#![forbid(unsafe_code)]`-equivalent — no `unsafe` blocks anywhere.
- **Embeddability.** A compiled Rust library can be linked into games, robotics firmware, or safety-critical applications where CPython is not an option.
- **Learning.** Understanding how backpropagation, replay buffers, and actor-critic methods work *without* PyTorch's autograd doing the heavy lifting is genuinely educational.

**When PyTorch + Gym is still the right choice:** large-scale training with GPUs, convolutional networks over pixels, environments from Gymnasium/MuJoCo/ALE, distributed training across machines — none of that is Asteria's territory (yet). Asteria is best suited for CPU-bound RL with low-dimensional state spaces, educational use, and Rust-embedded deployments.

## What Asteria Adds over Coeus

The comparison below is based on direct inspection of the Coeus source code (included in `./Coeus/`).

**What Coeus shipped:** TD, Q-Learning, SARSA, DQN, AC, CACLA, DDPG, ForwardModel (predictive-error model), Metacritic (surprise model), QAC (code present, not listed in Coeus README features), PolicyGradient, SGD (with momentum and Nesterov), Adam, RAdam. Two compute backends: Intel MKL (primary; optimizers use AVX/SIMD intrinsics via `_mm256_load_ps`) and CUDA/cuBLAS (optional `CuCLAB` module). Activations: Linear, Sigmoid, Tanh, TanhExp, ReLU, Softmax. `tensor::save_numpy()` via CNPy for saving training logs and reward tensors to `.npy` format. XOR example in the README; maze, mountain car, and CartPole experiment runners.

**What Coeus planned but never shipped:** A2C, A3C, PPO, convolutional networks, ALE support, MuJoCo support.

Asteria adds or corrects the following relative to Coeus:

| Category | Addition / Change |
|---|---|
| **RL Algorithms** | A2C (Synchronous Advantage Actor-Critic with batched rollouts) |
| | A3C (Asynchronous A-C; Hogwild-style gradient sharing via `Arc<Mutex<...>>`) |
| | PPO (Proximal Policy Optimization with clipped surrogate and multiple update epochs) |
| **Optimizers** | ADOPT (optimal $O(1/\sqrt{T})$ convergence without bounded-noise assumption; default in all examples) |
| **LR Schedulers** | `StepDecay`, `CosineAnnealing`, `LinearDecay`, `WarmupCosine`, `ReduceLROnPlateau` (Coeus had none) |
| **Loss Functions** | `SoftmaxCrossEntropyFunction` — fused logit→softmax→cross-entropy with correct combined backward |
| **Supervised Examples** | Iris, Wine, MNIST classifiers (mini-batch, shuffle, one-hot, per-epoch accuracy eval) |
| **Mini-batch Training** | `NeuralNetwork::forward` / `backward` both accept `[batch, features]` tensors |
| **Model Serialization** | `NeuralNetwork::save(path)` / `::load(path)` for model weights as sorted little-endian f32 bytes |
| **Bug Fixes** | **Softmax backward:** Coeus's active implementation only propagates through the first non-zero delta (correct for policy gradient but wrong for multi-class CE); the full Jacobian is commented out. Asteria uses the correct Jacobian $\partial L/\partial z_j = S_j(\delta_j - \sum_i \delta_i S_i)$ always. |
| | **Q-learning `delta`:** Coeus stores raw `Q(s,a)` as the actor advantage; Asteria fixes this to store the TD error `Q(s,a) − target`, consistent with how all other critics signal the actor. |
| | **`max_index(dim)`:** Returns per-row/per-column argmax indices (clean semantics); Coeus's equivalent was flat-buffer based. |
| **Compute backend** | Pure safe Rust (no `unsafe`, no AVX intrinsics, no CUDA) replaces MKL+AVX and optional CuBLAS |
| **Documentation** | Full `rustdoc` coverage across all public items; Coeus had Doxygen comments on most but not all items |

**What Coeus had that Asteria does not:**
- **TanhExp activation** — Coeus shipped a TanhExp (tanh-exponential) activation; Asteria does not include it.
- **CUDA/GPU backend** — Coeus had an optional `CuCLAB` CUDA module; Asteria is CPU-only.
- **NumPy-format tensor I/O** — Coeus saved tensors (training logs, rewards) as `.npy` via CNPy; Asteria uses a binary f32 format for model weights and does not yet support `.npy`/`.npz`.
- **Convolutional layers, ALE, MuJoCo** — planned in Coeus, not yet in Asteria either.

## Getting Started

This section is for newcomers: people who know Python and maybe some PyTorch or Gym, and are comfortable writing basic Rust, but haven't worked with Asteria before.

### Installation

Asteria is a library crate. Clone it and use it as a local path dependency:

```shell
git clone https://github.com/dassarthak18/asteria.git
```

In your `Cargo.toml`:

```toml
[dependencies]
asteria = { path = "../asteria" }
```

You only need `rand` and `rand_distr` as transitive dependencies — Asteria has no other external requirements.

To run the bundled examples from the repo root:

```shell
# Supervised learning
cargo run --release --example xor
cargo run --release --example iris
cargo run --release --example wine
cargo run --release --example mnist      # --release mandatory; debug is ~10x slower

# Reinforcement learning (discrete)
cargo run --release --example maze
cargo run --release --example maze_pg

# Reinforcement learning (continuous)
cargo run --release --example simple_continuous_env
cargo run --release --example mountain_car
cargo run --release --example cart_pole
```

Data files for the supervised examples (`iris.data`, `wine.dat`, four MNIST IDX binaries) live in `./data/`. All paths are relative to the repo root, so run from there.

### Your First Network (XOR)

If you've used PyTorch, the pattern maps like this:

| PyTorch concept | Asteria equivalent |
|---|---|
| `torch.Tensor` | `asteria::clab::tensor::Tensor` |
| `nn.Module` / `nn.Sequential` | `NeuralNetwork` (a DAG of `DenseLayer` nodes) |
| `nn.Linear` + activation | `DenseLayer::new(id, out, activation, initializer, in_dim)` |
| `torch.optim.Adam` | `ADOPT::new(lr)` or `Adam::new(lr)` |
| `optimizer.zero_grad()` | implicit — gradients are zeroed inside `backward` |
| `loss.backward()` | `network.backward(&mut delta)` where `delta = loss.backward(...)` |
| `optimizer.step()` | `optimizer.update(&mut network)` |

Here is a minimal XOR network. It is almost identical to `examples/xor.rs`:

```rust
use asteria::clab::tensor::Tensor;
use asteria::clab::activation_functions::ActivationType;
use asteria::clab::tensor_initializer::{TensorInitializer, InitializerType};
use asteria::clab::loss_functions::{MseFunction, LossFunction};
use asteria::core::neural_network::NeuralNetwork;
use asteria::core::dense_layer::DenseLayer;
use asteria::core::adopt::ADOPT;
use asteria::core::optimizer::Optimizer;

fn main() {
    // Build the dataset — all 4 XOR pairs at once (batch of 4)
    let input  = Tensor::from_data(vec![4, 2], vec![0.,0., 0.,1., 1.,0., 1.,1.]);
    let target = Tensor::from_data(vec![4, 1], vec![0., 1., 1., 0.]);

    // Build the network: 2 → 8 (Sigmoid) → 4 (Sigmoid) → 1 (Sigmoid)
    let mut net = NeuralNetwork::new();
    net.add_layer(DenseLayer::new("h0".into(), 8, ActivationType::Sigmoid,
                                  TensorInitializer::new(InitializerType::LecunUniform), 2));
    net.add_layer(DenseLayer::new("h1".into(), 4, ActivationType::Sigmoid,
                                  TensorInitializer::new(InitializerType::LecunUniform), 0));
    net.add_layer(DenseLayer::new("out".into(), 1, ActivationType::Sigmoid,
                                  TensorInitializer::new(InitializerType::LecunUniform), 0));
    net.add_connection("h0", "h1");
    net.add_connection("h1", "out");
    net.init();  // topological sort; must be called before first forward pass

    let loss = MseFunction;
    let mut opt = ADOPT::new(1e-2);  // ADOPT optimizer, lr = 0.01

    for _ in 0..500 {
        let output = net.forward(&input).clone();
        let error  = loss.forward(&output, &target);
        let mut delta = loss.backward(&output, &target);
        net.backward(&mut delta);  // computes gradients
        opt.update(&mut net);      // applies parameter update
        let _ = error;
    }

    println!("{}", net.forward(&input));
    // Typical output: 0.02, 0.95, 0.96, 0.03  (close to 0, 1, 1, 0)
}
```

A few things to note:
- **No `zero_grad()`**: gradients are automatically zeroed during `backward`. There is no accumulation.
- **`init()` is required**: it runs the topological sort that determines forward and backward execution order. Forgetting it will panic.
- **Cloning forward output**: `net.forward(&input)` returns a reference tied to the network's internal buffer. Clone it before calling `backward`, or the borrow checker will complain.
- **Convergence is stochastic**: XOR with random initialization sometimes requires more than one run to converge. The weight scale matters; `LecunUniform` with Sigmoid works well.

### Your First RL Agent (Maze)

In Gym, environments implement `reset()`, `step(action)` → `(obs, reward, done, info)`. Asteria's equivalent is the `IEnvironment` trait:

```rust
pub trait IEnvironment {
    fn reset(&mut self);
    fn get_state(&mut self) -> Tensor;
    fn do_action(&mut self, action: &Tensor);
    fn get_reward(&mut self) -> f32;
    fn is_finished(&mut self) -> bool;
    fn state_dim(&self) -> usize;
    fn action_dim(&self) -> usize;
}
```

Instead of `step(action) → (obs, reward, done)`, you call `do_action`, then query `get_state`, `get_reward`, and `is_finished` separately. This avoids heap allocation on every step.

A minimal Q-Learning agent on the bundled maze:

```rust
use asteria::rl::qlearning::Qlearning;
use asteria::rl::discrete_exploration::{DiscreteExploration, DiscreteExplorationMethod};
use asteria::clab::linear_interpolation::LinearInterpolation;
// ... (see examples/maze.rs for the full environment definition)

let mut agent = Qlearning::new(network, Box::new(ADOPT::new(1e-3)), 0.99);
let mut explore = DiscreteExploration::new(
    DiscreteExplorationMethod::Egreedy,
    0.5,  // initial epsilon
    Some(Box::new(LinearInterpolation::new(0.5, 0.0, episodes))),
);

for ep in 0..episodes {
    env.reset();
    explore.update(ep);

    while !env.is_finished() {
        let state  = env.get_state();
        let action = explore.explore(agent.get_action(&state));

        env.do_action(&action);
        let next_state = env.get_state();
        let reward     = env.get_reward();
        let done       = env.is_finished();

        agent.train(&state, &action, &next_state, reward, done);
    }
}
```

**Key differences from Gym:**
- Actions are `Tensor` objects. For discrete spaces, they are one-hot vectors (see `examples/maze.rs` for the pattern). For continuous spaces, they are plain real-valued tensors.
- There is no info dict. The environment is responsible for its own bookkeeping.
- Exploration is a separate object (`DiscreteExploration`, `ContinuousExploration`) rather than being baked into the environment or the algorithm.
- Learning rate scheduling is explicit: create a scheduler, call `sched.step()` each episode, and pass the result to `agent.optimizer.set_lr(...)`.

## External Dependencies

* [**rand**](https://crates.io/crates/rand) — random number generation and sampling.
* [**rand\_distr**](https://crates.io/crates/rand_distr) — statistical distributions, including the Normal distribution used in Ornstein-Uhlenbeck noise.

## Tensor Engine (CLAB)

CLAB is a native N-dimensional array implementation in safe Rust. A `Tensor` is stored in row-major order with explicit shape, stride, and a transpose flag that defers the physical layout change until the next matrix multiply. Supported operations:

* **Matrix multiplication** across all four transpose combinations $(AB,\ A^TB,\ AB^T,\ A^TB^T)$, implemented as cache-friendly loop-reordered kernels.
* **Elementwise operations** — addition, subtraction, scalar multiply, scalar divide — with broadcasting over a leading batch dimension.
* **Reductions** — `reduce_sum` along the batch axis, `mean` along either axis, `max`, `min`, `gather(indices)`, `max_index(dim)`.
* **Initialization** — zero, constant fill, identity, and named initializer schemes (see [Weight Initialization](#weight-initialization)).
* **Concatenation** — `Tensor::concat(slices, axis)` along rows (`axis=0`) or columns (`axis=1`).

**`max_index` semantics (important):** For a 2-D `[rows, cols]` tensor:

- `max_index(0)` iterates over **rows** and returns, for each row, the **column index** of the maximum value. On a `[batch, features]` tensor this gives per-sample argmax — e.g. the predicted class or the greedy Q-action.
- `max_index(1)` iterates over **columns** and returns, for each column, the **row index** of the maximum value.

For a 1-D tensor `max_index(0)` returns the single global argmax index.

DQN and DDPG convert these per-row column indices to flat buffer offsets before calling `gather`:
```rust
let col_indices = q_values.max_index(0);  // [batch] column indices
let flat: Vec<usize> = col_indices.iter().enumerate()
    .map(|(row, &col)| row * n_actions + col).collect();
let max_q = q_values.gather(&flat);
```

No heap allocations occur during a forward or backward pass beyond the initial parameter and gradient tensors.

## Neural Network Framework

Networks are built as directed acyclic graphs. Each `DenseLayer` is a node identified by a string ID; `add_connection(from, to)` adds a directed edge. `init()` runs a topological sort over the graph to derive the forward execution order and its reverse for backpropagation. Multi-input layers receive their inputs concatenated along the feature axis; the backward pass splits the gradient back along the same axis. Parameters are stored as `Arc<Mutex<Param>>` to support asynchronous training (A3C).

### Weight Initialization

| Scheme | Formula |
|---|---|
| `LecunUniform` | $U\!\left[-\sqrt{3/n_{\text{in}}},\ \sqrt{3/n_{\text{in}}}\right]$ |
| `LecunNormal` | $\mathcal{N}\!\left(0,\ \sqrt{1/n_{\text{in}}}\right)$ |
| `GlorotUniform` / `XavierUniform` | $U\!\left[-\sqrt{6/(n_{\text{in}}+n_{\text{out}})},\ \sqrt{6/(n_{\text{in}}+n_{\text{out}})}\right]$ |
| `GlorotNormal` / `XavierNormal` | $\mathcal{N}\!\left(0,\ \sqrt{2/(n_{\text{in}}+n_{\text{out}})}\right)$ |
| `Uniform(lo, hi)` | $U[\text{lo}, \text{hi}]$ (explicit range; e.g. $U[-3\times10^{-3},\ 3\times10^{-3}]$ for DDPG output layers) |

### Activation Functions

| Activation | Forward | Backward |
|---|---|---|
| `Linear` | $f(z) = z$ | $f'(z) = 1$ |
| `Sigmoid` | $f(z) = 1/(1+e^{-z})$ | $f'(z) = f(z)(1-f(z))$ |
| `Tanh` | $f(z) = \tanh(z)$ | $f'(z) = 1 - f(z)^2$ |
| `Relu` | $f(z) = \max(0, z)$ | $f'(z) = \mathbf{1}[z > 0]$ |
| `Softmax` | $f(z)_i = e^{z_i - \max z} / \sum_j e^{z_j - \max z}$ | Jacobian $\partial L/\partial z_j = S_j(\delta_j - \sum_i \delta_i S_i)$ |

> **Note on `Softmax` vs `SoftmaxCrossEntropyFunction`:** For multi-class classification, use `ActivationType::Linear` on the output layer paired with `SoftmaxCrossEntropyFunction`. The fused loss applies softmax internally and returns the correct combined gradient $(\sigma(z) - y)/N$, bypassing the Jacobian entirely and avoiding numerical issues. The `Softmax` activation is kept for policy networks (actor outputs) where the Jacobian is needed for policy-gradient updates.

### Loss Functions

| Loss | Forward | Backward | Typical use |
|---|---|---|---|
| `MseFunction` | $\frac{1}{2N}\sum(p_i - y_i)^2$ | $(p_i - y_i)/N$ | Regression, RL critic targets |
| `SoftmaxCrossEntropyFunction` | $-\frac{1}{N}\sum y_i \ln(\sigma(z)_i)$ | $(\sigma(z)_i - y_i)/N$ | Multi-class classification with `Linear` output |

`SoftmaxCrossEntropyFunction` takes raw **logits** (not softmax outputs). It applies softmax internally during both the forward and backward passes, so the output layer must use `ActivationType::Linear` to avoid double-application.

### Optimizers

All optimizers implement the `Optimizer` trait (`update`, `set_lr`, `get_lr`) and are interchangeable.

| Optimizer | Update rule | Notes |
|---|---|---|
| `Sgd` | $\theta \leftarrow \theta - \alpha g$ | Baseline; no momentum |
| `Adam` | $\theta \leftarrow \theta - \alpha\hat{m}/(\sqrt{\hat{v}}+\epsilon)$ | Bias-corrected; optional L2 weight decay |
| `RAdam` | Rectified Adam | Falls back to SGD-with-momentum when variance estimate is unreliable ($\rho_t \leq 5$); otherwise applies rectification term $r_t$ |
| `ADOPT` | See below | Default in all examples; optimal convergence rate without bounded-noise assumption |

**ADOPT** ([paper](https://arxiv.org/abs/2411.02853)) achieves the optimal $O(1/\sqrt{T})$ convergence rate for any $\beta_2$ without the bounded-noise assumption required by Adam. The key structural change: the second moment $v_t$ is updated **after** the parameter step, so the normalisation at step $t$ uses only gradients $g_1,\ldots,g_{t-1}$ — the current gradient never appears in its own denominator. Step 0 is a pure bootstrap: $v_1 \leftarrow g_0^2$ with no parameter update. From step 1 onward:

$$\hat{g}_t = \mathrm{clip}(g_t/\sqrt{v_{t-1}},\ \pm t^{1/4}), \quad m_t = \beta_1 m_{t-1} + (1-\beta_1)\hat{g}_t, \quad \theta_t = \theta_{t-1} - \alpha m_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$

Defaults: $\beta_1=0.9$, $\beta_2=0.9999$, $\epsilon=10^{-6}$, gradient clipping enabled.

### Learning Rate Schedulers

All schedulers (except `ReduceLROnPlateau`) implement the `LrScheduler` trait (`step() -> f32`, `current_lr() -> f32`). Call `opt.set_lr(sched.step())` once per epoch (supervised) or once per episode (RL). `ReduceLROnPlateau` is metric-driven and has a distinct `step(metric: f32) -> f32` signature.

| Scheduler | Schedule |
|---|---|
| `StepDecay` | $\alpha_t = \alpha_0 \cdot \gamma^{\lfloor t/k \rfloor}$ — multiplies by $\gamma$ every $k$ steps |
| `CosineAnnealing` | $\alpha_t = \alpha_{\min} + \tfrac{1}{2}(\alpha_{\max}-\alpha_{\min})(1 + \cos(\pi t/T_{\max}))$ |
| `LinearDecay` | $\alpha_t = \alpha_{\text{start}} + (\alpha_{\text{end}}-\alpha_{\text{start}}) \cdot \min(t/T, 1)$ |
| `WarmupCosine` | Linear ramp $0 \to \alpha_{\max}$ over `warmup_steps`, then cosine decay to $\alpha_{\min}$ |
| `ReduceLROnPlateau` | $\alpha \leftarrow \max(\alpha \cdot f,\ \alpha_{\min})$ after `patience` non-improving calls |

Usage pattern (supervised, cosine schedule):

```rust
use asteria::core::lr_scheduler::{CosineAnnealing, LrScheduler};
let mut sched = CosineAnnealing::new(1e-3, 1e-5, epochs);
for epoch in 0..epochs {
    opt.set_lr(sched.step());
    // training loop
}
```

Usage pattern (RL, linear decay):

```rust
use asteria::core::lr_scheduler::{LinearDecay, LrScheduler};
let mut sched = LinearDecay::new(1e-3, 1e-4, episodes);
for ep in 0..episodes {
    agent.optimizer.set_lr(sched.step());
    // episode loop
}
```

Supervised examples use `CosineAnnealing` (XOR uses `LinearDecay`). All RL examples use `LinearDecay` from $\alpha=10^{-3}$ to $\alpha=10^{-4}$ over the total episode count.

### Model Serialization

Parameters can be saved and reloaded across sessions:

```rust
net.save("model.bin").expect("save failed");
// later:
net.load("model.bin").expect("load failed");
```

Parameters are written in topologically sorted layer order as little-endian `f32` bytes. The file is portable across machines with the same network architecture.

## Reinforcement Learning Algorithms

All algorithms interact with environments through the `IEnvironment` trait. A shared `ReplayBuffer<T>` provides uniform random experience replay for off-policy methods.

### Environment Interface

```rust
pub trait IEnvironment {
    fn reset(&mut self);
    fn get_state(&mut self) -> Tensor;
    fn do_action(&mut self, action: &Tensor);
    fn get_reward(&mut self) -> f32;
    fn is_finished(&mut self) -> bool;
    fn state_dim(&self) -> usize;
    fn action_dim(&self) -> usize;
}
```

Implement this trait for any custom environment. State and action tensors can be any shape — the examples use `[1, n]` row vectors throughout.

### Exploration

Exploration is handled separately from the learning algorithm:

* **`DiscreteExploration`** — $\varepsilon$-greedy policy with a configurable annealing schedule (e.g. `LinearInterpolation` to decay $\varepsilon$ to zero). Also supports Boltzmann (softmax) sampling.
* **`ContinuousExploration`** — Gaussian noise or Ornstein-Uhlenbeck process $dX_t = \theta(\mu - X_t)\,dt + \sigma\,dW_t$ for temporally correlated action perturbation.

### Discrete Action Space

| Algorithm | Description |
|---|---|
| **TD** | One-step temporal difference: $\delta = V(s) − (r + \gamma V(s'))$ |
| **Q-Learning** | Off-policy TD control: $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$ |
| **SARSA** | On-policy TD control: $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma Q(s',a') - Q(s,a)]$ |
| **DQN** | Deep Q-Network with experience replay, target network, and periodic hard sync |
| **AC** | Vanilla Actor-Critic: TD critic provides the advantage $\delta$ to a REINFORCE actor |
| **A2C** | Synchronous Advantage Actor-Critic — batched rollouts with advantage $A = r + \gamma V(s') - V(s)$ |
| **A3C** | Asynchronous A-C — worker threads compute gradients locally on thread-local network copies and push them to a shared global model protected by `Arc<Mutex<...>>` |
| **PPO** | Proximal Policy Optimization — on-policy clipped surrogate $L^{\text{CLIP}} = \mathbb{E}[\min(r_t(\theta)A_t,\ \text{clip}(r_t(\theta),1-\varepsilon,1+\varepsilon)A_t)]$ with multiple minibatch epochs per rollout |
| **QAC** | Q-value Actor-Critic: Q-network critic, policy gradient actor using critic TD error as advantage signal |

**Note on QAC:** QAC combines a Q-learning (off-policy) critic with a policy-gradient actor. Because Q-learning estimates the optimal Q-function Q\* rather than the policy's own Q^π, the actor advantage signal can be noisy and training is slower than AC or A2C on small environments. QAC is included for completeness and as a research baseline; A2C or PPO are recommended for reliable convergence.

**Sign convention for actor advantages:** All actor-critic methods use the loss-gradient convention: a *negative* advantage signal increases the probability of the selected action; a *positive* signal decreases it. The TD critic returns `V(s) − target` (negative when the action was better than expected), which is consistent with standard gradient descent minimizing the critic loss.

### Continuous Action Space

| Algorithm | Description |
|---|---|
| **CACLA** | Continuous Actor-Critic Learning Automaton — updates actor only when $\delta > 0$ (actual return exceeded prediction), using the regression target $(a - \mu_\theta(s))$ as the actor loss |
| **DDPG** | Deep Deterministic Policy Gradient — off-policy actor-critic with experience replay and soft target updates $\theta' \leftarrow \tau\theta + (1-\tau)\theta'$; actor gradient via $\nabla_\theta J \approx \mathbb{E}[\nabla_a Q(s,a)\vert_{a=\mu_\theta(s)} \cdot \nabla_\theta\mu_\theta(s)]$ |

### Motivation Models

| Model | Description |
|---|---|
| **ForwardModel** | Trains a forward dynamics predictor $\hat{s}' = f_\phi(s, a)$; uses prediction error $\tanh(\|s'-\hat{s}'\|^2)$ as intrinsic reward, bounded to $[0,1)$ |
| **Metacritic** | Maintains a meta-learner that predicts the forward model's prediction error; rewards when actual error deviates from prediction by more than $\sigma$, encouraging genuinely novel exploration |

## Test Suite

`cargo test` runs 42 unit tests and 1 doctest, all passing. Tests are organized across three modules:

### CLAB Tests (34 tests)

**Tensor operations** (`src/clab/tensor.rs`):
- `from_data_roundtrip` — round-trip through `from_data` and `get` preserves values.
- `get_set_2d` — 2-D indexing with `vec![row, col]` reads and writes correctly.
- `zero_like_matches_shape` — `zero_like` produces a zero tensor with the same shape.
- `value_fills_constant` — `with_shape_val` fills all elements to the given constant.
- `identity_init` — identity initialization produces a 1-on-diagonal, 0-elsewhere matrix.
- `reshape_preserves_data` — `reshape` changes shape without altering the flat buffer.
- `max_min_correct` — `max()` and `min()` return the correct extremes.
- `max_index_dim0_returns_per_row_col_index` — `max_index(0)` on a 2×3 matrix returns one column index per row (per-sample argmax for classification/DQN).
- `max_index_dim1_returns_per_col_row_index` — `max_index(1)` returns one row index per column.
- `max_index_rank1` — `max_index(0)` on a rank-1 tensor returns the single global argmax.
- `gather_picks_correct_flat_indices` — `gather` extracts elements at the specified flat indices.
- `mean_dim0_averages_columns` — `mean(0)` computes the column means.
- `concat_rows_dim0` / `concat_cols_dim1` — `Tensor::concat` along both axes.

**Activation functions** (`src/clab/activation_functions.rs`):
- Linear (forward identity, backward pass-through delta unchanged).
- Sigmoid (output in unit interval; 0.5 at zero; correct derivative form).
- Tanh (zero at zero; range in (−1, 1); correct derivative at zero).
- ReLU (zeroes negatives; backward zeroes dead neurons).
- Softmax (numerically stable via `max` subtraction; row sums to 1; correct Jacobian).

**Loss functions** (`src/clab/loss_functions.rs`):
- MSE (zero on perfect prediction; known value check; gradient = $(p-y)/N$).
- `SoftmaxCrossEntropyFunction` (near-zero on confident correct prediction; high on wrong class; backward shape correct; backward sums near zero per row, verifying the $(\sigma(z)-y)/N$ formula).

### Core Tests (3 tests)

**ADOPT optimizer** (`src/core/adopt.rs`):
- `bootstrap_step_skips_param_update` — step 0 initialises the second moment $v_1 = g_0^2$ but makes **no parameter update** (the defining property of ADOPT).
- `step1_updates_params` — step 1 performs a real update and parameters change.
- `xor_converges` — ADOPT drives a 2→4→1 Sigmoid network to solve XOR to MSE < 0.01 within 1000 steps.

### RL Gradient Checks (5 tests)

All RL gradient checks use finite-difference numerical differentiation to verify that the analytical backward pass matches the numerical gradient to within 1%:

- `mse_grad_check` — MSE loss backward on a small network with Sigmoid output.
- `td_critic_grad_check` — TD critic backward produces the correct gradient w.r.t. value network parameters.
- `policy_gradient_grad_check` — Policy gradient backward produces the correct gradient w.r.t. actor parameters.
- `softmax_ce_grad_check` — Fused `SoftmaxCrossEntropyFunction` backward on a two-class problem.
- `ddpg_critic_grad_check` — DDPG critic backward (concatenated state-action input) passes the gradient check.

### Doctest (1 test)

- `LrScheduler` trait — the `LinearDecay` doctest in `src/core/lr_scheduler.rs` verifies that `step()` returns `lr_start` on the first call and that the rate decays after `total_steps` calls.

## Examples and Benchmarks

All results are from `cargo run --release --example <name>` on a single CPU core. LR scheduling is applied in every example. Optimizer: ADOPT throughout. Results represent a single representative run; due to random weight initialization, individual runs may vary slightly.

### Supervised Learning

#### XOR (`examples/xor.rs`)

All four XOR input pairs as a single batch of size 4. MSE loss. `LinearDecay` LR: $10^{-2} \to 10^{-3}$ over 500 steps.

| Architecture | LR Schedule | Steps | Final MSE | Typical outputs |
|---|---|---|---|---|
| 2 → 8 (Sigmoid) → 4 (Sigmoid) → 1 (Sigmoid) | LinearDecay | 500 | < 0.001 | ≈ 0.02, 0.95, 0.96, 0.03 |

> Convergence is stochastic (depends on random initialization). Typical successful runs reach MSE < 0.001 by step 200. On rare initializations a second run may be needed.

#### Iris (`examples/iris.rs`)

150 samples, 4 features, 3 classes. Min-max normalized. Every 5th sample held out (120 train / 30 test). `CosineAnnealing` LR: $10^{-3} \to 10^{-5}$ over 200 epochs.

| Architecture | LR Schedule | Batch | Epochs | Train Acc | Test Acc |
|---|---|---|---|---|---|
| 4 → 16 (ReLU) → 3 (Linear) | CosineAnnealing | 16 | 200 | 95.0% | **100.0%** |

Loss: `SoftmaxCrossEntropyFunction`. Converges to 100% test accuracy by epoch 100.

#### Wine (`examples/wine.rs`)

178 samples, 13 features, 3 classes (ARFF). Min-max normalized. Every 5th sample held out (142 train / 36 test). `CosineAnnealing` LR: $10^{-3} \to 10^{-5}$ over 300 epochs.

| Architecture | LR Schedule | Batch | Epochs | Train Acc | Test Acc |
|---|---|---|---|---|---|
| 13 → 32 (ReLU) → 16 (ReLU) → 3 (Linear) | CosineAnnealing | 8 | 300 | 100.0% | **100.0%** |

Loss: `SoftmaxCrossEntropyFunction`. Converges to 100% training and test accuracy by epoch 50. Due to random initialization, occasional runs may misclassify one or two test samples (97–100% range); re-running resolves this.

#### MNIST (`examples/mnist.rs`)

70,000 handwritten digit images (60k train / 10k test), 784 features, 10 classes. IDX binary format. Pixels normalized to $[0, 1]$. Shuffled each epoch. `CosineAnnealing` LR: $10^{-3} \to 10^{-5}$ over 10 epochs. During training, accuracy is tracked on a 2,000-sample subset for speed; final evaluation uses the full 10,000-sample test set.

| Architecture | LR Schedule | Batch | Epochs | Full Test Acc |
|---|---|---|---|---|
| 784 → 128 (ReLU) → 64 (ReLU) → 10 (Linear) | CosineAnnealing | 64 | 10 | **98.18%** |

Loss: `SoftmaxCrossEntropyFunction`. Always run with `--release`; each epoch takes ~1 minute in debug mode.

### Discrete Reinforcement Learning

#### Maze (`examples/maze.rs`)

4×4 grid world with walls, a hazard cell (reward −1), and a goal cell (reward +1). State: 16-dimensional one-hot. Actions: {Up, Right, Down, Left}. Step penalty: −0.01. Episode limit: 100 steps. 500 training episodes. $\varepsilon$-greedy exploration with linear decay from 0.5 to 0. LR: `LinearDecay` $10^{-3} \to 10^{-4}$ over 500 episodes.

```
F  W  F  F
F  W  F  H
F  F  F  W
F  F  F  G   (G = goal, H = hazard, W = wall, F = free)
```

| Algorithm | Architecture | $\gamma$ | LR Schedule | Reward at convergence | Converges by |
|---|---|---|---|---|---|
| Q-Learning | 16 → 32 (ReLU) → 16 (ReLU) → 4 (Linear) | 0.99 | LinearDecay | ~0.95 | ~ep 200 |
| SARSA | 16 → 32 (ReLU) → 16 (ReLU) → 4 (Linear) | 0.99 | LinearDecay | ~0.95 | ~ep 50 |
| DQN | 16 → 32 (ReLU) → 16 (ReLU) → 4 (Linear) | 0.99 | LinearDecay | ~0.95 | ~ep 200 |
| AC | 16 → 32 (ReLU) → 4 (Softmax) / 16 → 32 (ReLU) → 1 (Linear) | 0.99 | LinearDecay | ~0.95 | ~ep 200 |

All methods reach near-optimal reward (~0.95) after convergence. The optimal path from start to goal takes 5 steps, giving reward `1.0 − 5 × 0.01 = 0.95`. DQN uses a replay buffer of 10,000, batch size 64, and target network sync every 100 steps.

#### Maze Policy Gradient (`examples/maze_pg.rs`)

Same 4×4 maze environment. Exercises the four policy-gradient algorithms added in Asteria. 500 training episodes. $\varepsilon$-greedy exploration decaying from 0.5 to 0.05. LR: `LinearDecay` $10^{-3} \to 10^{-4}$ over 500 episodes. A3C uses 4 parallel worker threads, each running 100 episodes independently.

| Algorithm | Architecture (actor / critic) | $\gamma$ | Notes | Converges by |
|---|---|---|---|---|
| A2C | 16→32(ReLU)→4(Softmax) / 16→32(ReLU)→1(Linear) | 0.99 | Batched advantage, batch=32 | ~ep 100 |
| PPO | 16→32(ReLU)→4(Softmax) / 16→32(ReLU)→1(Linear) | 0.99 | ε=0.2, 4 update epochs, batch=32 | ~ep 100 |
| A3C | 16→32(ReLU)→4(Softmax) / 16→32(ReLU)→1(Linear) | 0.99 | 4 workers × 100 episodes | Parallel |
| QAC | 16→32(ReLU)→4(Softmax) / 16→32(ReLU)→4(Linear) | 0.99 | Q-network critic; slower convergence | Variable |

A2C and PPO reliably converge to near-optimal reward (~0.95) by episode 100. A3C converges across 400 total worker-episodes. QAC uses a Q-learning critic (off-policy) as an advantage signal for the actor; because Q-learning estimates Q* rather than Q^π, the actor advantage signal can be noisy, leading to slower and noisier convergence than A2C or PPO on this task.

### Continuous Reinforcement Learning

#### Simple Continuous Environment (`examples/simple_continuous_env.rs`)

1D reach-target task. State: [position, target] ∈ [−1, 1]². Action: velocity delta ∈ [−1, 1]. Reward: −|pos − target| per step; +10 on success (|pos − target| < 0.05). Episode limit: 200 steps. LR: `LinearDecay` $10^{-3} \to 10^{-4}$ over 500 episodes. Exploration: Gaussian noise $\sigma=0.2$.

| Algorithm | Architecture (actor / critic) | $\gamma$ | Episodes | Success rate |
|---|---|---|---|---|
| CACLA | 2 → 16 (Tanh) → 16 (Tanh) → 1 (Tanh) / 2 → 16 (Tanh) → 1 (Linear) | 0.99 | 500 | **448/500** |

CACLA reaches the target in 448 out of 500 episodes, demonstrating reliable convergence on this task. Success rate improves rapidly after episode 100.

#### Mountain Car (`examples/mountain_car.rs`)

Continuous mountain car. State: [position, velocity]. Action: engine force ∈ [−1, 1]. Reward: $-0.1\cdot\text{force}^2$ per step; $+100 - 0.1\cdot\text{force}^2$ on reaching the goal (position ≥ 0.45, velocity ≥ 0). Episode limit: 1000 steps. LR: `LinearDecay` $10^{-3} \to 10^{-4}$ over 300 episodes.

| Algorithm | Architecture | $\gamma$ | Memory | Batch | Consistent success? |
|---|---|---|---|---|---|
| CACLA | 2 → 32 (Tanh) → 1 (Tanh) / 2 → 32 (Tanh) → 1 (Linear) | 0.99 | — | — | No (0/300) |
| DDPG | Actor: 2 → 64 (Tanh) → 64 (Tanh) → 1 (Tanh) | 0.99 | 100,000 | 64 | **Yes (~ep 60+)** |
| DDPG + ForwardModel | Same as DDPG | 0.99 | 100,000 | 64 | **Yes (ep 0+)** |
| DDPG + Metacritic | Same as DDPG | 0.99 | 100,000 | 64 | **Yes (~ep 30+)** |

Critic architecture for all DDPG variants: (2+1) → 64 (ReLU) → 64 (ReLU) → 1 (Linear). Output layers initialized with $U[-3\times10^{-3}, 3\times10^{-3}]$. Soft update $\tau=0.005$. Exploration: OU noise ($\sigma=0.3$, $\theta=0.15$, dt=0.01) for DDPG; Gaussian $\sigma=0.3$ for CACLA.

CACLA does not solve mountain car: the hill-climbing problem requires temporally coordinated action sequences that CACLA's greedy update cannot discover. DDPG with the forward model consistently solves the task immediately (the intrinsic reward compensates for the sparse extrinsic reward during initial exploration).

#### CartPole (`examples/cart_pole.rs`)

Continuous cart-pole balancing. State: [x, ẋ, θ, θ̇]. Action: force ∈ [−1, 1] (scaled by 10 N). Episode fails on |x| > 2.4 m, |θ| > 12°, or after 200 steps; reward is −1 on failure, 0 otherwise. LR: `LinearDecay` $10^{-3} \to 10^{-4}$ over 300 episodes.

| Algorithm | Architecture | $\gamma$ | Memory | Batch | Sustained 200 steps by ep |
|---|---|---|---|---|---|
| DDPG | Actor: 4 → 64 (ReLU) → 64 (ReLU) → 1 (Tanh) | 0.99 | 100,000 | 64 | **~ep 180** |
| DDPG + ForwardModel | Same | 0.99 | 100,000 | 64 | **~ep 90** |
| DDPG + Metacritic | Same | 0.99 | 100,000 | 64 | Inconsistent |

Critic for all DDPG variants: (4+1) → 64 (ReLU) → 64 (ReLU) → 1 (Linear). Output layers: $U[-3\times10^{-3}, 3\times10^{-3}]$. Soft update $\tau=0.005$. Exploration: OU noise ($\sigma=0.2$, $\theta=0.15$, dt=0.01).

An episode reaching 200 steps is considered solved. DDPG with ForwardModel benefits from intrinsic exploration reward early in training, allowing it to discover the balancing policy ~90 episodes sooner than plain DDPG.

## Planned Features

The following features are planned for future releases:

* **ONNX support.** Export trained networks to `.onnx` format for inference in any ONNX-compatible runtime, and load pre-trained ONNX models for fine-tuning or evaluation. This is the standard interoperability format for neural networks and enables deployment outside the Rust ecosystem.

* **NumPy / CNPy format.** Save and load network parameters as `.npy` / `.npz` archives — the same format supported by [CNPy](https://github.com/rogersce/cnpy) in the original Coeus C++ library. This allows direct exchange of trained weights between Asteria, NumPy, and any Python deep learning framework.

## Acknowledgements

Asteria is a Rust port and extension of [Coeus](https://github.com/Iskandor/Coeus), a C++ reinforcement learning library developed by **Matej Pechac** (Iskandor). The port replaces the C++ backend — which used Intel MKL and optional CUDA/cuBLAS via AVX SIMD intrinsics — with a fully safe, zero-`unsafe` Rust implementation. Algorithmic content is faithfully ported; two bugs confirmed in the Coeus source are corrected (Softmax backward Jacobian and Q-learning delta sign convention). Beyond the port, Asteria extends the original suite with A2C, A3C, and PPO (planned but never shipped in Coeus), introduces ADOPT as a new optimizer, ships a full LR scheduler suite, and adds Iris/Wine/MNIST supervised learning benchmarks. QAC exists in both libraries; Asteria's version corrects the actor-advantage signal.
