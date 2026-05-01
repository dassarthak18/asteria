//! Reinforcement learning algorithms, environments, and exploration strategies.
//!
//! ## Environment
//! All agents interact with environments through the [`IEnvironment`](ienvironment::IEnvironment) trait,
//! which models a Markov Decision Process (MDP).
//!
//! ## Algorithms
//!
//! | Algorithm | Module | Action space | Notes |
//! |---|---|---|---|
//! | Q-Learning | [`qlearning`] | Discrete | Neural function approximator |
//! | SARSA | [`sarsa`] | Discrete | On-policy Q-learning |
//! | DQN | [`dqn`] | Discrete | Replay buffer + target network |
//! | AC | [`ac`] | Discrete | Online actor-critic |
//! | A2C | [`ac2`] | Discrete | Batched actor-critic |
//! | A3C | [`ac3`] | Discrete | Async multi-worker actor-critic |
//! | PPO | [`ppo`] | Discrete | Clipped surrogate objective |
//! | QAC | [`qac`] | Discrete | Q-value actor-critic |
//! | CACLA | [`cacla`] | Continuous | Conditional actor update |
//! | DDPG | [`ddpg`] | Continuous | Deterministic policy + soft target updates |
//!
//! ## Intrinsic motivation
//! [`ForwardModel`](forward_model::ForwardModel) and [`Metacritic`](metacritic::Metacritic)
//! compute curiosity-driven intrinsic rewards that can be summed with extrinsic rewards.
//!
//! ## Exploration
//! - Discrete: [`DiscreteExploration`](discrete_exploration::DiscreteExploration) (ε-greedy, Boltzmann).
//! - Continuous: [`ContinuousExploration`](continuous_exploration::ContinuousExploration) (Gaussian, OU noise),
//!   [`OuNoise`](continuous_exploration::OuNoise) (standalone OU process).

pub mod ienvironment;
pub mod replay_buffer;
pub mod discrete_exploration;
pub mod continuous_exploration;
pub mod td;
pub mod qlearning;
pub mod sarsa;
pub mod dqn;
pub mod ac;
pub mod ac2;
pub mod ppo;
pub mod ac3;
pub mod qac;
pub mod cacla;
pub mod ddpg;
pub mod policy_gradient;
pub mod metacritic;
pub mod forward_model;

#[cfg(test)]
mod tests {
    use crate::clab::activation_functions::ActivationType;
    use crate::clab::loss_functions::{LossFunction, MseFunction, SoftmaxCrossEntropyFunction};
    use crate::clab::tensor::Tensor;
    use crate::clab::tensor_initializer::{InitializerType, TensorInitializer};
    use crate::core::dense_layer::DenseLayer;
    use crate::core::neural_network::NeuralNetwork;

    fn small_net(in_dim: usize, hidden: usize, out_dim: usize, out_act: ActivationType) -> NeuralNetwork {
        let mut net = NeuralNetwork::new();
        // Debug(0.1): deterministic, small weights keep tanh in the linear regime
        net.add_layer(DenseLayer::new(
            "h0".to_string(), hidden, ActivationType::Tanh,
            TensorInitializer::new(InitializerType::Debug(0.1)), in_dim,
        ));
        net.add_layer(DenseLayer::new(
            "out".to_string(), out_dim, out_act,
            TensorInitializer::new(InitializerType::Debug(0.1)), 0,
        ));
        net.add_connection("h0", "out");
        net.init();
        net
    }

    /// Numerical gradient check via symmetric finite differences.
    ///
    /// Calls `backward` once with `loss_grad` to obtain analytic parameter gradients,
    /// then perturbs each scalar parameter ±`eps` and measures the finite-difference slope
    /// using `scalar_loss(output)`. Returns the maximum relative error over all parameters.
    fn grad_check<F>(
        net: &mut NeuralNetwork,
        x: &Tensor,
        loss_grad: &mut Tensor,
        mut scalar_loss: F,
    ) -> f32
    where
        F: FnMut(&Tensor) -> f32,
    {
        net.backward(loss_grad);

        let mut param_ids: Vec<String> = net.model.model.keys().cloned().collect();
        param_ids.sort_by_key(|k| {
            k.strip_prefix("param_").and_then(|n| n.parse::<u32>().ok()).unwrap_or(0)
        });

        let analytic: Vec<f32> = param_ids
            .iter()
            .flat_map(|id| net.model.model[id].lock().unwrap().gradient.data.clone())
            .collect();

        let eps = 1e-3_f32;
        let mut numerical = Vec::new();
        for id in &param_ids {
            let n = net.model.model[id].lock().unwrap().params.size;
            for i in 0..n {
                net.model.model[id].lock().unwrap().params.data[i] += eps;
                let out_p = net.forward(x).clone();
                let fp = scalar_loss(&out_p);

                net.model.model[id].lock().unwrap().params.data[i] -= 2.0 * eps;
                let out_m = net.forward(x).clone();
                let fm = scalar_loss(&out_m);

                net.model.model[id].lock().unwrap().params.data[i] += eps;
                numerical.push((fp - fm) / (2.0 * eps));
            }
        }

        // Use 1e-5 as the denominator floor so that near-zero gradients (both analytic and
        // numerical below ~1e-5 in absolute value) are compared on an absolute scale instead
        // of inflating the relative error with f32 float noise.
        analytic
            .iter()
            .zip(numerical.iter())
            .map(|(a, n)| (a - n).abs() / a.abs().max(n.abs()).max(1e-5))
            .fold(0.0_f32, f32::max)
    }

    // ── MSE loss gradient ─────────────────────────────────────────────────────

    #[test]
    fn mse_grad_check() {
        let mut net = small_net(3, 4, 2, ActivationType::Linear);
        let x = Tensor::from_data(vec![1, 3], vec![0.5, -0.3, 0.8]);
        let target = Tensor::from_data(vec![1, 2], vec![1.0, 0.0]);

        let output = net.forward(&x).clone();
        let mut loss_grad = MseFunction.backward(&output, &target);

        let err = grad_check(&mut net, &x, &mut loss_grad, |out| {
            MseFunction.forward(out, &target)
        });
        assert!(err < 0.01, "MSE grad check: max relative error = {err:.4}");
    }

    // ── Softmax + cross-entropy gradient ──────────────────────────────────────

    #[test]
    fn softmax_ce_grad_check() {
        let mut net = small_net(3, 4, 3, ActivationType::Linear);
        let x = Tensor::from_data(vec![1, 3], vec![0.2, -0.6, 1.0]);
        let target = Tensor::from_data(vec![1, 3], vec![0.0, 1.0, 0.0]);

        let output = net.forward(&x).clone();
        let mut loss_grad = SoftmaxCrossEntropyFunction.backward(&output, &target);

        let err = grad_check(&mut net, &x, &mut loss_grad, |out| {
            SoftmaxCrossEntropyFunction.forward(out, &target)
        });
        // 5% tolerance: softmax nonlinearity amplifies finite-difference error in f32
        assert!(err < 0.05, "SoftmaxCE grad check: max relative error = {err:.4}");
    }

    // ── Policy gradient (REINFORCE) ───────────────────────────────────────────

    #[test]
    fn policy_gradient_grad_check() {
        // REINFORCE gradient: δ/π(a|s) at the taken action, 0 elsewhere.
        // Scalar loss: δ · log π(a|s).
        let mut net = small_net(3, 4, 3, ActivationType::Softmax);
        let x = Tensor::from_data(vec![1, 3], vec![0.1, 0.5, -0.2]);
        let action_idx = 1usize;
        let delta_val = 0.5_f32;

        let output = net.forward(&x).clone();
        let pi_a = output.get(vec![0, action_idx]);

        let mut loss_grad = Tensor::with_shape_val(vec![1, 3], 0.0);
        loss_grad.set(vec![0, action_idx], delta_val / pi_a);

        let err = grad_check(&mut net, &x, &mut loss_grad, |out| {
            let pi = out.get(vec![0, action_idx]).max(1e-10);
            delta_val * pi.ln()
        });
        assert!(err < 0.01, "PolicyGradient grad check: max relative error = {err:.4}");
    }

    // ── TD critic (semi-gradient, fixed bootstrap target) ─────────────────────

    #[test]
    fn td_critic_grad_check() {
        // Semi-gradient TD: loss = 0.5·(V(s) − target)², target treated as constant.
        let mut net = small_net(3, 4, 1, ActivationType::Linear);
        let state = Tensor::from_data(vec![1, 3], vec![0.3, -0.1, 0.7]);
        let target = 1.0_f32 + 0.99 * 0.5_f32; // r + γ·V(s') with V(s') fixed at 0.5

        let v_s = net.forward(&state).get(vec![0, 0]);
        let td_error = v_s - target;
        let mut loss_grad = Tensor::from_data(vec![1, 1], vec![td_error]);

        let err = grad_check(&mut net, &state, &mut loss_grad, |out| {
            0.5 * (out.get(vec![0, 0]) - target).powi(2)
        });
        // 5% tolerance: Tanh hidden layer causes nonlinear finite-difference error in f32
        assert!(err < 0.05, "TD critic grad check: max relative error = {err:.4}");
    }

    // ── DDPG critic (MSE on concatenated state-action input) ─────────────────

    #[test]
    fn ddpg_critic_grad_check() {
        // DDPG critic Q(s,a): MSE against a fixed Bellman target.
        // Input: [state ∥ action] concatenated.
        let in_dim = 5usize; // 3-dim state + 2-dim action
        let mut net = small_net(in_dim, 8, 1, ActivationType::Linear);

        let sa = Tensor::from_data(vec![1, in_dim], vec![0.4, -0.2, 0.6, 0.1, -0.5]);
        let q_target = Tensor::from_data(vec![1, 1], vec![0.8]);

        let output = net.forward(&sa).clone();
        let mut loss_grad = MseFunction.backward(&output, &q_target);

        let err = grad_check(&mut net, &sa, &mut loss_grad, |out| {
            MseFunction.forward(out, &q_target)
        });
        // 5% tolerance: tanh + linear composition with concatenated input
        assert!(err < 0.05, "DDPG critic grad check: max relative error = {err:.4}");
    }
}
