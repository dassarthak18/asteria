// Exercises A2C, QAC, PPO, and A3C on the 4×4 maze environment.
//
// Same maze layout as maze.rs:
//  F  W  F  F
//  F  W  F  H
//  F  F  F  W
//  F  F  F  G   (G = goal, H = hazard, W = wall)
//
// Run with: cargo run --example maze_pg

use asteria::clab::activation_functions::ActivationType;
use asteria::clab::linear_interpolation::LinearInterpolation;
use asteria::clab::tensor::Tensor;
use asteria::clab::tensor_initializer::{InitializerType, TensorInitializer};
use asteria::core::adopt::ADOPT;
use asteria::core::dense_layer::DenseLayer;
use asteria::core::lr_scheduler::{LinearDecay, LrScheduler};
use asteria::core::neural_network::NeuralNetwork;
use asteria::rl::ac2::AC2;
use asteria::rl::ac3::AC3;
use asteria::rl::discrete_exploration::{DiscreteExploration, DiscreteExplorationMethod};
use asteria::rl::ienvironment::IEnvironment;
use asteria::rl::ppo::PPO;
use asteria::rl::qac::QAC;

const TOPOLOGY: [u8; 16] = [0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 0, 1, 0, 0, 0, 3];

const UP: usize = 0;
const RIGHT: usize = 1;
const DOWN: usize = 2;
const LEFT: usize = 3;

#[derive(Clone)]
struct Maze {
    state: usize,
    steps: usize,
    max_steps: usize,
    done: bool,
    reward: f32,
}

impl Maze {
    fn new() -> Self {
        Maze { state: 0, steps: 0, max_steps: 100, done: false, reward: 0.0 }
    }

    fn try_move(&self, pos: usize, action: usize) -> usize {
        let row = pos / 4;
        let col = pos % 4;
        let (nr, nc) = match action {
            UP    => (row.saturating_sub(1), col),
            DOWN  => ((row + 1).min(3), col),
            LEFT  => (row, col.saturating_sub(1)),
            RIGHT => (row, (col + 1).min(3)),
            _     => (row, col),
        };
        let next = nr * 4 + nc;
        if TOPOLOGY[next] == 1 { pos } else { next }
    }
}

impl IEnvironment for Maze {
    fn reset(&mut self) {
        self.state = 0;
        self.steps = 0;
        self.done = false;
        self.reward = 0.0;
    }

    fn get_state(&mut self) -> Tensor {
        let mut t = Tensor::with_shape_val(vec![1, 16], 0.0);
        t.set(vec![0, self.state], 1.0);
        t
    }

    fn do_action(&mut self, action: &Tensor) {
        let act = action.max_index(0)[0];
        let next = self.try_move(self.state, act);
        self.state = next;
        self.steps += 1;
        match TOPOLOGY[next] {
            3 => { self.reward = 1.0;  self.done = true; }
            2 => { self.reward = -1.0; self.done = true; }
            _ => {
                self.reward = -0.01;
                if self.steps >= self.max_steps { self.done = true; }
            }
        }
    }

    fn get_reward(&mut self) -> f32 { self.reward }
    fn is_finished(&mut self) -> bool { self.done }
    fn state_dim(&self) -> usize { 16 }
    fn action_dim(&self) -> usize { 4 }
}

fn one_hot(idx: usize, n: usize) -> Tensor {
    let mut t = Tensor::with_shape_val(vec![1, n], 0.0);
    t.set(vec![0, idx], 1.0);
    t
}

fn build_actor(inputs: usize, outputs: usize) -> NeuralNetwork {
    let mut net = NeuralNetwork::new();
    net.add_layer(DenseLayer::new(
        "ah0".to_string(), 32, ActivationType::Relu,
        TensorInitializer::new(InitializerType::LecunUniform), inputs,
    ));
    net.add_layer(DenseLayer::new(
        "aout".to_string(), outputs, ActivationType::Softmax,
        TensorInitializer::new(InitializerType::LecunUniform), 0,
    ));
    net.add_connection("ah0", "aout");
    net.init();
    net
}

fn build_critic(inputs: usize) -> NeuralNetwork {
    let mut net = NeuralNetwork::new();
    net.add_layer(DenseLayer::new(
        "ch0".to_string(), 32, ActivationType::Relu,
        TensorInitializer::new(InitializerType::LecunUniform), inputs,
    ));
    net.add_layer(DenseLayer::new(
        "cout".to_string(), 1, ActivationType::Linear,
        TensorInitializer::new(InitializerType::LecunUniform), 0,
    ));
    net.add_connection("ch0", "cout");
    net.init();
    net
}

fn build_q_network(inputs: usize, outputs: usize) -> NeuralNetwork {
    let mut net = NeuralNetwork::new();
    net.add_layer(DenseLayer::new(
        "h0".to_string(), 32, ActivationType::Relu,
        TensorInitializer::new(InitializerType::LecunUniform), inputs,
    ));
    net.add_layer(DenseLayer::new(
        "out".to_string(), outputs, ActivationType::Linear,
        TensorInitializer::new(InitializerType::LecunUniform), 0,
    ));
    net.add_connection("h0", "out");
    net.init();
    net
}

// ── A2C ───────────────────────────────────────────────────────────────────────

fn run_episode_a2c(env: &mut Maze, agent: &mut AC2, explore: &mut DiscreteExploration, ep: i32) -> (f32, usize) {
    env.reset();
    explore.update(ep);
    let mut total_reward = 0.0;
    let mut steps = 0;

    while !env.is_finished() {
        let state = env.get_state();
        let probs = agent.get_action(&state);
        let action = explore.explore(&probs);
        let action_oh = one_hot(action.max_index(0)[0], 4);

        env.do_action(&action_oh);
        let next_state = env.get_state();
        let reward = env.get_reward();
        let done = env.is_finished();

        agent.train(&state, &action_oh, &next_state, reward, done);
        total_reward += reward;
        steps += 1;
    }
    (total_reward, steps)
}

// ── QAC ───────────────────────────────────────────────────────────────────────

fn run_episode_qac(env: &mut Maze, agent: &mut QAC, explore: &mut DiscreteExploration, ep: i32) -> (f32, usize) {
    env.reset();
    explore.update(ep);
    let mut total_reward = 0.0;
    let mut steps = 0;

    while !env.is_finished() {
        let state = env.get_state();
        let probs = agent.get_action(&state);
        let action = explore.explore(&probs);
        let action_oh = one_hot(action.max_index(0)[0], 4);

        env.do_action(&action_oh);
        let next_state = env.get_state();
        let reward = env.get_reward();
        let done = env.is_finished();

        agent.train(&state, &action_oh, &next_state, reward, done);
        total_reward += reward;
        steps += 1;
    }
    (total_reward, steps)
}

// ── PPO ───────────────────────────────────────────────────────────────────────

fn run_episode_ppo(env: &mut Maze, agent: &mut PPO, explore: &mut DiscreteExploration, ep: i32) -> (f32, usize) {
    env.reset();
    explore.update(ep);
    let mut total_reward = 0.0;
    let mut steps = 0;

    while !env.is_finished() {
        let state = env.get_state();
        let probs = agent.get_action(&state);
        let action = explore.explore(&probs);
        let action_oh = one_hot(action.max_index(0)[0], 4);

        env.do_action(&action_oh);
        let next_state = env.get_state();
        let reward = env.get_reward();
        let done = env.is_finished();

        agent.train(&state, &action_oh, &next_state, reward, done);
        total_reward += reward;
        steps += 1;
    }
    (total_reward, steps)
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() {
    let episodes = 500;
    let gamma = 0.99;
    let lr = 1e-3;

    // ── A2C ───────────────────────────────────────────────────────────────────
    println!("=== Maze: A2C (LinearDecay LR) ===");
    {
        let mut env = Maze::new();
        let mut agent = AC2::new(
            build_actor(16, 4), Box::new(ADOPT::new(lr)),
            build_critic(16), Box::new(ADOPT::new(lr)),
            gamma, 32,
        );
        let mut explore = DiscreteExploration::new(
            DiscreteExplorationMethod::Egreedy, 0.5,
            Some(Box::new(LinearInterpolation::new(0.5, 0.05, episodes))),
        );
        let mut sched = LinearDecay::new(lr, lr * 0.1, episodes as usize);

        for ep in 0..episodes {
            let new_lr = sched.step();
            agent.actor_optimizer.set_lr(new_lr);
            agent.critic_optimizer.set_lr(new_lr);
            let (reward, steps) = run_episode_a2c(&mut env, &mut agent, &mut explore, ep);
            if ep % 50 == 0 {
                println!("  ep {:>4}  reward {:>6.2}  steps {:>4}", ep, reward, steps);
            }
        }
    }

    // ── QAC ───────────────────────────────────────────────────────────────────
    println!("\n=== Maze: QAC (LinearDecay LR) ===");
    {
        let mut env = Maze::new();
        let mut agent = QAC::new(
            build_actor(16, 4), Box::new(ADOPT::new(lr)),
            build_q_network(16, 4), Box::new(ADOPT::new(lr)),
            gamma,
        );
        let mut explore = DiscreteExploration::new(
            DiscreteExplorationMethod::Egreedy, 0.5,
            Some(Box::new(LinearInterpolation::new(0.5, 0.05, episodes))),
        );
        let mut sched = LinearDecay::new(lr, lr * 0.1, episodes as usize);

        for ep in 0..episodes {
            let new_lr = sched.step();
            agent.actor.optimizer.set_lr(new_lr);
            agent.critic.optimizer.set_lr(new_lr);
            let (reward, steps) = run_episode_qac(&mut env, &mut agent, &mut explore, ep);
            if ep % 50 == 0 {
                println!("  ep {:>4}  reward {:>6.2}  steps {:>4}", ep, reward, steps);
            }
        }
    }

    // ── PPO ───────────────────────────────────────────────────────────────────
    println!("\n=== Maze: PPO (LinearDecay LR) ===");
    {
        let mut env = Maze::new();
        let mut agent = PPO::new(
            build_actor(16, 4), Box::new(ADOPT::new(lr)),
            build_critic(16), Box::new(ADOPT::new(lr)),
            gamma, 0.2, 4, 32, 64,
        );
        let mut explore = DiscreteExploration::new(
            DiscreteExplorationMethod::Egreedy, 0.5,
            Some(Box::new(LinearInterpolation::new(0.5, 0.05, episodes))),
        );
        let mut sched = LinearDecay::new(lr, lr * 0.1, episodes as usize);

        for ep in 0..episodes {
            let new_lr = sched.step();
            agent.actor_optimizer.set_lr(new_lr);
            agent.critic_optimizer.set_lr(new_lr);
            let (reward, steps) = run_episode_ppo(&mut env, &mut agent, &mut explore, ep);
            if ep % 50 == 0 {
                println!("  ep {:>4}  reward {:>6.2}  steps {:>4}", ep, reward, steps);
            }
        }
    }

    // ── A3C ───────────────────────────────────────────────────────────────────
    println!("\n=== Maze: A3C (4 workers, no LR schedule) ===");
    {
        let mut agent = AC3::new(
            build_actor(16, 4), Box::new(ADOPT::new(lr)),
            build_critic(16), Box::new(ADOPT::new(lr)),
            gamma, 4,
        );
        // Each worker runs 100 episodes; env_factory creates a fresh Maze per worker.
        agent.train(Maze::new);
    }
}
