use asteria::clab::activation_functions::ActivationType;
use asteria::clab::linear_interpolation::LinearInterpolation;
use asteria::clab::tensor::Tensor;
use asteria::clab::tensor_initializer::{InitializerType, TensorInitializer};
use asteria::core::adopt::ADOPT;
use asteria::core::dense_layer::DenseLayer;
use asteria::core::lr_scheduler::{LinearDecay, LrScheduler};
use asteria::core::neural_network::NeuralNetwork;
use asteria::rl::ac::AC;
use asteria::rl::discrete_exploration::{DiscreteExploration, DiscreteExplorationMethod};
use asteria::rl::dqn::DQN;
use asteria::rl::ienvironment::IEnvironment;
use asteria::rl::qlearning::Qlearning;
use asteria::rl::sarsa::SARSA;

// 4x4 grid: 0=free, 1=wall, 2=hazard, 3=goal
// Row-major indexing: cell = row*4 + col
//
//  0  1  2  3
//  4  5  6  7
//  8  9 10 11
// 12 13 14 15
//
// Topology:
//  F  W  F  F
//  F  W  F  H
//  F  F  F  W
//  F  F  F  G
const TOPOLOGY: [u8; 16] = [0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 0, 1, 0, 0, 0, 3];

const UP: usize = 0;
const RIGHT: usize = 1;
const DOWN: usize = 2;
const LEFT: usize = 3;

struct Maze {
    state: usize,
    steps: usize,
    max_steps: usize,
    done: bool,
    reward: f32,
}

impl Maze {
    fn new() -> Self {
        Maze {
            state: 0,
            steps: 0,
            max_steps: 100,
            done: false,
            reward: 0.0,
        }
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

    fn get_reward(&mut self) -> f32 {
        self.reward
    }

    fn is_finished(&mut self) -> bool {
        self.done
    }

    fn state_dim(&self) -> usize { 16 }
    fn action_dim(&self) -> usize { 4 }
}

fn one_hot_action(idx: usize, n: usize) -> Tensor {
    let mut t = Tensor::with_shape_val(vec![1, n], 0.0);
    t.set(vec![0, idx], 1.0);
    t
}

fn build_q_network(inputs: usize, outputs: usize) -> NeuralNetwork {
    let mut net = NeuralNetwork::new();
    net.add_layer(DenseLayer::new(
        "h0".to_string(), 32, ActivationType::Relu,
        TensorInitializer::new(InitializerType::LecunUniform), inputs,
    ));
    net.add_layer(DenseLayer::new(
        "h1".to_string(), 16, ActivationType::Relu,
        TensorInitializer::new(InitializerType::LecunUniform), 0,
    ));
    net.add_layer(DenseLayer::new(
        "out".to_string(), outputs, ActivationType::Linear,
        TensorInitializer::new(InitializerType::LecunUniform), 0,
    ));
    net.add_connection("h0", "h1");
    net.add_connection("h1", "out");
    net.init();
    net
}

fn run_episode_qlearning(env: &mut Maze, agent: &mut Qlearning, explore: &mut DiscreteExploration, ep: i32) -> (f32, usize) {
    env.reset();
    explore.update(ep);
    let mut total_reward = 0.0;
    let mut steps = 0;

    while !env.is_finished() {
        let state = env.get_state();
        let q_vals = agent.network.forward(&state).clone();
        let action = explore.explore(&q_vals);
        let action_one_hot = one_hot_action(action.max_index(0)[0], 4);

        env.do_action(&action_one_hot);
        let next_state = env.get_state();
        let reward = env.get_reward();
        let done = env.is_finished();

        agent.train(&state, &action_one_hot, &next_state, reward, done);
        total_reward += reward;
        steps += 1;
    }
    (total_reward, steps)
}

fn run_episode_sarsa(env: &mut Maze, agent: &mut SARSA, explore: &mut DiscreteExploration, ep: i32) -> (f32, usize) {
    env.reset();
    explore.update(ep);

    let state0 = env.get_state();
    let q0 = agent.critic.forward(&state0).clone();
    let mut action = explore.explore(&q0);
    let mut total_reward = 0.0;
    let mut steps = 0;

    while !env.is_finished() {
        let state = env.get_state();
        let action_one_hot = one_hot_action(action.max_index(0)[0], 4);

        env.do_action(&action_one_hot);
        let next_state = env.get_state();
        let reward = env.get_reward();
        let done = env.is_finished();

        let q_next = agent.critic.forward(&next_state).clone();
        let next_action = explore.explore(&q_next);

        agent.train(&state, &action_one_hot, &next_state, &next_action, reward, done);
        action = next_action;
        total_reward += reward;
        steps += 1;
    }
    (total_reward, steps)
}

fn run_episode_dqn(env: &mut Maze, agent: &mut DQN, explore: &mut DiscreteExploration, ep: i32) -> (f32, usize) {
    env.reset();
    explore.update(ep);
    let mut total_reward = 0.0;
    let mut steps = 0;

    while !env.is_finished() {
        let state = env.get_state();
        let q_vals = agent.network.forward(&state).clone();
        let action = explore.explore(&q_vals);
        let action_one_hot = one_hot_action(action.max_index(0)[0], 4);

        env.do_action(&action_one_hot);
        let next_state = env.get_state();
        let reward = env.get_reward();
        let done = env.is_finished();

        agent.train(&state, &action_one_hot, &next_state, reward, done);
        total_reward += reward;
        steps += 1;
    }
    (total_reward, steps)
}

fn run_episode_ac(env: &mut Maze, agent: &mut AC, explore: &mut DiscreteExploration, ep: i32) -> (f32, usize) {
    env.reset();
    explore.update(ep);
    let mut total_reward = 0.0;
    let mut steps = 0;

    while !env.is_finished() {
        let state = env.get_state();
        let probs = agent.get_action(&state);
        let action = explore.explore(&probs);
        let action_one_hot = one_hot_action(action.max_index(0)[0], 4);

        env.do_action(&action_one_hot);
        let next_state = env.get_state();
        let reward = env.get_reward();
        let done = env.is_finished();

        agent.train(&state, &action_one_hot, &next_state, reward, done);
        total_reward += reward;
        steps += 1;
    }
    (total_reward, steps)
}

fn main() {
    let episodes = 500;
    let gamma = 0.99;
    let lr = 1e-3;

    println!("=== Maze: Q-Learning (LinearDecay LR) ===");
    {
        let mut env = Maze::new();
        let net = build_q_network(16, 4);
        let mut agent = Qlearning::new(net, Box::new(ADOPT::new(lr)), gamma);
        let mut explore = DiscreteExploration::new(
            DiscreteExplorationMethod::Egreedy, 0.5,
            Some(Box::new(LinearInterpolation::new(0.5, 0.0, episodes))),
        );
        let mut sched = LinearDecay::new(lr, lr * 0.1, episodes as usize);
        for ep in 0..episodes {
            agent.optimizer.set_lr(sched.step());
            let (reward, steps) = run_episode_qlearning(&mut env, &mut agent, &mut explore, ep);
            if ep % 50 == 0 {
                println!("  ep {:>4}  reward {:>6.2}  steps {:>4}", ep, reward, steps);
            }
        }
    }

    println!("\n=== Maze: SARSA (LinearDecay LR) ===");
    {
        let mut env = Maze::new();
        let net = build_q_network(16, 4);
        let mut agent = SARSA::new(net, Box::new(ADOPT::new(lr)), gamma);
        let mut explore = DiscreteExploration::new(
            DiscreteExplorationMethod::Egreedy, 0.5,
            Some(Box::new(LinearInterpolation::new(0.5, 0.0, episodes))),
        );
        let mut sched = LinearDecay::new(lr, lr * 0.1, episodes as usize);
        for ep in 0..episodes {
            agent.critic_optimizer.set_lr(sched.step());
            let (reward, steps) = run_episode_sarsa(&mut env, &mut agent, &mut explore, ep);
            if ep % 50 == 0 {
                println!("  ep {:>4}  reward {:>6.2}  steps {:>4}", ep, reward, steps);
            }
        }
    }

    println!("\n=== Maze: DQN (LinearDecay LR) ===");
    {
        let mut env = Maze::new();
        let net = build_q_network(16, 4);
        let mut agent = DQN::new(net, Box::new(ADOPT::new(lr)), gamma, 10000, 64, 100);
        let mut explore = DiscreteExploration::new(
            DiscreteExplorationMethod::Egreedy, 0.5,
            Some(Box::new(LinearInterpolation::new(0.5, 0.0, episodes))),
        );
        let mut sched = LinearDecay::new(lr, lr * 0.1, episodes as usize);
        for ep in 0..episodes {
            agent.optimizer.set_lr(sched.step());
            let (reward, steps) = run_episode_dqn(&mut env, &mut agent, &mut explore, ep);
            if ep % 50 == 0 {
                println!("  ep {:>4}  reward {:>6.2}  steps {:>4}", ep, reward, steps);
            }
        }
    }

    println!("\n=== Maze: AC (LinearDecay LR) ===");
    {
        let mut env = Maze::new();

        let mut actor_net = NeuralNetwork::new();
        actor_net.add_layer(DenseLayer::new(
            "ah0".to_string(), 32, ActivationType::Relu,
            TensorInitializer::new(InitializerType::LecunUniform), 16,
        ));
        actor_net.add_layer(DenseLayer::new(
            "aout".to_string(), 4, ActivationType::Softmax,
            TensorInitializer::new(InitializerType::LecunUniform), 0,
        ));
        actor_net.add_connection("ah0", "aout");
        actor_net.init();

        let mut critic_net = NeuralNetwork::new();
        critic_net.add_layer(DenseLayer::new(
            "ch0".to_string(), 32, ActivationType::Relu,
            TensorInitializer::new(InitializerType::LecunUniform), 16,
        ));
        critic_net.add_layer(DenseLayer::new(
            "cout".to_string(), 1, ActivationType::Linear,
            TensorInitializer::new(InitializerType::LecunUniform), 0,
        ));
        critic_net.add_connection("ch0", "cout");
        critic_net.init();

        let mut agent = AC::new(
            actor_net, Box::new(ADOPT::new(lr)),
            critic_net, Box::new(ADOPT::new(lr)),
            gamma,
        );
        let mut explore = DiscreteExploration::new(
            DiscreteExplorationMethod::Egreedy, 0.5,
            Some(Box::new(LinearInterpolation::new(0.5, 0.0, episodes))),
        );
        let mut sched = LinearDecay::new(lr, lr * 0.1, episodes as usize);
        for ep in 0..episodes {
            let new_lr = sched.step();
            agent.actor.optimizer.set_lr(new_lr);
            agent.critic.optimizer.set_lr(new_lr);
            let (reward, steps) = run_episode_ac(&mut env, &mut agent, &mut explore, ep);
            if ep % 50 == 0 {
                println!("  ep {:>4}  reward {:>6.2}  steps {:>4}", ep, reward, steps);
            }
        }
    }
}
