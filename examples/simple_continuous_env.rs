use asteria::clab::activation_functions::ActivationType;
use asteria::clab::tensor::Tensor;
use asteria::clab::tensor_initializer::{InitializerType, TensorInitializer};
use asteria::core::adopt::ADOPT;
use asteria::core::dense_layer::DenseLayer;
use asteria::core::lr_scheduler::{LinearDecay, LrScheduler};
use asteria::core::neural_network::NeuralNetwork;
use asteria::rl::cacla::CACLA;
use asteria::rl::continuous_exploration::ContinuousExploration;
use asteria::rl::ienvironment::IEnvironment;

// 1D continuous environment: agent controls velocity to reach target position.
// State:  [position, target]  in [-1, 1]²
// Action: velocity delta      in [-1, 1]
// Reward: -|position - target| per step; +10 bonus on success
// Episode ends when |position - target| < 0.05 or steps >= 200
struct SimpleContinuousEnv {
    position: f32,
    target: f32,
    steps: usize,
    max_steps: usize,
    done: bool,
    reward: f32,
    rng_state: u64,
}

impl SimpleContinuousEnv {
    fn new() -> Self {
        SimpleContinuousEnv {
            position: 0.0,
            target: 1.0,
            steps: 0,
            max_steps: 200,
            done: false,
            reward: 0.0,
            rng_state: 99991,
        }
    }

    fn rand_uniform(&mut self, lo: f32, hi: f32) -> f32 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        let t = (self.rng_state as f32) / (u64::MAX as f32);
        lo + t * (hi - lo)
    }
}

impl IEnvironment for SimpleContinuousEnv {
    fn reset(&mut self) {
        self.position = self.rand_uniform(-1.0, 0.5);
        self.target = self.rand_uniform(0.5, 1.0);
        self.steps = 0;
        self.done = false;
        self.reward = 0.0;
    }

    fn get_state(&mut self) -> Tensor {
        Tensor::from_data(vec![1, 2], vec![self.position, self.target])
    }

    fn do_action(&mut self, action: &Tensor) {
        let delta = action.get(vec![0, 0]).clamp(-1.0, 1.0) * 0.1;
        self.position = (self.position + delta).clamp(-1.0, 1.0);
        self.steps += 1;

        let dist = (self.position - self.target).abs();
        if dist < 0.05 {
            self.reward = 10.0;
            self.done = true;
        } else if self.steps >= self.max_steps {
            self.reward = -dist;
            self.done = true;
        } else {
            self.reward = -dist;
        }
    }

    fn get_reward(&mut self) -> f32 { self.reward }
    fn is_finished(&mut self) -> bool { self.done }
    fn state_dim(&self) -> usize { 2 }
    fn action_dim(&self) -> usize { 1 }
}

fn build_actor(state_dim: usize, action_dim: usize) -> NeuralNetwork {
    let mut net = NeuralNetwork::new();
    net.add_layer(DenseLayer::new(
        "ah0".to_string(), 16, ActivationType::Tanh,
        TensorInitializer::new(InitializerType::LecunUniform), state_dim,
    ));
    net.add_layer(DenseLayer::new(
        "ah1".to_string(), 16, ActivationType::Tanh,
        TensorInitializer::new(InitializerType::LecunUniform), 0,
    ));
    net.add_layer(DenseLayer::new(
        "aout".to_string(), action_dim, ActivationType::Tanh,
        TensorInitializer::new(InitializerType::LecunUniform), 0,
    ));
    net.add_connection("ah0", "ah1");
    net.add_connection("ah1", "aout");
    net.init();
    net
}

fn build_critic(state_dim: usize) -> NeuralNetwork {
    let mut net = NeuralNetwork::new();
    net.add_layer(DenseLayer::new(
        "ch0".to_string(), 16, ActivationType::Tanh,
        TensorInitializer::new(InitializerType::LecunUniform), state_dim,
    ));
    net.add_layer(DenseLayer::new(
        "cout".to_string(), 1, ActivationType::Linear,
        TensorInitializer::new(InitializerType::LecunUniform), 0,
    ));
    net.add_connection("ch0", "cout");
    net.init();
    net
}

fn main() {
    let episodes = 500;
    let state_dim = 2usize;
    let action_dim = 1usize;
    let gamma = 0.99;
    let lr = 1e-3;

    println!("=== Simple Continuous Env: CACLA (LinearDecay LR) ===");
    let mut env = SimpleContinuousEnv::new();
    let mut agent = CACLA::new(
        build_actor(state_dim, action_dim), Box::new(ADOPT::new(lr)),
        build_critic(state_dim), Box::new(ADOPT::new(lr)),
        gamma,
    );
    let mut explore = ContinuousExploration::new(None);
    explore.init_gaussian(0.2);
    let mut sched = LinearDecay::new(lr, lr * 0.1, episodes);

    let mut successes = 0u32;

    for ep in 0..episodes {
        let new_lr = sched.step();
        agent.actor_optimizer.set_lr(new_lr);
        agent.critic.optimizer.set_lr(new_lr);

        env.reset();
        explore.reset();
        let mut total_reward = 0.0;

        while !env.is_finished() {
            let state = env.get_state();
            let raw_action = agent.get_action(&state).clone();
            let action = explore.explore(&raw_action);

            env.do_action(&action);
            let next_state = env.get_state();
            let reward = env.get_reward();
            let done = env.is_finished();

            agent.train(&state, &action, &next_state, reward, done);
            total_reward += reward;
        }

        if (env.position - env.target).abs() < 0.05 {
            successes += 1;
        }

        if ep % 50 == 0 {
            println!(
                "  ep {:>4}  lr {:.2e}  reward {:>8.3}  successes so far: {}",
                ep, new_lr, total_reward, successes
            );
        }
    }

    println!("\nFinal success rate: {}/{}", successes, episodes);
}
