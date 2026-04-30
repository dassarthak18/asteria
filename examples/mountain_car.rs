use asteria::clab::activation_functions::ActivationType;
use asteria::clab::tensor::Tensor;
use asteria::clab::tensor_initializer::{InitializerType, TensorInitializer};
use asteria::core::adopt::ADOPT;
use asteria::core::dense_layer::DenseLayer;
use asteria::core::lr_scheduler::{LinearDecay, LrScheduler};
use asteria::core::neural_network::NeuralNetwork;
use asteria::rl::cacla::CACLA;
use asteria::rl::continuous_exploration::{ContinuousExploration, OuNoise};
use asteria::rl::ddpg::DDPG;
use asteria::rl::forward_model::ForwardModel;
use asteria::rl::ienvironment::IEnvironment;
use asteria::rl::metacritic::Metacritic;

const POS_MIN: f32 = -1.2;
const POS_MAX: f32 = 0.6;
const VEL_MIN: f32 = -0.07;
const VEL_MAX: f32 = 0.07;
const GOAL_POS: f32 = 0.45;
const GRAVITY: f32 = 0.0025;
const POWER: f32 = 0.001;
const MAX_STEPS: usize = 1000;

struct MountainCar {
    position: f32,
    velocity: f32,
    steps: usize,
    done: bool,
    reward: f32,
    last_action: f32,
    rng_state: u64,
}

impl MountainCar {
    fn new() -> Self {
        MountainCar {
            position: -0.5,
            velocity: 0.0,
            steps: 0,
            done: false,
            reward: 0.0,
            last_action: 0.0,
            rng_state: 42,
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

impl IEnvironment for MountainCar {
    fn reset(&mut self) {
        self.position = self.rand_uniform(POS_MIN, -0.4);
        self.velocity = 0.0;
        self.steps = 0;
        self.done = false;
        self.reward = 0.0;
        self.last_action = 0.0;
    }

    fn get_state(&mut self) -> Tensor {
        Tensor::from_data(vec![1, 2], vec![self.position, self.velocity])
    }

    fn do_action(&mut self, action: &Tensor) {
        let force = action.get(vec![0, 0]).clamp(-1.0, 1.0);
        self.last_action = force;

        self.velocity = (self.velocity + POWER * force - GRAVITY * (3.0 * self.position).cos())
            .clamp(VEL_MIN, VEL_MAX);
        self.position = (self.position + self.velocity).clamp(POS_MIN, POS_MAX);

        if self.position == POS_MIN {
            self.velocity = 0.0;
        }
        self.steps += 1;

        if self.position >= GOAL_POS && self.velocity >= 0.0 {
            self.reward = 100.0 - 0.1 * force * force;
            self.done = true;
        } else if self.steps >= MAX_STEPS {
            self.reward = -0.1 * force * force;
            self.done = true;
        } else {
            self.reward = -0.1 * force * force;
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
        "ah0".to_string(), 64, ActivationType::Tanh,
        TensorInitializer::new(InitializerType::XavierUniform), state_dim,
    ));
    net.add_layer(DenseLayer::new(
        "ah1".to_string(), 64, ActivationType::Tanh,
        TensorInitializer::new(InitializerType::XavierUniform), 0,
    ));
    net.add_layer(DenseLayer::new(
        "aout".to_string(), action_dim, ActivationType::Tanh,
        TensorInitializer::new(InitializerType::Uniform(-3e-3, 3e-3)), 0,
    ));
    net.add_connection("ah0", "ah1");
    net.add_connection("ah1", "aout");
    net.init();
    net
}

fn build_critic(state_dim: usize, action_dim: usize) -> NeuralNetwork {
    let mut net = NeuralNetwork::new();
    net.add_layer(DenseLayer::new(
        "ch0".to_string(), 64, ActivationType::Relu,
        TensorInitializer::new(InitializerType::XavierUniform), state_dim + action_dim,
    ));
    net.add_layer(DenseLayer::new(
        "ch1".to_string(), 64, ActivationType::Relu,
        TensorInitializer::new(InitializerType::XavierUniform), 0,
    ));
    net.add_layer(DenseLayer::new(
        "cout".to_string(), 1, ActivationType::Linear,
        TensorInitializer::new(InitializerType::Uniform(-3e-3, 3e-3)), 0,
    ));
    net.add_connection("ch0", "ch1");
    net.add_connection("ch1", "cout");
    net.init();
    net
}

fn build_cacla_actor(state_dim: usize, action_dim: usize) -> NeuralNetwork {
    let mut net = NeuralNetwork::new();
    net.add_layer(DenseLayer::new(
        "ah0".to_string(), 32, ActivationType::Tanh,
        TensorInitializer::new(InitializerType::LecunUniform), state_dim,
    ));
    net.add_layer(DenseLayer::new(
        "aout".to_string(), action_dim, ActivationType::Tanh,
        TensorInitializer::new(InitializerType::LecunUniform), 0,
    ));
    net.add_connection("ah0", "aout");
    net.init();
    net
}

fn build_cacla_critic(state_dim: usize) -> NeuralNetwork {
    let mut net = NeuralNetwork::new();
    net.add_layer(DenseLayer::new(
        "ch0".to_string(), 32, ActivationType::Tanh,
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

fn build_forward_model(state_dim: usize, action_dim: usize) -> ForwardModel {
    let mut net = NeuralNetwork::new();
    net.add_layer(DenseLayer::new(
        "fh0".to_string(), 64, ActivationType::Relu,
        TensorInitializer::new(InitializerType::XavierUniform), state_dim + action_dim,
    ));
    net.add_layer(DenseLayer::new(
        "fout".to_string(), state_dim, ActivationType::Linear,
        TensorInitializer::new(InitializerType::XavierUniform), 0,
    ));
    net.add_connection("fh0", "fout");
    net.init();
    ForwardModel::new(net, Box::new(ADOPT::new(1e-3)))
}

fn build_metacritic_net(state_dim: usize, action_dim: usize) -> NeuralNetwork {
    let mut net = NeuralNetwork::new();
    net.add_layer(DenseLayer::new(
        "mh0".to_string(), 64, ActivationType::Relu,
        TensorInitializer::new(InitializerType::XavierUniform), state_dim + action_dim,
    ));
    net.add_layer(DenseLayer::new(
        "mout".to_string(), 1, ActivationType::Linear,
        TensorInitializer::new(InitializerType::XavierUniform), 0,
    ));
    net.add_connection("mh0", "mout");
    net.init();
    net
}

fn run_episode_cacla(env: &mut MountainCar, agent: &mut CACLA, explore: &mut ContinuousExploration) -> (f32, bool) {
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
    let success = env.position >= GOAL_POS;
    (total_reward, success)
}

fn run_episode_ddpg(env: &mut MountainCar, agent: &mut DDPG, noise: &mut OuNoise) -> (f32, bool) {
    env.reset();
    noise.reset();
    let mut total_reward = 0.0;

    while !env.is_finished() {
        let state = env.get_state();
        let mut action = agent.get_action(&state).clone();
        noise.noise(&mut action);

        env.do_action(&action);
        let next_state = env.get_state();
        let reward = env.get_reward();
        let done = env.is_finished();

        agent.train(&state, &action, &next_state, reward, done);
        total_reward += reward;
    }
    let success = env.position >= GOAL_POS;
    (total_reward, success)
}

fn run_episode_ddpg_fm(
    env: &mut MountainCar,
    agent: &mut DDPG,
    forward_model: &mut ForwardModel,
    noise: &mut OuNoise,
) -> (f32, bool) {
    env.reset();
    noise.reset();
    let mut total_reward = 0.0;

    while !env.is_finished() {
        let state = env.get_state();
        let mut action = agent.get_action(&state).clone();
        noise.noise(&mut action);

        env.do_action(&action);
        let next_state = env.get_state();
        let extrinsic = env.get_reward();
        let intrinsic = forward_model.reward(&state, &action, &next_state).get(vec![0]);
        let reward = extrinsic + intrinsic;
        let done = env.is_finished();

        forward_model.train(&state, &action, &next_state);
        agent.train(&state, &action, &next_state, reward, done);
        total_reward += reward;
    }
    let success = env.position >= GOAL_POS;
    (total_reward, success)
}

fn run_episode_ddpg_mc(
    env: &mut MountainCar,
    agent: &mut DDPG,
    metacritic: &mut Metacritic,
    noise: &mut OuNoise,
) -> (f32, bool) {
    env.reset();
    noise.reset();
    let mut total_reward = 0.0;

    while !env.is_finished() {
        let state = env.get_state();
        let mut action = agent.get_action(&state).clone();
        noise.noise(&mut action);

        env.do_action(&action);
        let next_state = env.get_state();
        let extrinsic = env.get_reward();
        let intrinsic = metacritic.reward(&state, &action, &next_state).get(vec![0]);
        let reward = extrinsic + intrinsic;
        let done = env.is_finished();

        metacritic.train(&state, &action, &next_state);
        agent.train(&state, &action, &next_state, reward, done);
        total_reward += reward;
    }
    let success = env.position >= GOAL_POS;
    (total_reward, success)
}

fn main() {
    let episodes = 300;
    let state_dim = 2usize;
    let action_dim = 1usize;
    let gamma = 0.99;
    let lr = 1e-3;

    println!("=== MountainCar: CACLA (LinearDecay LR) ===");
    {
        let mut env = MountainCar::new();
        let mut agent = CACLA::new(
            build_cacla_actor(state_dim, action_dim), Box::new(ADOPT::new(lr)),
            build_cacla_critic(state_dim), Box::new(ADOPT::new(lr)),
            gamma,
        );
        let mut explore = ContinuousExploration::new(None);
        explore.init_gaussian(0.3);
        let mut sched = LinearDecay::new(lr, lr * 0.1, episodes);

        for ep in 0..episodes {
            let new_lr = sched.step();
            agent.actor_optimizer.set_lr(new_lr);
            agent.critic.optimizer.set_lr(new_lr);
            let (reward, success) = run_episode_cacla(&mut env, &mut agent, &mut explore);
            if ep % 30 == 0 {
                println!("  ep {:>4}  reward {:>8.2}  success: {}", ep, reward, success);
            }
        }
    }

    let memory_size = 100_000;
    let sample_size = 64;
    let tau = 0.005;

    println!("\n=== MountainCar: DDPG (LinearDecay LR) ===");
    {
        let mut env = MountainCar::new();
        let mut agent = DDPG::new(
            build_actor(state_dim, action_dim), Box::new(ADOPT::new(lr)),
            build_critic(state_dim, action_dim), Box::new(ADOPT::new(lr)),
            gamma, memory_size, sample_size, tau, state_dim, action_dim,
        );
        let mut noise = OuNoise::new(action_dim, 0.0, 0.3, 0.15, 0.01);
        let mut sched = LinearDecay::new(lr, lr * 0.1, episodes);

        for ep in 0..episodes {
            let new_lr = sched.step();
            agent.actor_optimizer.set_lr(new_lr);
            agent.critic_optimizer.set_lr(new_lr);
            let (reward, success) = run_episode_ddpg(&mut env, &mut agent, &mut noise);
            if ep % 30 == 0 {
                println!("  ep {:>4}  reward {:>8.2}  success: {}", ep, reward, success);
            }
        }
    }

    println!("\n=== MountainCar: DDPG + ForwardModel (LinearDecay LR) ===");
    {
        let mut env = MountainCar::new();
        let mut agent = DDPG::new(
            build_actor(state_dim, action_dim), Box::new(ADOPT::new(lr)),
            build_critic(state_dim, action_dim), Box::new(ADOPT::new(lr)),
            gamma, memory_size, sample_size, tau, state_dim, action_dim,
        );
        let mut forward_model = build_forward_model(state_dim, action_dim);
        let mut noise = OuNoise::new(action_dim, 0.0, 0.3, 0.15, 0.01);
        let mut sched = LinearDecay::new(lr, lr * 0.1, episodes);

        for ep in 0..episodes {
            let new_lr = sched.step();
            agent.actor_optimizer.set_lr(new_lr);
            agent.critic_optimizer.set_lr(new_lr);
            let (reward, success) = run_episode_ddpg_fm(&mut env, &mut agent, &mut forward_model, &mut noise);
            if ep % 30 == 0 {
                println!("  ep {:>4}  reward {:>8.2}  success: {}", ep, reward, success);
            }
        }
    }

    println!("\n=== MountainCar: DDPG + Metacritic (LinearDecay LR) ===");
    {
        let mut env = MountainCar::new();
        let mut agent = DDPG::new(
            build_actor(state_dim, action_dim), Box::new(ADOPT::new(lr)),
            build_critic(state_dim, action_dim), Box::new(ADOPT::new(lr)),
            gamma, memory_size, sample_size, tau, state_dim, action_dim,
        );
        let fm = build_forward_model(state_dim, action_dim);
        let mc_net = build_metacritic_net(state_dim, action_dim);
        let mut metacritic = Metacritic::new(mc_net, Box::new(ADOPT::new(lr)), fm, 0.1);
        let mut noise = OuNoise::new(action_dim, 0.0, 0.3, 0.15, 0.01);
        let mut sched = LinearDecay::new(lr, lr * 0.1, episodes);

        for ep in 0..episodes {
            let new_lr = sched.step();
            agent.actor_optimizer.set_lr(new_lr);
            agent.critic_optimizer.set_lr(new_lr);
            let (reward, success) = run_episode_ddpg_mc(&mut env, &mut agent, &mut metacritic, &mut noise);
            if ep % 30 == 0 {
                println!("  ep {:>4}  reward {:>8.2}  success: {}", ep, reward, success);
            }
        }
    }
}
