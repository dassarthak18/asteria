use asteria::clab::activation_functions::ActivationType;
use asteria::clab::tensor::Tensor;
use asteria::clab::tensor_initializer::{InitializerType, TensorInitializer};
use asteria::core::adopt::ADOPT;
use asteria::core::dense_layer::DenseLayer;
use asteria::core::lr_scheduler::{LinearDecay, LrScheduler};
use asteria::core::neural_network::NeuralNetwork;
use asteria::rl::continuous_exploration::OuNoise;
use asteria::rl::ddpg::DDPG;
use asteria::rl::forward_model::ForwardModel;
use asteria::rl::ienvironment::IEnvironment;
use asteria::rl::metacritic::Metacritic;

const MC: f32 = 1.0;
const MP: f32 = 0.1;
const L: f32 = 0.5;
const G: f32 = 9.8;
const DT: f32 = 0.02;
const FORCE_MAG: f32 = 10.0;
const X_LIMIT: f32 = 2.4;
const THETA_LIMIT: f32 = 12.0 * std::f32::consts::PI / 180.0;
const MAX_STEPS: usize = 200;

struct CartPole {
    x: f32,
    x_dot: f32,
    theta: f32,
    theta_dot: f32,
    steps: usize,
    done: bool,
    reward: f32,
    rng_state: u64,
}

impl CartPole {
    fn new() -> Self {
        CartPole {
            x: 0.0, x_dot: 0.0, theta: 0.0, theta_dot: 0.0,
            steps: 0, done: false, reward: 0.0,
            rng_state: 12345,
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

impl IEnvironment for CartPole {
    fn reset(&mut self) {
        self.x       = self.rand_uniform(-0.05, 0.05);
        self.x_dot   = self.rand_uniform(-0.05, 0.05);
        self.theta    = self.rand_uniform(-0.05, 0.05);
        self.theta_dot = self.rand_uniform(-0.05, 0.05);
        self.steps = 0;
        self.done = false;
        self.reward = 0.0;
    }

    fn get_state(&mut self) -> Tensor {
        Tensor::from_data(
            vec![1, 4],
            vec![self.x, self.x_dot, self.theta, self.theta_dot],
        )
    }

    fn do_action(&mut self, action: &Tensor) {
        let force = action.get(vec![0, 0]).clamp(-1.0, 1.0) * FORCE_MAG;
        let cos_th = self.theta.cos();
        let sin_th = self.theta.sin();
        let total_mass = MC + MP;
        let ml = MP * L;

        let temp = (force + ml * self.theta_dot * self.theta_dot * sin_th) / total_mass;
        let theta_acc = (G * sin_th - cos_th * temp)
            / (L * (4.0 / 3.0 - MP * cos_th * cos_th / total_mass));
        let x_acc = temp - ml * theta_acc * cos_th / total_mass;

        self.x        += DT * self.x_dot;
        self.x_dot    += DT * x_acc;
        self.theta     += DT * self.theta_dot;
        self.theta_dot += DT * theta_acc;
        self.steps += 1;

        let failed = self.x.abs() > X_LIMIT
            || self.theta.abs() > THETA_LIMIT
            || self.steps >= MAX_STEPS;

        if failed {
            self.reward = -1.0;
            self.done = true;
        } else {
            self.reward = 0.0;
        }
    }

    fn get_reward(&mut self) -> f32 { self.reward }
    fn is_finished(&mut self) -> bool { self.done }
    fn state_dim(&self) -> usize { 4 }
    fn action_dim(&self) -> usize { 1 }
}

fn build_actor(state_dim: usize, action_dim: usize) -> NeuralNetwork {
    let mut net = NeuralNetwork::new();
    net.add_layer(DenseLayer::new(
        "ah0".to_string(), 64, ActivationType::Relu,
        TensorInitializer::new(InitializerType::XavierUniform), state_dim,
    ));
    net.add_layer(DenseLayer::new(
        "ah1".to_string(), 64, ActivationType::Relu,
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

fn run_episode(
    env: &mut CartPole,
    agent: &mut DDPG,
    noise: &mut OuNoise,
) -> (f32, usize) {
    env.reset();
    noise.reset();
    let mut total_reward = 0.0;
    let mut steps = 0;

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
        steps += 1;
    }
    (total_reward, steps)
}

fn run_episode_fm(
    env: &mut CartPole,
    agent: &mut DDPG,
    forward_model: &mut ForwardModel,
    noise: &mut OuNoise,
) -> (f32, usize) {
    env.reset();
    noise.reset();
    let mut total_reward = 0.0;
    let mut steps = 0;

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
        steps += 1;
    }
    (total_reward, steps)
}

fn run_episode_mc(
    env: &mut CartPole,
    agent: &mut DDPG,
    metacritic: &mut Metacritic,
    noise: &mut OuNoise,
) -> (f32, usize) {
    env.reset();
    noise.reset();
    let mut total_reward = 0.0;
    let mut steps = 0;

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
        steps += 1;
    }
    (total_reward, steps)
}

fn main() {
    let episodes = 300;
    let state_dim = 4usize;
    let action_dim = 1usize;
    let memory_size = 100_000;
    let sample_size = 64;
    let tau = 0.005;
    let gamma = 0.99;
    let lr = 1e-3;

    println!("=== CartPole: DDPG (LinearDecay LR) ===");
    {
        let mut env = CartPole::new();
        let mut agent = DDPG::new(
            build_actor(state_dim, action_dim), Box::new(ADOPT::new(lr)),
            build_critic(state_dim, action_dim), Box::new(ADOPT::new(lr)),
            gamma, memory_size, sample_size, tau, state_dim, action_dim,
        );
        let mut noise = OuNoise::new(action_dim, 0.0, 0.2, 0.15, 0.01);
        let mut sched = LinearDecay::new(lr, lr * 0.1, episodes);

        for ep in 0..episodes {
            let new_lr = sched.step();
            agent.actor_optimizer.set_lr(new_lr);
            agent.critic_optimizer.set_lr(new_lr);
            let (reward, steps) = run_episode(&mut env, &mut agent, &mut noise);
            if ep % 30 == 0 {
                println!("  ep {:>4}  reward {:>7.2}  steps {:>4}", ep, reward, steps);
            }
        }
    }

    println!("\n=== CartPole: DDPG + ForwardModel (LinearDecay LR) ===");
    {
        let mut env = CartPole::new();
        let mut agent = DDPG::new(
            build_actor(state_dim, action_dim), Box::new(ADOPT::new(lr)),
            build_critic(state_dim, action_dim), Box::new(ADOPT::new(lr)),
            gamma, memory_size, sample_size, tau, state_dim, action_dim,
        );
        let mut forward_model = build_forward_model(state_dim, action_dim);
        let mut noise = OuNoise::new(action_dim, 0.0, 0.2, 0.15, 0.01);
        let mut sched = LinearDecay::new(lr, lr * 0.1, episodes);

        for ep in 0..episodes {
            let new_lr = sched.step();
            agent.actor_optimizer.set_lr(new_lr);
            agent.critic_optimizer.set_lr(new_lr);
            let (reward, steps) = run_episode_fm(&mut env, &mut agent, &mut forward_model, &mut noise);
            if ep % 30 == 0 {
                println!("  ep {:>4}  reward {:>7.2}  steps {:>4}", ep, reward, steps);
            }
        }
    }

    println!("\n=== CartPole: DDPG + Metacritic (LinearDecay LR) ===");
    {
        let mut env = CartPole::new();
        let mut agent = DDPG::new(
            build_actor(state_dim, action_dim), Box::new(ADOPT::new(lr)),
            build_critic(state_dim, action_dim), Box::new(ADOPT::new(lr)),
            gamma, memory_size, sample_size, tau, state_dim, action_dim,
        );
        let fm = build_forward_model(state_dim, action_dim);
        let mc_net = build_metacritic_net(state_dim, action_dim);
        let mut metacritic = Metacritic::new(mc_net, Box::new(ADOPT::new(lr)), fm, 0.1);
        let mut noise = OuNoise::new(action_dim, 0.0, 0.2, 0.15, 0.01);
        let mut sched = LinearDecay::new(lr, lr * 0.1, episodes);

        for ep in 0..episodes {
            let new_lr = sched.step();
            agent.actor_optimizer.set_lr(new_lr);
            agent.critic_optimizer.set_lr(new_lr);
            let (reward, steps) = run_episode_mc(&mut env, &mut agent, &mut metacritic, &mut noise);
            if ep % 30 == 0 {
                println!("  ep {:>4}  reward {:>7.2}  steps {:>4}", ep, reward, steps);
            }
        }
    }
}
