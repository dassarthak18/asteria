use crate::clab::tensor::Tensor;
use crate::core::neural_network::NeuralNetwork;
use crate::core::optimizer::Optimizer;
use crate::rl::ienvironment::IEnvironment;
use std::sync::{Arc, Mutex};
use std::thread;

/// Asynchronous Advantage Actor-Critic (A3C).
///
/// `AC3` parallelises training by spawning `num_workers` threads, each of which
/// runs its own copy of the environment and maintains *local* actor/critic weights.
/// After each transition the local gradients are pushed to the shared global networks
/// and the local weights are synchronised back.
///
/// This is the Hogwild-style gradient sharing scheme from the original A3C paper
/// (Mnih et al., 2016): no locks on the parameter update itself, only on the
/// gradient-copy/sync operation.
///
/// Call [`train`](AC3::train) with an environment factory closure; the factory is
/// called once per worker to produce independent environment instances.
pub struct AC3 {
    /// Shared global actor network π(a|s).
    pub global_actor: Arc<Mutex<NeuralNetwork>>,
    /// Shared global critic network V(s).
    pub global_critic: Arc<Mutex<NeuralNetwork>>,
    /// Optimizer for the global actor (shared across workers).
    pub actor_optimizer: Arc<Mutex<Box<dyn Optimizer + Send>>>,
    /// Optimizer for the global critic (shared across workers).
    pub critic_optimizer: Arc<Mutex<Box<dyn Optimizer + Send>>>,
    /// Discount factor γ ∈ [0, 1].
    pub gamma: f32,
    /// Number of parallel worker threads.
    pub num_workers: usize,
}

impl AC3 {
    /// Creates a new A3C agent.
    ///
    /// - `actor` / `actor_optimizer`: global policy network and its (shared) update rule.
    /// - `critic` / `critic_optimizer`: global value network and its (shared) update rule.
    /// - `gamma`: discount factor.
    /// - `num_workers`: number of parallel environment/worker threads.
    pub fn new(
        actor: NeuralNetwork,
        actor_optimizer: Box<dyn Optimizer + Send>,
        critic: NeuralNetwork,
        critic_optimizer: Box<dyn Optimizer + Send>,
        gamma: f32,
        num_workers: usize,
    ) -> Self {
        AC3 {
            global_actor: Arc::new(Mutex::new(actor)),
            global_critic: Arc::new(Mutex::new(critic)),
            actor_optimizer: Arc::new(Mutex::new(actor_optimizer)),
            critic_optimizer: Arc::new(Mutex::new(critic_optimizer)),
            gamma,
            num_workers,
        }
    }

    /// Spawns `num_workers` threads, each running 100 episodes in `env_factory()`.
    ///
    /// Each worker maintains local copies of the actor and critic, computes one-step
    /// TD gradients, pushes them to the global networks, and immediately syncs its
    /// local weights back. The method blocks until all workers finish.
    pub fn train<E: IEnvironment + Send + Clone + 'static>(&mut self, env_factory: impl Fn() -> E + Send + Sync + 'static) {
        let mut handles = Vec::new();

        for i in 0..self.num_workers {
            let actor = self.global_actor.clone();
            let critic = self.global_critic.clone();
            let actor_opt = self.actor_optimizer.clone();
            let critic_opt = self.critic_optimizer.clone();
            let gamma = self.gamma;
            let mut env = env_factory();

            let handle = thread::spawn(move || {
                let mut local_actor = actor.lock().unwrap().clone();
                let mut local_critic = critic.lock().unwrap().clone();

                println!("Worker {} started", i);

                for _ in 0..100 {
                    env.reset();
                    while !env.is_finished() {
                        let s0 = env.get_state();
                        let probs = local_actor.forward(&s0).clone();
                        let action_idx = probs.max_index(0)[0];
                        let mut action = Tensor::with_shape_val(vec![probs.size()], 0.0);
                        action.set(vec![action_idx], 1.0);

                        env.do_action(&action);
                        let s1 = env.get_state();
                        let r = env.get_reward();
                        let terminal = env.is_finished();

                        let v_s0 = local_critic.forward(&s0).get(vec![0, 0]);
                        let v_s1 = if terminal { 0.0 } else { local_critic.forward(&s1).get(vec![0, 0]) };
                        let td_error = r + gamma * v_s1 - v_s0;

                        let mut critic_delta = Tensor::with_shape_val(vec![1, 1], -td_error);
                        local_critic.backward(&mut critic_delta);

                        let mut actor_delta = Tensor::with_shape_val(probs.shape.clone(), 0.0);
                        actor_delta.set(vec![0, action_idx], -td_error / (probs.get(vec![0, action_idx]) + 1e-8));
                        local_actor.backward(&mut actor_delta);

                        {
                            let mut g_actor = actor.lock().unwrap();
                            for (id, local_p_arc) in &local_actor.model.model {
                                let local_p = local_p_arc.lock().unwrap();
                                let mut global_p = g_actor.model.model.get(id).unwrap().lock().unwrap();
                                global_p.gradient.data.copy_from_slice(&local_p.gradient.data);
                            }
                            actor_opt.lock().unwrap().update(&mut g_actor);
                            local_actor.copy_params(&g_actor, 1.0);
                        }

                        {
                            let mut g_critic = critic.lock().unwrap();
                            for (id, local_p_arc) in &local_critic.model.model {
                                let local_p = local_p_arc.lock().unwrap();
                                let mut global_p = g_critic.model.model.get(id).unwrap().lock().unwrap();
                                global_p.gradient.data.copy_from_slice(&local_p.gradient.data);
                            }
                            critic_opt.lock().unwrap().update(&mut g_critic);
                            local_critic.copy_params(&g_critic, 1.0);
                        }
                    }
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }
}
