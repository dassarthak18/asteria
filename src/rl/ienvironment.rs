use crate::clab::tensor::Tensor;

pub trait IEnvironment {
    fn get_state(&mut self) -> Tensor;
    fn do_action(&mut self, action: &Tensor);
    fn get_reward(&mut self) -> f32;
    fn reset(&mut self);
    fn is_finished(&mut self) -> bool;
    fn state_dim(&self) -> usize;
    fn action_dim(&self) -> usize;
}
