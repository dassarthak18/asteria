use asteria::clab::tensor::Tensor;
use asteria::clab::activation_functions::ActivationType;
use asteria::clab::tensor_initializer::{TensorInitializer, InitializerType};
use asteria::clab::loss_functions::{MseFunction, LossFunction};
use asteria::core::neural_network::NeuralNetwork;
use asteria::core::dense_layer::DenseLayer;
use asteria::core::adopt::ADOPT;
use asteria::core::optimizer::Optimizer;
use asteria::core::lr_scheduler::{LinearDecay, LrScheduler};

fn main() {
    let input_data = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let target_data = vec![0.0, 1.0, 1.0, 0.0];

    let input = Tensor::from_data(vec![4, 2], input_data);
    let target = Tensor::from_data(vec![4, 1], target_data);

    let mut network = NeuralNetwork::new();

    network.add_layer(DenseLayer::new(
        "hidden0".to_string(),
        8,
        ActivationType::Sigmoid,
        TensorInitializer::new(InitializerType::LecunUniform),
        2,
    ));
    network.add_layer(DenseLayer::new(
        "hidden1".to_string(),
        4,
        ActivationType::Sigmoid,
        TensorInitializer::new(InitializerType::LecunUniform),
        0,
    ));
    network.add_layer(DenseLayer::new(
        "output".to_string(),
        1,
        ActivationType::Sigmoid,
        TensorInitializer::new(InitializerType::LecunUniform),
        0,
    ));

    network.add_connection("hidden0", "hidden1");
    network.add_connection("hidden1", "output");
    network.init();

    let loss = MseFunction;
    let mut optimizer = ADOPT::new(1e-2);
    let mut sched = LinearDecay::new(1e-2, 1e-3, 500);

    for t in 0..500 {
        let lr = sched.step();
        optimizer.set_lr(lr);

        let output = network.forward(&input).clone();
        let error = loss.forward(&output, &target);
        let mut delta = loss.backward(&output, &target);
        network.backward(&mut delta);
        optimizer.update(&mut network);

        if t % 50 == 0 {
            println!("  step {:>3}  lr {:.2e}  error {:.6}", t, lr, error);
        }
    }

    println!("\n=== XOR: MSE Classifier (LinearDecay LR) ===");
    println!("Final output:\n{}", network.forward(&input));
}
