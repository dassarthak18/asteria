use asteria::clab::activation_functions::ActivationType;
use asteria::clab::loss_functions::{SoftmaxCrossEntropyFunction, LossFunction};
use asteria::clab::tensor::Tensor;
use asteria::clab::tensor_initializer::{InitializerType, TensorInitializer};
use asteria::core::adopt::ADOPT;
use asteria::core::dense_layer::DenseLayer;
use asteria::core::lr_scheduler::{CosineAnnealing, LrScheduler};
use asteria::core::neural_network::NeuralNetwork;
use asteria::core::optimizer::Optimizer;

fn load_wine(path: &str) -> (Vec<Vec<f32>>, Vec<usize>) {
    let content = std::fs::read_to_string(path).expect("cannot read wine.dat");
    let mut features = Vec::new();
    let mut labels = Vec::new();
    let mut in_data = false;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() { continue; }
        if line.starts_with('@') {
            if line.to_lowercase().starts_with("@data") { in_data = true; }
            continue;
        }
        if !in_data { continue; }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != 14 { continue; }
        let feats: Vec<f32> = parts[..13].iter().map(|s| s.trim().parse::<f32>().unwrap()).collect();
        let class: usize = parts[13].trim().parse::<usize>().unwrap() - 1; // 1/2/3 → 0/1/2
        features.push(feats);
        labels.push(class);
    }
    (features, labels)
}

fn normalize(data: &mut Vec<Vec<f32>>) {
    let n_feat = data[0].len();
    for j in 0..n_feat {
        let min = data.iter().map(|r| r[j]).fold(f32::INFINITY, f32::min);
        let max = data.iter().map(|r| r[j]).fold(f32::NEG_INFINITY, f32::max);
        let range = (max - min).max(1e-8);
        for row in data.iter_mut() { row[j] = (row[j] - min) / range; }
    }
}

fn one_hot(label: usize, n: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; n];
    v[label] = 1.0;
    v
}

fn accuracy(net: &mut NeuralNetwork, features: &[Vec<f32>], labels: &[usize]) -> f32 {
    let mut correct = 0;
    for (feat, &label) in features.iter().zip(labels.iter()) {
        let input = Tensor::from_data(vec![1, feat.len()], feat.clone());
        let out = net.forward(&input);
        let pred = out.max_index(0)[0];
        if pred == label { correct += 1; }
    }
    correct as f32 / features.len() as f32
}

fn main() {
    let (mut features, labels) = load_wine("data/wine.dat");
    normalize(&mut features);

    // stratified 80/20 split
    let mut train_x = Vec::new();
    let mut train_y = Vec::new();
    let mut test_x = Vec::new();
    let mut test_y = Vec::new();
    for (i, (f, l)) in features.iter().zip(labels.iter()).enumerate() {
        if i % 5 == 4 { test_x.push(f.clone()); test_y.push(*l); }
        else           { train_x.push(f.clone()); train_y.push(*l); }
    }

    let n_feat = 13usize;
    let n_class = 3usize;
    let lr = 1e-3;
    let epochs = 300;
    let batch_size = 8usize;

    let mut net = NeuralNetwork::new();
    net.add_layer(DenseLayer::new("h0".to_string(), 32, ActivationType::Relu,
        TensorInitializer::new(InitializerType::XavierUniform), n_feat));
    net.add_layer(DenseLayer::new("h1".to_string(), 16, ActivationType::Relu,
        TensorInitializer::new(InitializerType::XavierUniform), 0));
    net.add_layer(DenseLayer::new("out".to_string(), n_class, ActivationType::Linear,
        TensorInitializer::new(InitializerType::XavierUniform), 0));
    net.add_connection("h0", "h1");
    net.add_connection("h1", "out");
    net.init();

    let mut opt = ADOPT::new(lr);
    let mut sched = CosineAnnealing::new(lr, 1e-5, epochs);
    let loss_fn = SoftmaxCrossEntropyFunction;

    let mut rng: u64 = 99;
    let mut indices: Vec<usize> = (0..train_x.len()).collect();

    println!("=== Wine: Softmax Classifier (CosineAnnealing LR) ===");
    for epoch in 0..=epochs {
        let epoch_lr = sched.step();
        opt.set_lr(epoch_lr);

        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        for i in (1..indices.len()).rev() {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let j = (rng as usize) % (i + 1);
            indices.swap(i, j);
        }

        let mut total_loss = 0.0f32;
        for chunk in indices.chunks(batch_size) {
            let feats: Vec<f32> = chunk.iter().flat_map(|&i| train_x[i].iter().copied()).collect();
            let tgts: Vec<f32> = chunk.iter().flat_map(|&i| one_hot(train_y[i], n_class)).collect();
            let input  = Tensor::from_data(vec![chunk.len(), n_feat], feats);
            let target = Tensor::from_data(vec![chunk.len(), n_class], tgts);

            let pred = net.forward(&input).clone();
            total_loss += loss_fn.forward(&pred, &target);
            let mut delta = loss_fn.backward(&pred, &target);
            net.backward(&mut delta);
            opt.update(&mut net);
        }

        if epoch % 50 == 0 {
            let train_acc = accuracy(&mut net, &train_x, &train_y);
            let test_acc  = accuracy(&mut net, &test_x,  &test_y);
            println!("  epoch {:>3}  lr {:.2e}  loss {:>7.4}  train_acc {:.3}  test_acc {:.3}",
                epoch, epoch_lr, total_loss / train_x.len() as f32, train_acc, test_acc);
        }
    }

    let final_acc = accuracy(&mut net, &test_x, &test_y);
    println!("\nFinal test accuracy: {:.1}%", final_acc * 100.0);
}
