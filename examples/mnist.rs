use asteria::clab::activation_functions::ActivationType;
use asteria::clab::loss_functions::{SoftmaxCrossEntropyFunction, LossFunction};
use asteria::clab::tensor::Tensor;
use asteria::clab::tensor_initializer::{InitializerType, TensorInitializer};
use asteria::core::adopt::ADOPT;
use asteria::core::dense_layer::DenseLayer;
use asteria::core::lr_scheduler::{CosineAnnealing, LrScheduler};
use asteria::core::neural_network::NeuralNetwork;
use asteria::core::optimizer::Optimizer;

fn read_u32_be(bytes: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes([bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]])
}

fn load_images(path: &str) -> Vec<Vec<f32>> {
    let bytes = std::fs::read(path).expect("cannot read image file");
    assert_eq!(read_u32_be(&bytes, 0), 0x0803, "bad image magic");
    let n    = read_u32_be(&bytes, 4) as usize;
    let rows = read_u32_be(&bytes, 8) as usize;
    let cols = read_u32_be(&bytes, 12) as usize;
    let pixels = rows * cols;
    let mut images = Vec::with_capacity(n);
    for i in 0..n {
        let base = 16 + i * pixels;
        let img: Vec<f32> = bytes[base..base+pixels].iter().map(|&b| b as f32 / 255.0).collect();
        images.push(img);
    }
    images
}

fn load_labels(path: &str) -> Vec<usize> {
    let bytes = std::fs::read(path).expect("cannot read label file");
    assert_eq!(read_u32_be(&bytes, 0), 0x0801, "bad label magic");
    let n = read_u32_be(&bytes, 4) as usize;
    bytes[8..8+n].iter().map(|&b| b as usize).collect()
}

fn one_hot(label: usize, n: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; n];
    v[label] = 1.0;
    v
}

fn accuracy(net: &mut NeuralNetwork, images: &[Vec<f32>], labels: &[usize]) -> f32 {
    let mut correct = 0;
    for (img, &label) in images.iter().zip(labels.iter()) {
        let input = Tensor::from_data(vec![1, img.len()], img.clone());
        let out = net.forward(&input);
        let pred = out.max_index(0)[0];
        if pred == label { correct += 1; }
    }
    correct as f32 / images.len() as f32
}

fn main() {
    println!("Loading MNIST...");
    let train_images = load_images("Data/train-images-idx3-ubyte");
    let train_labels = load_labels("Data/train-labels-idx1-ubyte");
    let test_images  = load_images("Data/t10k-images-idx3-ubyte");
    let test_labels  = load_labels("Data/t10k-labels-idx1-ubyte");

    let n_feat  = 784usize;
    let n_class = 10usize;
    let lr = 1e-3;
    let epochs = 10;
    let batch_size = 64usize;

    let mut net = NeuralNetwork::new();
    net.add_layer(DenseLayer::new("h0".to_string(), 128, ActivationType::Relu,
        TensorInitializer::new(InitializerType::XavierUniform), n_feat));
    net.add_layer(DenseLayer::new("h1".to_string(), 64, ActivationType::Relu,
        TensorInitializer::new(InitializerType::XavierUniform), 0));
    net.add_layer(DenseLayer::new("out".to_string(), n_class, ActivationType::Linear,
        TensorInitializer::new(InitializerType::XavierUniform), 0));
    net.add_connection("h0", "h1");
    net.add_connection("h1", "out");
    net.init();

    let mut opt = ADOPT::new(lr);
    let mut sched = CosineAnnealing::new(lr, 1e-5, epochs);
    let loss_fn = SoftmaxCrossEntropyFunction;

    let mut rng: u64 = 12345;
    let mut indices: Vec<usize> = (0..train_images.len()).collect();

    println!("=== MNIST: MLP Classifier (batch {}, cosine LR) ===", batch_size);
    for epoch in 0..epochs {
        let epoch_lr = sched.step();
        opt.set_lr(epoch_lr);

        // shuffle
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        for i in (1..indices.len()).rev() {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let j = (rng as usize) % (i + 1);
            indices.swap(i, j);
        }

        let mut total_loss = 0.0f32;
        for chunk in indices.chunks(batch_size) {
            let feats: Vec<f32> = chunk.iter().flat_map(|&i| train_images[i].iter().copied()).collect();
            let tgts:  Vec<f32> = chunk.iter().flat_map(|&i| one_hot(train_labels[i], n_class)).collect();
            let input  = Tensor::from_data(vec![chunk.len(), n_feat],  feats);
            let target = Tensor::from_data(vec![chunk.len(), n_class], tgts);

            let pred = net.forward(&input).clone();
            total_loss += loss_fn.forward(&pred, &target);
            let mut delta = loss_fn.backward(&pred, &target);
            net.backward(&mut delta);
            opt.update(&mut net);
        }

        // evaluate on first 2000 test samples for speed
        let test_acc = accuracy(&mut net, &test_images[..2000], &test_labels[..2000]);
        println!("  epoch {:>2}  lr {:.2e}  loss {:>7.4}  test_acc(2k) {:.3}",
            epoch, epoch_lr, total_loss / train_images.len() as f32, test_acc);
    }

    println!("\nEvaluating on full test set...");
    let final_acc = accuracy(&mut net, &test_images, &test_labels);
    println!("Final test accuracy: {:.2}%", final_acc * 100.0);
}
