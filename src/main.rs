#![feature(array_chunks, generic_arg_infer, array_methods)]#![allow(non_snake_case)]
use rand::{Rng, thread_rng};

fn main() -> std::io::Result<()> {
	let images = |path| -> std::io::Result<_> {
		let images = std::fs::read(path)?;
		let images = &images[4*std::mem::size_of::<u32>()..];
		let images = Vec::from_iter(images.array_chunks::<{28*28}>().map(|image|
			image.map(|u8| f64::from(u8)/255.)
		));
		Ok(images)
	};

	let labels = |path| -> std::io::Result<_> {
		let labels = std::fs::read(path)?;
		let labels = &labels[2*std::mem::size_of::<u32>()..];
		let labels = Vec::from_iter(labels.iter().map(|&u8| usize::from(u8)));
		for &value in &labels { assert!(value <= 9); }
		Ok(labels)
	};

	let set = |name| -> std::io::Result<_> {
		let images = images(format!("{name}-images-idx3-ubyte"))?;
		let labels = labels(format!("{name}-labels-idx1-ubyte"))?;
		assert_eq!(images.len(), labels.len());
		Ok((images, labels))
	};

	let (train_images, train_labels) = set("train")?;

	// Stochastic gradient descent for the logistic regression model
	fn rand<const N: usize>() -> [f64; N] {thread_rng().gen::<[_; _]>().map(|u:f64| u*2.-1.)}
	let mut theta : [[f64; 28*28]; 10] = [(); 10].map(|_| rand());
	let mut b: [f64; 10] = rand();

	let F = |weights: &[[f64; _]; _], b: &[_; _], image:&[f64; 28*28]| -> [f64; 10] {
		let y = std::array::from_fn(|i| weights[i].iter().zip(image).map(|(w,i)| w*i).sum::<f64>() + b[i]);
		let y = y.map(|y| f64::exp(y));
		let sum = y.iter().sum::<f64>();
		y.map(|y| y/sum)
	};

	let start = std::time::Instant::now();
	for _ in 0..6 {
		let (mut G_theta, mut G_b) = ([[0.; 28*28]; 10], [0.; 10]);
		let batch_size = 10;
		for _ in 0..batch_size {
			let index = thread_rng().gen_range(0..10);
			let (ref x, y) = (train_images[index], train_labels[index]);
			let F = F(&theta, &b, x);
			for i in 0..10 {
				let ey_i= if i == y { 1. } else { 0. };
				for j in 0..28*28 {
					G_theta[i][j] += -(ey_i - F[i]*x[j]);
				}
				G_b[i] += -(ey_i - F[i]);
			}
		}
		let learn_rate = 0.005;
		for i in 0..10 {
			for j in 0..28*28 {
				theta[i][j] -= learn_rate * G_theta[i][j] / batch_size as f64;
			}
			b[i] -= learn_rate * G_b[i] / batch_size as f64;
		}
	}
	println!("{}s", start.elapsed().as_secs());

	let (images, labels) = set("t10k")?;
	let matches = images.iter().zip(labels.iter()).filter(|&(image, label)| {
		fn max_position(iter: impl IntoIterator<Item=f64>) -> usize {
			iter.into_iter().enumerate().max_by(|(_, a),(_, b)| f64::total_cmp(a,b)).map(|(i,_)| i).unwrap()
		}
		let prediction = F(&theta, &b, image);
		let prediction = max_position(prediction);
		prediction == *label
	}).count();
	println!("{matches}/{} = {}%", labels.len(), matches as f64/labels.len() as f64*100.);
	Ok(())
}
