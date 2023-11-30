#![feature(array_chunks, generic_arg_infer, array_methods)]#![allow(non_snake_case)]
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
	let train = || train_images.iter().zip(&train_labels);

	// Mean example
	let mut mean_examples = [[0.; 28*28]; 10];
	let mut label_example_count = [0; 10];
	for (image, &label) in train() {
		for j in 0..28*28 { mean_examples[label][j] += image[j]; }
		label_example_count[label] += 1;
	}
	for label in 0..10 {
		for j in 0..28*28 { mean_examples[label][j] /= label_example_count[label] as f64; }
	}

	// Stochastic gradient descent for the logistic regression model
	fn rand<const N: usize>() -> [f64; N] {use rand::Rng; rand::thread_rng().gen::<[_; _]>().map(|u:f64| u*2.-1.)}
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
		for (x, &y) in train() {
			let ey: [_; 10] = std::array::from_fn(|k| if k == y { 1. } else { 0. });
			let F = F(&theta, &b, x);
			for i in 0..10 {
				for j in 0..28*28 {
					theta[i][j] -= -(ey[i] - F[i]*x[j]);
				}
				b[i] -= -(ey[i] - F[i]);
			}
		}
	}
	println!("{}s", start.elapsed().as_secs());

	let (images, labels) = set("t10k")?;
	let matches = images.iter().zip(labels.iter()).filter(|&(image, label)| {
		let sq = |x| x*x;
		fn max_position(iter: impl IntoIterator<Item=f64>) -> usize {
			iter.into_iter().enumerate().max_by(|(_, a),(_, b)| f64::total_cmp(a,b)).map(|(i,_)| i).unwrap()
		}
		let distance = |a: &[f64; _], b| a.iter().zip(b).map(|(a,b)| sq(a-b)).sum::<f64>();
		let prediction = match "feedforward" {
			"nearest" => { // 82%
				let prediction = mean_examples.each_ref().map(|example| -distance(example, image));
				max_position(prediction)
			},
			"feedforward" => {
				let prediction = F(&theta, &b, image);
				max_position(prediction)
			},
			"kNN" => { // slow
				let mut distances = Vec::from_iter(train().map(|(example, &label)| (distance(image, example), label)));
				let (k_nearest, _, _) = distances.select_nth_unstable_by(5, |(a,_),(b,_)| f64::total_cmp(a,b));
				let counts: [_; 10] = std::array::from_fn(|i| k_nearest.into_iter().filter(|(_,j)| i==*j).count());
				counts.iter().enumerate().max_by(|(_, a),(_, b)| a.cmp(b)).map(|(i,_)| i).unwrap()
			},
			_ => unreachable!()
		};
		//println!("{prediction} {label}");
		prediction == *label
	}).count();
	println!("{matches}/{} = {}%", labels.len(), matches as f64/labels.len() as f64*100.);
	Ok(())
}
