#![feature(array_chunks, generic_arg_infer)]
fn main() -> std::io::Result<()> {
	let mut weights = [[0.; 28*28]; 10];

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

	let feedforward = |weights: &[[f64; _]; _], image:&[f64; 28*28]| -> [f64; 10] {
		std::array::from_fn(|i| weights[i].iter().zip(image).map(|(w,i)| w*i).sum())
	};
	
	let (train_images, train_labels) = set("train")?;
	let train = || train_images.iter().zip(&train_labels);
	let start = std::time::Instant::now();
	for _ in 0..2 {
		for (image, &label) in train() {
			let prediction = feedforward(&weights, image);
			let target: [_; 10] = std::array::from_fn(|i| if i == label { 1. } else { 0. });
			for i in 0..10 { 
				let e_i = target[i]-prediction[i];
				for j in 0..28*28 {
					weights[i][j] += 2.*e_i*image[j]
				}
			}
		}
	}
	println!("{}s", start.elapsed().as_secs());

	let (images, labels) = set("t10k")?;
	let matches = images.iter().zip(labels.iter()).filter(|&(image, label)| {
		let prediction = feedforward(&weights, image);
		fn max_position(iter: impl IntoIterator<Item=f64>) -> usize { 
			iter.into_iter().enumerate().max_by(|(_, a),(_, b)| f64::total_cmp(a,b)).map(|(i,_)| i).unwrap()
		}
		let prediction = max_position(prediction);
		// kNN
		//fn sq(x: f64) -> f64 { x*x }
		let sq = |x| x*x;
		let distance = |a: &[f64; _], b| a.iter().zip(b).map(|(a,b)| sq(a-b)).sum::<f64>();
		let mut distances = Vec::from_iter(train().map(|(example, &label)| (distance(image, example), label)));
		let (kNN, _, _) = distances.select_nth_unstable_by(5, |(a,_),(b,_)| f64::total_cmp(a,b));
		let counts: [_; 10] = std::array::from_fn(|i| kNN.into_iter().filter(|(_,j)| i==*j).count());
		let prediction = counts.iter().enumerate().max_by(|(_, a),(_, b)| a.cmp(b)).map(|(i,_)| i).unwrap();
		println!("{prediction} {label}");
		prediction == *label
	}).count();
	println!("{matches}/{} = {}%", labels.len(), matches as f64/labels.len() as f64*100.);
	Ok(())
}
