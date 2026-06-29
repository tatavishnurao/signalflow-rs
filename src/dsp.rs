pub fn normalize_samples(samples: &mut [f32]) {
    let max_abs = samples
        .iter()
        .fold(0.0_f32, |acc, &sample| acc.max(sample.abs()));

    if max_abs > 0.0 {
        for sample in samples {
            *sample /= max_abs;
        }
    }
}

pub fn rms_energy(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let sum_sq: f32 = samples.iter().map(|sample| sample * sample).sum();
    (sum_sq / samples.len() as f32).sqrt()
}
