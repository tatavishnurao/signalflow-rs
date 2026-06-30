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

#[cfg(test)]
mod tests {
    use super::{normalize_samples, rms_energy};

    #[test]
    fn rms_energy_of_empty_slice_is_zero() {
        assert_eq!(rms_energy(&[]), 0.0);
    }

    #[test]
    fn rms_energy_of_nonzero_samples_is_positive() {
        assert!(rms_energy(&[1.0, -1.0, 0.5]) > 0.0);
    }

    #[test]
    fn normalize_samples_scales_max_abs_to_one() {
        let mut samples = [2.0, -4.0, 1.0];

        normalize_samples(&mut samples);

        let max_abs = samples
            .iter()
            .fold(0.0_f32, |acc, &sample| acc.max(sample.abs()));
        assert!((max_abs - 1.0).abs() < 1e-6);
    }
}
