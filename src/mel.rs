#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MelConfig {
    pub sample_rate_hz: u32,
    pub fft_size: usize,
    pub num_mel_bins: usize,
    pub min_freq_hz: f32,
    pub max_freq_hz: f32,
}

impl MelConfig {
    pub fn speech_default(sample_rate_hz: u32, fft_size: usize) -> Self {
        Self {
            sample_rate_hz,
            fft_size,
            num_mel_bins: 40,
            min_freq_hz: 0.0,
            max_freq_hz: sample_rate_hz as f32 / 2.0,
        }
    }
}

pub fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

pub fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

pub fn build_mel_filterbank(config: MelConfig) -> Vec<Vec<f32>> {
    if config.fft_size == 0 || config.num_mel_bins == 0 || config.sample_rate_hz == 0 {
        return Vec::new();
    }

    let fft_bins = config.fft_size / 2 + 1;
    if fft_bins == 0 {
        return Vec::new();
    }

    let min_mel = hz_to_mel(config.min_freq_hz.max(0.0));
    let max_mel = hz_to_mel(config.max_freq_hz.max(config.min_freq_hz));
    let mel_points: Vec<f32> = (0..config.num_mel_bins + 2)
        .map(|i| min_mel + (i as f32) * (max_mel - min_mel) / (config.num_mel_bins + 1) as f32)
        .collect();
    let hz_points: Vec<f32> = mel_points.into_iter().map(mel_to_hz).collect();
    let mut bin_points: Vec<usize> = hz_points
        .into_iter()
        .map(|hz| {
            let bin = ((config.fft_size as f32 + 1.0) * hz / config.sample_rate_hz as f32).floor();
            bin.clamp(0.0, (fft_bins - 1) as f32) as usize
        })
        .collect();

    if bin_points.len() < 3 {
        return Vec::new();
    }

    for point in &mut bin_points {
        *point = (*point).min(fft_bins - 1);
    }

    let mut filterbank = Vec::with_capacity(config.num_mel_bins);
    for i in 1..bin_points.len() - 1 {
        let left = bin_points[i - 1];
        let center = bin_points[i];
        let right = bin_points[i + 1];

        let mut row = vec![0.0; fft_bins];

        if left == center && center == right {
            row[center] = 1.0;
            filterbank.push(row);
            continue;
        }

        if center > left {
            for (bin, slot) in row
                .iter_mut()
                .enumerate()
                .take(center.min(fft_bins - 1) + 1)
                .skip(left)
            {
                *slot = (bin - left) as f32 / (center - left) as f32;
            }
        } else if center < fft_bins {
            row[center] = 1.0;
        }

        if right > center {
            for (bin, slot) in row
                .iter_mut()
                .enumerate()
                .take(right.min(fft_bins - 1) + 1)
                .skip(center)
            {
                *slot = (*slot).max((right - bin) as f32 / (right - center) as f32);
            }
        } else if center < fft_bins {
            row[center] = row[center].max(1.0);
        }

        filterbank.push(row);
    }

    filterbank
}

pub fn apply_mel_filterbank(power_spectrum: &[f32], filterbank: &[Vec<f32>]) -> Vec<f32> {
    filterbank
        .iter()
        .map(|filter| {
            filter
                .iter()
                .zip(power_spectrum.iter())
                .map(|(weight, power)| weight * power)
                .sum()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{apply_mel_filterbank, build_mel_filterbank, hz_to_mel, mel_to_hz, MelConfig};

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn hz_to_mel_zero_is_zero() {
        assert!(approx_eq(hz_to_mel(0.0), 0.0, 1e-6));
    }

    #[test]
    fn mel_to_hz_zero_is_zero() {
        assert!(approx_eq(mel_to_hz(0.0), 0.0, 1e-6));
    }

    #[test]
    fn hz_mel_roundtrip() {
        let hz = 1000.0;
        let roundtrip = mel_to_hz(hz_to_mel(hz));

        assert!(approx_eq(roundtrip, hz, 1e-3));
    }

    #[test]
    fn build_mel_filterbank_empty_for_invalid_config() {
        assert!(build_mel_filterbank(MelConfig {
            sample_rate_hz: 0,
            fft_size: 400,
            num_mel_bins: 40,
            min_freq_hz: 0.0,
            max_freq_hz: 8000.0,
        })
        .is_empty());

        assert!(build_mel_filterbank(MelConfig {
            sample_rate_hz: 16_000,
            fft_size: 0,
            num_mel_bins: 40,
            min_freq_hz: 0.0,
            max_freq_hz: 8000.0,
        })
        .is_empty());

        assert!(build_mel_filterbank(MelConfig {
            sample_rate_hz: 16_000,
            fft_size: 400,
            num_mel_bins: 0,
            min_freq_hz: 0.0,
            max_freq_hz: 8000.0,
        })
        .is_empty());
    }

    #[test]
    fn build_mel_filterbank_has_expected_shape() {
        let filterbank = build_mel_filterbank(MelConfig::speech_default(16_000, 400));

        assert_eq!(filterbank.len(), 40);
        assert!(filterbank.iter().all(|row| row.len() == 201));
    }

    #[test]
    fn mel_filterbank_values_are_non_negative() {
        let filterbank = build_mel_filterbank(MelConfig::speech_default(16_000, 400));

        assert!(filterbank.iter().flatten().all(|&value| value >= 0.0));
    }

    #[test]
    fn apply_mel_filterbank_returns_one_value_per_filter() {
        let filterbank = build_mel_filterbank(MelConfig::speech_default(16_000, 400));
        let power_spectrum = vec![1.0; 201];

        assert_eq!(apply_mel_filterbank(&power_spectrum, &filterbank).len(), 40);
    }

    #[test]
    fn apply_mel_filterbank_handles_mismatched_lengths() {
        let filterbank = vec![vec![1.0, 2.0, 3.0], vec![0.5, 0.5, 0.5]];
        let power_spectrum = vec![2.0, 4.0];

        assert_eq!(
            apply_mel_filterbank(&power_spectrum, &filterbank),
            vec![10.0, 3.0]
        );
    }
}
