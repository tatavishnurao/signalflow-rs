use crate::{
    config::AppConfig,
    features::{log_mel_features, LogMelConfig},
    framing::{frame_signal, FrameConfig},
};

#[derive(Debug, Clone, PartialEq)]
pub struct FeatureMatrix {
    pub values: Vec<Vec<f32>>,
    pub num_frames: usize,
    pub num_bins: usize,
    pub frame_size_samples: usize,
    pub hop_size_samples: usize,
    pub sample_rate_hz: u32,
}

pub fn extract_log_mel_from_samples(samples: &[f32], config: &AppConfig) -> FeatureMatrix {
    let frame_size_samples = config.frame_size_samples();
    let hop_size_samples = config.hop_size_samples();
    let frame_config = FrameConfig::new(frame_size_samples, hop_size_samples);
    let frames = frame_signal(samples, frame_config);
    let values = log_mel_features(
        &frames,
        LogMelConfig::speech_default(config.sample_rate_hz, frame_size_samples),
    );
    let num_frames = values.len();
    let num_bins = values.first().map(|row| row.len()).unwrap_or(0);

    FeatureMatrix {
        values,
        num_frames,
        num_bins,
        frame_size_samples,
        hop_size_samples,
        sample_rate_hz: config.sample_rate_hz,
    }
}

#[cfg(test)]
mod tests {
    use super::extract_log_mel_from_samples;
    use crate::config::AppConfig;

    #[test]
    fn extractor_empty_input_returns_empty_matrix() {
        let matrix = extract_log_mel_from_samples(&[], &AppConfig::default());

        assert!(matrix.values.is_empty());
        assert_eq!(matrix.num_frames, 0);
        assert_eq!(matrix.num_bins, 0);
        assert_eq!(matrix.frame_size_samples, 400);
        assert_eq!(matrix.hop_size_samples, 160);
        assert_eq!(matrix.sample_rate_hz, 16_000);
    }

    #[test]
    fn extractor_short_input_returns_empty_matrix() {
        let matrix = extract_log_mel_from_samples(&vec![1.0; 399], &AppConfig::default());

        assert!(matrix.values.is_empty());
        assert_eq!(matrix.num_frames, 0);
        assert_eq!(matrix.num_bins, 0);
        assert_eq!(matrix.frame_size_samples, 400);
        assert_eq!(matrix.hop_size_samples, 160);
        assert_eq!(matrix.sample_rate_hz, 16_000);
    }

    #[test]
    fn extractor_default_audio_shape() {
        let samples: Vec<f32> = (0..1_600).map(|i| i as f32).collect();
        let matrix = extract_log_mel_from_samples(&samples, &AppConfig::default());

        assert_eq!(matrix.num_frames, 8);
        assert_eq!(matrix.num_bins, 40);
        assert_eq!(matrix.values.len(), 8);
        assert!(matrix.values.iter().all(|row| row.len() == 40));
    }

    #[test]
    fn extractor_values_are_finite() {
        let samples: Vec<f32> = (0..1_600).map(|i| i as f32).collect();
        let matrix = extract_log_mel_from_samples(&samples, &AppConfig::default());

        assert!(matrix
            .values
            .iter()
            .flatten()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn extractor_preserves_config_metadata() {
        let matrix = extract_log_mel_from_samples(&vec![1.0; 1_600], &AppConfig::default());

        assert_eq!(matrix.frame_size_samples, 400);
        assert_eq!(matrix.hop_size_samples, 160);
        assert_eq!(matrix.sample_rate_hz, 16_000);
    }
}
