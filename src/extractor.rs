use crate::{
    config::AppConfig,
    features::{log_mel_features, LogMelConfig},
    framing::{frame_signal, FrameConfig},
    metrics::ExtractionMetrics,
};
use std::time::Instant;

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

#[derive(Debug, Clone, PartialEq)]
pub struct BatchFeatureSet {
    pub items: Vec<FeatureMatrix>,
    pub num_items: usize,
    pub total_frames: usize,
    pub num_bins: usize,
    pub sample_rate_hz: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TimedFeatureMatrix {
    pub features: FeatureMatrix,
    pub metrics: ExtractionMetrics,
}

pub fn extract_log_mel_batch(buffers: &[Vec<f32>], config: &AppConfig) -> BatchFeatureSet {
    let items: Vec<FeatureMatrix> = buffers
        .iter()
        .map(|buffer| extract_log_mel_from_samples(buffer, config))
        .collect();
    let num_items = items.len();
    let total_frames = items.iter().map(|item| item.num_frames).sum();
    let num_bins = items
        .iter()
        .find(|item| item.num_bins > 0)
        .map(|item| item.num_bins)
        .unwrap_or(0);

    BatchFeatureSet {
        items,
        num_items,
        total_frames,
        num_bins,
        sample_rate_hz: config.sample_rate_hz,
    }
}

pub fn extract_log_mel_timed(samples: &[f32], config: &AppConfig) -> TimedFeatureMatrix {
    let start = Instant::now();
    let features = extract_log_mel_from_samples(samples, config);
    let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
    let metrics = ExtractionMetrics::new(
        elapsed_ms,
        samples.len(),
        features.num_frames,
        features.num_bins,
    );

    TimedFeatureMatrix { features, metrics }
}

#[cfg(test)]
mod tests {
    use super::{
        extract_log_mel_batch, extract_log_mel_from_samples, extract_log_mel_timed, BatchFeatureSet,
    };
    use crate::config::AppConfig;
    use crate::metrics::ExtractionMetrics;

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

    #[test]
    fn batch_empty_input_returns_empty_set() {
        let batch = extract_log_mel_batch(&[], &AppConfig::default());

        assert_eq!(
            batch,
            BatchFeatureSet {
                items: Vec::new(),
                num_items: 0,
                total_frames: 0,
                num_bins: 0,
                sample_rate_hz: 16_000,
            }
        );
    }

    #[test]
    fn batch_single_valid_buffer() {
        let batch = extract_log_mel_batch(&[vec![1.0; 1_600]], &AppConfig::default());

        assert_eq!(batch.num_items, 1);
        assert_eq!(batch.total_frames, 8);
        assert_eq!(batch.num_bins, 40);
        assert_eq!(batch.sample_rate_hz, 16_000);
        assert_eq!(batch.items.len(), 1);
        assert_eq!(batch.items[0].num_frames, 8);
        assert_eq!(batch.items[0].num_bins, 40);
    }

    #[test]
    fn batch_multiple_valid_buffers() {
        let batch =
            extract_log_mel_batch(&[vec![1.0; 1_600], vec![0.5; 1_600]], &AppConfig::default());

        assert_eq!(batch.num_items, 2);
        assert_eq!(batch.total_frames, 16);
        assert_eq!(batch.num_bins, 40);
        assert_eq!(batch.items.len(), 2);
    }

    #[test]
    fn batch_preserves_one_item_per_buffer() {
        let buffers = vec![vec![], vec![0.0; 100], vec![1.0; 1_600]];
        let batch = extract_log_mel_batch(&buffers, &AppConfig::default());

        assert_eq!(batch.num_items, 3);
        assert_eq!(batch.items.len(), 3);
        assert_eq!(batch.items[0].num_frames, 0);
        assert_eq!(batch.items[1].num_frames, 0);
        assert_eq!(batch.items[2].num_frames, 8);
    }

    #[test]
    fn batch_handles_short_and_empty_buffers() {
        let buffers = vec![vec![], vec![0.0; 100], vec![1.0; 1_600]];
        let batch = extract_log_mel_batch(&buffers, &AppConfig::default());

        assert_eq!(batch.total_frames, 8);
        assert_eq!(batch.num_bins, 40);
        assert_eq!(batch.items[0].num_frames, 0);
        assert_eq!(batch.items[1].num_frames, 0);
        assert_eq!(batch.items[2].num_frames, 8);
    }

    #[test]
    fn batch_total_frames_is_sum_of_items() {
        let batch =
            extract_log_mel_batch(&[vec![1.0; 1_600], vec![1.0; 1_600]], &AppConfig::default());

        let sum_of_items: usize = batch.items.iter().map(|item| item.num_frames).sum();

        assert_eq!(batch.total_frames, sum_of_items);
    }

    #[test]
    fn batch_values_are_finite() {
        let batch =
            extract_log_mel_batch(&[vec![1.0; 1_600], vec![0.5; 1_600]], &AppConfig::default());

        assert!(batch
            .items
            .iter()
            .flat_map(|item| item.values.iter())
            .flat_map(|row| row.iter())
            .all(|value| value.is_finite()));
    }

    #[test]
    fn extraction_metrics_zero_elapsed_has_zero_rates() {
        let metrics = ExtractionMetrics::new(0.0, 1_600, 8, 40);

        assert_eq!(metrics.elapsed_ms, 0.0);
        assert_eq!(metrics.samples_per_second, 0.0);
        assert_eq!(metrics.frames_per_second, 0.0);
    }

    #[test]
    fn extraction_metrics_computes_rates() {
        let metrics = ExtractionMetrics::new(200.0, 2_000, 10, 40);

        assert_eq!(metrics.input_samples, 2_000);
        assert_eq!(metrics.output_frames, 10);
        assert_eq!(metrics.output_bins, 40);
        assert_eq!(metrics.samples_per_second, 10_000.0);
        assert_eq!(metrics.frames_per_second, 50.0);
    }

    #[test]
    fn timed_extractor_returns_features_and_metrics() {
        let timed = extract_log_mel_timed(&vec![1.0; 1_600], &AppConfig::default());

        assert_eq!(timed.features.num_frames, 8);
        assert_eq!(timed.features.num_bins, 40);
        assert_eq!(timed.metrics.input_samples, 1_600);
        assert_eq!(timed.metrics.output_frames, 8);
        assert_eq!(timed.metrics.output_bins, 40);
        assert!(timed.metrics.elapsed_ms >= 0.0);
    }
}
