use crate::{
    config::AppConfig,
    extractor::FeatureMatrix,
    features::{LogMelConfig, LogMelProcessor},
    framing::{frame_signal, FrameConfig},
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CachedExtractorConfig {
    pub app: AppConfig,
    pub num_mel_bins: usize,
    pub min_freq_hz: f32,
    pub max_freq_hz: f32,
    pub epsilon: f32,
}

impl CachedExtractorConfig {
    pub fn speech_default(app: AppConfig) -> Self {
        Self {
            app,
            num_mel_bins: 40,
            min_freq_hz: 0.0,
            max_freq_hz: app.sample_rate_hz as f32 / 2.0,
            epsilon: 1e-6,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CachedLogMelExtractor {
    config: CachedExtractorConfig,
    frame_size_samples: usize,
    hop_size_samples: usize,
    processor: LogMelProcessor,
}

impl CachedLogMelExtractor {
    pub fn new(config: CachedExtractorConfig) -> Self {
        let frame_size_samples = config.app.frame_size_samples();
        let hop_size_samples = config.app.hop_size_samples();
        let processor = LogMelProcessor::new(LogMelConfig {
            sample_rate_hz: config.app.sample_rate_hz,
            fft_size: frame_size_samples,
            num_mel_bins: config.num_mel_bins,
            min_freq_hz: config.min_freq_hz,
            max_freq_hz: config.max_freq_hz,
            epsilon: config.epsilon,
        });

        Self {
            config,
            frame_size_samples,
            hop_size_samples,
            processor,
        }
    }

    pub fn extract_frame(&mut self, frame: &[f32]) -> Vec<f32> {
        self.processor.process_frame(frame)
    }

    pub fn extract_frames(&mut self, frames: &[Vec<f32>]) -> Vec<Vec<f32>> {
        frames
            .iter()
            .map(|frame| self.extract_frame(frame))
            .filter(|row| !row.is_empty())
            .collect()
    }

    pub fn extract_samples(&mut self, samples: &[f32]) -> FeatureMatrix {
        let frames = frame_signal(
            samples,
            FrameConfig::new(self.frame_size_samples, self.hop_size_samples),
        );
        let values = self.extract_frames(&frames);
        let num_frames = values.len();
        let num_bins = values.first().map(|row| row.len()).unwrap_or(0);

        FeatureMatrix {
            values,
            num_frames,
            num_bins,
            frame_size_samples: self.frame_size_samples,
            hop_size_samples: self.hop_size_samples,
            sample_rate_hz: self.config.app.sample_rate_hz,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CachedExtractorConfig, CachedLogMelExtractor};
    use crate::{
        config::AppConfig,
        extractor::extract_log_mel_from_samples,
        features::{LogMelConfig, LogMelProcessor},
    };

    #[test]
    fn cached_config_speech_default_matches_app_config() {
        let app = AppConfig::default();
        let config = CachedExtractorConfig::speech_default(app);

        assert_eq!(config.app, app);
        assert_eq!(config.num_mel_bins, 40);
        assert_eq!(config.min_freq_hz, 0.0);
        assert_eq!(config.max_freq_hz, app.sample_rate_hz as f32 / 2.0);
        assert_eq!(config.epsilon, 1e-6);
    }

    #[test]
    fn cached_extractor_extract_frame_returns_40_bins() {
        let mut extractor =
            CachedLogMelExtractor::new(CachedExtractorConfig::speech_default(AppConfig::default()));

        let features = extractor.extract_frame(&vec![1.0; 400]);

        assert_eq!(features.len(), 40);
    }

    #[test]
    fn cached_extractor_extract_frame_values_are_finite() {
        let mut extractor =
            CachedLogMelExtractor::new(CachedExtractorConfig::speech_default(AppConfig::default()));

        let features = extractor.extract_frame(&vec![1.0; 400]);

        assert!(features.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn cached_extractor_rejects_wrong_frame_size() {
        let mut extractor =
            CachedLogMelExtractor::new(CachedExtractorConfig::speech_default(AppConfig::default()));

        assert!(extractor.extract_frame(&vec![1.0; 399]).is_empty());
    }

    #[test]
    fn cached_extractor_extract_samples_default_shape() {
        let mut extractor =
            CachedLogMelExtractor::new(CachedExtractorConfig::speech_default(AppConfig::default()));
        let samples: Vec<f32> = (0..1_600).map(|i| i as f32).collect();

        let features = extractor.extract_samples(&samples);

        assert_eq!(features.num_frames, 8);
        assert_eq!(features.num_bins, 40);
        assert_eq!(features.values.len(), 8);
        assert!(features.values.iter().all(|row| row.len() == 40));
    }

    #[test]
    fn cached_extractor_extract_samples_values_are_finite() {
        let mut extractor =
            CachedLogMelExtractor::new(CachedExtractorConfig::speech_default(AppConfig::default()));
        let samples: Vec<f32> = (0..1_600).map(|i| i as f32).collect();

        let features = extractor.extract_samples(&samples);

        assert!(features
            .values
            .iter()
            .flatten()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn cached_extractor_matches_existing_extractor_shape() {
        let app = AppConfig::default();
        let mut extractor = CachedLogMelExtractor::new(CachedExtractorConfig::speech_default(app));
        let samples: Vec<f32> = (0..1_600).map(|i| i as f32).collect();

        let cached = extractor.extract_samples(&samples);
        let existing = extract_log_mel_from_samples(&samples, &app);

        assert_eq!(cached.num_frames, existing.num_frames);
        assert_eq!(cached.num_bins, existing.num_bins);
        assert_eq!(cached.frame_size_samples, existing.frame_size_samples);
        assert_eq!(cached.hop_size_samples, existing.hop_size_samples);
        assert_eq!(cached.sample_rate_hz, existing.sample_rate_hz);
    }

    #[test]
    fn cached_extractor_uses_same_shape_as_existing_extractor() {
        let app = AppConfig::default();
        let mut cached = CachedLogMelExtractor::new(CachedExtractorConfig::speech_default(app));
        let samples: Vec<f32> = (0..1_600).map(|i| i as f32).collect();
        let cached_features = cached.extract_samples(&samples);
        let existing = extract_log_mel_from_samples(&samples, &app);

        assert_eq!(cached_features.values.len(), existing.values.len());
        assert!(cached_features.values.iter().all(|row| row.len() == 40));
    }

    #[test]
    fn cached_extractor_multiple_calls_are_stable() {
        let mut extractor =
            CachedLogMelExtractor::new(CachedExtractorConfig::speech_default(AppConfig::default()));
        let samples: Vec<f32> = (0..1_600).map(|i| (i as f32).sin()).collect();

        let first = extractor.extract_samples(&samples);
        let second = extractor.extract_samples(&samples);

        assert_eq!(first.num_frames, second.num_frames);
        assert_eq!(first.num_bins, second.num_bins);
        assert_eq!(first.values.len(), second.values.len());
        assert!(second
            .values
            .iter()
            .flatten()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn cached_extractor_matches_log_mel_processor_for_one_frame() {
        let config = LogMelConfig::speech_default(16_000, 400);
        let frame: Vec<f32> = (0..400).map(|index| (index as f32 * 0.1).sin()).collect();
        let mut cached =
            CachedLogMelExtractor::new(CachedExtractorConfig::speech_default(AppConfig::default()));
        let mut processor = LogMelProcessor::new(config);

        assert_eq!(
            cached.extract_frame(&frame),
            processor.process_frame(&frame)
        );
    }
}
