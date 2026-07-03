use crate::{
    config::AppConfig,
    features::{log_mel_frame, LogMelConfig},
};

#[derive(Debug, Clone, PartialEq)]
pub struct StreamingOutput {
    pub features: Vec<Vec<f32>>,
    pub num_frames: usize,
    pub num_bins: usize,
    pub consumed_samples: usize,
    pub pending_samples: usize,
}

#[derive(Debug, Clone)]
pub struct StreamingExtractor {
    config: AppConfig,
    frame_size_samples: usize,
    hop_size_samples: usize,
    log_mel_config: LogMelConfig,
    pending: Vec<f32>,
    total_input_samples: usize,
    total_emitted_frames: usize,
}

impl StreamingExtractor {
    pub fn new(config: AppConfig) -> Self {
        let frame_size_samples = config.frame_size_samples();
        let hop_size_samples = config.hop_size_samples();
        let log_mel_config =
            LogMelConfig::speech_default(config.sample_rate_hz, frame_size_samples);

        Self {
            config,
            frame_size_samples,
            hop_size_samples,
            log_mel_config,
            pending: Vec::new(),
            total_input_samples: 0,
            total_emitted_frames: 0,
        }
    }

    pub fn push_samples(&mut self, samples: &[f32]) -> StreamingOutput {
        let _ = self.config.sample_rate_hz;
        self.pending.extend_from_slice(samples);
        self.total_input_samples += samples.len();

        let mut features = Vec::new();
        let mut consumed_samples = 0;

        if self.frame_size_samples == 0 || self.hop_size_samples == 0 {
            return StreamingOutput {
                num_frames: 0,
                num_bins: 0,
                consumed_samples,
                pending_samples: self.pending.len(),
                features,
            };
        }

        while self.pending.len() >= self.frame_size_samples {
            let frame = self.pending[..self.frame_size_samples].to_vec();
            let row = log_mel_frame(&frame, self.log_mel_config);
            features.push(row);
            self.pending.drain(0..self.hop_size_samples);
            consumed_samples += self.hop_size_samples;
            self.total_emitted_frames += 1;
        }

        let num_frames = features.len();
        let num_bins = features.first().map(|row| row.len()).unwrap_or(0);

        StreamingOutput {
            features,
            num_frames,
            num_bins,
            consumed_samples,
            pending_samples: self.pending.len(),
        }
    }

    pub fn pending_samples(&self) -> usize {
        self.pending.len()
    }

    pub fn total_input_samples(&self) -> usize {
        self.total_input_samples
    }

    pub fn total_emitted_frames(&self) -> usize {
        self.total_emitted_frames
    }
}

#[cfg(test)]
mod tests {
    use super::{StreamingExtractor, StreamingOutput};
    use crate::{
        audio::generate_dummy_audio, config::AppConfig, extractor::extract_log_mel_from_samples,
    };

    fn flatten_features(outputs: &[StreamingOutput]) -> Vec<Vec<f32>> {
        outputs
            .iter()
            .flat_map(|output| output.features.iter().cloned())
            .collect()
    }

    #[test]
    fn streaming_empty_push_returns_no_frames() {
        let mut extractor = StreamingExtractor::new(AppConfig::default());
        let output = extractor.push_samples(&[]);

        assert_eq!(output.num_frames, 0);
        assert_eq!(output.num_bins, 0);
        assert_eq!(output.consumed_samples, 0);
        assert_eq!(output.pending_samples, 0);
        assert_eq!(extractor.pending_samples(), 0);
    }

    #[test]
    fn streaming_short_push_buffers_samples() {
        let mut extractor = StreamingExtractor::new(AppConfig::default());
        let output = extractor.push_samples(&vec![1.0; 100]);

        assert_eq!(output.num_frames, 0);
        assert_eq!(output.pending_samples, 100);
        assert_eq!(extractor.pending_samples(), 100);
    }

    #[test]
    fn streaming_emits_first_frame_after_enough_samples() {
        let mut extractor = StreamingExtractor::new(AppConfig::default());
        let output = extractor.push_samples(&vec![1.0; 400]);

        assert_eq!(output.num_frames, 1);
        assert_eq!(output.num_bins, 40);
        assert_eq!(output.consumed_samples, 160);
        assert_eq!(output.pending_samples, 240);
    }

    #[test]
    fn streaming_emits_expected_frames_for_1600_samples_single_push() {
        let mut extractor = StreamingExtractor::new(AppConfig::default());
        let output = extractor.push_samples(&vec![1.0; 1_600]);

        assert_eq!(output.num_frames, 8);
        assert_eq!(output.num_bins, 40);
        assert_eq!(output.pending_samples, 320);
        assert_eq!(extractor.total_emitted_frames(), 8);
    }

    #[test]
    fn streaming_emits_expected_frames_for_1600_samples_chunked_push() {
        let mut extractor = StreamingExtractor::new(AppConfig::default());
        let mut total_frames = 0;
        let mut last_output = StreamingOutput {
            features: Vec::new(),
            num_frames: 0,
            num_bins: 0,
            consumed_samples: 0,
            pending_samples: 0,
        };

        for _ in 0..10 {
            last_output = extractor.push_samples(&vec![1.0; 160]);
            total_frames += last_output.num_frames;
        }

        assert_eq!(total_frames, 8);
        assert_eq!(extractor.total_emitted_frames(), 8);
        assert_eq!(last_output.pending_samples, 320);
    }

    #[test]
    fn streaming_outputs_have_40_bins() {
        let mut extractor = StreamingExtractor::new(AppConfig::default());
        let output = extractor.push_samples(&vec![1.0; 1_600]);

        assert!(output.features.iter().all(|row| row.len() == 40));
        assert_eq!(output.num_bins, 40);
    }

    #[test]
    fn streaming_values_are_finite() {
        let mut extractor = StreamingExtractor::new(AppConfig::default());
        let output = extractor.push_samples(&vec![1.0; 1_600]);

        assert!(output
            .features
            .iter()
            .flatten()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn streaming_tracks_total_input_samples() {
        let mut extractor = StreamingExtractor::new(AppConfig::default());
        extractor.push_samples(&vec![1.0; 1_600]);

        assert_eq!(extractor.total_input_samples(), 1_600);
    }

    #[test]
    fn streaming_tracks_total_emitted_frames() {
        let mut extractor = StreamingExtractor::new(AppConfig::default());
        extractor.push_samples(&vec![1.0; 1_600]);

        assert_eq!(extractor.total_emitted_frames(), 8);
    }

    #[test]
    fn streaming_preserves_overlap_across_pushes() {
        let samples: Vec<f32> = (0..1_600).map(|i| i as f32).collect();
        let reference = extract_log_mel_from_samples(&samples, &AppConfig::default());
        let mut extractor = StreamingExtractor::new(AppConfig::default());
        let mut outputs = Vec::new();

        for chunk in samples.chunks(160) {
            outputs.push(extractor.push_samples(chunk));
        }

        let streamed = flatten_features(&outputs);

        assert_eq!(streamed, reference.values);
        assert_eq!(extractor.total_emitted_frames(), reference.num_frames);
        assert_eq!(extractor.pending_samples(), 320);
    }

    #[test]
    fn streaming_demo_like_path_handles_dummy_audio() {
        let mut extractor = StreamingExtractor::new(AppConfig::default());
        let chunk = generate_dummy_audio(&AppConfig::default(), 100);

        for slice in chunk.samples.chunks(160) {
            extractor.push_samples(slice);
        }

        assert_eq!(extractor.total_input_samples(), 1_600);
        assert_eq!(extractor.total_emitted_frames(), 8);
    }
}
