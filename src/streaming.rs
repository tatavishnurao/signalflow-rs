use crate::{
    cached::{CachedExtractorConfig, CachedLogMelExtractor},
    config::AppConfig,
    features::{LogMelConfig, LogMelProcessor},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamingConfig {
    pub app: AppConfig,
    pub max_pending_samples: Option<usize>,
}

impl StreamingConfig {
    pub fn new(app: AppConfig) -> Self {
        Self {
            app,
            max_pending_samples: None,
        }
    }

    pub fn with_max_pending_samples(app: AppConfig, max_pending_samples: usize) -> Self {
        Self {
            app,
            max_pending_samples: Some(max_pending_samples),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StreamingOutput {
    pub features: Vec<Vec<f32>>,
    pub num_frames: usize,
    pub num_bins: usize,
    pub consumed_samples: usize,
    pub pending_samples: usize,
    pub dropped_samples: usize,
    pub dropped_frames: usize,
    pub peak_pending_samples: usize,
}

#[derive(Debug, Clone)]
pub struct StreamingExtractor {
    streaming_config: StreamingConfig,
    frame_size_samples: usize,
    hop_size_samples: usize,
    log_mel_processor: LogMelProcessor,
    pending: Vec<f32>,
    read_offset: usize,
    total_input_samples: usize,
    total_emitted_frames: usize,
    total_consumed_samples: usize,
    total_dropped_samples: usize,
    total_dropped_frames: usize,
    peak_pending_samples: usize,
}

pub struct CachedStreamingExtractor {
    streaming_config: StreamingConfig,
    frame_size_samples: usize,
    hop_size_samples: usize,
    cached_extractor: CachedLogMelExtractor,
    pending: Vec<f32>,
    read_offset: usize,
    total_input_samples: usize,
    total_emitted_frames: usize,
    total_consumed_samples: usize,
    total_dropped_samples: usize,
    total_dropped_frames: usize,
    peak_pending_samples: usize,
}

impl StreamingExtractor {
    pub fn new(config: AppConfig) -> Self {
        Self::with_streaming_config(StreamingConfig::new(config))
    }

    pub fn with_streaming_config(streaming_config: StreamingConfig) -> Self {
        let app = streaming_config.app;
        let frame_size_samples = app.frame_size_samples();
        let hop_size_samples = app.hop_size_samples();
        let log_mel_processor = LogMelProcessor::new(LogMelConfig::speech_default(
            app.sample_rate_hz,
            frame_size_samples,
        ));

        Self {
            streaming_config,
            frame_size_samples,
            hop_size_samples,
            log_mel_processor,
            pending: Vec::new(),
            read_offset: 0,
            total_input_samples: 0,
            total_emitted_frames: 0,
            total_consumed_samples: 0,
            total_dropped_samples: 0,
            total_dropped_frames: 0,
            peak_pending_samples: 0,
        }
    }

    pub fn push_samples(&mut self, samples: &[f32]) -> StreamingOutput {
        self.pending.extend_from_slice(samples);
        self.total_input_samples += samples.len();

        let logical_pending_after_append = self.pending.len().saturating_sub(self.read_offset);
        if logical_pending_after_append > self.peak_pending_samples {
            self.peak_pending_samples = logical_pending_after_append;
        }

        let mut features = Vec::new();
        let mut consumed_samples = 0;

        if self.frame_size_samples != 0 && self.hop_size_samples != 0 {
            while self.pending.len().saturating_sub(self.read_offset) >= self.frame_size_samples {
                let frame =
                    &self.pending[self.read_offset..self.read_offset + self.frame_size_samples];
                let row = self.log_mel_processor.process_frame(frame);
                features.push(row);
                self.read_offset += self.hop_size_samples;
                self.total_consumed_samples += self.hop_size_samples;
                consumed_samples += self.hop_size_samples;
                self.total_emitted_frames += 1;
            }
        }

        self.compact_if_needed();
        let (dropped_samples, dropped_frames) = self.enforce_pending_limit();
        self.total_dropped_samples += dropped_samples;
        self.total_dropped_frames += dropped_frames;

        let num_frames = features.len();
        let num_bins = features.first().map(|row| row.len()).unwrap_or(0);

        StreamingOutput {
            features,
            num_frames,
            num_bins,
            consumed_samples,
            pending_samples: self.pending_samples(),
            dropped_samples,
            dropped_frames,
            peak_pending_samples: self.peak_pending_samples,
        }
    }

    pub fn pending_samples(&self) -> usize {
        self.pending.len().saturating_sub(self.read_offset)
    }

    pub fn total_input_samples(&self) -> usize {
        self.total_input_samples
    }

    pub fn total_emitted_frames(&self) -> usize {
        self.total_emitted_frames
    }

    pub fn total_consumed_samples(&self) -> usize {
        self.total_consumed_samples
    }

    pub fn total_dropped_samples(&self) -> usize {
        self.total_dropped_samples
    }

    pub fn total_dropped_frames(&self) -> usize {
        self.total_dropped_frames
    }

    pub fn peak_pending_samples(&self) -> usize {
        self.peak_pending_samples
    }

    pub fn max_pending_samples(&self) -> usize {
        self.peak_pending_samples
    }

    fn compact_if_needed(&mut self) {
        if self.read_offset == 0 {
            return;
        }

        if self.read_offset >= self.hop_size_samples.saturating_mul(4)
            || self.read_offset >= self.pending.len() / 2
        {
            self.pending.drain(0..self.read_offset);
            self.read_offset = 0;
        }
    }

    fn enforce_pending_limit(&mut self) -> (usize, usize) {
        let Some(limit) = self.streaming_config.max_pending_samples else {
            return (0, 0);
        };

        if self.read_offset > 0 {
            self.pending.drain(0..self.read_offset);
            self.read_offset = 0;
        }

        let logical_pending = self.pending.len();
        if logical_pending <= limit || self.hop_size_samples == 0 {
            return (0, 0);
        }

        let mut dropped_samples = logical_pending - limit;
        let hop = self.hop_size_samples;
        let remainder = dropped_samples % hop;
        if remainder != 0 {
            dropped_samples += hop - remainder;
        }
        dropped_samples = dropped_samples.min(logical_pending);

        if dropped_samples == 0 {
            return (0, 0);
        }

        self.pending.drain(0..dropped_samples);
        let dropped_frames = dropped_samples / hop;
        (dropped_samples, dropped_frames)
    }
}

impl CachedStreamingExtractor {
    pub fn new(config: AppConfig) -> Self {
        Self::with_streaming_config(StreamingConfig::new(config))
    }

    pub fn with_streaming_config(streaming_config: StreamingConfig) -> Self {
        let app = streaming_config.app;
        let frame_size_samples = app.frame_size_samples();
        let hop_size_samples = app.hop_size_samples();
        let cached_extractor =
            CachedLogMelExtractor::new(CachedExtractorConfig::speech_default(app));

        Self {
            streaming_config,
            frame_size_samples,
            hop_size_samples,
            cached_extractor,
            pending: Vec::new(),
            read_offset: 0,
            total_input_samples: 0,
            total_emitted_frames: 0,
            total_consumed_samples: 0,
            total_dropped_samples: 0,
            total_dropped_frames: 0,
            peak_pending_samples: 0,
        }
    }

    pub fn push_samples(&mut self, samples: &[f32]) -> StreamingOutput {
        self.pending.extend_from_slice(samples);
        self.total_input_samples += samples.len();

        let logical_pending_after_append = self.pending.len().saturating_sub(self.read_offset);
        if logical_pending_after_append > self.peak_pending_samples {
            self.peak_pending_samples = logical_pending_after_append;
        }

        let mut features = Vec::new();
        let mut consumed_samples = 0;

        if self.frame_size_samples != 0 && self.hop_size_samples != 0 {
            while self.pending.len().saturating_sub(self.read_offset) >= self.frame_size_samples {
                let frame =
                    &self.pending[self.read_offset..self.read_offset + self.frame_size_samples];
                let row = self.cached_extractor.extract_frame(frame);
                features.push(row);
                self.read_offset += self.hop_size_samples;
                self.total_consumed_samples += self.hop_size_samples;
                consumed_samples += self.hop_size_samples;
                self.total_emitted_frames += 1;
            }
        }

        self.compact_if_needed();
        let (dropped_samples, dropped_frames) = self.enforce_pending_limit();
        self.total_dropped_samples += dropped_samples;
        self.total_dropped_frames += dropped_frames;

        let num_frames = features.len();
        let num_bins = features.first().map(|row| row.len()).unwrap_or(0);

        StreamingOutput {
            features,
            num_frames,
            num_bins,
            consumed_samples,
            pending_samples: self.pending_samples(),
            dropped_samples,
            dropped_frames,
            peak_pending_samples: self.peak_pending_samples,
        }
    }

    pub fn pending_samples(&self) -> usize {
        self.pending.len().saturating_sub(self.read_offset)
    }

    pub fn total_input_samples(&self) -> usize {
        self.total_input_samples
    }

    pub fn total_emitted_frames(&self) -> usize {
        self.total_emitted_frames
    }

    pub fn total_consumed_samples(&self) -> usize {
        self.total_consumed_samples
    }

    pub fn total_dropped_samples(&self) -> usize {
        self.total_dropped_samples
    }

    pub fn total_dropped_frames(&self) -> usize {
        self.total_dropped_frames
    }

    pub fn peak_pending_samples(&self) -> usize {
        self.peak_pending_samples
    }

    pub fn max_pending_samples(&self) -> usize {
        self.peak_pending_samples
    }

    fn compact_if_needed(&mut self) {
        if self.read_offset == 0 {
            return;
        }

        if self.read_offset >= self.hop_size_samples.saturating_mul(4)
            || self.read_offset >= self.pending.len() / 2
        {
            self.pending.drain(0..self.read_offset);
            self.read_offset = 0;
        }
    }

    fn enforce_pending_limit(&mut self) -> (usize, usize) {
        let Some(limit) = self.streaming_config.max_pending_samples else {
            return (0, 0);
        };

        if self.read_offset > 0 {
            self.pending.drain(0..self.read_offset);
            self.read_offset = 0;
        }

        let logical_pending = self.pending.len();
        if logical_pending <= limit || self.hop_size_samples == 0 {
            return (0, 0);
        }

        let mut dropped_samples = logical_pending - limit;
        let hop = self.hop_size_samples;
        let remainder = dropped_samples % hop;
        if remainder != 0 {
            dropped_samples += hop - remainder;
        }
        dropped_samples = dropped_samples.min(logical_pending);

        if dropped_samples == 0 {
            return (0, 0);
        }

        self.pending.drain(0..dropped_samples);
        let dropped_frames = dropped_samples / hop;
        (dropped_samples, dropped_frames)
    }
}

#[cfg(test)]
mod tests {
    use super::{CachedStreamingExtractor, StreamingConfig, StreamingExtractor, StreamingOutput};
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
    fn streaming_config_new_is_unbounded() {
        let config = StreamingConfig::new(AppConfig::default());

        assert_eq!(config.app, AppConfig::default());
        assert_eq!(config.max_pending_samples, None);
    }

    #[test]
    fn streaming_config_with_max_pending_samples_sets_limit() {
        let config = StreamingConfig::with_max_pending_samples(AppConfig::default(), 512);

        assert_eq!(config.max_pending_samples, Some(512));
    }

    #[test]
    fn streaming_empty_push_returns_no_frames() {
        let mut extractor = StreamingExtractor::new(AppConfig::default());
        let output = extractor.push_samples(&[]);

        assert_eq!(output.num_frames, 0);
        assert_eq!(output.num_bins, 0);
        assert_eq!(output.consumed_samples, 0);
        assert_eq!(output.pending_samples, 0);
        assert_eq!(output.dropped_samples, 0);
        assert_eq!(output.dropped_frames, 0);
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
        assert_eq!(extractor.total_consumed_samples(), 1_280);
        assert_eq!(extractor.peak_pending_samples(), 1_600);
    }

    #[test]
    fn streaming_emits_expected_frames_for_1600_samples_chunked_push() {
        let mut extractor = StreamingExtractor::new(AppConfig::default());
        let mut total_frames = 0;

        for _ in 0..10 {
            total_frames += extractor.push_samples(&vec![1.0; 160]).num_frames;
        }

        assert_eq!(total_frames, 8);
        assert_eq!(extractor.total_emitted_frames(), 8);
        assert_eq!(extractor.total_consumed_samples(), 1_280);
        assert_eq!(extractor.pending_samples(), 320);
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
    fn streaming_tracks_total_consumed_samples() {
        let mut extractor = StreamingExtractor::new(AppConfig::default());
        extractor.push_samples(&vec![1.0; 1_600]);

        assert_eq!(extractor.total_consumed_samples(), 1_280);
    }

    #[test]
    fn cached_streaming_empty_push_returns_no_frames() {
        let mut extractor = CachedStreamingExtractor::new(AppConfig::default());
        let output = extractor.push_samples(&[]);

        assert_eq!(output.num_frames, 0);
        assert_eq!(output.num_bins, 0);
        assert_eq!(output.consumed_samples, 0);
        assert_eq!(output.pending_samples, 0);
        assert_eq!(output.dropped_samples, 0);
        assert_eq!(output.dropped_frames, 0);
        assert_eq!(extractor.pending_samples(), 0);
    }

    #[test]
    fn cached_streaming_emits_expected_frames_for_1600_samples_single_push() {
        let mut extractor = CachedStreamingExtractor::new(AppConfig::default());
        let output = extractor.push_samples(&vec![1.0; 1_600]);

        assert_eq!(output.num_frames, 8);
        assert_eq!(output.num_bins, 40);
        assert_eq!(output.pending_samples, 320);
        assert_eq!(extractor.total_emitted_frames(), 8);
        assert_eq!(extractor.total_consumed_samples(), 1_280);
        assert_eq!(extractor.peak_pending_samples(), 1_600);
    }

    #[test]
    fn cached_streaming_values_are_finite() {
        let mut extractor = CachedStreamingExtractor::new(AppConfig::default());
        let output = extractor.push_samples(&vec![1.0; 1_600]);

        assert!(output
            .features
            .iter()
            .flatten()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn cached_streaming_preserves_overlap_across_pushes() {
        let samples: Vec<f32> = (0..1_600).map(|i| i as f32).collect();
        let reference = extract_log_mel_from_samples(&samples, &AppConfig::default());
        let mut extractor = CachedStreamingExtractor::new(AppConfig::default());
        let mut outputs = Vec::new();

        for chunk in samples.chunks(160) {
            outputs.push(extractor.push_samples(chunk));
        }

        let streamed = flatten_features(&outputs);

        assert_eq!(streamed.len(), reference.values.len());
        assert_eq!(extractor.total_emitted_frames(), reference.num_frames);
        assert_eq!(extractor.pending_samples(), 320);
        assert_eq!(extractor.total_consumed_samples(), 1_280);
    }

    #[test]
    fn peak_pending_samples_is_tracked() {
        let mut extractor = StreamingExtractor::new(AppConfig::default());
        extractor.push_samples(&vec![1.0; 100]);
        extractor.push_samples(&vec![1.0; 300]);
        extractor.push_samples(&vec![1.0; 1_200]);

        assert!(extractor.peak_pending_samples() >= 1_200);
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
        assert_eq!(extractor.total_consumed_samples(), 1_280);
    }

    #[test]
    fn streaming_pending_samples_are_logical_after_processing() {
        let mut extractor = StreamingExtractor::new(AppConfig::default());
        extractor.push_samples(&vec![1.0; 1_600]);

        assert_eq!(extractor.pending_samples(), 320);
    }

    #[test]
    fn streaming_compaction_preserves_future_frames() {
        let samples: Vec<f32> = (0..1_600).map(|i| i as f32).collect();
        let reference = extract_log_mel_from_samples(&samples, &AppConfig::default());
        let mut extractor = StreamingExtractor::new(AppConfig::default());

        let mut streamed = Vec::new();
        for chunk in samples.chunks(160) {
            let output = extractor.push_samples(chunk);
            streamed.extend(output.features);
        }

        assert_eq!(streamed, reference.values);
        assert_eq!(extractor.total_emitted_frames(), 8);
    }

    #[test]
    fn streaming_chunked_output_matches_single_push_frame_count() {
        let mut single = StreamingExtractor::new(AppConfig::default());
        let mut chunked = StreamingExtractor::new(AppConfig::default());
        let samples: Vec<f32> = (0..1_600).map(|i| i as f32).collect();

        let single_output = single.push_samples(&samples);
        let mut chunked_frames = 0;
        for chunk in samples.chunks(160) {
            chunked_frames += chunked.push_samples(chunk).num_frames;
        }

        assert_eq!(single_output.num_frames, chunked_frames);
        assert_eq!(
            single.total_emitted_frames(),
            chunked.total_emitted_frames()
        );
    }

    #[test]
    fn streaming_chunked_output_matches_single_push_feature_shape() {
        let samples: Vec<f32> = (0..1_600).map(|i| i as f32).collect();
        let mut single = StreamingExtractor::new(AppConfig::default());
        let mut chunked = StreamingExtractor::new(AppConfig::default());

        let single_output = single.push_samples(&samples);
        let mut chunked_features = Vec::new();
        for chunk in samples.chunks(160) {
            chunked_features.extend(chunked.push_samples(chunk).features);
        }

        assert_eq!(single_output.features.len(), chunked_features.len());
        assert!(chunked_features.iter().all(|row| row.len() == 40));
    }

    #[test]
    fn streaming_handles_many_small_pushes_without_losing_frames() {
        let samples: Vec<f32> = (0..1_600).map(|i| i as f32).collect();
        let mut extractor = StreamingExtractor::new(AppConfig::default());
        let mut total_frames = 0;

        for chunk in samples.chunks(32) {
            total_frames += extractor.push_samples(chunk).num_frames;
        }

        assert_eq!(total_frames, 8);
        assert_eq!(extractor.total_emitted_frames(), 8);
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

    #[test]
    fn unbounded_streaming_does_not_drop_samples() {
        let mut extractor = StreamingExtractor::new(AppConfig::default());
        let output = extractor.push_samples(&vec![1.0; 1_600]);

        assert_eq!(output.dropped_samples, 0);
        assert_eq!(output.dropped_frames, 0);
        assert_eq!(extractor.total_dropped_samples(), 0);
        assert_eq!(extractor.total_dropped_frames(), 0);
    }

    #[test]
    fn bounded_streaming_drops_when_pending_exceeds_limit() {
        let config = StreamingConfig::with_max_pending_samples(AppConfig::default(), 200);
        let mut extractor = StreamingExtractor::with_streaming_config(config);
        let output = extractor.push_samples(&vec![1.0; 1_600]);

        assert!(output.dropped_samples > 0);
        assert!(output.dropped_frames > 0);
    }

    #[test]
    fn bounded_streaming_tracks_total_dropped_samples() {
        let config = StreamingConfig::with_max_pending_samples(AppConfig::default(), 200);
        let mut extractor = StreamingExtractor::with_streaming_config(config);

        extractor.push_samples(&vec![1.0; 1_600]);

        assert!(extractor.total_dropped_samples() > 0);
    }

    #[test]
    fn bounded_streaming_tracks_total_dropped_frames() {
        let config = StreamingConfig::with_max_pending_samples(AppConfig::default(), 200);
        let mut extractor = StreamingExtractor::with_streaming_config(config);

        extractor.push_samples(&vec![1.0; 1_600]);

        assert!(extractor.total_dropped_frames() > 0);
    }

    #[test]
    fn bounded_streaming_keeps_pending_within_limit() {
        let limit = 200;
        let config = StreamingConfig::with_max_pending_samples(AppConfig::default(), limit);
        let mut extractor = StreamingExtractor::with_streaming_config(config);

        extractor.push_samples(&vec![1.0; 1_600]);

        assert!(extractor.pending_samples() <= limit);
    }

    #[test]
    fn bounded_streaming_still_emits_valid_features() {
        let config = StreamingConfig::with_max_pending_samples(AppConfig::default(), 200);
        let mut extractor = StreamingExtractor::with_streaming_config(config);
        let output = extractor.push_samples(&vec![1.0; 1_600]);

        assert!(output.features.iter().all(|row| row.len() == 40));
    }

    #[test]
    fn bounded_streaming_outputs_finite_values() {
        let config = StreamingConfig::with_max_pending_samples(AppConfig::default(), 200);
        let mut extractor = StreamingExtractor::with_streaming_config(config);
        let output = extractor.push_samples(&vec![1.0; 1_600]);

        assert!(output
            .features
            .iter()
            .flatten()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn peak_pending_samples_is_reported_in_output() {
        let config = StreamingConfig::with_max_pending_samples(AppConfig::default(), 200);
        let mut extractor = StreamingExtractor::with_streaming_config(config);
        let output = extractor.push_samples(&vec![1.0; 1_600]);

        assert_eq!(output.peak_pending_samples, 1_600);
        assert_eq!(extractor.peak_pending_samples(), 1_600);
    }
}
