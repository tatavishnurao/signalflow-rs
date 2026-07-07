use crate::{
    audio::generate_dummy_audio,
    config::AppConfig,
    streaming::{CachedStreamingExtractor, StreamingConfig},
};
use std::time::Instant;

#[derive(Debug, Clone, PartialEq)]
pub struct StreamingStressReport {
    pub duration_ms: u32,
    pub chunk_size_samples: usize,
    pub input_samples: usize,
    pub emitted_frames: usize,
    pub bins: usize,
    pub pending_samples: usize,
    pub consumed_samples: usize,
    pub dropped_samples: usize,
    pub dropped_frames: usize,
    pub peak_pending_samples: usize,
    pub total_elapsed_ms: f64,
    pub avg_chunk_ms: f64,
    pub p50_chunk_ms: f64,
    pub p95_chunk_ms: f64,
    pub p99_chunk_ms: f64,
    pub realtime_factor: f64,
}

impl StreamingStressReport {
    fn empty(duration_ms: u32, chunk_size_samples: usize) -> Self {
        Self {
            duration_ms,
            chunk_size_samples,
            input_samples: 0,
            emitted_frames: 0,
            bins: 0,
            pending_samples: 0,
            consumed_samples: 0,
            dropped_samples: 0,
            dropped_frames: 0,
            peak_pending_samples: 0,
            total_elapsed_ms: 0.0,
            avg_chunk_ms: 0.0,
            p50_chunk_ms: 0.0,
            p95_chunk_ms: 0.0,
            p99_chunk_ms: 0.0,
            realtime_factor: 0.0,
        }
    }
}

fn percentile_nearest_rank(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let clamped = percentile.clamp(0.0, 1.0);
    let rank = (clamped * values.len() as f64).ceil().max(1.0) as usize - 1;
    values[rank.min(values.len() - 1)]
}

pub fn run_cached_streaming_stress(
    config: AppConfig,
    duration_ms: u32,
    chunk_size_samples: usize,
    max_pending_samples: Option<usize>,
) -> StreamingStressReport {
    if chunk_size_samples == 0 {
        return StreamingStressReport::empty(duration_ms, chunk_size_samples);
    }

    let audio = generate_dummy_audio(&config, duration_ms);
    let mut extractor = match max_pending_samples {
        Some(limit) => CachedStreamingExtractor::with_streaming_config(
            StreamingConfig::with_max_pending_samples(config, limit),
        ),
        None => CachedStreamingExtractor::new(config),
    };

    let wall_start = Instant::now();
    let mut chunk_durations_ms = Vec::new();
    let mut bins = 0;

    for chunk in audio.samples.chunks(chunk_size_samples) {
        let chunk_start = Instant::now();
        let output = extractor.push_samples(chunk);
        let elapsed_ms = chunk_start.elapsed().as_secs_f64() * 1_000.0;
        chunk_durations_ms.push(elapsed_ms);
        if output.num_bins > 0 {
            bins = output.num_bins;
        }
    }

    let total_elapsed_ms = wall_start.elapsed().as_secs_f64() * 1_000.0;
    let chunk_sum_ms: f64 = chunk_durations_ms.iter().sum();
    let avg_chunk_ms = if chunk_durations_ms.is_empty() {
        0.0
    } else {
        chunk_sum_ms / chunk_durations_ms.len() as f64
    };
    let mut sorted_chunk_durations = chunk_durations_ms;
    sorted_chunk_durations.sort_by(|left, right| left.total_cmp(right));

    let audio_ms = duration_ms as f64;
    let realtime_factor = if total_elapsed_ms > 0.0 {
        audio_ms / total_elapsed_ms
    } else {
        0.0
    };

    StreamingStressReport {
        duration_ms,
        chunk_size_samples,
        input_samples: audio.samples.len(),
        emitted_frames: extractor.total_emitted_frames(),
        bins,
        pending_samples: extractor.pending_samples(),
        consumed_samples: extractor.total_consumed_samples(),
        dropped_samples: extractor.total_dropped_samples(),
        dropped_frames: extractor.total_dropped_frames(),
        peak_pending_samples: extractor.peak_pending_samples(),
        total_elapsed_ms,
        avg_chunk_ms,
        p50_chunk_ms: percentile_nearest_rank(&sorted_chunk_durations, 0.50),
        p95_chunk_ms: percentile_nearest_rank(&sorted_chunk_durations, 0.95),
        p99_chunk_ms: percentile_nearest_rank(&sorted_chunk_durations, 0.99),
        realtime_factor,
    }
}

#[cfg(test)]
mod tests {
    use super::run_cached_streaming_stress;
    use crate::config::AppConfig;

    #[test]
    fn stress_zero_chunk_size_returns_empty_report() {
        let report = run_cached_streaming_stress(AppConfig::default(), 100, 0, None);

        assert_eq!(report.input_samples, 0);
        assert_eq!(report.emitted_frames, 0);
        assert_eq!(report.total_elapsed_ms, 0.0);
    }

    #[test]
    fn stress_100ms_emits_expected_frames() {
        let report = run_cached_streaming_stress(AppConfig::default(), 100, 160, None);

        assert_eq!(report.emitted_frames, 8);
        assert_eq!(report.bins, 40);
    }

    #[test]
    fn stress_1s_emits_nonzero_frames() {
        let report = run_cached_streaming_stress(AppConfig::default(), 1_000, 160, None);

        assert!(report.emitted_frames > 0);
        assert_eq!(report.bins, 40);
    }

    #[test]
    fn stress_report_values_are_finite() {
        let report = run_cached_streaming_stress(AppConfig::default(), 100, 160, None);

        assert!(report.total_elapsed_ms.is_finite());
        assert!(report.avg_chunk_ms.is_finite());
        assert!(report.p50_chunk_ms.is_finite());
        assert!(report.p95_chunk_ms.is_finite());
        assert!(report.p99_chunk_ms.is_finite());
        assert!(report.realtime_factor.is_finite());
    }

    #[test]
    fn stress_unbounded_has_zero_drops() {
        let report = run_cached_streaming_stress(AppConfig::default(), 100, 160, None);

        assert_eq!(report.dropped_samples, 0);
        assert_eq!(report.dropped_frames, 0);
    }

    #[test]
    fn stress_bounded_reports_drop_fields() {
        let report = run_cached_streaming_stress(AppConfig::default(), 1_000, 160, Some(200));

        assert!(report.dropped_samples > 0 || report.dropped_frames > 0);
    }

    #[test]
    fn stress_percentiles_are_ordered() {
        let report = run_cached_streaming_stress(AppConfig::default(), 1_000, 160, None);

        assert!(report.p50_chunk_ms <= report.p95_chunk_ms);
        assert!(report.p95_chunk_ms <= report.p99_chunk_ms);
    }
}
