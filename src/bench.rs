use crate::{
    cached::{CachedExtractorConfig, CachedLogMelExtractor},
    config::AppConfig,
    streaming::CachedStreamingExtractor,
};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BenchReport {
    pub iterations: usize,
    pub audio_ms_per_iter: f64,
    pub elapsed_ms: f64,
    pub avg_ms_per_iter: f64,
    pub realtime_factor: f64,
    pub frames_per_iter: usize,
    pub bins: usize,
}

pub fn benchmark_cached_extractor(
    samples: &[f32],
    config: &AppConfig,
    iterations: usize,
    audio_ms_per_iter: f64,
) -> BenchReport {
    let mut extractor = CachedLogMelExtractor::new(CachedExtractorConfig::speech_default(*config));
    let start = Instant::now();
    let mut frames_per_iter = 0;
    let mut bins = 0;

    for _ in 0..iterations {
        let features = extractor.extract_samples(samples);
        frames_per_iter = features.num_frames;
        bins = features.num_bins;
    }

    let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
    let avg_ms_per_iter = if iterations == 0 {
        0.0
    } else {
        elapsed_ms / iterations as f64
    };
    let realtime_factor = if avg_ms_per_iter > 0.0 {
        audio_ms_per_iter / avg_ms_per_iter
    } else {
        0.0
    };

    BenchReport {
        iterations,
        audio_ms_per_iter,
        elapsed_ms,
        avg_ms_per_iter,
        realtime_factor,
        frames_per_iter,
        bins,
    }
}

pub fn benchmark_cached_streaming_extractor(
    samples: &[f32],
    config: &AppConfig,
    iterations: usize,
    audio_ms_per_iter: f64,
) -> BenchReport {
    let hop_size = config.hop_size_samples();
    let start = Instant::now();
    let mut frames_per_iter = 0;
    let mut bins = 0;

    for _ in 0..iterations {
        let mut extractor = CachedStreamingExtractor::new(*config);
        let mut last_output_bins = 0;
        for chunk in samples.chunks(hop_size) {
            let output = extractor.push_samples(chunk);
            last_output_bins = output.num_bins;
        }
        frames_per_iter = extractor.total_emitted_frames();
        bins = last_output_bins;
    }

    let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
    let avg_ms_per_iter = if iterations == 0 {
        0.0
    } else {
        elapsed_ms / iterations as f64
    };
    let realtime_factor = if avg_ms_per_iter > 0.0 {
        audio_ms_per_iter / avg_ms_per_iter
    } else {
        0.0
    };

    BenchReport {
        iterations,
        audio_ms_per_iter,
        elapsed_ms,
        avg_ms_per_iter,
        realtime_factor,
        frames_per_iter,
        bins,
    }
}

#[cfg(test)]
mod tests {
    use super::{benchmark_cached_extractor, benchmark_cached_streaming_extractor, BenchReport};
    use crate::config::AppConfig;

    #[test]
    fn bench_report_has_expected_shape() {
        let report = BenchReport {
            iterations: 10,
            audio_ms_per_iter: 100.0,
            elapsed_ms: 5.0,
            avg_ms_per_iter: 0.5,
            realtime_factor: 200.0,
            frames_per_iter: 8,
            bins: 40,
        };

        assert_eq!(report.iterations, 10);
        assert_eq!(report.frames_per_iter, 8);
        assert_eq!(report.bins, 40);
    }

    #[test]
    fn benchmark_cached_extractor_runs() {
        let samples: Vec<f32> = (0..1_600).map(|i| i as f32).collect();
        let report = benchmark_cached_extractor(&samples, &AppConfig::default(), 4, 100.0);

        assert_eq!(report.frames_per_iter, 8);
        assert_eq!(report.bins, 40);
    }

    #[test]
    fn benchmark_cached_extractor_reports_positive_iterations() {
        let samples: Vec<f32> = (0..1_600).map(|i| i as f32).collect();
        let report = benchmark_cached_extractor(&samples, &AppConfig::default(), 4, 100.0);

        assert_eq!(report.iterations, 4);
        assert!(report.elapsed_ms >= 0.0);
    }

    #[test]
    fn benchmark_cached_extractor_avoids_divide_by_zero() {
        let samples: Vec<f32> = (0..1_600).map(|i| i as f32).collect();
        let report = benchmark_cached_extractor(&samples, &AppConfig::default(), 0, 100.0);

        assert_eq!(report.iterations, 0);
        assert_eq!(report.avg_ms_per_iter, 0.0);
        assert_eq!(report.realtime_factor, 0.0);
    }

    #[test]
    fn benchmark_cached_streaming_extractor_runs() {
        let samples: Vec<f32> = (0..1_600).map(|i| i as f32).collect();
        let report =
            benchmark_cached_streaming_extractor(&samples, &AppConfig::default(), 4, 100.0);

        assert_eq!(report.frames_per_iter, 8);
        assert_eq!(report.bins, 40);
    }

    #[test]
    fn benchmark_cached_streaming_extractor_reports_expected_shape() {
        let samples: Vec<f32> = (0..1_600).map(|i| i as f32).collect();
        let report =
            benchmark_cached_streaming_extractor(&samples, &AppConfig::default(), 4, 100.0);

        assert_eq!(report.iterations, 4);
        assert!(report.elapsed_ms >= 0.0);
        assert_eq!(report.audio_ms_per_iter, 100.0);
    }
}
