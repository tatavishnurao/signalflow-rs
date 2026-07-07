use criterion::{
    criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use signalflow_rs::{
    audio::generate_dummy_audio,
    cached::{CachedExtractorConfig, CachedLogMelExtractor},
    config::AppConfig,
    extractor::extract_log_mel_from_samples,
    streaming::{CachedStreamingExtractor, StreamingExtractor},
};
use std::hint::black_box;

const DURATIONS_MS: [u32; 3] = [100, 1_000, 60_000];

fn expected_frames(sample_count: usize, frame_size: usize, hop_size: usize) -> usize {
    if sample_count < frame_size || frame_size == 0 || hop_size == 0 {
        return 0;
    }

    1 + (sample_count - frame_size) / hop_size
}

fn benchmark_batch_extraction(criterion: &mut Criterion) {
    let config = AppConfig::default();
    let mut uncached_group = criterion.benchmark_group("batch_log_mel_uncached");

    for duration_ms in DURATIONS_MS {
        let audio = generate_dummy_audio(&config, duration_ms);
        let expected = expected_frames(
            audio.samples.len(),
            config.frame_size_samples(),
            config.hop_size_samples(),
        );
        let features = extract_log_mel_from_samples(&audio.samples, &config);
        assert_eq!(features.num_frames, expected);
        assert_eq!(features.num_bins, 40);

        uncached_group.throughput(Throughput::Elements(audio.samples.len() as u64));
        uncached_group.bench_with_input(
            BenchmarkId::from_parameter(format!("{duration_ms}ms")),
            &audio.samples,
            |bencher, samples| {
                bencher.iter(|| extract_log_mel_from_samples(black_box(samples), &config));
            },
        );
    }

    uncached_group.finish();

    let mut cached_group = criterion.benchmark_group("batch_log_mel_cached");
    for duration_ms in DURATIONS_MS {
        let audio = generate_dummy_audio(&config, duration_ms);
        let expected = expected_frames(
            audio.samples.len(),
            config.frame_size_samples(),
            config.hop_size_samples(),
        );
        let features = extract_log_mel_from_samples(&audio.samples, &config);
        assert_eq!(features.num_frames, expected);
        assert_eq!(features.num_bins, 40);

        cached_group.throughput(Throughput::Elements(audio.samples.len() as u64));
        cached_group.bench_with_input(
            BenchmarkId::from_parameter(format!("{duration_ms}ms")),
            &audio.samples,
            |bencher, samples| {
                bencher.iter_batched(
                    || CachedLogMelExtractor::new(CachedExtractorConfig::speech_default(config)),
                    |mut extractor| {
                        let features = extractor.extract_samples(black_box(samples));
                        black_box(features.num_frames)
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    cached_group.finish();
}

fn benchmark_streaming_extraction(criterion: &mut Criterion) {
    let config = AppConfig::default();
    let hop_size = config.hop_size_samples();
    let frame_size = config.frame_size_samples();
    let mut uncached_group = criterion.benchmark_group("streaming_log_mel_uncached");

    for duration_ms in DURATIONS_MS {
        let audio = generate_dummy_audio(&config, duration_ms);
        let expected = expected_frames(audio.samples.len(), frame_size, hop_size);
        let mut extractor = StreamingExtractor::new(config);
        let mut emitted_frames = 0;
        for chunk in audio.samples.chunks(hop_size) {
            emitted_frames += extractor.push_samples(chunk).num_frames;
        }
        assert_eq!(emitted_frames, expected);
        assert_eq!(extractor.total_emitted_frames(), expected);

        uncached_group.throughput(Throughput::Elements(audio.samples.len() as u64));
        uncached_group.bench_with_input(
            BenchmarkId::from_parameter(format!("{duration_ms}ms")),
            &audio.samples,
            |bencher, samples| {
                bencher.iter(|| {
                    let mut extractor = StreamingExtractor::new(config);
                    let mut emitted_frames = 0;
                    for chunk in samples.chunks(hop_size) {
                        emitted_frames +=
                            black_box(extractor.push_samples(black_box(chunk))).num_frames;
                    }
                    black_box(emitted_frames);
                    black_box(extractor.total_emitted_frames())
                });
            },
        );
    }

    uncached_group.finish();

    let mut cached_group = criterion.benchmark_group("streaming_log_mel_cached");
    for duration_ms in DURATIONS_MS {
        let audio = generate_dummy_audio(&config, duration_ms);
        let expected = expected_frames(audio.samples.len(), frame_size, hop_size);
        let mut extractor = CachedStreamingExtractor::new(config);
        let mut emitted_frames = 0;
        for chunk in audio.samples.chunks(hop_size) {
            emitted_frames += extractor.push_samples(chunk).num_frames;
        }
        assert_eq!(emitted_frames, expected);
        assert_eq!(extractor.total_emitted_frames(), expected);

        cached_group.throughput(Throughput::Elements(audio.samples.len() as u64));
        cached_group.bench_with_input(
            BenchmarkId::from_parameter(format!("{duration_ms}ms")),
            &audio.samples,
            |bencher, samples| {
                bencher.iter(|| {
                    let mut extractor = CachedStreamingExtractor::new(config);
                    let mut emitted_frames = 0;
                    for chunk in samples.chunks(hop_size) {
                        emitted_frames +=
                            black_box(extractor.push_samples(black_box(chunk))).num_frames;
                    }
                    black_box(emitted_frames);
                    black_box(extractor.total_emitted_frames())
                });
            },
        );
    }

    cached_group.finish();
}

criterion_group!(
    extraction_benches,
    benchmark_batch_extraction,
    benchmark_streaming_extraction
);
criterion_main!(extraction_benches);
