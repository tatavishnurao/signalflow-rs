use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use signalflow_rs::{
    audio::generate_dummy_audio, config::AppConfig, extractor::extract_log_mel_from_samples,
    streaming::StreamingExtractor,
};
use std::hint::black_box;

const DURATIONS_MS: [u32; 3] = [100, 1_000, 60_000];

fn benchmark_batch_extraction(criterion: &mut Criterion) {
    let config = AppConfig::default();
    let mut group = criterion.benchmark_group("batch_log_mel");

    for duration_ms in DURATIONS_MS {
        let audio = generate_dummy_audio(&config, duration_ms);
        group.throughput(Throughput::Elements(audio.samples.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{duration_ms}ms")),
            &audio.samples,
            |bencher, samples| {
                bencher.iter(|| extract_log_mel_from_samples(black_box(samples), &config));
            },
        );
    }

    group.finish();
}

fn benchmark_streaming_extraction(criterion: &mut Criterion) {
    let config = AppConfig::default();
    let hop_size = config.hop_size_samples();
    let mut group = criterion.benchmark_group("streaming_log_mel");

    for duration_ms in DURATIONS_MS {
        let audio = generate_dummy_audio(&config, duration_ms);
        group.throughput(Throughput::Elements(audio.samples.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{duration_ms}ms")),
            &audio.samples,
            |bencher, samples| {
                bencher.iter(|| {
                    let mut extractor = StreamingExtractor::new(config);
                    for chunk in samples.chunks(hop_size) {
                        black_box(extractor.push_samples(black_box(chunk)));
                    }
                    black_box(extractor.total_emitted_frames())
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    extraction_benches,
    benchmark_batch_extraction,
    benchmark_streaming_extraction
);
criterion_main!(extraction_benches);
