use signalflow_rs::{config::AppConfig, stress::run_cached_streaming_stress};

fn main() {
    let config = AppConfig::default();
    for duration_ms in [100, 1_000, 60_000] {
        let report =
            run_cached_streaming_stress(config, duration_ms, config.hop_size_samples(), None);
        println!(
            "stress: duration_ms={}, frames={}, bins={}, avg_chunk_ms={:.4}, p95_ms={:.4}, p99_ms={:.4}, rt_factor={:.2}, drops={}/{}",
            report.duration_ms,
            report.emitted_frames,
            report.bins,
            report.avg_chunk_ms,
            report.p95_chunk_ms,
            report.p99_chunk_ms,
            report.realtime_factor,
            report.dropped_samples,
            report.dropped_frames
        );
    }
}
