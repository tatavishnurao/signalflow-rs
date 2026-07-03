use signalflow_rs::{
    audio::generate_dummy_audio, config::AppConfig, extractor::extract_log_mel_timed,
};

fn main() -> anyhow::Result<()> {
    let config = AppConfig::default();
    let chunk = generate_dummy_audio(&config, 100);
    let timed = extract_log_mel_timed(&chunk.samples, &config);

    println!(
        "extractor demo: input_samples={}, frames={}, bins={}, elapsed_ms={:.3}, samples_per_second={:.2}, frames_per_second={:.2}",
        timed.metrics.input_samples,
        timed.features.num_frames,
        timed.features.num_bins,
        timed.metrics.elapsed_ms,
        timed.metrics.samples_per_second,
        timed.metrics.frames_per_second
    );

    Ok(())
}
