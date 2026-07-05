use signalflow_rs::{
    audio::generate_dummy_audio,
    config::AppConfig,
    extractor::extract_log_mel_timed,
    preprocess::{preprocess_audio, PreprocessConfig},
    streaming::StreamingExtractor,
};
use std::env;

fn main() -> anyhow::Result<()> {
    let config = AppConfig::default();

    if env::var("SIGNALFLOW_CAPTURE").as_deref() == Ok("1") {
        if let Err(error) = run_capture_demo() {
            eprintln!("capture demo failed: {error}");
        }
        return Ok(());
    }

    let audio_chunk = generate_dummy_audio(&config, 100);
    let timed = extract_log_mel_timed(&audio_chunk.samples, &config);

    println!(
        "extractor demo: input_samples={}, frames={}, bins={}, elapsed_ms={:.3}, samples_per_second={:.2}, frames_per_second={:.2}",
        timed.metrics.input_samples,
        timed.features.num_frames,
        timed.features.num_bins,
        timed.metrics.elapsed_ms,
        timed.metrics.samples_per_second,
        timed.metrics.frames_per_second
    );

    let mut streaming = StreamingExtractor::new(config);
    let mut total_streaming_frames = 0;
    for sample_chunk in audio_chunk.samples.chunks(160) {
        total_streaming_frames += streaming.push_samples(sample_chunk).num_frames;
    }

    println!(
        "streaming demo: frames={}, pending={}, consumed={}, dropped_samples={}, dropped_frames={}, peak_pending={}",
        total_streaming_frames,
        streaming.pending_samples(),
        streaming.total_consumed_samples(),
        streaming.total_dropped_samples(),
        streaming.total_dropped_frames(),
        streaming.peak_pending_samples()
    );

    Ok(())
}

fn run_capture_demo() -> anyhow::Result<()> {
    let error =
        anyhow::anyhow!("microphone capture backend is not available in this repository build");
    let _ = preprocess_audio(&[], 0, 0, PreprocessConfig::default());
    Err(error)
}
