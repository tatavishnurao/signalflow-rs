use signalflow_rs::{
    audio::{generate_dummy_audio, AudioChunk},
    bench::benchmark_cached_extractor,
    config::AppConfig,
    extractor::extract_log_mel_timed,
    preprocess::{preprocess_audio, PreprocessConfig, PreprocessedAudio},
    streaming::StreamingExtractor,
    wav::read_wav_mono_or_interleaved_f32,
};
use std::{env, path::Path, time::Instant};

fn main() -> anyhow::Result<()> {
    if env::var("SIGNALFLOW_CAPTURE").as_deref() == Ok("1") {
        if let Err(error) = run_capture_demo() {
            eprintln!("microphone demo unavailable: {error}");
        }
        return Ok(());
    }

    if let Some(path) = env::args_os().nth(1) {
        run_wav_demo(Path::new(&path))
    } else {
        run_synthetic_demo();
        Ok(())
    }
}

fn run_synthetic_demo() {
    let config = AppConfig::default();
    let audio = generate_dummy_audio(&config, 100);
    let processed = preprocess_chunk(&audio);

    println!(
        "synthetic input: samples={}, rate_hz={}, channels={}",
        audio.samples.len(),
        audio.sample_rate_hz,
        audio.channels
    );
    run_extractors(&processed, &config);
}

fn run_wav_demo(path: &Path) -> anyhow::Result<()> {
    let audio = read_wav_mono_or_interleaved_f32(path)?;
    let processed = preprocess_audio(
        &audio.samples,
        audio.sample_rate_hz,
        audio.channels,
        PreprocessConfig::default(),
    );

    println!(
        "WAV input: samples={}, rate_hz={}, channels={}",
        audio.samples.len(),
        audio.sample_rate_hz,
        audio.channels
    );
    println!(
        "preprocessed: samples={}, rate_hz={}, channels={}",
        processed.samples.len(),
        processed.sample_rate_hz,
        processed.channels
    );
    run_extractors(&processed, &AppConfig::default());
    Ok(())
}

fn preprocess_chunk(audio: &AudioChunk) -> PreprocessedAudio {
    preprocess_audio(
        &audio.samples,
        audio.sample_rate_hz,
        audio.channels,
        PreprocessConfig::default(),
    )
}

fn run_extractors(audio: &PreprocessedAudio, config: &AppConfig) {
    let timed = extract_log_mel_timed(&audio.samples, config);
    println!(
        "extractor: input_samples={}, frames={}, bins={}, elapsed_ms={:.3}, samples_per_second={:.2}",
        timed.metrics.input_samples,
        timed.features.num_frames,
        timed.features.num_bins,
        timed.metrics.elapsed_ms,
        timed.metrics.samples_per_second
    );
    let audio_ms = (audio.samples.len() as f64 / config.sample_rate_hz as f64) * 1_000.0;
    let bench_iterations = if audio.samples.len() <= config.sample_rate_hz as usize {
        1_000
    } else {
        100
    };
    let cached = benchmark_cached_extractor(&audio.samples, config, bench_iterations, audio_ms);
    println!(
        "cached: avg_ms={:.3}, rt_factor={:.2}, frames={}, bins={}",
        cached.avg_ms_per_iter, cached.realtime_factor, cached.frames_per_iter, cached.bins
    );

    let streaming_start = Instant::now();
    let mut streaming = StreamingExtractor::new(*config);
    for chunk in audio.samples.chunks(config.hop_size_samples()) {
        streaming.push_samples(chunk);
    }
    let streaming_elapsed_ms = streaming_start.elapsed().as_secs_f64() * 1_000.0;
    let streaming_realtime_factor = if streaming_elapsed_ms > 0.0 {
        audio_ms / streaming_elapsed_ms
    } else {
        0.0
    };
    println!(
        "streaming: frames={}, elapsed_ms={:.3}, rt_factor={:.2}, pending={}, consumed={}, dropped_samples={}, dropped_frames={}, peak_pending={}",
        streaming.total_emitted_frames(),
        streaming_elapsed_ms,
        streaming_realtime_factor,
        streaming.pending_samples(),
        streaming.total_consumed_samples(),
        streaming.total_dropped_samples(),
        streaming.total_dropped_frames(),
        streaming.peak_pending_samples()
    );
}

fn run_capture_demo() -> anyhow::Result<()> {
    anyhow::bail!(
        "this build has no microphone backend; use a WAV path or run the synthetic demo instead"
    )
}
