use signalflow_rs::{
    cached::{CachedExtractorConfig, CachedLogMelExtractor},
    preprocess::{preprocess_audio, PreprocessConfig},
    wav::read_wav_mono_or_interleaved_f32,
};
use std::{env, path::Path};

fn main() -> anyhow::Result<()> {
    let Some(path) = env::args_os().nth(1) else {
        println!("usage: cargo run --example wav_file -- path/to/audio.wav");
        return Ok(());
    };

    let path = Path::new(&path);
    let audio = read_wav_mono_or_interleaved_f32(path)?;
    let processed = preprocess_audio(
        &audio.samples,
        audio.sample_rate_hz,
        audio.channels,
        PreprocessConfig::default(),
    );
    let mut extractor = CachedLogMelExtractor::new(CachedExtractorConfig::speech_default(
        signalflow_rs::config::AppConfig::default(),
    ));
    let features = extractor.extract_samples(&processed.samples);

    println!(
        "wav: input_rate_hz={}, input_channels={}, input_samples={}, processed_rate_hz={}, processed_channels={}, processed_samples={}, frames={}, bins={}",
        audio.sample_rate_hz,
        audio.channels,
        audio.samples.len(),
        processed.sample_rate_hz,
        processed.channels,
        processed.samples.len(),
        features.num_frames,
        features.num_bins
    );

    Ok(())
}
