use signalflow_rs::{
    audio::generate_dummy_audio,
    cached::{CachedExtractorConfig, CachedLogMelExtractor},
    config::AppConfig,
    preprocess::{preprocess_audio, PreprocessConfig},
};

fn main() {
    let config = AppConfig::default();
    let audio = generate_dummy_audio(&config, 100);
    let processed = preprocess_audio(
        &audio.samples,
        audio.sample_rate_hz,
        audio.channels,
        PreprocessConfig::default(),
    );
    let mut extractor = CachedLogMelExtractor::new(CachedExtractorConfig::speech_default(config));
    let features = extractor.extract_samples(&processed.samples);

    println!(
        "synthetic: frames={}, bins={}, samples={}",
        features.num_frames,
        features.num_bins,
        processed.samples.len()
    );
}
