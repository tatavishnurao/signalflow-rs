# signalflow-rs
Real-Time Audio DSP Accelerator in Rust

## Current capabilities

- dummy audio generation
- framing
- Hann windowing
- FFT power spectrum
- Mel filterbank
- log-Mel features
- single-buffer extraction
- batch extraction
- extraction timing metrics
- streaming / buffered extraction for chunked input
- streaming extractor uses cursor-based buffering with periodic compaction
- bounded streaming mode with backpressure/drop metrics
- preprocessing: mono conversion and simple linear resampling

This streaming layer is stateful chunked extraction for real-time pipelines; it is not microphone capture yet.
The microphone path preprocesses audio to 16 kHz mono before extraction. High-quality resampling is not implemented yet.

## Pipeline

raw samples -> overlapping frames -> Hann windowing -> FFT power spectrum -> Mel filterbank -> log-Mel features
raw samples -> streaming buffer -> overlapping frames -> Hann windowing -> FFT power spectrum -> Mel filterbank -> log-Mel features

## Demo

```bash
cargo run
```

## Tests

```bash
cargo test
```

The repository is still scaffold-first and will expand toward streaming inference support next.
