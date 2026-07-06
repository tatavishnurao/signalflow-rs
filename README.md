# SignalFlow-rs

A Rust real-time audio DSP frontend for log-Mel feature extraction.

## What it does

SignalFlow-rs converts raw or WAV audio into finite, model-ready log-Mel feature
matrices. It supports whole-buffer, batch, timed, and stateful streaming
extraction.

## Current pipeline

```text
raw samples / WAV input
  -> preprocessing to 16 kHz mono
  -> streaming buffer
  -> overlapping frames
  -> Hann windowing
  -> FFT power spectrum
  -> Mel filterbank
  -> log-Mel features
  -> metrics/demo
```

## Current capabilities

- dummy audio generation
- preprocessing to 16 kHz mono
- framing and Hann windowing
- FFT power spectrum
- Mel filterbank and log-Mel features
- single-buffer and batch extraction
- timed extraction metrics
- streaming extraction
- cursor-based streaming buffer compaction
- bounded streaming with backpressure and drop metrics
- integer PCM and 32-bit float WAV input

## Quick start

Requires a stable Rust toolchain.

```bash
cargo run
cargo test
```

## Synthetic demo

The default demo generates 100 ms of audio, preprocesses it, and runs both
timed and streaming extraction:

```bash
cargo run
```

## WAV demo

Pass a WAV path to preserve its source metadata, preprocess it to 16 kHz mono,
and extract features:

```bash
cargo run -- path/to/audio.wav
```

## Optional microphone demo

The environment-variable entry point is reserved, but this v0.1 build does not
include a microphone backend:

```bash
SIGNALFLOW_CAPTURE=1 cargo run
```

It exits cleanly with an explanatory message. Use WAV input for captured audio.

## Testing

```bash
cargo fmt --check
cargo test
```

Tests cover DSP stages, preprocessing and resampling, extraction shapes,
streaming behavior and metrics, WAV decoding, and WAV-to-feature extraction.
They do not require audio hardware or external files.

## Current limitations

- simple linear resampling only
- no model inference yet
- no SIMD or GPU acceleration yet
- microphone capture remains experimental and is not included in this build
- no production real-time scheduling yet

## Roadmap

- better resampling
- inference stub
- ONNX or Candle integration
- Criterion benchmarks
- real-time audio callback integration
- SIMD optimization
