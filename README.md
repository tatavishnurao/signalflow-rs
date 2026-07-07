# SignalFlow-rs

A Rust real-time audio DSP frontend for log-Mel feature extraction.

## What it does

SignalFlow-rs converts raw or WAV audio into finite, model-ready log-Mel feature
matrices. It supports whole-buffer, batch, timed, and stateful streaming
extraction, plus a cached/prepared extractor for repeated log-Mel work.

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
- cached/prepared log-Mel extraction
- cached streaming extraction
- precomputed Hann window reuse
- cached FFT planning reuse
- cached Mel filterbank reuse
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
cargo run --release
```

## Synthetic demo

The default demo generates 100 ms of audio, preprocesses it, and runs both
timed, cached, and streaming extraction:

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

## Benchmarking

Criterion compares API paths for free-function batch extraction, reusable
processor batch extraction, processor-backed streaming extraction, and cached
wrapper streaming extraction for 100 ms, 1 second, and 60 seconds of 16 kHz
mono audio:

```bash
cargo bench --bench extraction
cargo run --release
```

`StreamingExtractor` already uses the shared cached `LogMelProcessor` kernel.
`CachedLogMelExtractor` is a reusable whole-buffer API built on that processor.
`CachedStreamingExtractor` is a named wrapper API, not a promise of better
speed than `StreamingExtractor`. The exact numbers depend on machine and build
mode, and this is still not production real-time scheduling.

## Current limitations

- simple linear resampling only
- no model inference yet
- no SIMD or GPU acceleration yet
- real-time-capable DSP frontend, but not yet a production real-time audio accelerator
- no real-time scheduling guarantees yet
- microphone capture remains experimental and is not included in this build

## Roadmap

- better resampling
- inference stub
- ONNX or Candle integration
- real-time audio callback integration
- SIMD optimization
