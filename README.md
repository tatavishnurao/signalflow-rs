# SignalFlow-rs

Real-time-capable Rust DSP/log-Mel frontend for audio feature extraction.

## Current pipeline

```text
WAV / raw samples
  -> preprocessing to 16 kHz mono
  -> framing
  -> Hann window
  -> FFT power spectrum
  -> Mel filterbank
  -> log compression
  -> feature matrix
  -> streaming buffer / drop metrics
```

## Capabilities

- dummy audio generation
- preprocessing to 16 kHz mono
- WAV input
- framing
- Hann windowing
- FFT power spectrum
- Mel filterbank and log-Mel features
- reusable batch extractor API
- streaming extraction
- bounded streaming with backpressure and drop metrics
- timed extraction metrics
- cached/prepared batch extraction
- cached streaming wrapper API
- std-only benchmark helpers
- Criterion benchmarks

## Quick Start

```bash
cargo run
cargo test
cargo run --release
```

## Synthetic Demo

The default demo generates 100 ms of synthetic audio, preprocesses it, and
prints timed, cached batch, cached streaming wrapper, and streaming results.

```bash
cargo run
```

## WAV Demo

Pass a WAV path to read audio, preprocess it to 16 kHz mono, and extract
features.

```bash
cargo run -- path/to/audio.wav
```

Microphone capture is not included in v0.1. The environment variable entry
point remains a stub and exits with an explanatory message.

## Streaming Stress Example

Run the synthetic streaming stress report from the example target.

```bash
cargo run --example streaming_stress --release
```

## Benchmarking

Criterion compares API paths for free-function batch extraction, reusable
processor batch extraction, processor-backed streaming extraction, and cached
wrapper streaming extraction.

```bash
cargo bench --bench extraction
cargo run --release
```

`StreamingExtractor` already uses the shared cached `LogMelProcessor` kernel.
`CachedLogMelExtractor` is a reusable whole-buffer API built on that processor.
`CachedStreamingExtractor` is a named wrapper API, not a guarantee of better
speed than `StreamingExtractor`.

Performance numbers are machine-dependent and should be read in release mode.
This is real-time-capable feature extraction, not real-time scheduling.

## Examples

- `cargo run --example synthetic --release`
- `cargo run --example wav_file -- path/to/audio.wav`
- `cargo run --example streaming_stress --release`

## Architecture Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/PERFORMANCE.md](docs/PERFORMANCE.md)
- [docs/RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md)

## Performance Notes

- Debug builds are for correctness, not performance claims.
- Release mode is required for meaningful timing.
- Criterion results vary by machine and compiler settings.
- Streaming and batch extraction share the same cached DSP kernel through
  `LogMelProcessor`.

## Current Limitations

- no microphone backend in v0.1
- no model inference yet
- no real-time scheduling guarantees
- simple linear resampling only
- no SIMD or GPU acceleration

## Roadmap

- better resampling
- inference stub
- ONNX or Candle integration
- real-time audio callback integration
- SIMD optimization

## v0.1 Status

SignalFlow-rs is usable as a small, reproducible DSP frontend for log-Mel
feature extraction with demos, benchmarks, examples, and documentation in place.
It is not a production real-time audio accelerator.
