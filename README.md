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

## Pipeline

raw samples -> overlapping frames -> Hann windowing -> FFT power spectrum -> Mel filterbank -> log-Mel features

## Demo

```bash
cargo run
```

## Tests

```bash
cargo test
```

The repository is still scaffold-first and will expand toward streaming inference support next.
