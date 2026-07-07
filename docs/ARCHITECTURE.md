# Architecture

## Pipeline

```text
WAV / raw samples
  -> preprocessing
  -> framing
  -> Hann window
  -> FFT power spectrum
  -> Mel filterbank
  -> log compression
  -> feature matrix
  -> streaming buffer / drop metrics
```

## Main APIs

- `extract_log_mel_from_samples` is the free-function batch path.
- `CachedLogMelExtractor` is the reusable batch extractor built on
  `LogMelProcessor`.
- `StreamingExtractor` is the stateful streaming path and already uses cached
  DSP state through `LogMelProcessor`.
- `CachedStreamingExtractor` is a named wrapper API over the same cached DSP
  kernel.
- `run_cached_streaming_stress` provides a std-only streaming stress report for
  synthetic audio.

## Why the shared processor matters

`LogMelProcessor` is the canonical cached DSP kernel. It owns the FFT plan,
window, filterbank, and scratch buffers used by both batch and streaming
feature extraction. That keeps the public APIs small and avoids maintaining two
separate DSP implementations.
