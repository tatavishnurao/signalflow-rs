# Performance

## How to run

```bash
cargo run --release
cargo bench --bench extraction
cargo run --example streaming_stress --release
```

## Notes

- Debug builds are useful for correctness, not for performance claims.
- Release mode is the baseline for timing and throughput comparisons.
- Criterion results vary by machine, CPU governor, and build environment.
- SignalFlow-rs is real-time-capable for feature extraction, but it does not
  provide real-time scheduling guarantees.
