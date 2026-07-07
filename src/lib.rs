//! SignalFlow-rs is a Rust audio DSP frontend for log-Mel feature extraction.
//!
//! It provides preprocessing, WAV input, batch extraction, cached extraction,
//! streaming extraction, bounded/drop metrics, and benchmark helpers.
//!
//! This crate is real-time-capable for feature extraction workloads, but it does
//! not provide production real-time thread scheduling guarantees.

pub mod audio;
pub mod bench;
pub mod cached;
pub mod config;
pub mod dsp;
pub mod extractor;
pub mod features;
pub mod framing;
pub mod mel;
pub mod metrics;
pub mod pipeline;
pub mod preprocess;
pub mod spectrum;
pub mod streaming;
pub mod stress;
pub mod wav;
pub mod window;
