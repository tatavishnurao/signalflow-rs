#[derive(Debug, Default, Clone, Copy)]
pub struct LatencyMetrics {
    pub processed_samples: usize,
    pub dropped_frames: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExtractionMetrics {
    pub elapsed_ms: f64,
    pub input_samples: usize,
    pub output_frames: usize,
    pub output_bins: usize,
    pub samples_per_second: f64,
    pub frames_per_second: f64,
}

impl ExtractionMetrics {
    pub fn new(
        elapsed_ms: f64,
        input_samples: usize,
        output_frames: usize,
        output_bins: usize,
    ) -> Self {
        let (samples_per_second, frames_per_second) = if elapsed_ms > 0.0 {
            let elapsed_seconds = elapsed_ms / 1_000.0;
            (
                input_samples as f64 / elapsed_seconds,
                output_frames as f64 / elapsed_seconds,
            )
        } else {
            (0.0, 0.0)
        };

        Self {
            elapsed_ms,
            input_samples,
            output_frames,
            output_bins,
            samples_per_second,
            frames_per_second,
        }
    }
}
