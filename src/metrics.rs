#[derive(Debug, Default, Clone, Copy)]
pub struct LatencyMetrics {
    pub processed_samples: usize,
    pub dropped_frames: usize,
}
