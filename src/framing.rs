#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameConfig {
    pub frame_size_samples: usize,
    pub hop_size_samples: usize,
}

impl FrameConfig {
    pub fn new(frame_size_samples: usize, hop_size_samples: usize) -> Self {
        Self {
            frame_size_samples,
            hop_size_samples,
        }
    }
}

pub fn frame_signal(samples: &[f32], config: FrameConfig) -> Vec<Vec<f32>> {
    if config.frame_size_samples == 0 || config.hop_size_samples == 0 {
        return Vec::new();
    }

    if samples.len() < config.frame_size_samples {
        return Vec::new();
    }

    let mut frames = Vec::new();
    let mut start = 0;

    while start + config.frame_size_samples <= samples.len() {
        frames.push(samples[start..start + config.frame_size_samples].to_vec());
        start += config.hop_size_samples;
    }

    frames
}

#[cfg(test)]
mod tests {
    use super::{frame_signal, FrameConfig};

    #[test]
    fn returns_empty_for_short_input() {
        let samples = [1.0_f32; 399];

        assert!(frame_signal(&samples, FrameConfig::new(400, 160)).is_empty());
    }

    #[test]
    fn returns_empty_for_zero_frame_size() {
        let samples = [1.0_f32; 10];

        assert!(frame_signal(&samples, FrameConfig::new(0, 160)).is_empty());
    }

    #[test]
    fn returns_empty_for_zero_hop_size() {
        let samples = [1.0_f32; 10];

        assert!(frame_signal(&samples, FrameConfig::new(400, 0)).is_empty());
    }

    #[test]
    fn frames_exact_default_audio_shape() {
        let samples: Vec<f32> = (0..1_600).map(|i| i as f32).collect();
        let frames = frame_signal(&samples, FrameConfig::new(400, 160));

        assert_eq!(frames.len(), 8);
        assert!(frames.iter().all(|frame| frame.len() == 400));
    }

    #[test]
    fn preserves_frame_contents() {
        let samples: Vec<f32> = (0..1_600).map(|i| i as f32).collect();
        let frames = frame_signal(&samples, FrameConfig::new(400, 160));

        assert_eq!(frames[0][0], 0.0);
        assert_eq!(frames[1][0], 160.0);
        assert_eq!(frames[7][0], 1_120.0);
    }
}
