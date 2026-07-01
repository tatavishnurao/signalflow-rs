#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFunction {
    Hann,
}

pub fn hann_window(size: usize) -> Vec<f32> {
    match size {
        0 => Vec::new(),
        1 => vec![1.0],
        _ => {
            let n_minus_1 = (size - 1) as f32;
            (0..size)
                .map(|n| 0.5 * (1.0 - ((2.0 * std::f32::consts::PI * n as f32) / n_minus_1).cos()))
                .collect()
        }
    }
}

pub fn apply_window(frame: &[f32], window: &[f32]) -> Vec<f32> {
    frame
        .iter()
        .zip(window.iter())
        .map(|(sample, weight)| sample * weight)
        .collect()
}

pub fn window_frame(frame: &[f32], function: WindowFunction) -> Vec<f32> {
    match function {
        WindowFunction::Hann => {
            let window = hann_window(frame.len());
            apply_window(frame, &window)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{apply_window, hann_window, window_frame, WindowFunction};

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn hann_window_empty() {
        assert!(hann_window(0).is_empty());
    }

    #[test]
    fn hann_window_single_sample() {
        assert_eq!(hann_window(1), vec![1.0]);
    }

    #[test]
    fn hann_window_has_expected_length() {
        assert_eq!(hann_window(8).len(), 8);
    }

    #[test]
    fn hann_window_starts_and_ends_near_zero() {
        let window = hann_window(8);

        assert!(approx_eq(window[0], 0.0, 1e-6));
        assert!(approx_eq(window[7], 0.0, 1e-6));
    }

    #[test]
    fn hann_window_center_is_near_one() {
        let window = hann_window(5);

        assert!(approx_eq(window[2], 1.0, 1e-6));
    }

    #[test]
    fn apply_window_multiplies_samples() {
        let frame = [1.0, 2.0, 3.0];
        let window = [0.5, 0.25, 0.0];

        assert_eq!(apply_window(&frame, &window), vec![0.5, 0.5, 0.0]);
    }

    #[test]
    fn apply_window_handles_mismatched_lengths() {
        let frame = [1.0, 2.0, 3.0];
        let window = [0.5, 0.25];

        assert_eq!(apply_window(&frame, &window), vec![0.5, 0.5]);
    }

    #[test]
    fn window_frame_hann_preserves_length() {
        let frame = [1.0, 2.0, 3.0, 4.0];

        assert_eq!(
            window_frame(&frame, WindowFunction::Hann).len(),
            frame.len()
        );
    }
}
