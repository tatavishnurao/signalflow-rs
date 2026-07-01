use rustfft::{num_complex::Complex, FftPlanner};

pub fn magnitude_spectrum(frame: &[f32]) -> Vec<f32> {
    if frame.is_empty() {
        return Vec::new();
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(frame.len());
    let mut buffer: Vec<Complex<f32>> = frame
        .iter()
        .copied()
        .map(|sample| Complex::new(sample, 0.0))
        .collect();

    fft.process(&mut buffer);

    buffer
        .iter()
        .take(frame.len() / 2 + 1)
        .map(|bin| (bin.re * bin.re + bin.im * bin.im).sqrt())
        .collect()
}

pub fn power_spectrum(frame: &[f32]) -> Vec<f32> {
    magnitude_spectrum(frame)
        .into_iter()
        .map(|magnitude| magnitude * magnitude)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{magnitude_spectrum, power_spectrum};

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn magnitude_spectrum_empty_input() {
        assert!(magnitude_spectrum(&[]).is_empty());
    }

    #[test]
    fn magnitude_spectrum_has_expected_length_even() {
        assert_eq!(magnitude_spectrum(&vec![0.0; 400]).len(), 201);
    }

    #[test]
    fn magnitude_spectrum_has_expected_length_odd() {
        assert_eq!(magnitude_spectrum(&vec![0.0; 401]).len(), 201);
    }

    #[test]
    fn magnitude_spectrum_detects_dc_signal() {
        let spectrum = magnitude_spectrum(&vec![1.0; 16]);

        assert!(spectrum[0] > 0.0);
        assert!(spectrum[0] > spectrum[1]);
    }

    #[test]
    fn power_spectrum_empty_input() {
        assert!(power_spectrum(&[]).is_empty());
    }

    #[test]
    fn power_spectrum_matches_squared_magnitude() {
        let frame = [0.0, 1.0, 0.0, -1.0];
        let magnitudes = magnitude_spectrum(&frame);
        let powers = power_spectrum(&frame);

        assert_eq!(magnitudes.len(), powers.len());
        for (magnitude, power) in magnitudes.iter().zip(powers.iter()) {
            assert!(approx_eq(*power, magnitude * magnitude, 1e-6));
        }
    }
}
