use anyhow::{bail, Context};
use hound::{SampleFormat, WavReader};
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct WavAudio {
    pub samples: Vec<f32>,
    pub sample_rate_hz: u32,
    pub channels: u16,
}

pub fn read_wav_mono_or_interleaved_f32<P: AsRef<Path>>(path: P) -> anyhow::Result<WavAudio> {
    let path = path.as_ref();
    let mut reader =
        WavReader::open(path).with_context(|| format!("failed to open WAV {}", path.display()))?;
    let spec = reader.spec();

    if spec.channels == 0 || spec.sample_rate == 0 {
        bail!("WAV has invalid sample rate or channel count");
    }

    let samples = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 1..=16) => {
            let scale = (1_u32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i16>()
                .map(|sample| sample.map(|value| value as f32 / scale))
                .collect::<Result<Vec<_>, _>>()
        }
        (SampleFormat::Int, 17..=32) => {
            let scale = (1_u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 / scale))
                .collect::<Result<Vec<_>, _>>()
        }
        (SampleFormat::Float, 32) => reader.samples::<f32>().collect::<Result<Vec<_>, _>>(),
        (format, bits) => bail!("unsupported WAV format: {format:?}, {bits} bits per sample"),
    }
    .with_context(|| format!("failed to decode WAV samples from {}", path.display()))?;

    Ok(WavAudio {
        samples,
        sample_rate_hz: spec.sample_rate,
        channels: spec.channels,
    })
}

#[cfg(test)]
mod tests {
    use super::read_wav_mono_or_interleaved_f32;
    use crate::{
        config::AppConfig, extractor::extract_log_mel_from_samples, preprocess::preprocess_audio,
        preprocess::PreprocessConfig,
    };
    use hound::{SampleFormat, WavSpec, WavWriter};
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn temp_wav_path(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "signalflow-rs-{label}-{}-{nonce}.wav",
            std::process::id()
        ))
    }

    fn write_i16_wav(path: &PathBuf, sample_rate: u32, channels: u16, samples: &[i16]) {
        let spec = WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(path, spec).expect("test WAV should be created");
        for sample in samples {
            writer
                .write_sample(*sample)
                .expect("test WAV sample should be written");
        }
        writer.finalize().expect("test WAV should be finalized");
    }

    #[test]
    fn reads_tiny_16_bit_mono_wav() {
        let path = temp_wav_path("mono");
        write_i16_wav(&path, 16_000, 1, &[0, i16::MAX, i16::MIN]);

        let audio = read_wav_mono_or_interleaved_f32(&path).expect("test WAV should be readable");
        fs::remove_file(path).expect("test WAV should be removed");

        assert_eq!(audio.sample_rate_hz, 16_000);
        assert_eq!(audio.channels, 1);
        assert_eq!(audio.samples.len(), 3);
        assert!(audio.samples.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn reads_tiny_stereo_wav_as_interleaved_samples() {
        let path = temp_wav_path("stereo");
        write_i16_wav(&path, 48_000, 2, &[1_000, -1_000, 2_000, -2_000]);

        let audio = read_wav_mono_or_interleaved_f32(&path).expect("test WAV should be readable");
        fs::remove_file(path).expect("test WAV should be removed");

        assert_eq!(audio.sample_rate_hz, 48_000);
        assert_eq!(audio.channels, 2);
        assert_eq!(audio.samples.len(), 4);
    }

    #[test]
    fn wav_read_preprocess_and_extract_is_finite() {
        let path = temp_wav_path("end-to-end");
        let samples: Vec<i16> = (0..4_800)
            .map(|index| ((index as f32 * 0.1).sin() * 10_000.0) as i16)
            .collect();
        write_i16_wav(&path, 48_000, 1, &samples);

        let audio = read_wav_mono_or_interleaved_f32(&path).expect("test WAV should be readable");
        fs::remove_file(path).expect("test WAV should be removed");
        let processed = preprocess_audio(
            &audio.samples,
            audio.sample_rate_hz,
            audio.channels,
            PreprocessConfig::default(),
        );
        let features = extract_log_mel_from_samples(&processed.samples, &AppConfig::default());

        assert!(features.num_frames > 0);
        assert!(features
            .values
            .iter()
            .flatten()
            .all(|value| value.is_finite()));
    }
}
