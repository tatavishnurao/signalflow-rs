use signalflow_rs::{config::AppConfig, pipeline::run_dummy_pipeline};

fn main() -> anyhow::Result<()> {
    let config = AppConfig::default();
    let report = run_dummy_pipeline(&config)?;

    println!(
        "dummy pipeline: samples={}, frame={} hop={} frames={} first_windowed_frame_rms={} spectrum_bins={} spectrum_peak={:.6} mel_bins={} mel_peak={:.6} log_mel_frames={} log_mel_bins={} first_log_mel={:.6} rms={:.6}",
        report.num_samples,
        report.frame_size_samples,
        report.hop_size_samples,
        report.num_frames,
        report.first_windowed_frame_rms,
        report.first_spectrum_bins,
        report.first_spectrum_peak,
        report.mel_bins,
        report.first_mel_energy_peak,
        report.log_mel_frames,
        report.log_mel_bins,
        report.first_log_mel_value,
        report.rms_energy
    );

    Ok(())
}
