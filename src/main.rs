use signalflow_rs::{config::AppConfig, pipeline::run_dummy_pipeline};

fn main() -> anyhow::Result<()> {
    let config = AppConfig::default();
    let report = run_dummy_pipeline(&config)?;

    println!(
        "dummy pipeline: samples={}, frame={} hop={} frames={} first_windowed_frame_rms={} spectrum_bins={} spectrum_peak={:.6} rms={:.6}",
        report.num_samples,
        report.frame_size_samples,
        report.hop_size_samples,
        report.num_frames,
        report.first_windowed_frame_rms,
        report.first_spectrum_bins,
        report.first_spectrum_peak,
        report.rms_energy
    );

    Ok(())
}
