use signalflow_rs::{config::AppConfig, pipeline::run_dummy_pipeline};

fn main() -> anyhow::Result<()> {
    let config = AppConfig::default();
    let report = run_dummy_pipeline(&config)?;

    println!(
        "dummy pipeline: samples={}, frame={} hop={} frames={} rms={:.6}",
        report.num_samples,
        report.frame_size_samples,
        report.hop_size_samples,
        report.num_frames,
        report.rms_energy
    );

    Ok(())
}
