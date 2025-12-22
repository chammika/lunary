use anyhow::Result;
use std::time::Instant;

use crate::{AdaptiveBatchConfig, AdaptiveBatchMetrics, AdaptiveBatchProcessor};

use super::utils::calculate_throughput;

pub fn bench_adaptive(data: &[u8]) -> Result<(u64, f64, f64)> {
    let mut processor = AdaptiveBatchProcessor::new();
    let t0 = Instant::now();
    let messages = processor.process_all(data)?;
    let wall = t0.elapsed();
    let (elapsed_ms, mps) = calculate_throughput(messages.len() as u64, wall);
    Ok((messages.len() as u64, elapsed_ms, mps))
}

pub fn bench_adaptive_config(data: &[u8], config: AdaptiveBatchConfig) -> Result<(u64, f64, f64)> {
    let mut processor = AdaptiveBatchProcessor::with_config(config);
    let t0 = Instant::now();
    let messages = processor.process_all(data)?;
    let wall = t0.elapsed();
    let (elapsed_ms, mps) = calculate_throughput(messages.len() as u64, wall);
    Ok((messages.len() as u64, elapsed_ms, mps))
}

pub fn run_adaptive(data: &[u8]) -> Result<()> {
    let mut processor = AdaptiveBatchProcessor::new();
    let t0 = Instant::now();
    let messages = processor.process_all(data)?;
    let wall = t0.elapsed();

    let mps = messages.len() as f64 / wall.as_secs_f64();
    println!(
        "[adaptive] Parsed {} messages ({} bytes) in {:?} => {:.2} M msg/sec",
        messages.len(),
        data.len(),
        wall,
        mps / 1_000_000.0
    );
    println!(
        "  Final batch size: {}, Avg throughput: {:.2} msg/sec",
        processor.current_batch_size(),
        processor.avg_throughput()
    );

    Ok(())
}

pub fn run_adaptive_strategies(data: &[u8]) -> Result<()> {
    println!("=== Adaptive Strategies Comparison ===");
    println!();

    let strategies = [
        ("Default (Balanced)", AdaptiveBatchConfig::default()),
        ("Low Latency", AdaptiveBatchConfig::low_latency()),
        ("High Throughput", AdaptiveBatchConfig::high_throughput()),
        ("Conservative", AdaptiveBatchConfig::conservative()),
    ];

    let mut results = Vec::new();

    for (name, config) in strategies {
        let mut processor = AdaptiveBatchProcessor::with_config(config);
        let t0 = Instant::now();
        let messages = processor.process_all(data)?;
        let wall = t0.elapsed();
        let metrics = processor.metrics();

        let mps = messages.len() as f64 / wall.as_secs_f64() / 1_000_000.0;
        results.push((name, messages.len(), wall, mps, metrics));
    }

    println!(
        "{:<20} {:>10} {:>12} {:>10} {:>12} {:>10}",
        "Strategy", "Messages", "Time (ms)", "M msg/s", "Batch Range", "Changes"
    );
    println!("{}", "-".repeat(80));

    for (name, msgs, wall, mps, metrics) in &results {
        let (min_b, max_b) = metrics.batch_size_range();
        println!(
            "{:<20} {:>10} {:>12.2} {:>10.2} {:>5}-{:<6} {:>10}",
            name,
            msgs,
            wall.as_secs_f64() * 1000.0,
            mps,
            min_b,
            max_b,
            metrics.batch_size_changes
        );
    }

    println!();

    if let Some((name, _, _, mps, _)) = results.iter().max_by(|a, b| a.3.partial_cmp(&b.3).unwrap())
    {
        println!("Best strategy: {} @ {:.2} M msg/sec", name, mps);
    }

    Ok(())
}

pub fn run_latency_analysis(data: &[u8]) -> Result<()> {
    println!("=== Latency Distribution Analysis ===");
    println!();

    let mut processor =
        AdaptiveBatchProcessor::with_config(AdaptiveBatchConfig::default().with_warmup(0));

    let t0 = Instant::now();
    let messages = processor.process_all(data)?;
    let wall = t0.elapsed();
    let metrics = processor.metrics();

    println!("Processed {} messages in {:?}", messages.len(), wall);
    println!();

    println!("Latency Histogram:");
    let total: u64 = metrics.latency_histogram.iter().sum();
    for (i, count) in metrics.latency_histogram.iter().enumerate() {
        let pct = if total > 0 {
            *count as f64 / total as f64 * 100.0
        } else {
            0.0
        };
        let bar_len = (pct / 2.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!(
            "  {:>10}: {:>8} ({:>5.1}%) {}",
            AdaptiveBatchMetrics::latency_bucket_name(i),
            count,
            pct,
            bar
        );
    }

    println!();
    println!("Batch Size Dynamics:");
    println!(
        "  Initial: {}",
        (processor.config().min_batch_size + processor.config().max_batch_size) / 2
    );
    println!("  Final: {}", metrics.current_batch_size);
    println!(
        "  Range: {} - {}",
        metrics.min_batch_observed, metrics.max_batch_observed
    );
    println!("  Changes: {}", metrics.batch_size_changes);
    println!();

    println!("Quality Metrics:");
    println!(
        "  Stability score: {:.2} (1.0 = perfect)",
        metrics.stability_score()
    );
    println!(
        "  Efficiency score: {:.2} (1.0 = perfect)",
        metrics.efficiency_score()
    );
    println!("  Throughput variance: {:.0}", metrics.throughput_variance);

    Ok(())
}
