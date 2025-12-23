use anyhow::Result;
use std::time::Instant;

use crate::{is_valid_message_type, scan_boundaries_auto, simd_info};

use super::utils::calculate_throughput;

pub fn bench_simd_scan(data: &[u8]) -> Result<(u64, f64, f64)> {
    let t0 = Instant::now();
    let result = scan_boundaries_auto(data, usize::MAX);
    let wall = t0.elapsed();
    let count = result.len() as u64;
    let (elapsed_ms, mps) = calculate_throughput(count, wall);
    Ok((count, elapsed_ms, mps))
}

pub fn bench_simd_validate(data: &[u8]) -> Result<(u64, f64, f64)> {
    let boundaries = scan_boundaries_auto(data, usize::MAX);
    let t0 = Instant::now();
    let mut valid_count = 0u64;
    for (offset, _len) in boundaries.boundaries.iter() {
        if *offset + 3 < data.len() {
            let msg_type = data[*offset + 2];
            if is_valid_message_type(msg_type) {
                valid_count += 1;
            }
        }
    }
    let wall = t0.elapsed();
    let (elapsed_ms, mps) = calculate_throughput(valid_count, wall);
    Ok((valid_count, elapsed_ms, mps))
}

pub fn run_simd_bench(data: &[u8]) -> Result<()> {
    let info = simd_info();
    println!("=== SIMD Boundary Scan Benchmark ===");
    println!("Using: {}", info.best_available());
    println!();

    let iterations = 10;
    let mut total_time = std::time::Duration::ZERO;
    let mut result_count = 0u64;

    for i in 0..iterations {
        let t0 = Instant::now();
        let result = scan_boundaries_auto(data, usize::MAX);
        let elapsed = t0.elapsed();
        total_time += elapsed;
        result_count = result.len() as u64;

        if i == 0 {
            let diag = &result.diagnostics;
            println!("Diagnostics:");
            println!("  Messages scanned: {}", diag.messages_scanned);
            println!("  SIMD bytes: {}", diag.simd_bytes);
            println!("  Scalar bytes: {}", diag.scalar_bytes);
            println!("  Prefetch count: {}", diag.prefetch_count);
            println!(
                "  SIMD utilization: {:.1}%",
                diag.simd_utilization() * 100.0
            );
            if let Some(level) = diag.level_used {
                println!("  SIMD level: {}", level);
            }
        }
    }

    let avg_time = total_time / iterations as u32;
    let avg_mps = result_count as f64 / avg_time.as_secs_f64() / 1_000_000.0;

    println!();
    println!("Results ({} iterations):", iterations);
    println!("  Messages: {}", result_count);
    println!("  Avg time: {:?}", avg_time);
    println!("  Throughput: {:.2} M msg/sec", avg_mps);

    Ok(())
}

pub fn run_simd_validate(data: &[u8]) -> Result<()> {
    println!("=== SIMD Message Validation Benchmark ===");

    let boundaries = scan_boundaries_auto(data, usize::MAX);
    println!("Scanned {} message boundaries", boundaries.len());

    let iterations = 10;
    let mut total_time = std::time::Duration::ZERO;
    let mut valid = 0u64;
    let mut invalid = 0u64;

    for i in 0..iterations {
        let t0 = Instant::now();
        valid = 0;
        invalid = 0;

        for (offset, _len) in boundaries.boundaries.iter() {
            if *offset + 3 < data.len() {
                let msg_type = data[*offset + 2];
                if is_valid_message_type(msg_type) {
                    valid += 1;
                } else {
                    invalid += 1;
                }
            }
        }

        total_time += t0.elapsed();

        if i == 0 {
            println!("  Valid messages: {}", valid);
            println!("  Invalid messages: {}", invalid);
        }
    }

    let avg_time = total_time / iterations as u32;
    let total_validated = valid + invalid;
    let avg_mps = total_validated as f64 / avg_time.as_secs_f64() / 1_000_000.0;

    println!();
    println!("Results ({} iterations):", iterations);
    println!("  Avg time: {:?}", avg_time);
    println!("  Throughput: {:.2} M validations/sec", avg_mps);

    Ok(())
}

pub fn print_simd_info() {
    let info = simd_info();
    println!("=== SIMD Capabilities ===");
    println!("  SSE2:     {}", if info.sse2 { "✓" } else { "✗" });
    println!("  SSSE3:    {}", if info.ssse3 { "✓" } else { "✗" });
    println!("  AVX2:     {}", if info.avx2 { "✓" } else { "✗" });
    println!("  AVX512:   {}", if info.avx512 { "✓" } else { "✗" });
    println!("  Best:     {}", info.best_available());
    println!("  Width:    {} bytes", info.register_width());
    println!();
}
