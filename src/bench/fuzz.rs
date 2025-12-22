use anyhow::Result;

use crate::{validate_message_stream_simd, BatchProcessor, ZeroCopyParser};

pub fn run_fuzzing_test(data: &[u8]) -> Result<()> {
    println!("=== Fuzzing and Error Injection Test ===");
    println!();

    println!("--- Baseline (Valid Data) ---");
    let validation = validate_message_stream_simd(data, 1000);
    println!(
        "Valid: {}, Invalid: {}",
        validation.valid_count, validation.invalid_count
    );
    println!();

    println!("--- Truncated Data Test ---");
    for truncate_pct in [10, 25, 50, 75, 90] {
        let truncate_at = data.len() * truncate_pct / 100;
        let truncated = &data[..truncate_at];
        let result = validate_message_stream_simd(truncated, 100);
        println!(
            "  {}% ({} bytes): valid={}, invalid={}",
            truncate_pct,
            truncated.len(),
            result.valid_count,
            result.invalid_count
        );
    }
    println!();

    println!("--- Corrupted Message Types Test ---");
    let mut corrupted = data.to_vec();
    let mut message_index = 0;
    let mut actual_corrupted = 0;
    let mut offset = 0;
    while offset + 3 < corrupted.len() && message_index < 100 {
        let len = u16::from_be_bytes([corrupted[offset], corrupted[offset + 1]]) as usize;
        if len == 0 || offset + 2 + len > corrupted.len() {
            break;
        }
        if message_index % 10 == 5 {
            corrupted[offset + 2] = 0xFF;
            actual_corrupted += 1;
        }
        message_index += 1;
        offset += 2 + len;
    }

    let result = validate_message_stream_simd(&corrupted, 1000);
    println!("  Corrupted {} message types", actual_corrupted);
    println!(
        "  Valid: {}, Invalid: {}",
        result.valid_count, result.invalid_count
    );
    if !result.error_types.is_empty() {
        let mut error_counts: std::collections::HashMap<_, usize> =
            std::collections::HashMap::new();
        for err in &result.error_types {
            *error_counts.entry(format!("{:?}", err)).or_insert(0) += 1;
        }
        println!("  Error breakdown:");
        for (err_type, count) in error_counts {
            println!("    {}: {}", err_type, count);
        }
    }
    println!();

    println!("--- Random Noise Injection Test ---");
    let mut noisy = data.to_vec();
    let noise_positions: Vec<usize> = (0..100)
        .map(|i| (i * data.len() / 100 + 7) % data.len())
        .collect();

    for pos in &noise_positions {
        if *pos < noisy.len() {
            noisy[*pos] = (noisy[*pos].wrapping_add(0x55)) ^ 0xAA;
        }
    }

    let result = validate_message_stream_simd(&noisy, 1000);
    println!("  Injected noise at {} positions", noise_positions.len());
    println!(
        "  Valid: {}, Invalid: {}",
        result.valid_count, result.invalid_count
    );
    println!();

    println!("--- Parser Robustness Test ---");

    let zerocopy_result = std::panic::catch_unwind(|| {
        let parser = ZeroCopyParser::new(&noisy);
        let mut count = 0u64;
        for _msg in parser {
            count += 1;
        }
        count
    });

    match zerocopy_result {
        Ok(count) => println!("  ZeroCopyParser: Processed {} messages (no panic)", count),
        Err(_) => println!("  ZeroCopyParser: Panicked on corrupted data"),
    }

    let batch_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut processor = BatchProcessor::new(1024);
        processor.process_all(&noisy).map(|m| m.len()).unwrap_or(0)
    }));

    match batch_result {
        Ok(count) => println!("  BatchProcessor: Processed {} messages (no panic)", count),
        Err(_) => println!("  BatchProcessor: Panicked on corrupted data"),
    }

    println!();
    println!("Fuzzing tests complete.");

    Ok(())
}
