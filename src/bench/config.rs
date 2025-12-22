use std::path::PathBuf;

use crate::Config;

#[derive(Clone)]
pub struct BenchConfig {
    pub iterations: usize,
    pub warmup_iterations: usize,
    pub output_format: OutputFormat,
    pub validate: bool,
    pub max_buffer_size: usize,
}

#[derive(Clone, Copy, PartialEq)]
pub enum OutputFormat {
    Human,
    Json,
    Csv,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            iterations: 1,
            warmup_iterations: 0,
            output_format: OutputFormat::Human,
            validate: true,
            max_buffer_size: 512 * 1024 * 1024,
        }
    }
}

pub struct CliArgs {
    pub path: PathBuf,
    pub mode: String,
    pub config: BenchConfig,
}

pub fn parse_cli_args() -> Option<CliArgs> {
    let args: Vec<String> = std::env::args().collect();

    let path = args.get(1).map(PathBuf::from)?;

    if path.as_os_str() == "--help" || path.as_os_str() == "-h" {
        print_usage();
        return None;
    }

    let mode = args
        .get(2)
        .map(|s| s.to_string())
        .unwrap_or_else(|| "all".to_string());

    let max_size_mb: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(512);
    let lunary_config = Config::from_size_mb(max_size_mb);

    let mut bench_config = BenchConfig {
        max_buffer_size: lunary_config.max_buffer_size,
        ..Default::default()
    };

    let mut i = 4;
    while i < args.len() {
        match args[i].as_str() {
            "--json" => bench_config.output_format = OutputFormat::Json,
            "--csv" => bench_config.output_format = OutputFormat::Csv,
            "--iterations" => {
                if let Some(n) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                    bench_config.iterations = n;
                    i += 1;
                }
            }
            "--warmup" => {
                if let Some(n) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                    bench_config.warmup_iterations = n;
                    i += 1;
                }
            }
            "--no-validate" => bench_config.validate = false,
            "--validate" => bench_config.validate = true,
            _ => {}
        }
        i += 1;
    }

    Some(CliArgs {
        path,
        mode,
        config: bench_config,
    })
}

pub fn print_usage() {
    eprintln!("Usage: itch-bench <path-to-itch-file> [mode] [max-size-mb] [options]");
    eprintln!();
    eprintln!("Basic Modes:");
    eprintln!("  simple, batch, adaptive, parallel, worksteal, mmap, decode, zerocopy, spsc, simd");
    eprintln!();
    eprintln!("Advanced Modes:");
    eprintln!("  zerocopy-ref   - Zero-copy reference parsing with MessageRef");
    eprintln!("  adaptive-all   - Compare all adaptive strategies");
    eprintln!("  worker-stats   - Parallel parsing with per-worker stats");
    eprintln!("  simd-validate  - SIMD message validation benchmark");
    eprintln!("  latency        - Latency distribution analysis");
    eprintln!("  realworld      - Real-world simulation benchmark");
    eprintln!("  diagnostics    - Full SIMD/cache diagnostics");
    eprintln!("  feature-cmp    - Per-feature comparison (zerocopy vs owned, SIMD vs scalar)");
    eprintln!("  fuzzing        - Fuzzing and error injection test");
    eprintln!();
    eprintln!("  all (default)  - Run all benchmarks");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  max-size-mb    - Maximum file/buffer size in MB (default: 512)");
    eprintln!("  --iterations N - Run each benchmark N times (default: 1)");
    eprintln!("  --warmup N     - Run N warmup iterations before measurement (default: 0)");
    eprintln!("  --json         - Output results in JSON format");
    eprintln!("  --csv          - Output results in CSV format");
}

pub fn print_config_summary(data_len: usize, max_size_mb: usize, mmap_len: usize) {
    println!("=== Configuration ===");
    println!("  Max buffer size: {} MB", max_size_mb);
    println!("  File size: {:.2} MB", mmap_len as f64 / (1024.0 * 1024.0));
    println!(
        "  Data loaded: {:.2} MB",
        data_len as f64 / (1024.0 * 1024.0)
    );
    println!();
}
