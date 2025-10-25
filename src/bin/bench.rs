use anyhow::Result;
use lunyn_itch_lite::Parser;
use memmap2::MmapOptions;
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let path = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        eprintln!("Usage: itch-bench <path-to-itch-file>");
        std::process::exit(2);
    });

    let file = File::open(&path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };

    let mut parser = Parser::default();
    let t0 = Instant::now();
    let stats = parser.parse(&mmap)?;
    let wall = t0.elapsed();

    let mps = stats.messages as f64 / wall.as_secs_f64();
    println!(
        "Parsed {} messages ({} bytes) in {:?} => {:.2} M msg/sec",
        stats.messages,
        stats.bytes,
        wall,
        mps / 1_000_000.0
    );

    Ok(())
}
