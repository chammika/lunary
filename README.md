# Lunyn ITCH Lite

Open-source, minimal ITCH (NASDAQ TotalView-ITCH) parser designed to be easy to read and reproduce. This version intentionally avoids advanced optimizations (no SIMD, no lock-free queues) and targets roughly 10M messages/second on commodity hardware.

- Focus: clarity, correctness, and reproducibility
- Not included: vendor-specific tweaks, SIMD intrinsics, NUMA pinning, lock-free structures, custom allocators
- Good for: education, research, basic ETL, and as a baseline for your own optimization work

## Quickstart

```bash
# Build
cargo build --release

# Run the simple benchmark against an ITCH binary file
# (Requires a licensed NASDAQ ITCH file; obey your data license.)
./target/release/itch-bench /path/to/itch-file.bin
```

The bench utility will:
- memory-map the file
- scan and count length-prefixed ITCH messages (u16 big-endian length)
- report the total message count, elapsed time, and messages/sec

## Format assumptions

- Messages are length-prefixed: first 2 bytes = message length (big-endian), followed by type and payload.
- This is consistent with NASDAQ TotalView-ITCH framing.
- The parser validates that lengths do not overrun the buffer; deeper field decoding is left as an exercise.

## API sketch

```rust
use lunyn_itch_lite::{Parser, ParseStats};

let buf = std::fs::read("/path/to/itch.bin")?;
let mut parser = Parser::default();
let stats: ParseStats = parser.parse(&buf)?;
println!("{} msgs in {:?} (~{:.2} M/s)", stats.messages, stats.elapsed, stats.mps() / 1_000_000.0);
```

## Performance notes

This version aims for simplicity rather than peak performance. On a modern x86_64 workstation, you should see multi-million messages/sec. Achieving 10M+/sec typically requires:
- release build
- memory-mapped input
- minimal per-message work (e.g., counting only)

For 100M+/sec parsing with low-nanosecond latency, see the commercial version.

## License

MIT © Lunyn
