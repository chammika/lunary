#![no_main]

use libfuzzer_sys::fuzz_target;
use lunary::{BatchProcessor, ZeroCopyParser};

fuzz_target!(|data: &[u8]| {
    // Fuzz the ZeroCopyParser
    let _ = std::panic::catch_unwind(|| {
        let parser = ZeroCopyParser::new(data);
        let mut count = 0;
        for _msg in parser {
            count += 1;
            if count > 1000 {
                break;
            }
        }
    });

    let _ = std::panic::catch_unwind(|| {
        let mut processor = BatchProcessor::new(1024);
        let _ = processor.process_all(data);
    });
});