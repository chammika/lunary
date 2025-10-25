use std::time::{Duration, Instant};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("buffer too small for length field at offset {0}")]
    TooSmall(usize),
    #[error("declared length {decl} overruns buffer at offset {offset}")]
    Overrun { offset: usize, decl: usize },
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ParseStats {
    pub messages: u64,
    pub bytes: u64,
    pub elapsed: Duration,
}

impl ParseStats {
    pub fn mps(&self) -> f64 {
        if self.elapsed.as_secs_f64() > 0.0 {
            self.messages as f64 / self.elapsed.as_secs_f64()
        } else {
            0.0
        }
    }
}

#[derive(Debug, Default)]
pub struct Parser;

impl Parser {
    /// Parse a buffer of length-prefixed ITCH messages (u16 big-endian length).
    /// Returns basic stats; does not decode message fields.
    pub fn parse(&mut self, buf: &[u8]) -> Result<ParseStats, ParseError> {
        let start = Instant::now();
        let mut offset = 0usize;
        let mut messages: u64 = 0;

        while offset < buf.len() {
            if offset + 2 > buf.len() {
                return Err(ParseError::TooSmall(offset));
            }
            let len = u16::from_be_bytes([buf[offset], buf[offset + 1]]) as usize;
            let next = offset + 2 + len;
            if next > buf.len() {
                return Err(ParseError::Overrun { offset, decl: len });
            }
            // Minimal work: count and skip. Users can inspect &buf[offset+2..next]
            messages += 1;
            offset = next;
        }

        Ok(ParseStats {
            messages,
            bytes: buf.len() as u64,
            elapsed: start.elapsed(),
        })
    }
}
