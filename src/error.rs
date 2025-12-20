use thiserror::Error;

#[derive(Error, Debug)]
pub enum ParseError {
    #[error("Invalid message type: {0}")]
    InvalidMessageType(u8),

    #[error("Insufficient data: expected {expected}, got {actual}")]
    InsufficientData { expected: usize, actual: usize },

    #[error("Buffer overflow: input exceeds maximum allowed size")]
    BufferOverflow,

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, ParseError>;
