use std::time::Duration;

pub fn median_f64(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

pub fn variance_f64(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let sum_sq_diff: f64 = values.iter().map(|x| (x - mean).powi(2)).sum();
    sum_sq_diff / (values.len() - 1) as f64
}

pub fn calculate_throughput(count: u64, elapsed: Duration) -> (f64, f64) {
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let mps = count as f64 / elapsed.as_secs_f64() / 1_000_000.0;
    (elapsed_ms, mps)
}

pub fn find_message_boundary(data: &[u8], target: usize) -> usize {
    if target >= data.len() {
        return data.len();
    }

    let mut offset = 0;
    while offset < data.len() {
        if offset + 2 > data.len() {
            return data.len();
        }
        let len = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
        let next = offset + 2 + len;
        if next > data.len() {
            return offset;
        }
        if next >= target {
            return next;
        }
        offset = next;
    }
    data.len()
}

pub fn build_message_boundaries(data: &[u8]) -> Vec<usize> {
    let mut boundaries = Vec::new();
    let mut offset = 0;

    while offset + 2 <= data.len() {
        let len = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
        let next = offset + 2 + len;
        if next > data.len() {
            break;
        }
        boundaries.push(next);
        offset = next;
    }

    boundaries
}

pub fn split_into_chunks(data: &[u8], chunk_size: usize) -> Vec<(usize, usize)> {
    let boundaries = build_message_boundaries(data);
    if boundaries.is_empty() {
        return vec![(0, data.len())];
    }

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < data.len() {
        let target = (start + chunk_size).min(data.len());

        let pos = match boundaries.binary_search(&target) {
            Ok(idx) => idx,
            Err(idx) => {
                if idx >= boundaries.len() {
                    boundaries.len() - 1
                } else {
                    idx
                }
            }
        };

        let end = boundaries.get(pos).copied().unwrap_or(data.len());
        if end > start {
            chunks.push((start, end));
            start = end;
        } else {
            break;
        }
    }

    if chunks.is_empty() && !data.is_empty() {
        chunks.push((0, data.len()));
    }

    chunks
}

#[cfg(test)]
pub fn make_buffer(messages: &[(u8, usize)]) -> Vec<u8> {
    let mut buf = Vec::new();
    for (msg_type, payload_len) in messages {
        let len = 1 + payload_len;
        buf.extend_from_slice(&(len as u16).to_be_bytes());
        buf.push(*msg_type);
        buf.extend(std::iter::repeat_n(0xABu8, *payload_len));
    }
    buf
}
