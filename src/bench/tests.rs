use super::utils::{build_message_boundaries, find_message_boundary, make_buffer};

#[test]
fn test_build_message_boundaries_empty() {
    let buf: &[u8] = &[];
    let boundaries = build_message_boundaries(buf);
    assert_eq!(boundaries, vec![]);
}

#[test]
fn test_build_message_boundaries_single() {
    let buf = make_buffer(&[(b'A', 5)]);
    let boundaries = build_message_boundaries(&buf);
    assert_eq!(boundaries, vec![8]);
}

#[test]
fn test_build_message_boundaries_multiple() {
    let buf = make_buffer(&[(b'A', 5), (b'B', 3), (b'C', 10)]);
    let boundaries = build_message_boundaries(&buf);
    assert_eq!(boundaries, vec![8, 14, 27]);
}

#[test]
fn test_find_message_boundary_at_start() {
    let buf = make_buffer(&[(b'A', 5), (b'B', 3)]);
    assert_eq!(find_message_boundary(&buf, 0), 8);
}

#[test]
fn test_find_message_boundary_mid_message() {
    let buf = make_buffer(&[(b'A', 5), (b'B', 3)]);
    let result = find_message_boundary(&buf, 4);
    assert_eq!(result, 8);
}

#[test]
fn test_split_into_chunks_single_thread() {
    use super::utils::split_into_chunks;
    let buf = make_buffer(&[(b'A', 5); 10]);
    let chunks = split_into_chunks(&buf, buf.len() + 1);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0], (0, buf.len()));
}

#[test]
fn test_split_into_chunks_preserves_boundaries() {
    use super::utils::split_into_chunks;
    let buf = make_buffer(&[(b'A', 5); 100]);
    let chunks = split_into_chunks(&buf, 4);

    assert_eq!(chunks.first().unwrap().0, 0);
    assert_eq!(chunks.last().unwrap().1, buf.len());

    for i in 1..chunks.len() {
        assert_eq!(chunks[i].0, chunks[i - 1].1);
    }

    for (start, _) in &chunks {
        if *start > 0 && *start < buf.len() {
            let len = u16::from_be_bytes([buf[*start], buf[*start + 1]]) as usize;
            assert!(*start + 2 + len <= buf.len());
        }
    }
}
