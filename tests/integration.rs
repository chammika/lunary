mod fixtures;

#[test]
fn test_message_boundary_detection() {
    let buf = fixtures::create_test_buffer(&[
        (b'A', &[1, 2, 3, 4, 5]),
        (b'B', &[10, 20]),
        (b'C', &[100]),
    ]);

    let mut boundaries = vec![0usize];
    let mut pos = 0;
    while pos + 2 <= buf.len() {
        let len = u16::from_be_bytes([buf[pos], buf[pos + 1]]) as usize;
        if pos + 2 + len > buf.len() {
            break;
        }
        pos += 2 + len;
        boundaries.push(pos);
    }

    assert_eq!(boundaries, vec![0, 8, 13, 17]);
}

#[test]
fn test_parser_count_consistency() {
    let buf = fixtures::standard_fixture();

    let mut expected_count = 0;
    let mut pos = 0;
    while pos + 2 <= buf.len() {
        let len = u16::from_be_bytes([buf[pos], buf[pos + 1]]) as usize;
        if pos + 2 + len > buf.len() {
            break;
        }
        pos += 2 + len;
        expected_count += 1;
    }

    assert_eq!(expected_count, 1000);
}
