mod fixtures;

use lunary::simd::{
    compute_checksum_simd, count_messages_fast, find_message_boundary, memcpy_simd,
    read_timestamp_simd, read_u16_be_simd, read_u32_be_simd, read_u64_be_simd,
    scan_boundaries_with_diagnostics, validate_boundaries_simd, SimdInfo, SimdLevel,
};

#[test]
fn test_simd_fallback_on_all_targets() {
    let data = fixtures::standard_fixture();

    let checksum = compute_checksum_simd(&data);
    assert!(checksum > 0);

    let count = count_messages_fast(&data);
    assert_eq!(count, 1000);

    let boundaries = scan_boundaries_with_diagnostics(&data, usize::MAX);
    assert_eq!(boundaries.len(), 1000);
    assert!(!boundaries.is_empty());

    let valid = validate_boundaries_simd(&data, &boundaries.boundaries);
    assert!(valid);
}

#[test]
fn test_simd_consistency_across_levels() {
    let data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

    let mut dst1 = [0u8; 16];
    let mut dst2 = [0u8; 16];
    memcpy_simd(&mut dst1, &data);
    data.iter().enumerate().for_each(|(i, &b)| dst2[i] = b);
    assert_eq!(dst1, dst2);

    let val_u64 = read_u64_be_simd(&data);
    let expected_u64 = u64::from_be_bytes([1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(val_u64, expected_u64);

    let val_u32 = read_u32_be_simd(&data);
    let expected_u32 = u32::from_be_bytes([1, 2, 3, 4]);
    assert_eq!(val_u32, expected_u32);

    let val_u16 = read_u16_be_simd(&data);
    let expected_u16 = u16::from_be_bytes([1, 2]);
    assert_eq!(val_u16, expected_u16);

    let val_ts = read_timestamp_simd(&data);
    let expected_ts = u64::from_be_bytes([0, 0, 1, 2, 3, 4, 5, 6]);
    assert_eq!(val_ts, expected_ts);
}

#[test]
fn test_simd_info_properties() {
    let info = SimdInfo {
        available: true,
        sse2: true,
        ssse3: true,
        avx2: true,
        avx512: false,
    };

    assert_eq!(info.best_available(), SimdLevel::Avx2);
    assert_eq!(info.register_width(), 32);

    let scalar_info = SimdInfo {
        available: false,
        sse2: false,
        ssse3: false,
        avx2: false,
        avx512: false,
    };

    assert_eq!(scalar_info.best_available(), SimdLevel::Scalar);
    assert_eq!(scalar_info.register_width(), 8);
}

#[test]
fn test_simd_edge_cases() {
    assert_eq!(find_message_boundary(&[], 255), None);
    assert_eq!(compute_checksum_simd(&[]), 0);

    let small_data = [1, 2, 3];
    let _ = find_message_boundary(&small_data, 255);
    assert_eq!(compute_checksum_simd(&small_data), 6);

    let boundary_data = [0, 0, 255, 0, 0];
    assert_eq!(find_message_boundary(&boundary_data, 255), Some(2));

    let large_data = vec![42; 10000];
    let count = count_messages_fast(&large_data);
    assert_eq!(count, 0);
}

#[test]
fn test_simd_memory_safety() {
    let data = vec![0u8; 4096];
    let slice = &data[4090..];
    let _ = find_message_boundary(slice, 255);

    let src = vec![1, 2, 3, 4, 5];
    let mut dst = vec![0; 5];
    memcpy_simd(&mut dst, &src);
    assert_eq!(dst, src);

    let unaligned_data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let slice = &unaligned_data[1..];
    let val = read_u64_be_simd(slice);
    let expected = u64::from_be_bytes([1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(val, expected);
}

#[test]
fn test_simd_fallback_consistency() {
    let data = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0];

    let result_simd = read_u64_be_simd(&data);
    let result_fallback = u64::from_be_bytes(data);
    assert_eq!(result_simd, result_fallback);

    let result_simd_u32 = read_u32_be_simd(&data);
    let result_fallback_u32 = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
    assert_eq!(result_simd_u32, result_fallback_u32);

    let result_simd_u16 = read_u16_be_simd(&data);
    let result_fallback_u16 = u16::from_be_bytes([data[0], data[1]]);
    assert_eq!(result_simd_u16, result_fallback_u16);
}
