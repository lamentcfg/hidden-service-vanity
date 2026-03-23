// Precomputed base point table for Ed25519 scalar multiplication
// 32*8 = 256 ge_precomp entries, each with 3 field elements of 10 int32_t
// Total size: 32 * 8 * 3 * 10 * 4 = 30,720 bytes
pub const BASE_TABLE: &[u8] = include_bytes!("precomp_base_table.bin");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_size() {
        assert_eq!(BASE_TABLE.len(), 32 * 8 * 3 * 10 * 4);
    }
}
