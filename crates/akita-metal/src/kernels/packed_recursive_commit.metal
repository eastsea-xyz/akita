// Appended after recursive_fold.metal; its unsigned Fp128 magnitude helpers
// implement the CPU's asymmetric-centering overflow case without signed overflow.
inline int akita_packed_recursive_commit_digit(
    device const char* source, constant RecursiveCommitParams& matrix,
    constant PackedRecursiveCommitParams& p, ulong expanded_index)
{
    if (p.bit_width == 0ul) { return (int)source[expanded_index]; }
    ulong raw_index = expanded_index;
    ulong plane = 0ul;
    if (p.num_digits != 1ul) {
        ulong coefficient = expanded_index % matrix.ring_d;
        ulong expanded_ring = expanded_index / matrix.ring_d;
        plane = expanded_ring % p.num_digits;
        raw_index = (expanded_ring / p.num_digits) * matrix.ring_d + coefficient;
    }
    if (raw_index >= p.stored_coefficients || raw_index >= p.physical_coefficients) { return 0; }
    int value = akita_field_packing_digit((device const uchar*)source, raw_index, p.bit_width);
    if (p.num_digits == 1ul) { return value; }

    uint bits = (uint)p.log_basis;
    int basis = 1 << bits;
    int mask = basis - 1;
    // Normally raw signed i8 coefficients are already the CPU's centered value.
    // Only a tiny asymmetric threshold can turn a positive digit into digit-q.
    bool ordinary = value <= 0 || p.threshold.limb.y != 0u || p.threshold.limb.z != 0u
        || p.threshold.limb.w != 0u || (uint)value <= p.threshold.limb.x;
    if (ordinary) {
        for (ulong digit = 0ul; digit <= plane; ++digit) {
            int raw = value & mask;
            int balanced = raw >= basis / 2 ? raw - basis : raw;
            if (digit == plane) { return balanced; }
            value = (value >> bits) + (int)(balanced < 0);
        }
    } else {
        uint4 magnitude = akita_recursive_usub(
            uint4(0x00005809u, 0xffffffffu, 0xffffffffu, 0xffffffffu), uint4((uint)value, 0u, 0u, 0u));
        for (ulong digit = 0ul; digit <= plane; ++digit) {
            uint residue = magnitude.x & (uint)mask;
            uint raw = (0u - residue) & (uint)mask;
            int balanced = raw >= (uint)(basis / 2) ? (int)raw - basis : (int)raw;
            if (digit == plane) { return balanced; }
            magnitude = akita_recursive_ushr(magnitude, bits);
            if (residue != 0u && balanced >= 0) { magnitude = akita_recursive_uinc(magnitude); }
        }
    }
    return 0; // Admission ensures a positive digit count and an in-range plane.
}
