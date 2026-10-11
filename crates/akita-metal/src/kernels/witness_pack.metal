// Direct balanced centered-i32 Z decomposition into exact signed packed bytes.
// Each group owns one coefficient tile; each payload byte has one writer.
struct AkitaWitnessPackParams {
    ulong ring_d;
    ulong tile_width;
    ulong fold_planes;
    ulong log_basis;
    ulong bit_width;
};

kernel void akita_centered_witness_z_pack(
    device const int *centered [[buffer(0)]],
    device uchar *packed [[buffer(1)]],
    device atomic_uint *bounds [[buffer(2)]],
    constant AkitaWitnessPackParams &p [[buffer(3)]],
    threadgroup uchar *digits [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const ulong tile_width = p.tile_width;
    const ulong tiles_per_row = p.ring_d / tile_width;
    const ulong row = ulong(group) / tiles_per_row;
    const ulong tile = ulong(group) % tiles_per_row;
    const uint base = 1u << uint(p.log_basis);
    const uint midpoint = base >> 1;
    const uint signed_mask = (1u << uint(p.bit_width)) - 1;
    uint negative_abs_max = 0;
    uint positive_max = 0;
    if (ulong(lane) < tile_width) {
        int carry = centered[row * p.ring_d + tile * tile_width + ulong(lane)];
        for (ulong plane = 0; plane < p.fold_planes; ++plane) {
            const uint raw = uint(carry) & (base - 1);
            const int digit = raw >= midpoint ? int(raw) - int(base) : int(raw);
            // This is (carry - digit) >> basis without the i32::MAX overflow.
            carry = (carry >> uint(p.log_basis)) + int(raw >= midpoint);
            digits[plane * tile_width + ulong(lane)] = uchar(uint(digit) & signed_mask);
            negative_abs_max = max(negative_abs_max, digit < 0 ? uint(-digit) : 0u);
            positive_max = max(positive_max, digit > 0 ? uint(digit) : 0u);
        }
    }
    const uint simd_negative = simd_max(negative_abs_max);
    const uint simd_positive = simd_max(positive_max);
    if ((lane & 31u) == 0) {
        atomic_fetch_max_explicit(&bounds[0], simd_negative, memory_order_relaxed);
        atomic_fetch_max_explicit(&bounds[1], simd_positive, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ulong plane_bytes = tile_width * p.bit_width / 8;
    const ulong tile_bytes = plane_bytes * p.fold_planes;
    for (ulong byte = ulong(lane); byte < tile_bytes; byte += 256) {
        const ulong plane = byte / plane_bytes;
        const ulong local_byte = byte % plane_bytes;
        const ulong first_bit = local_byte * 8;
        const ulong first_digit = first_bit / p.bit_width;
        const uint shift = uint(first_bit % p.bit_width);
        uint bits = 0;
        // A byte spans at most eight one-bit digits, fewer for wider payloads.
        for (ulong digit = first_digit; digit < tile_width; ++digit) {
            const ulong offset = (digit - first_digit) * p.bit_width;
            if (offset >= ulong(shift) + 8) break;
            bits |= uint(digits[plane * tile_width + digit]) << uint(offset);
        }
        const ulong destination = ((row * p.fold_planes + plane) * p.ring_d
            + tile * tile_width) * p.bit_width / 8 + local_byte;
        packed[destination] = uchar(bits >> shift);
    }
}
