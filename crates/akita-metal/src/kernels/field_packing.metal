// Compiled after onehot.metal, sharing its exact Fp128 arithmetic.
struct FieldCoefficientPackingParams {
    ulong live_coefficients;
    ulong num_live_positions;
    ulong positions_per_block;
    ulong num_blocks;
    ulong ring_d;
    ulong stride;
    ulong subring_dimension;
    ulong extension_degree;
    ulong output_coefficients;
    ulong bit_width;
};

inline int akita_field_packing_digit(device const uchar* source, ulong index, ulong width) {
    ulong bit = index * width;
    uint shift = (uint)(bit % 8ul);
    uint raw = (uint)source[bit / 8ul] >> shift;
    // Do not read beyond the exact payload's last byte.
    if ((ulong)shift + width > 8ul) {
        raw |= (uint)source[bit / 8ul + 1ul] << (8u - shift);
    }
    uint sign = 1u << ((uint)width - 1u);
    raw &= (1u << (uint)width) - 1u;
    return (int)(raw ^ sign) - (int)sign;
}

inline void akita_field_packing_reduce(
    threadgroup AkitaFp128* reduction, uint thread_index,
    AkitaFp128 accumulator, device AkitaFp128* output, ulong output_index)
{
    reduction[thread_index] = accumulator;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128u; offset != 0u; offset >>= 1u) {
        if (thread_index < offset) {
            reduction[thread_index] = akita_add(reduction[thread_index], reduction[thread_index + offset]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (thread_index == 0u) { output[output_index] = reduction[0]; }
}

kernel void akita_fp128_field_coefficient_packing(
    device const AkitaFp128* source [[buffer(0)]],
    device const AkitaFp128* combined_weights [[buffer(1)]],
    device AkitaFp128* output [[buffer(2)]],
    constant FieldCoefficientPackingParams& params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    threadgroup AkitaFp128 reduction[256];
    ulong output_index = (ulong)group.x;
    ulong partial_width = params.subring_dimension * params.extension_degree;
    ulong block = output_index / partial_width;
    ulong coordinate = (output_index % partial_width) / params.subring_dimension;
    ulong subring = output_index % params.subring_dimension;
    ulong first_position = block * params.positions_per_block;
    ulong terms = params.positions_per_block * params.stride;
    AkitaFp128 accumulator = akita_zero();
    for (ulong term = (ulong)thread_index; term < terms; term += 256ul) {
        ulong position = first_position + term / params.stride;
        ulong coefficient = subring * params.stride + term % params.stride;
        ulong flat = position * params.ring_d + coefficient;
        if (position < params.num_live_positions && flat < params.live_coefficients) {
            accumulator = akita_add(accumulator, akita_mul(source[flat], combined_weights[coordinate * terms + term]));
        }
    }
    akita_field_packing_reduce(reduction, thread_index, accumulator, output, output_index);
}

kernel void akita_fp128_packed_digit_coefficient_packing(
    device const uchar* source [[buffer(0)]],
    device const AkitaFp128* combined_weights [[buffer(1)]],
    device AkitaFp128* output [[buffer(2)]],
    constant FieldCoefficientPackingParams& params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    threadgroup AkitaFp128 reduction[256];
    ulong output_index = (ulong)group.x;
    ulong partial_width = params.subring_dimension * params.extension_degree;
    ulong block = output_index / partial_width;
    ulong coordinate = (output_index % partial_width) / params.subring_dimension;
    ulong subring = output_index % params.subring_dimension;
    ulong first_position = block * params.positions_per_block;
    ulong terms = params.positions_per_block * params.stride;
    AkitaFp128 accumulator = akita_zero();
    for (ulong term = (ulong)thread_index; term < terms; term += 256ul) {
        ulong position = first_position + term / params.stride;
        ulong coefficient = subring * params.stride + term % params.stride;
        ulong flat = position * params.ring_d + coefficient;
        if (position < params.num_live_positions && flat < params.live_coefficients) {
            int digit = akita_field_packing_digit(source, flat, params.bit_width);
            AkitaFp128 value = akita_mul_signed_small(combined_weights[coordinate * terms + term], (long)digit);
            accumulator = akita_add(accumulator, value);
        }
    }
    akita_field_packing_reduce(reduction, thread_index, accumulator, output, output_index);
}
