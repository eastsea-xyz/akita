// Appended after onehot.metal and field_packing.metal; reuse canonical Fp128 arithmetic.
struct RecursiveFoldParams {
    ulong stored_coefficients;
    ulong live_rings;
    ulong physical_coefficients;
    ulong positions_per_block;
    ulong num_blocks;
    ulong ring_d;
    ulong bit_width;
    ulong num_digits;
    ulong log_basis;
    AkitaFp128 threshold;
};

struct RecursiveWeightTerm { AkitaFp128 weight; ulong position; ulong shift; };
struct RecursiveSparseTerm { ulong block; uint shift; int coefficient; };

inline AkitaFp128 akita_recursive_source(
    device const uchar* bytes, constant RecursiveFoldParams& p, ulong coefficient)
{
    if (coefficient >= p.stored_coefficients || coefficient >= p.physical_coefficients) {
        return akita_zero();
    }
    if (p.bit_width == 0ul) {
        return ((device const AkitaFp128*)bytes)[coefficient];
    }
    return akita_mul_signed_small(akita_from_u32(1u),
        (long)akita_field_packing_digit(bytes, coefficient, p.bit_width));
}

kernel void akita_fp128_recursive_evaluate_fold(
    device const uchar* source [[buffer(0)]],
    device const RecursiveWeightTerm* terms [[buffer(1)]],
    device AkitaFp128* folded [[buffer(2)]],
    constant RecursiveFoldParams& p [[buffer(3)]],
    constant ulong& term_count [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup AkitaFp128 reduction[256];
    ulong output_index = (ulong)group.x;
    ulong block = output_index / p.ring_d;
    ulong coefficient = output_index % p.ring_d;
    AkitaFp128 accumulator = akita_zero();
    for (ulong t = lane; t < term_count; t += 256ul) {
        RecursiveWeightTerm term = terms[t];
        bool positive = coefficient >= term.shift;
        ulong source_coefficient = (coefficient + p.ring_d - term.shift) % p.ring_d;
        ulong flat = (block * p.positions_per_block + term.position) * p.ring_d
            + source_coefficient;
        AkitaFp128 product;
        if (p.bit_width != 0ul && flat < p.stored_coefficients && flat < p.physical_coefficients) {
            product = akita_mul_signed_small(term.weight,
                (long)akita_field_packing_digit(source, flat, p.bit_width));
        } else {
            product = akita_mul(akita_recursive_source(source, p, flat), term.weight);
        }
        accumulator = positive ? akita_add(accumulator, product) : akita_sub(accumulator, product);
    }
    akita_field_packing_reduce(reduction, lane, accumulator, folded, output_index);
}

kernel void akita_fp128_recursive_fold_evaluation(
    device const AkitaFp128* folded [[buffer(0)]],
    device const RecursiveWeightTerm* terms [[buffer(1)]],
    device AkitaFp128* evaluation [[buffer(2)]],
    constant RecursiveFoldParams& p [[buffer(3)]],
    constant ulong& term_count [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup AkitaFp128 reduction[256];
    ulong coefficient = (ulong)group.x;
    AkitaFp128 accumulator = akita_zero();
    for (ulong t = lane; t < term_count; t += 256ul) {
        RecursiveWeightTerm term = terms[t];
        bool positive = coefficient >= term.shift;
        ulong source_coefficient = (coefficient + p.ring_d - term.shift) % p.ring_d;
        AkitaFp128 product = akita_mul(folded[term.position * p.ring_d + source_coefficient], term.weight);
        accumulator = positive ? akita_add(accumulator, product) : akita_sub(accumulator, product);
    }
    akita_field_packing_reduce(reduction, lane, accumulator, evaluation, coefficient);
}

inline bool akita_recursive_gt(uint4 a, uint4 b) {
    for (int limb = 3; limb >= 0; --limb) {
        if (a[limb] != b[limb]) { return a[limb] > b[limb]; }
    }
    return false;
}

// Unsigned integer subtraction. This must not use modular-field subtraction.
inline uint4 akita_recursive_usub(uint4 a, uint4 b) {
    uint4 result;
    ulong borrow = 0ul;
    for (uint limb = 0; limb < 4u; ++limb) {
        ulong subtrahend = (ulong)b[limb] + borrow;
        result[limb] = (uint)((ulong)a[limb] - subtrahend);
        borrow = (ulong)a[limb] < subtrahend;
    }
    return result;
}

inline uint4 akita_recursive_ushr(uint4 value, uint bits) {
    return uint4((value.x >> bits) | (value.y << (32u - bits)),
        (value.y >> bits) | (value.z << (32u - bits)),
        (value.z >> bits) | (value.w << (32u - bits)), value.w >> bits);
}

inline uint4 akita_recursive_uinc(uint4 value) {
    for (uint limb = 0; limb < 4u; ++limb) {
        value[limb] += 1u;
        if (value[limb] != 0u) { break; }
    }
    return value;
}

kernel void akita_fp128_recursive_balanced_planes(
    device const AkitaFp128* fields [[buffer(0)]],
    device short* planes [[buffer(1)]],
    constant RecursiveFoldParams& p [[buffer(2)]],
    uint flat [[thread_position_in_grid]])
{
    if ((ulong)flat >= p.stored_coefficients) { return; }
    uint4 value = fields[flat].limb;
    bool negative = akita_recursive_gt(value, p.threshold.limb);
    if (negative) {
        value = akita_recursive_usub(uint4(0x00005809u, 0xffffffffu, 0xffffffffu, 0xffffffffu), value);
    }
    uint bits = (uint)p.log_basis;
    uint basis = 1u << bits;
    uint mask = basis - 1u;
    for (ulong digit = 0ul; digit < p.num_digits; ++digit) {
        uint residue = value.x & mask;
        uint raw = negative ? (0u - residue) & mask : residue;
        int balanced = raw >= basis / 2u ? (int)raw - (int)basis : (int)raw;
        ulong ring = (ulong)flat / p.ring_d;
        ulong coefficient = (ulong)flat % p.ring_d;
        planes[(ring * p.num_digits + digit) * p.ring_d + coefficient] = (short)balanced;
        value = akita_recursive_ushr(value, bits);
        // For negative magnitude, -(ceil(m/B)) + carry; for positive, floor(m/B) + carry.
        bool increment = negative ? residue != 0u && balanced >= 0 : balanced < 0;
        if (increment) { value = akita_recursive_uinc(value); }
    }
}

kernel void akita_fp128_recursive_sparse_fold(
    device const uchar* source [[buffer(0)]],
    device const RecursiveSparseTerm* terms [[buffer(1)]],
    device int* output [[buffer(2)]],
    constant RecursiveFoldParams& p [[buffer(3)]],
    constant ulong& term_count [[buffer(4)]],
    uint flat [[thread_position_in_grid]])
{
    ulong count = p.positions_per_block * p.num_digits * p.ring_d;
    if ((ulong)flat >= count) { return; }
    ulong coefficient = (ulong)flat % p.ring_d;
    ulong digit = ((ulong)flat / p.ring_d) % p.num_digits;
    ulong position = (ulong)flat / (p.ring_d * p.num_digits);
    int accumulator = 0;
    for (ulong t = 0ul; t < term_count; ++t) {
        RecursiveSparseTerm term = terms[t];
        ulong ring = term.block * p.positions_per_block + position;
        if (ring >= p.live_rings) { continue; }
        ulong source_coefficient = (coefficient + p.ring_d - term.shift) % p.ring_d;
        ulong source_flat = ring * p.ring_d + source_coefficient;
        if (source_flat >= p.stored_coefficients) { continue; }
        int value = p.bit_width == 0ul
            ? (int)((device const short*)source)[(ring * p.num_digits + digit) * p.ring_d + source_coefficient]
            : akita_field_packing_digit(source, source_flat, p.bit_width);
        int contribution = value * term.coefficient;
        accumulator += coefficient >= term.shift ? contribution : -contribution;
    }
    output[flat] = accumulator;
}

kernel void akita_recursive_sparse_batch_reduce(
    device const int* inputs [[buffer(0)]],
    device int* output [[buffer(1)]],
    constant ulong& source_count [[buffer(2)]],
    constant ulong& output_count [[buffer(3)]],
    uint flat [[thread_position_in_grid]])
{
    if ((ulong)flat >= output_count) { return; }
    int value = 0;
    for (ulong source = 0ul; source < source_count; ++source) {
        value += inputs[source * output_count + flat];
    }
    output[flat] = value;
}
