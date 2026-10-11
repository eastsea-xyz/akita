// Appended to onehot.metal: use its canonical AkitaFp128 arithmetic.
struct DigitTreeParams {
    ulong explicit_rows;
    ulong pair_count;
    ulong lanes;
    ulong first_len;
    ulong parent_count;
    ulong arity;
    ulong degree;
    ulong workgroups;
};

inline void akita_digit_tree_polynomial_from_lanes(
    thread const AkitaFp128 *left_lanes,
    thread const AkitaFp128 *right_lanes,
    device const AkitaFp128 *polynomial,
    constant DigitTreeParams &params,
    thread AkitaFp128 *result)
{
    for (uint degree = 0; degree < 5; ++degree) result[degree] = akita_zero();
    if (params.arity != 0ul) {
        for (ulong parent = 0; parent < params.parent_count; ++parent) {
            AkitaFp128 product[5];
            for (uint degree = 0; degree < 5; ++degree) product[degree] = akita_zero();
            product[0] = akita_from_u32(1u);
            for (uint lane = 0; lane < params.arity; ++lane) {
                ulong column = parent * params.arity + lane;
                AkitaFp128 left = left_lanes[column];
                AkitaFp128 right = right_lanes[column];
                AkitaFp128 slope = akita_sub(right, left);
                for (int degree = int(lane) + 1; degree >= 0; --degree) {
                    AkitaFp128 value = degree <= int(lane)
                        ? akita_mul(product[degree], left) : akita_zero();
                    if (degree != 0) value = akita_add(value, akita_mul(product[degree - 1], slope));
                    product[degree] = value;
                }
            }
            for (uint degree = 0; degree <= params.degree; ++degree)
                result[degree] = akita_add(result[degree], akita_mul(polynomial[parent], product[degree]));
        }
    } else {
        AkitaFp128 left = left_lanes[0];
        AkitaFp128 right = right_lanes[0];
        AkitaFp128 slope = akita_sub(right, left);
        AkitaFp128 power[5];
        for (uint degree = 0; degree < 5; ++degree) power[degree] = akita_zero();
        power[0] = akita_from_u32(1u);
        for (uint term = 0; term <= params.degree; ++term) {
            for (uint degree = 0; degree <= term; ++degree)
                result[degree] = akita_add(result[degree], akita_mul(polynomial[term], power[degree]));
            if (term != params.degree) {
                for (int degree = int(term) + 1; degree >= 0; --degree) {
                    AkitaFp128 value = degree <= int(term) ? akita_mul(power[degree], left) : akita_zero();
                    if (degree != 0) value = akita_add(value, akita_mul(power[degree - 1], slope));
                    power[degree] = value;
                }
            }
        }
    }
}

inline void akita_digit_tree_pair_polynomial(
    device const AkitaFp128 *source,
    device const AkitaFp128 *padding,
    device const AkitaFp128 *polynomial,
    constant DigitTreeParams &params,
    ulong pair,
    thread AkitaFp128 *result)
{
    AkitaFp128 left[8], right[8];
    for (uint lane = 0; lane < params.lanes; ++lane) {
        left[lane] = source[2ul * pair * params.lanes + lane];
        right[lane] = 2ul * pair + 1ul < params.explicit_rows
            ? source[(2ul * pair + 1ul) * params.lanes + lane] : padding[lane];
    }
    akita_digit_tree_polynomial_from_lanes(left, right, polynomial, params, result);
}

kernel void akita_fp128_digit_tree_partials(
    device const AkitaFp128 *source [[buffer(0)]],
    device const AkitaFp128 *padding [[buffer(1)]],
    device const AkitaFp128 *polynomial [[buffer(2)]],
    device const AkitaFp128 *first [[buffer(3)]],
    device const AkitaFp128 *second [[buffer(4)]],
    device AkitaFp128 *partials [[buffer(5)]],
    constant DigitTreeParams &params [[buffer(6)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]])
{
    threadgroup AkitaFp128 sums[40];
    AkitaFp128 accumulated[5];
    for (uint degree = 0; degree < 5; ++degree) accumulated[degree] = akita_zero();
    for (ulong pair = ulong(group.x) * 256ul + thread_index;
         pair < params.pair_count; pair += params.workgroups * 256ul) {
        AkitaFp128 row[5];
        akita_digit_tree_pair_polynomial(source, padding, polynomial, params, pair, row);
        AkitaFp128 weight = akita_mul(first[pair % params.first_len], second[pair / params.first_len]);
        for (uint degree = 0; degree < 5; ++degree)
            accumulated[degree] = akita_add(accumulated[degree], akita_mul(weight, row[degree]));
    }
    for (uint degree = 0; degree < 5; ++degree) {
        AkitaFp128 sum = akita_simd_sum_fp128(accumulated[degree]);
        if (simd_lane == 0) sums[degree * 8u + simd_index] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index < 5u) {
        AkitaFp128 sum = akita_zero();
        for (uint simd = 0; simd < 8; ++simd) sum = akita_add(sum, sums[thread_index * 8u + simd]);
        partials[ulong(group.x) * 5ul + thread_index] = sum;
    }
}

kernel void akita_fp128_digit_tree_reduce(
    device const AkitaFp128 *partials [[buffer(0)]],
    device AkitaFp128 *output [[buffer(1)]],
    constant DigitTreeParams &params [[buffer(2)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]])
{
    threadgroup AkitaFp128 sums[8];
    AkitaFp128 value = akita_zero();
    for (ulong partial = thread_index; partial < params.workgroups; partial += 256ul)
        value = akita_add(value, partials[partial * 5ul + group.x]);
    value = akita_simd_sum_fp128(value);
    if (simd_lane == 0) sums[simd_index] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0) {
        AkitaFp128 sum = akita_zero();
        for (uint simd = 0; simd < 8; ++simd) sum = akita_add(sum, sums[simd]);
        output[group.x] = sum;
    }
}

kernel void akita_fp128_digit_tree_bind(
    device const AkitaFp128 *source [[buffer(0)]],
    device const AkitaFp128 *padding [[buffer(1)]],
    device AkitaFp128 *destination [[buffer(2)]],
    constant DigitTreeParams &params [[buffer(3)]],
    constant AkitaFp128 &challenge [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    ulong next_rows = (params.explicit_rows + 1ul) / 2ul;
    if (ulong(index) >= next_rows * params.lanes) return;
    ulong pair = ulong(index) / params.lanes;
    ulong lane = ulong(index) % params.lanes;
    AkitaFp128 left = source[2ul * pair * params.lanes + lane];
    AkitaFp128 right = 2ul * pair + 1ul < params.explicit_rows
        ? source[(2ul * pair + 1ul) * params.lanes + lane] : padding[lane];
    destination[index] = akita_add(left, akita_mul(challenge, akita_sub(right, left)));
}

struct CompactDigitTreeParams {
    DigitTreeParams table;
    ulong pair_lut_count;
    ulong phase;
    ulong use_quartet_lut;
};

static_assert(sizeof(DigitTreeParams) == 64, "digit-tree table params ABI");
static_assert(alignof(DigitTreeParams) == 8, "digit-tree table params alignment");
static_assert(sizeof(CompactDigitTreeParams) == 88, "compact digit-tree params ABI");
static_assert(alignof(CompactDigitTreeParams) == 8, "compact digit-tree params alignment");

kernel void akita_fp128_digit_tree_compact_partials(
    device const ushort *indices [[buffer(0)]],
    device const AkitaFp128 *folded [[buffer(1)]],
    device const AkitaFp128 *padding [[buffer(2)]],
    device const AkitaFp128 *polynomial [[buffer(3)]],
    device const AkitaFp128 *coefficients [[buffer(4)]],
    device const AkitaFp128 *first [[buffer(5)]],
    device const AkitaFp128 *second [[buffer(6)]],
    device AkitaFp128 *partials [[buffer(7)]],
    constant CompactDigitTreeParams &compact [[buffer(8)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]])
{
    constant DigitTreeParams &params = compact.table;
    threadgroup AkitaFp128 sums[40];
    AkitaFp128 accumulated[5];
    for (uint degree = 0; degree < 5; ++degree) accumulated[degree] = akita_zero();
    for (ulong pair = ulong(group.x) * 256ul + thread_index;
         pair < params.pair_count; pair += params.workgroups * 256ul) {
        AkitaFp128 row[5];
        if (compact.phase == 0ul) {
            ulong index = ulong(indices[pair]) * 5ul;
            for (uint degree = 0; degree < 5; ++degree) row[degree] = coefficients[index + degree];
        } else {
            ulong left_index = indices[2ul * pair];
            ulong right_index = 2ul * pair + 1ul < params.explicit_rows ? indices[2ul * pair + 1ul] : 0ul;
            if (compact.use_quartet_lut != 0ul) {
                ulong index = (left_index * compact.pair_lut_count + right_index) * 5ul;
                for (uint degree = 0; degree < 5; ++degree) row[degree] = coefficients[index + degree];
            } else {
                AkitaFp128 left[8], right[8];
                for (uint lane = 0; lane < params.lanes; ++lane) {
                    left[lane] = folded[left_index * params.lanes + lane];
                    right[lane] = folded[right_index * params.lanes + lane];
                }
                akita_digit_tree_polynomial_from_lanes(left, right, polynomial, params, row);
            }
        }
        AkitaFp128 weight = akita_mul(first[pair % params.first_len], second[pair / params.first_len]);
        for (uint degree = 0; degree < 5; ++degree)
            accumulated[degree] = akita_add(accumulated[degree], akita_mul(weight, row[degree]));
    }
    for (uint degree = 0; degree < 5; ++degree) {
        AkitaFp128 sum = akita_simd_sum_fp128(accumulated[degree]);
        if (simd_lane == 0) sums[degree * 8u + simd_index] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index < 5u) {
        AkitaFp128 sum = akita_zero();
        for (uint simd = 0; simd < 8; ++simd) sum = akita_add(sum, sums[thread_index * 8u + simd]);
        partials[ulong(group.x) * 5ul + thread_index] = sum;
    }
}

kernel void akita_fp128_digit_tree_compact_materialize(
    device const ushort *indices [[buffer(0)]],
    device const AkitaFp128 *folded [[buffer(1)]],
    device AkitaFp128 *destination [[buffer(2)]],
    constant CompactDigitTreeParams &compact [[buffer(3)]],
    constant AkitaFp128 &challenge [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    constant DigitTreeParams &params = compact.table;
    ulong next_rows = (params.explicit_rows + 1ul) / 2ul;
    if (ulong(index) >= next_rows * params.lanes) return;
    ulong row = ulong(index) / params.lanes, lane = ulong(index) % params.lanes;
    ulong left_index = indices[2ul * row];
    ulong right_index = 2ul * row + 1ul < params.explicit_rows ? indices[2ul * row + 1ul] : 0ul;
    AkitaFp128 left = folded[left_index * params.lanes + lane];
    AkitaFp128 right = folded[right_index * params.lanes + lane];
    destination[index] = akita_add(left, akita_mul(challenge, akita_sub(right, left)));
}
