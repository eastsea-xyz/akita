// Canonical Fp128 ordinary-product high halves. One group owns each output;
// lane zero writes once, after an exact canonical-field reduction.
struct AkitaSparseHighHalfTerm {
    ulong challenge_index;
    uint shift;
    int coefficient;
};
struct AkitaSparseHighHalfRow {
    ulong source_start;
    ulong source_end;
    ulong first_ring;
    ulong ring_stride;
    ulong live_challenges;
};
struct AkitaSparseHighHalfParams {
    ulong ring_d;
    ulong batch_count;
    ulong term_count;
};
kernel void akita_fp128_sparse_product_high_half_batch(
    device const AkitaFp128 *sources [[buffer(0)]],
    device const AkitaSparseHighHalfTerm *terms [[buffer(1)]],
    device const AkitaSparseHighHalfRow *rows [[buffer(2)]],
    device const AkitaFp128 *subtract [[buffer(3)]],
    device AkitaFp128 *output [[buffer(4)]],
    constant AkitaSparseHighHalfParams &p [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const ulong flat = ulong(group);
    if (flat >= p.batch_count * p.ring_d) return;
    const ulong coefficient = flat % p.ring_d;
    const AkitaSparseHighHalfRow row = rows[flat / p.ring_d];
    AkitaFp128 accumulator = akita_zero();
    for (ulong term_index = ulong(lane); term_index < p.term_count; term_index += 128) {
        const AkitaSparseHighHalfTerm term = terms[term_index];
        if (term.challenge_index >= row.live_challenges
            || coefficient >= ulong(term.shift) || term.coefficient == 0) continue;
        const ulong ring = row.first_ring + term.challenge_index * row.ring_stride;
        const ulong source = row.source_start + ring * p.ring_d
            + p.ring_d + coefficient - ulong(term.shift);
        accumulator = akita_add(accumulator,
            akita_mul_signed_small(sources[source], long(term.coefficient)));
    }
    threadgroup AkitaFp128 partials[128];
    partials[lane] = accumulator;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint distance = 64; distance != 0; distance >>= 1) {
        if (lane < distance) partials[lane] = akita_add(partials[lane], partials[lane + distance]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) output[flat] = akita_sub(partials[0], subtract[flat]);
}
