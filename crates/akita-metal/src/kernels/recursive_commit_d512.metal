// Shared-spectrum D256/D512 path: one 512-thread group owns one block and prime.
// Inactive threads at D256 participate in every barrier without touching source,
// matrix, twiddle, or output buffers. D64/D128 keep their optimized SIMD path.
kernel void akita_fp128_recursive_commit_d512_matvec(
    device const char *digits [[buffer(0)]],
    device const int *matrix_ntt [[buffer(1)]],
    device uint *residues [[buffer(2)]],
    device const D512LinearNttPrime *primes [[buffer(3)]],
    device const int *fwd_twiddles [[buffer(4)]],
    device const int *inv_twiddles [[buffer(5)]],
    device const int *psi_pows [[buffer(6)]],
    device const int *inverse_scale [[buffer(7)]],
    constant RecursiveCommitParams &params [[buffer(8)]],
    constant PackedRecursiveCommitParams &packed [[buffer(9)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_index [[threadgroup_position_in_grid]])
{
    threadgroup int values[512];
    int accumulators[RECURSIVE_COMMIT_MAX_ROWS];
    for (uint row = 0u; row < RECURSIVE_COMMIT_MAX_ROWS; ++row) {
        accumulators[row] = 0;
    }
    ulong group = (ulong)threadgroup_index.x;
    uint prime_index = (uint)(group % params.num_primes);
    ulong block = group / params.num_primes;
    uint ring_d = (uint)params.ring_d;
    bool active = thread_index < ring_d;
    D512LinearNttPrime prime = primes[prime_index];
    ulong table_base = (ulong)prime_index * params.ring_d;

    for (ulong column = 0ul; column < params.num_cols; ++column) {
        if (active) {
            ulong source_index = (block * params.num_cols + column) * params.ring_d
                + (ulong)thread_index;
            values[thread_index] = d512_ntt_mul(
                d512_i32_to_mont(akita_packed_recursive_commit_digit(digits, params, packed, source_index), prime),
                psi_pows[table_base + (ulong)thread_index], prime);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // DIF forward transform matches the cached matrix's spectral order.
        for (uint len = ring_d >> 1u; len != 0u; len >>= 1u) {
            if (thread_index < ring_d >> 1u) {
                uint butterfly_block = thread_index / len;
                uint offset = thread_index - butterfly_block * len;
                uint left = butterfly_block * (len << 1u) + offset;
                uint right = left + len;
                int lhs = values[left];
                int rhs = values[right];
                int twiddle = fwd_twiddles[table_base + (ulong)(len - 1u + offset)];
                values[left] = d512_ntt_add(lhs, rhs, prime.p);
                values[right] = d512_ntt_mul(d512_ntt_sub(lhs, rhs, prime.p), twiddle, prime);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (active) {
            int rhs = values[thread_index];
            for (uint row = 0u; row < (uint)params.num_rows; ++row) {
                ulong matrix_ring = (ulong)row * params.num_cols + column;
                ulong matrix_index = ((ulong)prime_index * params.matrix_rings + matrix_ring)
                    * params.ring_d + (ulong)thread_index;
                accumulators[row] = d512_ntt_add(accumulators[row],
                    d512_ntt_mul(matrix_ntt[matrix_index], rhs, prime), prime.p);
            }
        }
        // Every thread reaches this barrier before the shared spectrum changes.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint row = 0u; row < (uint)params.num_rows; ++row) {
        if (active) { values[thread_index] = accumulators[row]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // DIT inverse consumes the same bit-reversed spectral order.
        for (uint len = 1u; len < ring_d; len <<= 1u) {
            if (thread_index < ring_d >> 1u) {
                uint butterfly_block = thread_index / len;
                uint offset = thread_index - butterfly_block * len;
                uint left = butterfly_block * (len << 1u) + offset;
                uint right = left + len;
                int lhs = values[left];
                int rhs = d512_ntt_mul(values[right],
                    inv_twiddles[table_base + (ulong)(len - 1u + offset)], prime);
                values[left] = d512_ntt_add(lhs, rhs, prime.p);
                values[right] = d512_ntt_sub(lhs, rhs, prime.p);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (active) {
            int scaled = d512_ntt_mul(values[thread_index],
                inverse_scale[table_base + (ulong)thread_index], prime);
            int canonical = d512_ntt_mul_raw(scaled, 1, prime);
            canonical = d512_ntt_reduce((long)canonical, prime.p);
            ulong coefficient = (block * params.num_rows + (ulong)row) * params.ring_d
                + (ulong)thread_index;
            // Existing CRT reconstruct uses coefficient-major, interleaved primes.
            residues[coefficient * params.num_primes + (ulong)prime_index] = (uint)canonical;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
