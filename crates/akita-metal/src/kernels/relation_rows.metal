struct RelationRowsParams {
    ulong ring_d;
    ulong ntt_size;
    ulong num_rows;
    ulong num_cols;
    ulong matrix_rings;
    ulong num_primes;
    ulong output_coefficients;
    ulong mode;
    ulong rhs_kind;
    ulong rhs_abs_bound;
};

inline void akita_relation_rows_forward(threadgroup int* values,
    device const int* twiddles, uint n, ulong base, uint tid, D512LinearNttPrime prime)
{
    for (uint len = n >> 1u; len != 0u; len >>= 1u) {
        if (tid < n >> 1u) {
            uint block = tid / len;
            uint offset = tid - block * len;
            uint left = block * (len << 1u) + offset;
            uint right = left + len;
            int lhs = values[left];
            int rhs = values[right];
            values[left] = d512_ntt_add(lhs, rhs, prime.p);
            values[right] = d512_ntt_mul(d512_ntt_sub(lhs, rhs, prime.p),
                twiddles[base + (ulong)(len - 1u + offset)], prime);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void akita_fp128_relation_rows_matrix_ntt(
    device const AkitaFp128* matrix [[buffer(0)]],
    device int* matrix_ntt [[buffer(1)]],
    device const D512LinearNttPrime* primes [[buffer(2)]],
    device const int* limb_weights [[buffer(3)]],
    device const int* field_moduli [[buffer(4)]],
    device const int* fwd_twiddles [[buffer(5)]],
    constant RelationRowsParams& params [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]])
{
    threadgroup int values[1024];
    uint prime_index = (uint)((ulong)group.x % params.num_primes);
    ulong ring = (ulong)group.x / params.num_primes;
    D512LinearNttPrime prime = primes[prime_index];
    ulong base = (ulong)prime_index * params.ntt_size;
    for (ulong coefficient = (ulong)tid; coefficient < params.ntt_size; coefficient += 512ul) {
        values[coefficient] = coefficient < params.ring_d
            ? d512_fp128_to_mont(matrix[ring * params.ring_d + coefficient],
                prime_index, prime, limb_weights, field_moduli)
            : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    akita_relation_rows_forward(values, fwd_twiddles, (uint)params.ntt_size, base, tid, prime);
    for (ulong coefficient = (ulong)tid; coefficient < params.ntt_size; coefficient += 512ul) {
        matrix_ntt[((ulong)prime_index * params.matrix_rings + ring) * params.ntt_size + coefficient] = values[coefficient];
    }
}

kernel void akita_fp128_relation_rows_rhs_ntt(
    device const char* rhs_i8 [[buffer(0)]],
    device const int* rhs_i32 [[buffer(1)]],
    device int* rhs_ntt [[buffer(2)]],
    device const D512LinearNttPrime* primes [[buffer(3)]],
    device const int* fwd_twiddles [[buffer(4)]],
    constant RelationRowsParams& params [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]])
{
    threadgroup int values[1024];
    uint prime_index = (uint)((ulong)group.x % params.num_primes);
    ulong column = (ulong)group.x / params.num_primes;
    D512LinearNttPrime prime = primes[prime_index];
    ulong base = (ulong)prime_index * params.ntt_size;
    for (ulong coefficient = (ulong)tid; coefficient < params.ntt_size; coefficient += 512ul) {
        int value = 0;
        if (coefficient < params.ring_d) {
            ulong index = column * params.ring_d + coefficient;
            value = params.rhs_kind == 1ul ? (int)rhs_i8[index] : rhs_i32[index];
        }
        // Full i32 input, including INT_MIN, can span more than one prime.
        int canonical = (int)d512_positive_mod((long)value, (long)prime.p);
        values[coefficient] = d512_ntt_mul(canonical, prime.montsq, prime);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    akita_relation_rows_forward(values, fwd_twiddles, (uint)params.ntt_size, base, tid, prime);
    for (ulong coefficient = (ulong)tid; coefficient < params.ntt_size; coefficient += 512ul) {
        rhs_ntt[((ulong)prime_index * params.num_cols + column) * params.ntt_size + coefficient] = values[coefficient];
    }
}

kernel void akita_fp128_relation_rows_matvec_inverse(
    device const int* matrix_ntt [[buffer(0)]],
    device const int* rhs_ntt [[buffer(1)]],
    device uint* residues [[buffer(2)]],
    device const D512LinearNttPrime* primes [[buffer(3)]],
    device const int* inv_twiddles [[buffer(4)]],
    device const int* d_inv [[buffer(5)]],
    constant RelationRowsParams& params [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]])
{
    threadgroup int values[1024];
    uint prime_index = (uint)((ulong)group.x % params.num_primes);
    ulong row = (ulong)group.x / params.num_primes;
    D512LinearNttPrime prime = primes[prime_index];
    ulong base = (ulong)prime_index * params.ntt_size;
    for (ulong coefficient = (ulong)tid; coefficient < params.ntt_size; coefficient += 512ul) {
        int acc = 0;
        for (ulong column = 0ul; column < params.num_cols; ++column) {
            ulong matrix_ring = row * params.num_cols + column;
            ulong matrix_index = ((ulong)prime_index * params.matrix_rings + matrix_ring) * params.ntt_size + coefficient;
            ulong rhs_index = ((ulong)prime_index * params.num_cols + column) * params.ntt_size + coefficient;
            acc = d512_ntt_add(acc, d512_ntt_mul(matrix_ntt[matrix_index], rhs_ntt[rhs_index], prime), prime.p);
        }
        values[coefficient] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint len = 1u; len < (uint)params.ntt_size; len <<= 1u) {
        if (tid < (uint)params.ntt_size >> 1u) {
            uint block = tid / len;
            uint offset = tid - block * len;
            uint left = block * (len << 1u) + offset;
            uint right = left + len;
            int lhs = values[left];
            int rhs = d512_ntt_mul(values[right], inv_twiddles[base + (ulong)(len - 1u + offset)], prime);
            values[left] = d512_ntt_add(lhs, rhs, prime.p);
            values[right] = d512_ntt_sub(lhs, rhs, prime.p);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (ulong coefficient = (ulong)tid; coefficient < params.ntt_size; coefficient += 512ul) {
        int scaled = d512_ntt_mul(values[coefficient], d_inv[prime_index], prime);
        int canonical = d512_ntt_reduce((long)d512_ntt_mul_raw(scaled, 1, prime), prime.p);
        residues[(row * params.ntt_size + coefficient) * params.num_primes + (ulong)prime_index] = (uint)canonical;
    }
}

inline AkitaFp128 akita_relation_rows_crt(device const uint* residues,
    device const D512LinearNttPrime* primes, device const uint* gamma,
    device const AkitaFp128* products, ulong low, ulong high, uint mode,
    constant RelationRowsParams& params)
{
    long digits[D512_LINEAR_NTT_PRIMES];
    for (uint p = 0u; p < D512_LINEAR_NTT_PRIMES; ++p) {
        long modulus = (long)primes[p].p;
        long h = (long)residues[high * params.num_primes + (ulong)p];
        long residue = h;
        if (mode != 0u) {
            long l = (long)residues[low * params.num_primes + (ulong)p];
            residue = d512_positive_mod(mode == 1u ? l + h : l - h, modulus);
        }
        for (uint prior = 0u; prior < p; ++prior) {
            residue = d512_positive_mod(residue - digits[prior], modulus);
            residue = residue * (long)gamma[p * D512_LINEAR_NTT_PRIMES + prior] % modulus;
        }
        digits[p] = residue > modulus / 2l ? residue - modulus : residue;
    }
    AkitaFp128 value = akita_zero();
    for (uint p = 0u; p < D512_LINEAR_NTT_PRIMES; ++p) {
        value = akita_add(value, akita_mul_signed_small(products[p], digits[p]));
    }
    return value;
}

kernel void akita_fp128_relation_rows_reconstruct(
    device const uint* residues [[buffer(0)]],
    device AkitaFp128* output [[buffer(1)]],
    device const D512LinearNttPrime* primes [[buffer(2)]],
    device const uint* gamma [[buffer(3)]],
    device const AkitaFp128* products [[buffer(4)]],
    constant RelationRowsParams& params [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]])
{
    ulong index = (ulong)group.x * 256ul + (ulong)tid;
    ulong plane_width = params.num_rows * params.ring_d;
    if (index >= plane_width) { return; }
    ulong row = index / params.ring_d;
    ulong coefficient = index % params.ring_d;
    ulong low = row * params.ntt_size + coefficient;
    ulong high = low + params.ring_d;
    output[index] = akita_relation_rows_crt(residues, primes, gamma, products, low, high, (uint)params.mode, params);
    if (params.mode == 2ul) {
        output[plane_width + index] = akita_relation_rows_crt(residues, primes, gamma, products, low, high, 1u, params);
    }
}
