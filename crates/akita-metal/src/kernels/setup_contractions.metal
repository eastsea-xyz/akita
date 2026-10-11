struct SetupContractionParams {
    ulong ring_d;
    ulong matrix_columns;
    ulong first_column;
    ulong column_count;
    ulong selected_rows;
    ulong batch_count;
    ulong output_count;
};

kernel void akita_fp128_setup_column_contractions(
    device const AkitaFp128* matrix [[buffer(0)]],
    device const ulong* row_ids [[buffer(1)]],
    device const AkitaFp128* row_weights [[buffer(2)]],
    device const AkitaFp128* alpha_powers [[buffer(3)]],
    device AkitaFp128* output [[buffer(4)]],
    constant SetupContractionParams& p [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup AkitaFp128 reduction[256];
    ulong index = (ulong)group.x;
    ulong column = p.first_column + index / p.batch_count;
    ulong batch = index % p.batch_count;
    ulong terms = p.selected_rows * p.ring_d;
    AkitaFp128 accumulator = akita_zero();
    for (ulong term = lane; term < terms; term += 256ul) {
        ulong row_slot = term / p.ring_d;
        ulong coefficient = term % p.ring_d;
        AkitaFp128 weight = row_weights[row_slot * p.batch_count + batch];
        ulong source = (row_ids[row_slot] * p.matrix_columns + column) * p.ring_d + coefficient;
        AkitaFp128 evaluated = akita_mul(matrix[source], alpha_powers[coefficient]);
        accumulator = akita_add(accumulator, akita_mul(weight, evaluated));
    }
    akita_field_packing_reduce(reduction, lane, accumulator, output, index);
}
