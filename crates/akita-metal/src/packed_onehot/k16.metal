#include <metal_stdlib>
using namespace metal;

// The ABI is deliberately separate from the legacy K256 fold parameters.
struct K16FoldParams {
    ulong num_columns;
    ulong num_positions;
    ulong position_start;
    ulong position_count;
    ulong blocks_per_column;
    ulong ring_dimension;
    ulong rows_per_ring;
    ulong zero_column_mask;
    ulong tiled_source;
};

// Each group owns a ring position and sums signed challenge shifts in
// Z[X]/(X^D+1). Host admission bounds every intermediate integer sum by i32.
kernel void akita_k16_decompose_fold(
    device const uchar* lanes [[buffer(0)]],
    device const ulong* active_zero_rows [[buffer(1)]],
    device const uint* challenge_offsets [[buffer(2)]],
    device const ushort* challenge_positions [[buffer(3)]],
    device const char* challenge_coefficients [[buffer(4)]],
    device int* output [[buffer(5)]],
    constant K16FoldParams& params [[buffer(6)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint local_position [[threadgroup_position_in_grid]])
{
    threadgroup atomic_int accumulators[512];
    uint d = (uint)params.ring_dimension;
    for (uint coefficient = thread_index; coefficient < d; coefficient += 256u)
        atomic_store_explicit(&accumulators[coefficient], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ulong position = params.position_start + (ulong)local_position;
    ulong tasks_per_block = params.num_columns * params.rows_per_ring;
    ulong tasks = params.blocks_per_column * tasks_per_block;
    for (ulong task = (ulong)thread_index; task < tasks; task += 256ul) {
        ulong block = task / tasks_per_block;
        ulong local = task % tasks_per_block;
        ulong column = local / params.rows_per_ring;
        ulong row_in_ring = local % params.rows_per_ring;
        ulong row = (block * params.num_positions + position) * params.rows_per_ring
            + row_in_ring;
        ulong source_row = params.tiled_source == 0ul ? row
            : (block * params.position_count + (ulong)local_position) * params.rows_per_ring
                + row_in_ring;
        uint hot = lanes[source_row * params.num_columns + column];
        bool committed = hot != 0u;
        if (!committed && ((params.zero_column_mask >> column) & 1ul) != 0ul)
            committed = ((active_zero_rows[row >> 6ul] >> (row & 63ul)) & 1ul) != 0ul;
        if (!committed) continue;

        uint source_coefficient = (uint)row_in_ring * 16u + hot;
        ulong challenge = column * params.blocks_per_column + block;
        for (uint term = challenge_offsets[challenge]; term < challenge_offsets[challenge + 1ul]; term++) {
            uint destination = source_coefficient + (uint)challenge_positions[term];
            int value = (int)challenge_coefficients[term];
            if (destination >= d) {
                destination -= d;
                value = -value;
            }
            atomic_fetch_add_explicit(&accumulators[destination], value, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint coefficient = thread_index; coefficient < d; coefficient += 256u)
        output[(ulong)local_position * params.ring_dimension + coefficient]
            = atomic_load_explicit(&accumulators[coefficient], memory_order_relaxed);
}
