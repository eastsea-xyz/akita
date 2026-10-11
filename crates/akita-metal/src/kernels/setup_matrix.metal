// Canonical SHAKE256 page expansion. uint2 holds one 64-bit lane [lo, hi].
constant uint AKITA_SETUP_KECCAK_RHO[24] = {
    1u, 3u, 6u, 10u, 15u, 21u, 28u, 36u, 45u, 55u, 2u, 14u,
    27u, 41u, 56u, 8u, 25u, 43u, 62u, 18u, 39u, 61u, 20u, 44u
};
constant uint AKITA_SETUP_KECCAK_PI[24] = {
    10u, 7u, 11u, 17u, 18u, 3u, 5u, 16u, 8u, 21u, 24u, 4u,
    15u, 23u, 19u, 13u, 12u, 2u, 20u, 14u, 22u, 9u, 6u, 1u
};
constant uint2 AKITA_SETUP_KECCAK_RC[24] = {
    uint2(0x00000001u, 0x00000000u), uint2(0x00008082u, 0x00000000u),
    uint2(0x0000808au, 0x80000000u), uint2(0x80008000u, 0x80000000u),
    uint2(0x0000808bu, 0x00000000u), uint2(0x80000001u, 0x00000000u),
    uint2(0x80008081u, 0x80000000u), uint2(0x00008009u, 0x80000000u),
    uint2(0x0000008au, 0x00000000u), uint2(0x00000088u, 0x00000000u),
    uint2(0x80008009u, 0x00000000u), uint2(0x8000000au, 0x00000000u),
    uint2(0x8000808bu, 0x00000000u), uint2(0x0000008bu, 0x80000000u),
    uint2(0x00008089u, 0x80000000u), uint2(0x00008003u, 0x80000000u),
    uint2(0x00008002u, 0x80000000u), uint2(0x00000080u, 0x80000000u),
    uint2(0x0000800au, 0x00000000u), uint2(0x8000000au, 0x80000000u),
    uint2(0x80008081u, 0x80000000u), uint2(0x00008080u, 0x80000000u),
    uint2(0x80000001u, 0x00000000u), uint2(0x80008008u, 0x80000000u)
};

inline uint2 akita_setup_rotl(uint2 value, uint shift) {
    if (shift == 0u) return value;
    if (shift >= 32u) {
        value = value.yx;
        shift -= 32u;
        if (shift == 0u) return value;
    }
    return uint2((value.x << shift) | (value.y >> (32u - shift)),
                 (value.y << shift) | (value.x >> (32u - shift)));
}

inline void akita_setup_keccak_f(thread uint2* state) {
    for (uint round = 0u; round < 24u; ++round) {
        uint2 column[5];
        #pragma unroll
        for (uint x = 0u; x < 5u; ++x) {
            column[x] = uint2(0u);
            #pragma unroll
            for (uint y = 0u; y < 5u; ++y) column[x] ^= state[5u * y + x];
        }
        #pragma unroll
        for (uint x = 0u; x < 5u; ++x) {
            uint2 delta = column[(x + 4u) % 5u]
                ^ akita_setup_rotl(column[(x + 1u) % 5u], 1u);
            #pragma unroll
            for (uint y = 0u; y < 5u; ++y) state[5u * y + x] ^= delta;
        }
        uint2 last = state[1];
        #pragma unroll
        for (uint i = 0u; i < 24u; ++i) {
            uint position = AKITA_SETUP_KECCAK_PI[i];
            uint2 next = state[position];
            state[position] = akita_setup_rotl(last, AKITA_SETUP_KECCAK_RHO[i]);
            last = next;
        }
        #pragma unroll
        for (uint y = 0u; y < 5u; ++y) {
            #pragma unroll
            for (uint x = 0u; x < 5u; ++x) column[x] = state[5u * y + x];
            #pragma unroll
            for (uint x = 0u; x < 5u; ++x) {
                state[5u * y + x] = column[x]
                    ^ (~column[(x + 1u) % 5u] & column[(x + 2u) % 5u]);
            }
        }
        state[0] ^= AKITA_SETUP_KECCAK_RC[round];
    }
}

struct SetupMatrixParams {
    uint first_page_lo;
    uint first_page_hi;
    uint total_fields;
    uint page_count;
    uint draw_limit;
    uint reserved;
};

inline uchar akita_setup_input_byte(
    device const uchar* message, uint position, uint2 page)
{
    // The page's eight LE bytes straddle absorb blocks: offsets 269..276.
    if (position >= 269u && position < 277u) {
        uint offset = position - 269u;
        uint word = offset < 4u ? page.x : page.y;
        return (uchar)(word >> (8u * (offset & 3u)));
    }
    if (position < 277u) return message[position];
    if (position == 277u) return 0x1fu;
    if (position == 407u) return 0x80u;
    return 0u;
}

inline uint2 akita_setup_squeeze_lane(thread uint2* state, thread uint& cursor) {
    if (cursor == 17u) {
        akita_setup_keccak_f(state);
        cursor = 0u;
    }
    return state[cursor++];
}

inline bool akita_setup_accept_candidate(
    uint4 candidate, device AkitaFp128* output, uint wanted, thread uint& accepted)
{
    // p = ffffffffffffffffffffffff00005809. Reject, never reduce, >= p.
    bool rejected = candidate.y == 0xffffffffu && candidate.z == 0xffffffffu
        && candidate.w == 0xffffffffu && candidate.x >= 0x00005809u;
    if (!rejected) {
        output[accepted].limb = candidate;
        ++accepted;
    }
    return accepted == wanted;
}

kernel void akita_fp128_setup_matrix(
    device const uchar* message [[buffer(0)]],
    device AkitaFp128* output [[buffer(1)]],
    device uint2* status [[buffer(2)]],
    constant SetupMatrixParams& p [[buffer(3)]],
    uint local_page [[thread_position_in_grid]])
{
    if (local_page >= p.page_count) return;
    uint page_low = p.first_page_lo + local_page;
    uint2 page = uint2(page_low, p.first_page_hi + (page_low < p.first_page_lo ? 1u : 0u));
    uint2 state[25];
    #pragma unroll
    for (uint i = 0u; i < 25u; ++i) state[i] = uint2(0u);
    for (uint block = 0u; block < 3u; ++block) {
        #pragma unroll
        for (uint lane = 0u; lane < 17u; ++lane) {
            uint2 value = uint2(0u);
            #pragma unroll
            for (uint byte = 0u; byte < 8u; ++byte) {
                uint part = (uint)akita_setup_input_byte(message, block * 136u + lane * 8u + byte, page);
                if (byte < 4u) value.x |= part << (byte * 8u);
                else value.y |= part << ((byte - 4u) * 8u);
            }
            state[lane] ^= value;
        }
        akita_setup_keccak_f(state);
    }
    uint output_start = local_page * 4096u;
    uint wanted = min(4096u, p.total_fields - output_start);
    uint accepted = 0u;
    uint draws = 0u;
    uint cursor = 0u;
    while (draws < p.draw_limit && accepted < wanted) {
        uint2 low = akita_setup_squeeze_lane(state, cursor);
        uint2 high = akita_setup_squeeze_lane(state, cursor);
        ++draws;
        if (akita_setup_accept_candidate(uint4(low.x, low.y, high.x, high.y),
                output + output_start, wanted, accepted)) break;
    }
    status[local_page] = uint2(accepted, draws);
}

// A small primitive oracle uses the same rejection/output routine, while a
// scripted stream exercises paths that natural 128-bit SHAKE pages rarely hit.
kernel void akita_fp128_setup_matrix_sampling_oracle(
    device const AkitaFp128* candidates [[buffer(0)]],
    device AkitaFp128* output [[buffer(1)]],
    device uint2* status [[buffer(2)]],
    constant uint2& p [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index != 0u) return;
    uint accepted = 0u;
    uint consumed = 0u;
    while (consumed < p.x && accepted < p.y) {
        uint4 candidate = candidates[consumed++].limb;
        if (akita_setup_accept_candidate(candidate, output, p.y, accepted)) break;
    }
    status[0] = uint2(accepted, consumed);
}
