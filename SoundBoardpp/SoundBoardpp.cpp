#define MINIMP3_IMPLEMENTATION
#define MINIMP3_NO_STDIO
#ifndef MINIMP3_H
#define MINIMP3_H
/*
    https://github.com/lieff/minimp3
    To the extent possible under law, the author(s) have dedicated all copyright and related and neighboring rights to this software to the public domain worldwide.
    This software is distributed without any warranty.
    See <http://creativecommons.org/publicdomain/zero/1.0/>.
*/
#include <stdint.h>

#define MINIMP3_MAX_SAMPLES_PER_FRAME (1152*2)

typedef struct
{
    int frame_bytes, frame_offset, channels, hz, layer, bitrate_kbps;
} mp3dec_frame_info_t;

typedef struct
{
    float mdct_overlap[2][9*32], qmf_state[15*2*32];
    int reserv, free_format_bytes;
    unsigned char header[4], reserv_buf[511];
} mp3dec_t;

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

void mp3dec_init(mp3dec_t *dec);
#ifndef MINIMP3_FLOAT_OUTPUT
typedef int16_t mp3d_sample_t;
#else /* MINIMP3_FLOAT_OUTPUT */
typedef float mp3d_sample_t;
void mp3dec_f32_to_s16(const float *in, int16_t *out, int num_samples);
#endif /* MINIMP3_FLOAT_OUTPUT */
int mp3dec_decode_frame(mp3dec_t *dec, const uint8_t *mp3, int mp3_bytes, mp3d_sample_t *pcm, mp3dec_frame_info_t *info);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* MINIMP3_H */
#if defined(MINIMP3_IMPLEMENTATION) && !defined(_MINIMP3_IMPLEMENTATION_GUARD)
#define _MINIMP3_IMPLEMENTATION_GUARD

#include <stdlib.h>
#include <string.h>

#define MAX_FREE_FORMAT_FRAME_SIZE  2304    /* more than ISO spec's */
#ifndef MAX_FRAME_SYNC_MATCHES
#define MAX_FRAME_SYNC_MATCHES      10
#endif /* MAX_FRAME_SYNC_MATCHES */

#define MAX_L3_FRAME_PAYLOAD_BYTES  MAX_FREE_FORMAT_FRAME_SIZE /* MUST be >= 320000/8/32000*1152 = 1440 */

#define MAX_BITRESERVOIR_BYTES      511
#define SHORT_BLOCK_TYPE            2
#define STOP_BLOCK_TYPE             3
#define MODE_MONO                   3
#define MODE_JOINT_STEREO           1
#define HDR_SIZE                    4
#define HDR_IS_MONO(h)              (((h[3]) & 0xC0) == 0xC0)
#define HDR_IS_MS_STEREO(h)         (((h[3]) & 0xE0) == 0x60)
#define HDR_IS_FREE_FORMAT(h)       (((h[2]) & 0xF0) == 0)
#define HDR_IS_CRC(h)               (!((h[1]) & 1))
#define HDR_TEST_PADDING(h)         ((h[2]) & 0x2)
#define HDR_TEST_MPEG1(h)           ((h[1]) & 0x8)
#define HDR_TEST_NOT_MPEG25(h)      ((h[1]) & 0x10)
#define HDR_TEST_I_STEREO(h)        ((h[3]) & 0x10)
#define HDR_TEST_MS_STEREO(h)       ((h[3]) & 0x20)
#define HDR_GET_STEREO_MODE(h)      (((h[3]) >> 6) & 3)
#define HDR_GET_STEREO_MODE_EXT(h)  (((h[3]) >> 4) & 3)
#define HDR_GET_LAYER(h)            (((h[1]) >> 1) & 3)
#define HDR_GET_BITRATE(h)          ((h[2]) >> 4)
#define HDR_GET_SAMPLE_RATE(h)      (((h[2]) >> 2) & 3)
#define HDR_GET_MY_SAMPLE_RATE(h)   (HDR_GET_SAMPLE_RATE(h) + (((h[1] >> 3) & 1) + ((h[1] >> 4) & 1))*3)
#define HDR_IS_FRAME_576(h)         ((h[1] & 14) == 2)
#define HDR_IS_LAYER_1(h)           ((h[1] & 6) == 6)

#define BITS_DEQUANTIZER_OUT        -1
#define MAX_SCF                     (255 + BITS_DEQUANTIZER_OUT*4 - 210)
#define MAX_SCFI                    ((MAX_SCF + 3) & ~3)

#define MINIMP3_MIN(a, b)           ((a) > (b) ? (b) : (a))
#define MINIMP3_MAX(a, b)           ((a) < (b) ? (b) : (a))

#if !defined(MINIMP3_NO_SIMD)

#if !defined(MINIMP3_ONLY_SIMD) && (defined(_M_X64) || defined(__x86_64__) || defined(__aarch64__) || defined(_M_ARM64))
/* x64 always have SSE2, arm64 always have neon, no need for generic code */
#define MINIMP3_ONLY_SIMD
#endif /* SIMD checks... */

#if (defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))) || ((defined(__i386__) || defined(__x86_64__)) && defined(__SSE2__))
#if defined(_MSC_VER)
#include <intrin.h>
#endif /* defined(_MSC_VER) */
#include <immintrin.h>
#define HAVE_SSE 1
#define HAVE_SIMD 1
#define VSTORE _mm_storeu_ps
#define VLD _mm_loadu_ps
#define VSET _mm_set1_ps
#define VADD _mm_add_ps
#define VSUB _mm_sub_ps
#define VMUL _mm_mul_ps
#define VMAC(a, x, y) _mm_add_ps(a, _mm_mul_ps(x, y))
#define VMSB(a, x, y) _mm_sub_ps(a, _mm_mul_ps(x, y))
#define VMUL_S(x, s)  _mm_mul_ps(x, _mm_set1_ps(s))
#define VREV(x) _mm_shuffle_ps(x, x, _MM_SHUFFLE(0, 1, 2, 3))
typedef __m128 f4;
#if defined(_MSC_VER) || defined(MINIMP3_ONLY_SIMD)
#define minimp3_cpuid __cpuid
#else /* defined(_MSC_VER) || defined(MINIMP3_ONLY_SIMD) */
static __inline__ __attribute__((always_inline)) void minimp3_cpuid(int CPUInfo[], const int InfoType)
{
#if defined(__PIC__)
    __asm__ __volatile__(
#if defined(__x86_64__)
        "push %%rbx\n"
        "cpuid\n"
        "xchgl %%ebx, %1\n"
        "pop  %%rbx\n"
#else /* defined(__x86_64__) */
        "xchgl %%ebx, %1\n"
        "cpuid\n"
        "xchgl %%ebx, %1\n"
#endif /* defined(__x86_64__) */
        : "=a" (CPUInfo[0]), "=r" (CPUInfo[1]), "=c" (CPUInfo[2]), "=d" (CPUInfo[3])
        : "a" (InfoType));
#else /* defined(__PIC__) */
    __asm__ __volatile__(
        "cpuid"
        : "=a" (CPUInfo[0]), "=b" (CPUInfo[1]), "=c" (CPUInfo[2]), "=d" (CPUInfo[3])
        : "a" (InfoType));
#endif /* defined(__PIC__)*/
}
#endif /* defined(_MSC_VER) || defined(MINIMP3_ONLY_SIMD) */
static int have_simd(void)
{
#ifdef MINIMP3_ONLY_SIMD
    return 1;
#else /* MINIMP3_ONLY_SIMD */
    static int g_have_simd;
    int CPUInfo[4];
#ifdef MINIMP3_TEST
    static int g_counter;
    if (g_counter++ > 100)
        return 0;
#endif /* MINIMP3_TEST */
    if (g_have_simd)
        goto end;
    minimp3_cpuid(CPUInfo, 0);
    g_have_simd = 1;
    if (CPUInfo[0] > 0)
    {
        minimp3_cpuid(CPUInfo, 1);
        g_have_simd = (CPUInfo[3] & (1 << 26)) + 1; /* SSE2 */
    }
end:
    return g_have_simd - 1;
#endif /* MINIMP3_ONLY_SIMD */
}
#elif defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define HAVE_SSE 0
#define HAVE_SIMD 1
#define VSTORE vst1q_f32
#define VLD vld1q_f32
#define VSET vmovq_n_f32
#define VADD vaddq_f32
#define VSUB vsubq_f32
#define VMUL vmulq_f32
#define VMAC(a, x, y) vmlaq_f32(a, x, y)
#define VMSB(a, x, y) vmlsq_f32(a, x, y)
#define VMUL_S(x, s)  vmulq_f32(x, vmovq_n_f32(s))
#define VREV(x) vcombine_f32(vget_high_f32(vrev64q_f32(x)), vget_low_f32(vrev64q_f32(x)))
typedef float32x4_t f4;
static int have_simd()
{   /* TODO: detect neon for !MINIMP3_ONLY_SIMD */
    return 1;
}
#else /* SIMD checks... */
#define HAVE_SSE 0
#define HAVE_SIMD 0
#ifdef MINIMP3_ONLY_SIMD
#error MINIMP3_ONLY_SIMD used, but SSE/NEON not enabled
#endif /* MINIMP3_ONLY_SIMD */
#endif /* SIMD checks... */
#else /* !defined(MINIMP3_NO_SIMD) */
#define HAVE_SIMD 0
#endif /* !defined(MINIMP3_NO_SIMD) */

#if defined(__ARM_ARCH) && (__ARM_ARCH >= 6) && !defined(__aarch64__) && !defined(_M_ARM64)
#define HAVE_ARMV6 1
static __inline__ __attribute__((always_inline)) int32_t minimp3_clip_int16_arm(int32_t a)
{
    int32_t x = 0;
    __asm__ ("ssat %0, #16, %1" : "=r"(x) : "r"(a));
    return x;
}
#else
#define HAVE_ARMV6 0
#endif

typedef struct
{
    const uint8_t *buf;
    int pos, limit;
} bs_t;

typedef struct
{
    float scf[3*64];
    uint8_t total_bands, stereo_bands, bitalloc[64], scfcod[64];
} L12_scale_info;

typedef struct
{
    uint8_t tab_offset, code_tab_width, band_count;
} L12_subband_alloc_t;

typedef struct
{
    const uint8_t *sfbtab;
    uint16_t part_23_length, big_values, scalefac_compress;
    uint8_t global_gain, block_type, mixed_block_flag, n_long_sfb, n_short_sfb;
    uint8_t table_select[3], region_count[3], subblock_gain[3];
    uint8_t preflag, scalefac_scale, count1_table, scfsi;
} L3_gr_info_t;

typedef struct
{
    bs_t bs;
    uint8_t maindata[MAX_BITRESERVOIR_BYTES + MAX_L3_FRAME_PAYLOAD_BYTES];
    L3_gr_info_t gr_info[4];
    float grbuf[2][576], scf[40], syn[18 + 15][2*32];
    uint8_t ist_pos[2][39];
} mp3dec_scratch_t;

static void bs_init(bs_t *bs, const uint8_t *data, int bytes)
{
    bs->buf   = data;
    bs->pos   = 0;
    bs->limit = bytes*8;
}

static uint32_t get_bits(bs_t *bs, int n)
{
    uint32_t next, cache = 0, s = bs->pos & 7;
    int shl = n + s;
    const uint8_t *p = bs->buf + (bs->pos >> 3);
    if ((bs->pos += n) > bs->limit)
        return 0;
    next = *p++ & (255 >> s);
    while ((shl -= 8) > 0)
    {
        cache |= next << shl;
        next = *p++;
    }
    return cache | (next >> -shl);
}

static int hdr_valid(const uint8_t *h)
{
    return h[0] == 0xff &&
        ((h[1] & 0xF0) == 0xf0 || (h[1] & 0xFE) == 0xe2) &&
        (HDR_GET_LAYER(h) != 0) &&
        (HDR_GET_BITRATE(h) != 15) &&
        (HDR_GET_SAMPLE_RATE(h) != 3);
}

static int hdr_compare(const uint8_t *h1, const uint8_t *h2)
{
    return hdr_valid(h2) &&
        ((h1[1] ^ h2[1]) & 0xFE) == 0 &&
        ((h1[2] ^ h2[2]) & 0x0C) == 0 &&
        !(HDR_IS_FREE_FORMAT(h1) ^ HDR_IS_FREE_FORMAT(h2));
}

static unsigned hdr_bitrate_kbps(const uint8_t *h)
{
    static const uint8_t halfrate[2][3][15] = {
        { { 0,4,8,12,16,20,24,28,32,40,48,56,64,72,80 }, { 0,4,8,12,16,20,24,28,32,40,48,56,64,72,80 }, { 0,16,24,28,32,40,48,56,64,72,80,88,96,112,128 } },
        { { 0,16,20,24,28,32,40,48,56,64,80,96,112,128,160 }, { 0,16,24,28,32,40,48,56,64,80,96,112,128,160,192 }, { 0,16,32,48,64,80,96,112,128,144,160,176,192,208,224 } },
    };
    return 2*halfrate[!!HDR_TEST_MPEG1(h)][HDR_GET_LAYER(h) - 1][HDR_GET_BITRATE(h)];
}

static unsigned hdr_sample_rate_hz(const uint8_t *h)
{
    static const unsigned g_hz[3] = { 44100, 48000, 32000 };
    return g_hz[HDR_GET_SAMPLE_RATE(h)] >> (int)!HDR_TEST_MPEG1(h) >> (int)!HDR_TEST_NOT_MPEG25(h);
}

static unsigned hdr_frame_samples(const uint8_t *h)
{
    return HDR_IS_LAYER_1(h) ? 384 : (1152 >> (int)HDR_IS_FRAME_576(h));
}

static int hdr_frame_bytes(const uint8_t *h, int free_format_size)
{
    int frame_bytes = hdr_frame_samples(h)*hdr_bitrate_kbps(h)*125/hdr_sample_rate_hz(h);
    if (HDR_IS_LAYER_1(h))
    {
        frame_bytes &= ~3; /* slot align */
    }
    return frame_bytes ? frame_bytes : free_format_size;
}

static int hdr_padding(const uint8_t *h)
{
    return HDR_TEST_PADDING(h) ? (HDR_IS_LAYER_1(h) ? 4 : 1) : 0;
}

#ifndef MINIMP3_ONLY_MP3
static const L12_subband_alloc_t *L12_subband_alloc_table(const uint8_t *hdr, L12_scale_info *sci)
{
    const L12_subband_alloc_t *alloc;
    int mode = HDR_GET_STEREO_MODE(hdr);
    int nbands, stereo_bands = (mode == MODE_MONO) ? 0 : (mode == MODE_JOINT_STEREO) ? (HDR_GET_STEREO_MODE_EXT(hdr) << 2) + 4 : 32;

    if (HDR_IS_LAYER_1(hdr))
    {
        static const L12_subband_alloc_t g_alloc_L1[] = { { 76, 4, 32 } };
        alloc = g_alloc_L1;
        nbands = 32;
    } else if (!HDR_TEST_MPEG1(hdr))
    {
        static const L12_subband_alloc_t g_alloc_L2M2[] = { { 60, 4, 4 }, { 44, 3, 7 }, { 44, 2, 19 } };
        alloc = g_alloc_L2M2;
        nbands = 30;
    } else
    {
        static const L12_subband_alloc_t g_alloc_L2M1[] = { { 0, 4, 3 }, { 16, 4, 8 }, { 32, 3, 12 }, { 40, 2, 7 } };
        int sample_rate_idx = HDR_GET_SAMPLE_RATE(hdr);
        unsigned kbps = hdr_bitrate_kbps(hdr) >> (int)(mode != MODE_MONO);
        if (!kbps) /* free-format */
        {
            kbps = 192;
        }

        alloc = g_alloc_L2M1;
        nbands = 27;
        if (kbps < 56)
        {
            static const L12_subband_alloc_t g_alloc_L2M1_lowrate[] = { { 44, 4, 2 }, { 44, 3, 10 } };
            alloc = g_alloc_L2M1_lowrate;
            nbands = sample_rate_idx == 2 ? 12 : 8;
        } else if (kbps >= 96 && sample_rate_idx != 1)
        {
            nbands = 30;
        }
    }

    sci->total_bands = (uint8_t)nbands;
    sci->stereo_bands = (uint8_t)MINIMP3_MIN(stereo_bands, nbands);

    return alloc;
}

static void L12_read_scalefactors(bs_t *bs, uint8_t *pba, uint8_t *scfcod, int bands, float *scf)
{
    static const float g_deq_L12[18*3] = {
#define DQ(x) 9.53674316e-07f/x, 7.56931807e-07f/x, 6.00777173e-07f/x
        DQ(3),DQ(7),DQ(15),DQ(31),DQ(63),DQ(127),DQ(255),DQ(511),DQ(1023),DQ(2047),DQ(4095),DQ(8191),DQ(16383),DQ(32767),DQ(65535),DQ(3),DQ(5),DQ(9)
    };
    int i, m;
    for (i = 0; i < bands; i++)
    {
        float s = 0;
        int ba = *pba++;
        int mask = ba ? 4 + ((19 >> scfcod[i]) & 3) : 0;
        for (m = 4; m; m >>= 1)
        {
            if (mask & m)
            {
                int b = get_bits(bs, 6);
                s = g_deq_L12[ba*3 - 6 + b % 3]*(1 << 21 >> b/3);
            }
            *scf++ = s;
        }
    }
}

static void L12_read_scale_info(const uint8_t *hdr, bs_t *bs, L12_scale_info *sci)
{
    static const uint8_t g_bitalloc_code_tab[] = {
        0,17, 3, 4, 5,6,7, 8,9,10,11,12,13,14,15,16,
        0,17,18, 3,19,4,5, 6,7, 8, 9,10,11,12,13,16,
        0,17,18, 3,19,4,5,16,
        0,17,18,16,
        0,17,18,19, 4,5,6, 7,8, 9,10,11,12,13,14,15,
        0,17,18, 3,19,4,5, 6,7, 8, 9,10,11,12,13,14,
        0, 2, 3, 4, 5,6,7, 8,9,10,11,12,13,14,15,16
    };
    const L12_subband_alloc_t *subband_alloc = L12_subband_alloc_table(hdr, sci);

    int i, k = 0, ba_bits = 0;
    const uint8_t *ba_code_tab = g_bitalloc_code_tab;

    for (i = 0; i < sci->total_bands; i++)
    {
        uint8_t ba;
        if (i == k)
        {
            k += subband_alloc->band_count;
            ba_bits = subband_alloc->code_tab_width;
            ba_code_tab = g_bitalloc_code_tab + subband_alloc->tab_offset;
            subband_alloc++;
        }
        ba = ba_code_tab[get_bits(bs, ba_bits)];
        sci->bitalloc[2*i] = ba;
        if (i < sci->stereo_bands)
        {
            ba = ba_code_tab[get_bits(bs, ba_bits)];
        }
        sci->bitalloc[2*i + 1] = sci->stereo_bands ? ba : 0;
    }

    for (i = 0; i < 2*sci->total_bands; i++)
    {
        sci->scfcod[i] = sci->bitalloc[i] ? HDR_IS_LAYER_1(hdr) ? 2 : get_bits(bs, 2) : 6;
    }

    L12_read_scalefactors(bs, sci->bitalloc, sci->scfcod, sci->total_bands*2, sci->scf);

    for (i = sci->stereo_bands; i < sci->total_bands; i++)
    {
        sci->bitalloc[2*i + 1] = 0;
    }
}

static int L12_dequantize_granule(float *grbuf, bs_t *bs, L12_scale_info *sci, int group_size)
{
    int i, j, k, choff = 576;
    for (j = 0; j < 4; j++)
    {
        float *dst = grbuf + group_size*j;
        for (i = 0; i < 2*sci->total_bands; i++)
        {
            int ba = sci->bitalloc[i];
            if (ba != 0)
            {
                if (ba < 17)
                {
                    int half = (1 << (ba - 1)) - 1;
                    for (k = 0; k < group_size; k++)
                    {
                        dst[k] = (float)((int)get_bits(bs, ba) - half);
                    }
                } else
                {
                    unsigned mod = (2 << (ba - 17)) + 1;    /* 3, 5, 9 */
                    unsigned code = get_bits(bs, mod + 2 - (mod >> 3));  /* 5, 7, 10 */
                    for (k = 0; k < group_size; k++, code /= mod)
                    {
                        dst[k] = (float)((int)(code % mod - mod/2));
                    }
                }
            }
            dst += choff;
            choff = 18 - choff;
        }
    }
    return group_size*4;
}

static void L12_apply_scf_384(L12_scale_info *sci, const float *scf, float *dst)
{
    int i, k;
    memcpy(dst + 576 + sci->stereo_bands*18, dst + sci->stereo_bands*18, (sci->total_bands - sci->stereo_bands)*18*sizeof(float));
    for (i = 0; i < sci->total_bands; i++, dst += 18, scf += 6)
    {
        for (k = 0; k < 12; k++)
        {
            dst[k + 0]   *= scf[0];
            dst[k + 576] *= scf[3];
        }
    }
}
#endif /* MINIMP3_ONLY_MP3 */

static int L3_read_side_info(bs_t *bs, L3_gr_info_t *gr, const uint8_t *hdr)
{
    static const uint8_t g_scf_long[8][23] = {
        { 6,6,6,6,6,6,8,10,12,14,16,20,24,28,32,38,46,52,60,68,58,54,0 },
        { 12,12,12,12,12,12,16,20,24,28,32,40,48,56,64,76,90,2,2,2,2,2,0 },
        { 6,6,6,6,6,6,8,10,12,14,16,20,24,28,32,38,46,52,60,68,58,54,0 },
        { 6,6,6,6,6,6,8,10,12,14,16,18,22,26,32,38,46,54,62,70,76,36,0 },
        { 6,6,6,6,6,6,8,10,12,14,16,20,24,28,32,38,46,52,60,68,58,54,0 },
        { 4,4,4,4,4,4,6,6,8,8,10,12,16,20,24,28,34,42,50,54,76,158,0 },
        { 4,4,4,4,4,4,6,6,6,8,10,12,16,18,22,28,34,40,46,54,54,192,0 },
        { 4,4,4,4,4,4,6,6,8,10,12,16,20,24,30,38,46,56,68,84,102,26,0 }
    };
    static const uint8_t g_scf_short[8][40] = {
        { 4,4,4,4,4,4,4,4,4,6,6,6,8,8,8,10,10,10,12,12,12,14,14,14,18,18,18,24,24,24,30,30,30,40,40,40,18,18,18,0 },
        { 8,8,8,8,8,8,8,8,8,12,12,12,16,16,16,20,20,20,24,24,24,28,28,28,36,36,36,2,2,2,2,2,2,2,2,2,26,26,26,0 },
        { 4,4,4,4,4,4,4,4,4,6,6,6,6,6,6,8,8,8,10,10,10,14,14,14,18,18,18,26,26,26,32,32,32,42,42,42,18,18,18,0 },
        { 4,4,4,4,4,4,4,4,4,6,6,6,8,8,8,10,10,10,12,12,12,14,14,14,18,18,18,24,24,24,32,32,32,44,44,44,12,12,12,0 },
        { 4,4,4,4,4,4,4,4,4,6,6,6,8,8,8,10,10,10,12,12,12,14,14,14,18,18,18,24,24,24,30,30,30,40,40,40,18,18,18,0 },
        { 4,4,4,4,4,4,4,4,4,4,4,4,6,6,6,8,8,8,10,10,10,12,12,12,14,14,14,18,18,18,22,22,22,30,30,30,56,56,56,0 },
        { 4,4,4,4,4,4,4,4,4,4,4,4,6,6,6,6,6,6,10,10,10,12,12,12,14,14,14,16,16,16,20,20,20,26,26,26,66,66,66,0 },
        { 4,4,4,4,4,4,4,4,4,4,4,4,6,6,6,8,8,8,12,12,12,16,16,16,20,20,20,26,26,26,34,34,34,42,42,42,12,12,12,0 }
    };
    static const uint8_t g_scf_mixed[8][40] = {
        { 6,6,6,6,6,6,6,6,6,8,8,8,10,10,10,12,12,12,14,14,14,18,18,18,24,24,24,30,30,30,40,40,40,18,18,18,0 },
        { 12,12,12,4,4,4,8,8,8,12,12,12,16,16,16,20,20,20,24,24,24,28,28,28,36,36,36,2,2,2,2,2,2,2,2,2,26,26,26,0 },
        { 6,6,6,6,6,6,6,6,6,6,6,6,8,8,8,10,10,10,14,14,14,18,18,18,26,26,26,32,32,32,42,42,42,18,18,18,0 },
        { 6,6,6,6,6,6,6,6,6,8,8,8,10,10,10,12,12,12,14,14,14,18,18,18,24,24,24,32,32,32,44,44,44,12,12,12,0 },
        { 6,6,6,6,6,6,6,6,6,8,8,8,10,10,10,12,12,12,14,14,14,18,18,18,24,24,24,30,30,30,40,40,40,18,18,18,0 },
        { 4,4,4,4,4,4,6,6,4,4,4,6,6,6,8,8,8,10,10,10,12,12,12,14,14,14,18,18,18,22,22,22,30,30,30,56,56,56,0 },
        { 4,4,4,4,4,4,6,6,4,4,4,6,6,6,6,6,6,10,10,10,12,12,12,14,14,14,16,16,16,20,20,20,26,26,26,66,66,66,0 },
        { 4,4,4,4,4,4,6,6,4,4,4,6,6,6,8,8,8,12,12,12,16,16,16,20,20,20,26,26,26,34,34,34,42,42,42,12,12,12,0 }
    };

    unsigned tables, scfsi = 0;
    int main_data_begin, part_23_sum = 0;
    int sr_idx = HDR_GET_MY_SAMPLE_RATE(hdr); sr_idx -= (sr_idx != 0);
    int gr_count = HDR_IS_MONO(hdr) ? 1 : 2;

    if (HDR_TEST_MPEG1(hdr))
    {
        gr_count *= 2;
        main_data_begin = get_bits(bs, 9);
        scfsi = get_bits(bs, 7 + gr_count);
    } else
    {
        main_data_begin = get_bits(bs, 8 + gr_count) >> gr_count;
    }

    do
    {
        if (HDR_IS_MONO(hdr))
        {
            scfsi <<= 4;
        }
        gr->part_23_length = (uint16_t)get_bits(bs, 12);
        part_23_sum += gr->part_23_length;
        gr->big_values = (uint16_t)get_bits(bs,  9);
        if (gr->big_values > 288)
        {
            return -1;
        }
        gr->global_gain = (uint8_t)get_bits(bs, 8);
        gr->scalefac_compress = (uint16_t)get_bits(bs, HDR_TEST_MPEG1(hdr) ? 4 : 9);
        gr->sfbtab = g_scf_long[sr_idx];
        gr->n_long_sfb  = 22;
        gr->n_short_sfb = 0;
        if (get_bits(bs, 1))
        {
            gr->block_type = (uint8_t)get_bits(bs, 2);
            if (!gr->block_type)
            {
                return -1;
            }
            gr->mixed_block_flag = (uint8_t)get_bits(bs, 1);
            gr->region_count[0] = 7;
            gr->region_count[1] = 255;
            if (gr->block_type == SHORT_BLOCK_TYPE)
            {
                scfsi &= 0x0F0F;
                if (!gr->mixed_block_flag)
                {
                    gr->region_count[0] = 8;
                    gr->sfbtab = g_scf_short[sr_idx];
                    gr->n_long_sfb = 0;
                    gr->n_short_sfb = 39;
                } else
                {
                    gr->sfbtab = g_scf_mixed[sr_idx];
                    gr->n_long_sfb = HDR_TEST_MPEG1(hdr) ? 8 : 6;
                    gr->n_short_sfb = 30;
                }
            }
            tables = get_bits(bs, 10);
            tables <<= 5;
            gr->subblock_gain[0] = (uint8_t)get_bits(bs, 3);
            gr->subblock_gain[1] = (uint8_t)get_bits(bs, 3);
            gr->subblock_gain[2] = (uint8_t)get_bits(bs, 3);
        } else
        {
            gr->block_type = 0;
            gr->mixed_block_flag = 0;
            tables = get_bits(bs, 15);
            gr->region_count[0] = (uint8_t)get_bits(bs, 4);
            gr->region_count[1] = (uint8_t)get_bits(bs, 3);
            gr->region_count[2] = 255;
        }
        gr->table_select[0] = (uint8_t)(tables >> 10);
        gr->table_select[1] = (uint8_t)((tables >> 5) & 31);
        gr->table_select[2] = (uint8_t)((tables) & 31);
        gr->preflag = HDR_TEST_MPEG1(hdr) ? get_bits(bs, 1) : (gr->scalefac_compress >= 500);
        gr->scalefac_scale = (uint8_t)get_bits(bs, 1);
        gr->count1_table = (uint8_t)get_bits(bs, 1);
        gr->scfsi = (uint8_t)((scfsi >> 12) & 15);
        scfsi <<= 4;
        gr++;
    } while(--gr_count);

    if (part_23_sum + bs->pos > bs->limit + main_data_begin*8)
    {
        return -1;
    }

    return main_data_begin;
}

static void L3_read_scalefactors(uint8_t *scf, uint8_t *ist_pos, const uint8_t *scf_size, const uint8_t *scf_count, bs_t *bitbuf, int scfsi)
{
    int i, k;
    for (i = 0; i < 4 && scf_count[i]; i++, scfsi *= 2)
    {
        int cnt = scf_count[i];
        if (scfsi & 8)
        {
            memcpy(scf, ist_pos, cnt);
        } else
        {
            int bits = scf_size[i];
            if (!bits)
            {
                memset(scf, 0, cnt);
                memset(ist_pos, 0, cnt);
            } else
            {
                int max_scf = (scfsi < 0) ? (1 << bits) - 1 : -1;
                for (k = 0; k < cnt; k++)
                {
                    int s = get_bits(bitbuf, bits);
                    ist_pos[k] = (s == max_scf ? -1 : s);
                    scf[k] = s;
                }
            }
        }
        ist_pos += cnt;
        scf += cnt;
    }
    scf[0] = scf[1] = scf[2] = 0;
}

static float L3_ldexp_q2(float y, int exp_q2)
{
    static const float g_expfrac[4] = { 9.31322575e-10f,7.83145814e-10f,6.58544508e-10f,5.53767716e-10f };
    int e;
    do
    {
        e = MINIMP3_MIN(30*4, exp_q2);
        y *= g_expfrac[e & 3]*(1 << 30 >> (e >> 2));
    } while ((exp_q2 -= e) > 0);
    return y;
}

static void L3_decode_scalefactors(const uint8_t *hdr, uint8_t *ist_pos, bs_t *bs, const L3_gr_info_t *gr, float *scf, int ch)
{
    static const uint8_t g_scf_partitions[3][28] = {
        { 6,5,5, 5,6,5,5,5,6,5, 7,3,11,10,0,0, 7, 7, 7,0, 6, 6,6,3, 8, 8,5,0 },
        { 8,9,6,12,6,9,9,9,6,9,12,6,15,18,0,0, 6,15,12,0, 6,12,9,6, 6,18,9,0 },
        { 9,9,6,12,9,9,9,9,9,9,12,6,18,18,0,0,12,12,12,0,12, 9,9,6,15,12,9,0 }
    };
    const uint8_t *scf_partition = g_scf_partitions[!!gr->n_short_sfb + !gr->n_long_sfb];
    uint8_t scf_size[4], iscf[40];
    int i, scf_shift = gr->scalefac_scale + 1, gain_exp, scfsi = gr->scfsi;
    float gain;

    if (HDR_TEST_MPEG1(hdr))
    {
        static const uint8_t g_scfc_decode[16] = { 0,1,2,3, 12,5,6,7, 9,10,11,13, 14,15,18,19 };
        int part = g_scfc_decode[gr->scalefac_compress];
        scf_size[1] = scf_size[0] = (uint8_t)(part >> 2);
        scf_size[3] = scf_size[2] = (uint8_t)(part & 3);
    } else
    {
        static const uint8_t g_mod[6*4] = { 5,5,4,4,5,5,4,1,4,3,1,1,5,6,6,1,4,4,4,1,4,3,1,1 };
        int k, modprod, sfc, ist = HDR_TEST_I_STEREO(hdr) && ch;
        sfc = gr->scalefac_compress >> ist;
        for (k = ist*3*4; sfc >= 0; sfc -= modprod, k += 4)
        {
            for (modprod = 1, i = 3; i >= 0; i--)
            {
                scf_size[i] = (uint8_t)(sfc / modprod % g_mod[k + i]);
                modprod *= g_mod[k + i];
            }
        }
        scf_partition += k;
        scfsi = -16;
    }
    L3_read_scalefactors(iscf, ist_pos, scf_size, scf_partition, bs, scfsi);

    if (gr->n_short_sfb)
    {
        int sh = 3 - scf_shift;
        for (i = 0; i < gr->n_short_sfb; i += 3)
        {
            iscf[gr->n_long_sfb + i + 0] += gr->subblock_gain[0] << sh;
            iscf[gr->n_long_sfb + i + 1] += gr->subblock_gain[1] << sh;
            iscf[gr->n_long_sfb + i + 2] += gr->subblock_gain[2] << sh;
        }
    } else if (gr->preflag)
    {
        static const uint8_t g_preamp[10] = { 1,1,1,1,2,2,3,3,3,2 };
        for (i = 0; i < 10; i++)
        {
            iscf[11 + i] += g_preamp[i];
        }
    }

    gain_exp = gr->global_gain + BITS_DEQUANTIZER_OUT*4 - 210 - (HDR_IS_MS_STEREO(hdr) ? 2 : 0);
    gain = L3_ldexp_q2(1 << (MAX_SCFI/4),  MAX_SCFI - gain_exp);
    for (i = 0; i < (int)(gr->n_long_sfb + gr->n_short_sfb); i++)
    {
        scf[i] = L3_ldexp_q2(gain, iscf[i] << scf_shift);
    }
}

static const float g_pow43[129 + 16] = {
    0,-1,-2.519842f,-4.326749f,-6.349604f,-8.549880f,-10.902724f,-13.390518f,-16.000000f,-18.720754f,-21.544347f,-24.463781f,-27.473142f,-30.567351f,-33.741992f,-36.993181f,
    0,1,2.519842f,4.326749f,6.349604f,8.549880f,10.902724f,13.390518f,16.000000f,18.720754f,21.544347f,24.463781f,27.473142f,30.567351f,33.741992f,36.993181f,40.317474f,43.711787f,47.173345f,50.699631f,54.288352f,57.937408f,61.644865f,65.408941f,69.227979f,73.100443f,77.024898f,81.000000f,85.024491f,89.097188f,93.216975f,97.382800f,101.593667f,105.848633f,110.146801f,114.487321f,118.869381f,123.292209f,127.755065f,132.257246f,136.798076f,141.376907f,145.993119f,150.646117f,155.335327f,160.060199f,164.820202f,169.614826f,174.443577f,179.305980f,184.201575f,189.129918f,194.090580f,199.083145f,204.107210f,209.162385f,214.248292f,219.364564f,224.510845f,229.686789f,234.892058f,240.126328f,245.389280f,250.680604f,256.000000f,261.347174f,266.721841f,272.123723f,277.552547f,283.008049f,288.489971f,293.998060f,299.532071f,305.091761f,310.676898f,316.287249f,321.922592f,327.582707f,333.267377f,338.976394f,344.709550f,350.466646f,356.247482f,362.051866f,367.879608f,373.730522f,379.604427f,385.501143f,391.420496f,397.362314f,403.326427f,409.312672f,415.320884f,421.350905f,427.402579f,433.475750f,439.570269f,445.685987f,451.822757f,457.980436f,464.158883f,470.357960f,476.577530f,482.817459f,489.077615f,495.357868f,501.658090f,507.978156f,514.317941f,520.677324f,527.056184f,533.454404f,539.871867f,546.308458f,552.764065f,559.238575f,565.731879f,572.243870f,578.774440f,585.323483f,591.890898f,598.476581f,605.080431f,611.702349f,618.342238f,625.000000f,631.675540f,638.368763f,645.079578f
};

static float L3_pow_43(int x)
{
    float frac;
    int sign, mult = 256;

    if (x < 129)
    {
        return g_pow43[16 + x];
    }

    if (x < 1024)
    {
        mult = 16;
        x <<= 3;
    }

    sign = 2*x & 64;
    frac = (float)((x & 63) - sign) / ((x & ~63) + sign);
    return g_pow43[16 + ((x + sign) >> 6)]*(1.f + frac*((4.f/3) + frac*(2.f/9)))*mult;
}

static void L3_huffman(float *dst, bs_t *bs, const L3_gr_info_t *gr_info, const float *scf, int layer3gr_limit)
{
    static const int16_t tabs[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        785,785,785,785,784,784,784,784,513,513,513,513,513,513,513,513,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,
        -255,1313,1298,1282,785,785,785,785,784,784,784,784,769,769,769,769,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,290,288,
        -255,1313,1298,1282,769,769,769,769,529,529,529,529,529,529,529,529,528,528,528,528,528,528,528,528,512,512,512,512,512,512,512,512,290,288,
        -253,-318,-351,-367,785,785,785,785,784,784,784,784,769,769,769,769,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,819,818,547,547,275,275,275,275,561,560,515,546,289,274,288,258,
        -254,-287,1329,1299,1314,1312,1057,1057,1042,1042,1026,1026,784,784,784,784,529,529,529,529,529,529,529,529,769,769,769,769,768,768,768,768,563,560,306,306,291,259,
        -252,-413,-477,-542,1298,-575,1041,1041,784,784,784,784,769,769,769,769,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,-383,-399,1107,1092,1106,1061,849,849,789,789,1104,1091,773,773,1076,1075,341,340,325,309,834,804,577,577,532,532,516,516,832,818,803,816,561,561,531,531,515,546,289,289,288,258,
        -252,-429,-493,-559,1057,1057,1042,1042,529,529,529,529,529,529,529,529,784,784,784,784,769,769,769,769,512,512,512,512,512,512,512,512,-382,1077,-415,1106,1061,1104,849,849,789,789,1091,1076,1029,1075,834,834,597,581,340,340,339,324,804,833,532,532,832,772,818,803,817,787,816,771,290,290,290,290,288,258,
        -253,-349,-414,-447,-463,1329,1299,-479,1314,1312,1057,1057,1042,1042,1026,1026,785,785,785,785,784,784,784,784,769,769,769,769,768,768,768,768,-319,851,821,-335,836,850,805,849,341,340,325,336,533,533,579,579,564,564,773,832,578,548,563,516,321,276,306,291,304,259,
        -251,-572,-733,-830,-863,-879,1041,1041,784,784,784,784,769,769,769,769,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,-511,-527,-543,1396,1351,1381,1366,1395,1335,1380,-559,1334,1138,1138,1063,1063,1350,1392,1031,1031,1062,1062,1364,1363,1120,1120,1333,1348,881,881,881,881,375,374,359,373,343,358,341,325,791,791,1123,1122,-703,1105,1045,-719,865,865,790,790,774,774,1104,1029,338,293,323,308,-799,-815,833,788,772,818,803,816,322,292,307,320,561,531,515,546,289,274,288,258,
        -251,-525,-605,-685,-765,-831,-846,1298,1057,1057,1312,1282,785,785,785,785,784,784,784,784,769,769,769,769,512,512,512,512,512,512,512,512,1399,1398,1383,1367,1382,1396,1351,-511,1381,1366,1139,1139,1079,1079,1124,1124,1364,1349,1363,1333,882,882,882,882,807,807,807,807,1094,1094,1136,1136,373,341,535,535,881,775,867,822,774,-591,324,338,-671,849,550,550,866,864,609,609,293,336,534,534,789,835,773,-751,834,804,308,307,833,788,832,772,562,562,547,547,305,275,560,515,290,290,
        -252,-397,-477,-557,-622,-653,-719,-735,-750,1329,1299,1314,1057,1057,1042,1042,1312,1282,1024,1024,785,785,785,785,784,784,784,784,769,769,769,769,-383,1127,1141,1111,1126,1140,1095,1110,869,869,883,883,1079,1109,882,882,375,374,807,868,838,881,791,-463,867,822,368,263,852,837,836,-543,610,610,550,550,352,336,534,534,865,774,851,821,850,805,593,533,579,564,773,832,578,578,548,548,577,577,307,276,306,291,516,560,259,259,
        -250,-2107,-2507,-2764,-2909,-2974,-3007,-3023,1041,1041,1040,1040,769,769,769,769,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,-767,-1052,-1213,-1277,-1358,-1405,-1469,-1535,-1550,-1582,-1614,-1647,-1662,-1694,-1726,-1759,-1774,-1807,-1822,-1854,-1886,1565,-1919,-1935,-1951,-1967,1731,1730,1580,1717,-1983,1729,1564,-1999,1548,-2015,-2031,1715,1595,-2047,1714,-2063,1610,-2079,1609,-2095,1323,1323,1457,1457,1307,1307,1712,1547,1641,1700,1699,1594,1685,1625,1442,1442,1322,1322,-780,-973,-910,1279,1278,1277,1262,1276,1261,1275,1215,1260,1229,-959,974,974,989,989,-943,735,478,478,495,463,506,414,-1039,1003,958,1017,927,942,987,957,431,476,1272,1167,1228,-1183,1256,-1199,895,895,941,941,1242,1227,1212,1135,1014,1014,490,489,503,487,910,1013,985,925,863,894,970,955,1012,847,-1343,831,755,755,984,909,428,366,754,559,-1391,752,486,457,924,997,698,698,983,893,740,740,908,877,739,739,667,667,953,938,497,287,271,271,683,606,590,712,726,574,302,302,738,736,481,286,526,725,605,711,636,724,696,651,589,681,666,710,364,467,573,695,466,466,301,465,379,379,709,604,665,679,316,316,634,633,436,436,464,269,424,394,452,332,438,363,347,408,393,448,331,422,362,407,392,421,346,406,391,376,375,359,1441,1306,-2367,1290,-2383,1337,-2399,-2415,1426,1321,-2431,1411,1336,-2447,-2463,-2479,1169,1169,1049,1049,1424,1289,1412,1352,1319,-2495,1154,1154,1064,1064,1153,1153,416,390,360,404,403,389,344,374,373,343,358,372,327,357,342,311,356,326,1395,1394,1137,1137,1047,1047,1365,1392,1287,1379,1334,1364,1349,1378,1318,1363,792,792,792,792,1152,1152,1032,1032,1121,1121,1046,1046,1120,1120,1030,1030,-2895,1106,1061,1104,849,849,789,789,1091,1076,1029,1090,1060,1075,833,833,309,324,532,532,832,772,818,803,561,561,531,560,515,546,289,274,288,258,
        -250,-1179,-1579,-1836,-1996,-2124,-2253,-2333,-2413,-2477,-2542,-2574,-2607,-2622,-2655,1314,1313,1298,1312,1282,785,785,785,785,1040,1040,1025,1025,768,768,768,768,-766,-798,-830,-862,-895,-911,-927,-943,-959,-975,-991,-1007,-1023,-1039,-1055,-1070,1724,1647,-1103,-1119,1631,1767,1662,1738,1708,1723,-1135,1780,1615,1779,1599,1677,1646,1778,1583,-1151,1777,1567,1737,1692,1765,1722,1707,1630,1751,1661,1764,1614,1736,1676,1763,1750,1645,1598,1721,1691,1762,1706,1582,1761,1566,-1167,1749,1629,767,766,751,765,494,494,735,764,719,749,734,763,447,447,748,718,477,506,431,491,446,476,461,505,415,430,475,445,504,399,460,489,414,503,383,474,429,459,502,502,746,752,488,398,501,473,413,472,486,271,480,270,-1439,-1455,1357,-1471,-1487,-1503,1341,1325,-1519,1489,1463,1403,1309,-1535,1372,1448,1418,1476,1356,1462,1387,-1551,1475,1340,1447,1402,1386,-1567,1068,1068,1474,1461,455,380,468,440,395,425,410,454,364,467,466,464,453,269,409,448,268,432,1371,1473,1432,1417,1308,1460,1355,1446,1459,1431,1083,1083,1401,1416,1458,1445,1067,1067,1370,1457,1051,1051,1291,1430,1385,1444,1354,1415,1400,1443,1082,1082,1173,1113,1186,1066,1185,1050,-1967,1158,1128,1172,1097,1171,1081,-1983,1157,1112,416,266,375,400,1170,1142,1127,1065,793,793,1169,1033,1156,1096,1141,1111,1155,1080,1126,1140,898,898,808,808,897,897,792,792,1095,1152,1032,1125,1110,1139,1079,1124,882,807,838,881,853,791,-2319,867,368,263,822,852,837,866,806,865,-2399,851,352,262,534,534,821,836,594,594,549,549,593,593,533,533,848,773,579,579,564,578,548,563,276,276,577,576,306,291,516,560,305,305,275,259,
        -251,-892,-2058,-2620,-2828,-2957,-3023,-3039,1041,1041,1040,1040,769,769,769,769,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,-511,-527,-543,-559,1530,-575,-591,1528,1527,1407,1526,1391,1023,1023,1023,1023,1525,1375,1268,1268,1103,1103,1087,1087,1039,1039,1523,-604,815,815,815,815,510,495,509,479,508,463,507,447,431,505,415,399,-734,-782,1262,-815,1259,1244,-831,1258,1228,-847,-863,1196,-879,1253,987,987,748,-767,493,493,462,477,414,414,686,669,478,446,461,445,474,429,487,458,412,471,1266,1264,1009,1009,799,799,-1019,-1276,-1452,-1581,-1677,-1757,-1821,-1886,-1933,-1997,1257,1257,1483,1468,1512,1422,1497,1406,1467,1496,1421,1510,1134,1134,1225,1225,1466,1451,1374,1405,1252,1252,1358,1480,1164,1164,1251,1251,1238,1238,1389,1465,-1407,1054,1101,-1423,1207,-1439,830,830,1248,1038,1237,1117,1223,1148,1236,1208,411,426,395,410,379,269,1193,1222,1132,1235,1221,1116,976,976,1192,1162,1177,1220,1131,1191,963,963,-1647,961,780,-1663,558,558,994,993,437,408,393,407,829,978,813,797,947,-1743,721,721,377,392,844,950,828,890,706,706,812,859,796,960,948,843,934,874,571,571,-1919,690,555,689,421,346,539,539,944,779,918,873,932,842,903,888,570,570,931,917,674,674,-2575,1562,-2591,1609,-2607,1654,1322,1322,1441,1441,1696,1546,1683,1593,1669,1624,1426,1426,1321,1321,1639,1680,1425,1425,1305,1305,1545,1668,1608,1623,1667,1592,1638,1666,1320,1320,1652,1607,1409,1409,1304,1304,1288,1288,1664,1637,1395,1395,1335,1335,1622,1636,1394,1394,1319,1319,1606,1621,1392,1392,1137,1137,1137,1137,345,390,360,375,404,373,1047,-2751,-2767,-2783,1062,1121,1046,-2799,1077,-2815,1106,1061,789,789,1105,1104,263,355,310,340,325,354,352,262,339,324,1091,1076,1029,1090,1060,1075,833,833,788,788,1088,1028,818,818,803,803,561,561,531,531,816,771,546,546,289,274,288,258,
        -253,-317,-381,-446,-478,-509,1279,1279,-811,-1179,-1451,-1756,-1900,-2028,-2189,-2253,-2333,-2414,-2445,-2511,-2526,1313,1298,-2559,1041,1041,1040,1040,1025,1025,1024,1024,1022,1007,1021,991,1020,975,1019,959,687,687,1018,1017,671,671,655,655,1016,1015,639,639,758,758,623,623,757,607,756,591,755,575,754,559,543,543,1009,783,-575,-621,-685,-749,496,-590,750,749,734,748,974,989,1003,958,988,973,1002,942,987,957,972,1001,926,986,941,971,956,1000,910,985,925,999,894,970,-1071,-1087,-1102,1390,-1135,1436,1509,1451,1374,-1151,1405,1358,1480,1420,-1167,1507,1494,1389,1342,1465,1435,1450,1326,1505,1310,1493,1373,1479,1404,1492,1464,1419,428,443,472,397,736,526,464,464,486,457,442,471,484,482,1357,1449,1434,1478,1388,1491,1341,1490,1325,1489,1463,1403,1309,1477,1372,1448,1418,1433,1476,1356,1462,1387,-1439,1475,1340,1447,1402,1474,1324,1461,1371,1473,269,448,1432,1417,1308,1460,-1711,1459,-1727,1441,1099,1099,1446,1386,1431,1401,-1743,1289,1083,1083,1160,1160,1458,1445,1067,1067,1370,1457,1307,1430,1129,1129,1098,1098,268,432,267,416,266,400,-1887,1144,1187,1082,1173,1113,1186,1066,1050,1158,1128,1143,1172,1097,1171,1081,420,391,1157,1112,1170,1142,1127,1065,1169,1049,1156,1096,1141,1111,1155,1080,1126,1154,1064,1153,1140,1095,1048,-2159,1125,1110,1137,-2175,823,823,1139,1138,807,807,384,264,368,263,868,838,853,791,867,822,852,837,866,806,865,790,-2319,851,821,836,352,262,850,805,849,-2399,533,533,835,820,336,261,578,548,563,577,532,532,832,772,562,562,547,547,305,275,560,515,290,290,288,258 };
    static const uint8_t tab32[] = { 130,162,193,209,44,28,76,140,9,9,9,9,9,9,9,9,190,254,222,238,126,94,157,157,109,61,173,205 };
    static const uint8_t tab33[] = { 252,236,220,204,188,172,156,140,124,108,92,76,60,44,28,12 };
    static const int16_t tabindex[2*16] = { 0,32,64,98,0,132,180,218,292,364,426,538,648,746,0,1126,1460,1460,1460,1460,1460,1460,1460,1460,1842,1842,1842,1842,1842,1842,1842,1842 };
    static const uint8_t g_linbits[] =  { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,3,4,6,8,10,13,4,5,6,7,8,9,11,13 };

#define PEEK_BITS(n)  (bs_cache >> (32 - n))
#define FLUSH_BITS(n) { bs_cache <<= (n); bs_sh += (n); }
#define CHECK_BITS    while (bs_sh >= 0) { bs_cache |= (uint32_t)*bs_next_ptr++ << bs_sh; bs_sh -= 8; }
#define BSPOS         ((bs_next_ptr - bs->buf)*8 - 24 + bs_sh)

    float one = 0.0f;
    int ireg = 0, big_val_cnt = gr_info->big_values;
    const uint8_t *sfb = gr_info->sfbtab;
    const uint8_t *bs_next_ptr = bs->buf + bs->pos/8;
    uint32_t bs_cache = (((bs_next_ptr[0]*256u + bs_next_ptr[1])*256u + bs_next_ptr[2])*256u + bs_next_ptr[3]) << (bs->pos & 7);
    int pairs_to_decode, np, bs_sh = (bs->pos & 7) - 8;
    bs_next_ptr += 4;

    while (big_val_cnt > 0)
    {
        int tab_num = gr_info->table_select[ireg];
        int sfb_cnt = gr_info->region_count[ireg++];
        const int16_t *codebook = tabs + tabindex[tab_num];
        int linbits = g_linbits[tab_num];
        if (linbits)
        {
            do
            {
                np = *sfb++ / 2;
                pairs_to_decode = MINIMP3_MIN(big_val_cnt, np);
                one = *scf++;
                do
                {
                    int j, w = 5;
                    int leaf = codebook[PEEK_BITS(w)];
                    while (leaf < 0)
                    {
                        FLUSH_BITS(w);
                        w = leaf & 7;
                        leaf = codebook[PEEK_BITS(w) - (leaf >> 3)];
                    }
                    FLUSH_BITS(leaf >> 8);

                    for (j = 0; j < 2; j++, dst++, leaf >>= 4)
                    {
                        int lsb = leaf & 0x0F;
                        if (lsb == 15)
                        {
                            lsb += PEEK_BITS(linbits);
                            FLUSH_BITS(linbits);
                            CHECK_BITS;
                            *dst = one*L3_pow_43(lsb)*((int32_t)bs_cache < 0 ? -1: 1);
                        } else
                        {
                            *dst = g_pow43[16 + lsb - 16*(bs_cache >> 31)]*one;
                        }
                        FLUSH_BITS(lsb ? 1 : 0);
                    }
                    CHECK_BITS;
                } while (--pairs_to_decode);
            } while ((big_val_cnt -= np) > 0 && --sfb_cnt >= 0);
        } else
        {
            do
            {
                np = *sfb++ / 2;
                pairs_to_decode = MINIMP3_MIN(big_val_cnt, np);
                one = *scf++;
                do
                {
                    int j, w = 5;
                    int leaf = codebook[PEEK_BITS(w)];
                    while (leaf < 0)
                    {
                        FLUSH_BITS(w);
                        w = leaf & 7;
                        leaf = codebook[PEEK_BITS(w) - (leaf >> 3)];
                    }
                    FLUSH_BITS(leaf >> 8);

                    for (j = 0; j < 2; j++, dst++, leaf >>= 4)
                    {
                        int lsb = leaf & 0x0F;
                        *dst = g_pow43[16 + lsb - 16*(bs_cache >> 31)]*one;
                        FLUSH_BITS(lsb ? 1 : 0);
                    }
                    CHECK_BITS;
                } while (--pairs_to_decode);
            } while ((big_val_cnt -= np) > 0 && --sfb_cnt >= 0);
        }
    }

    for (np = 1 - big_val_cnt;; dst += 4)
    {
        const uint8_t *codebook_count1 = (gr_info->count1_table) ? tab33 : tab32;
        int leaf = codebook_count1[PEEK_BITS(4)];
        if (!(leaf & 8))
        {
            leaf = codebook_count1[(leaf >> 3) + (bs_cache << 4 >> (32 - (leaf & 3)))];
        }
        FLUSH_BITS(leaf & 7);
        if (BSPOS > layer3gr_limit)
        {
            break;
        }
#define RELOAD_SCALEFACTOR  if (!--np) { np = *sfb++/2; if (!np) break; one = *scf++; }
#define DEQ_COUNT1(s) if (leaf & (128 >> s)) { dst[s] = ((int32_t)bs_cache < 0) ? -one : one; FLUSH_BITS(1) }
        RELOAD_SCALEFACTOR;
        DEQ_COUNT1(0);
        DEQ_COUNT1(1);
        RELOAD_SCALEFACTOR;
        DEQ_COUNT1(2);
        DEQ_COUNT1(3);
        CHECK_BITS;
    }

    bs->pos = layer3gr_limit;
}

static void L3_midside_stereo(float *left, int n)
{
    int i = 0;
    float *right = left + 576;
#if HAVE_SIMD
    if (have_simd())
    {
        for (; i < n - 3; i += 4)
        {
            f4 vl = VLD(left + i);
            f4 vr = VLD(right + i);
            VSTORE(left + i, VADD(vl, vr));
            VSTORE(right + i, VSUB(vl, vr));
        }
#ifdef __GNUC__
        /* Workaround for spurious -Waggressive-loop-optimizations warning from gcc.
         * For more info see: https://github.com/lieff/minimp3/issues/88
         */
        if (__builtin_constant_p(n % 4 == 0) && n % 4 == 0)
            return;
#endif
    }
#endif /* HAVE_SIMD */
    for (; i < n; i++)
    {
        float a = left[i];
        float b = right[i];
        left[i] = a + b;
        right[i] = a - b;
    }
}

static void L3_intensity_stereo_band(float *left, int n, float kl, float kr)
{
    int i;
    for (i = 0; i < n; i++)
    {
        left[i + 576] = left[i]*kr;
        left[i] = left[i]*kl;
    }
}

static void L3_stereo_top_band(const float *right, const uint8_t *sfb, int nbands, int max_band[3])
{
    int i, k;

    max_band[0] = max_band[1] = max_band[2] = -1;

    for (i = 0; i < nbands; i++)
    {
        for (k = 0; k < sfb[i]; k += 2)
        {
            if (right[k] != 0 || right[k + 1] != 0)
            {
                max_band[i % 3] = i;
                break;
            }
        }
        right += sfb[i];
    }
}

static void L3_stereo_process(float *left, const uint8_t *ist_pos, const uint8_t *sfb, const uint8_t *hdr, int max_band[3], int mpeg2_sh)
{
    static const float g_pan[7*2] = { 0,1,0.21132487f,0.78867513f,0.36602540f,0.63397460f,0.5f,0.5f,0.63397460f,0.36602540f,0.78867513f,0.21132487f,1,0 };
    unsigned i, max_pos = HDR_TEST_MPEG1(hdr) ? 7 : 64;

    for (i = 0; sfb[i]; i++)
    {
        unsigned ipos = ist_pos[i];
        if ((int)i > max_band[i % 3] && ipos < max_pos)
        {
            float kl, kr, s = HDR_TEST_MS_STEREO(hdr) ? 1.41421356f : 1;
            if (HDR_TEST_MPEG1(hdr))
            {
                kl = g_pan[2*ipos];
                kr = g_pan[2*ipos + 1];
            } else
            {
                kl = 1;
                kr = L3_ldexp_q2(1, (ipos + 1) >> 1 << mpeg2_sh);
                if (ipos & 1)
                {
                    kl = kr;
                    kr = 1;
                }
            }
            L3_intensity_stereo_band(left, sfb[i], kl*s, kr*s);
        } else if (HDR_TEST_MS_STEREO(hdr))
        {
            L3_midside_stereo(left, sfb[i]);
        }
        left += sfb[i];
    }
}

static void L3_intensity_stereo(float *left, uint8_t *ist_pos, const L3_gr_info_t *gr, const uint8_t *hdr)
{
    int max_band[3], n_sfb = gr->n_long_sfb + gr->n_short_sfb;
    int i, max_blocks = gr->n_short_sfb ? 3 : 1;

    L3_stereo_top_band(left + 576, gr->sfbtab, n_sfb, max_band);
    if (gr->n_long_sfb)
    {
        max_band[0] = max_band[1] = max_band[2] = MINIMP3_MAX(MINIMP3_MAX(max_band[0], max_band[1]), max_band[2]);
    }
    for (i = 0; i < max_blocks; i++)
    {
        int default_pos = HDR_TEST_MPEG1(hdr) ? 3 : 0;
        int itop = n_sfb - max_blocks + i;
        int prev = itop - max_blocks;
        ist_pos[itop] = max_band[i] >= prev ? default_pos : ist_pos[prev];
    }
    L3_stereo_process(left, ist_pos, gr->sfbtab, hdr, max_band, gr[1].scalefac_compress & 1);
}

static void L3_reorder(float *grbuf, float *scratch, const uint8_t *sfb)
{
    int i, len;
    float *src = grbuf, *dst = scratch;

    for (;0 != (len = *sfb); sfb += 3, src += 2*len)
    {
        for (i = 0; i < len; i++, src++)
        {
            *dst++ = src[0*len];
            *dst++ = src[1*len];
            *dst++ = src[2*len];
        }
    }
    memcpy(grbuf, scratch, (dst - scratch)*sizeof(float));
}

static void L3_antialias(float *grbuf, int nbands)
{
    static const float g_aa[2][8] = {
        {0.85749293f,0.88174200f,0.94962865f,0.98331459f,0.99551782f,0.99916056f,0.99989920f,0.99999316f},
        {0.51449576f,0.47173197f,0.31337745f,0.18191320f,0.09457419f,0.04096558f,0.01419856f,0.00369997f}
    };

    for (; nbands > 0; nbands--, grbuf += 18)
    {
        int i = 0;
#if HAVE_SIMD
        if (have_simd()) for (; i < 8; i += 4)
        {
            f4 vu = VLD(grbuf + 18 + i);
            f4 vd = VLD(grbuf + 14 - i);
            f4 vc0 = VLD(g_aa[0] + i);
            f4 vc1 = VLD(g_aa[1] + i);
            vd = VREV(vd);
            VSTORE(grbuf + 18 + i, VSUB(VMUL(vu, vc0), VMUL(vd, vc1)));
            vd = VADD(VMUL(vu, vc1), VMUL(vd, vc0));
            VSTORE(grbuf + 14 - i, VREV(vd));
        }
#endif /* HAVE_SIMD */
#ifndef MINIMP3_ONLY_SIMD
        for(; i < 8; i++)
        {
            float u = grbuf[18 + i];
            float d = grbuf[17 - i];
            grbuf[18 + i] = u*g_aa[0][i] - d*g_aa[1][i];
            grbuf[17 - i] = u*g_aa[1][i] + d*g_aa[0][i];
        }
#endif /* MINIMP3_ONLY_SIMD */
    }
}

static void L3_dct3_9(float *y)
{
    float s0, s1, s2, s3, s4, s5, s6, s7, s8, t0, t2, t4;

    s0 = y[0]; s2 = y[2]; s4 = y[4]; s6 = y[6]; s8 = y[8];
    t0 = s0 + s6*0.5f;
    s0 -= s6;
    t4 = (s4 + s2)*0.93969262f;
    t2 = (s8 + s2)*0.76604444f;
    s6 = (s4 - s8)*0.17364818f;
    s4 += s8 - s2;

    s2 = s0 - s4*0.5f;
    y[4] = s4 + s0;
    s8 = t0 - t2 + s6;
    s0 = t0 - t4 + t2;
    s4 = t0 + t4 - s6;

    s1 = y[1]; s3 = y[3]; s5 = y[5]; s7 = y[7];

    s3 *= 0.86602540f;
    t0 = (s5 + s1)*0.98480775f;
    t4 = (s5 - s7)*0.34202014f;
    t2 = (s1 + s7)*0.64278761f;
    s1 = (s1 - s5 - s7)*0.86602540f;

    s5 = t0 - s3 - t2;
    s7 = t4 - s3 - t0;
    s3 = t4 + s3 - t2;

    y[0] = s4 - s7;
    y[1] = s2 + s1;
    y[2] = s0 - s3;
    y[3] = s8 + s5;
    y[5] = s8 - s5;
    y[6] = s0 + s3;
    y[7] = s2 - s1;
    y[8] = s4 + s7;
}

static void L3_imdct36(float *grbuf, float *overlap, const float *window, int nbands)
{
    int i, j;
    static const float g_twid9[18] = {
        0.73727734f,0.79335334f,0.84339145f,0.88701083f,0.92387953f,0.95371695f,0.97629601f,0.99144486f,0.99904822f,0.67559021f,0.60876143f,0.53729961f,0.46174861f,0.38268343f,0.30070580f,0.21643961f,0.13052619f,0.04361938f
    };

    for (j = 0; j < nbands; j++, grbuf += 18, overlap += 9)
    {
        float co[9], si[9];
        co[0] = -grbuf[0];
        si[0] = grbuf[17];
        for (i = 0; i < 4; i++)
        {
            si[8 - 2*i] =   grbuf[4*i + 1] - grbuf[4*i + 2];
            co[1 + 2*i] =   grbuf[4*i + 1] + grbuf[4*i + 2];
            si[7 - 2*i] =   grbuf[4*i + 4] - grbuf[4*i + 3];
            co[2 + 2*i] = -(grbuf[4*i + 3] + grbuf[4*i + 4]);
        }
        L3_dct3_9(co);
        L3_dct3_9(si);

        si[1] = -si[1];
        si[3] = -si[3];
        si[5] = -si[5];
        si[7] = -si[7];

        i = 0;

#if HAVE_SIMD
        if (have_simd()) for (; i < 8; i += 4)
        {
            f4 vovl = VLD(overlap + i);
            f4 vc = VLD(co + i);
            f4 vs = VLD(si + i);
            f4 vr0 = VLD(g_twid9 + i);
            f4 vr1 = VLD(g_twid9 + 9 + i);
            f4 vw0 = VLD(window + i);
            f4 vw1 = VLD(window + 9 + i);
            f4 vsum = VADD(VMUL(vc, vr1), VMUL(vs, vr0));
            VSTORE(overlap + i, VSUB(VMUL(vc, vr0), VMUL(vs, vr1)));
            VSTORE(grbuf + i, VSUB(VMUL(vovl, vw0), VMUL(vsum, vw1)));
            vsum = VADD(VMUL(vovl, vw1), VMUL(vsum, vw0));
            VSTORE(grbuf + 14 - i, VREV(vsum));
        }
#endif /* HAVE_SIMD */
        for (; i < 9; i++)
        {
            float ovl  = overlap[i];
            float sum  = co[i]*g_twid9[9 + i] + si[i]*g_twid9[0 + i];
            overlap[i] = co[i]*g_twid9[0 + i] - si[i]*g_twid9[9 + i];
            grbuf[i]      = ovl*window[0 + i] - sum*window[9 + i];
            grbuf[17 - i] = ovl*window[9 + i] + sum*window[0 + i];
        }
    }
}

static void L3_idct3(float x0, float x1, float x2, float *dst)
{
    float m1 = x1*0.86602540f;
    float a1 = x0 - x2*0.5f;
    dst[1] = x0 + x2;
    dst[0] = a1 + m1;
    dst[2] = a1 - m1;
}

static void L3_imdct12(float *x, float *dst, float *overlap)
{
    static const float g_twid3[6] = { 0.79335334f,0.92387953f,0.99144486f, 0.60876143f,0.38268343f,0.13052619f };
    float co[3], si[3];
    int i;

    L3_idct3(-x[0], x[6] + x[3], x[12] + x[9], co);
    L3_idct3(x[15], x[12] - x[9], x[6] - x[3], si);
    si[1] = -si[1];

    for (i = 0; i < 3; i++)
    {
        float ovl  = overlap[i];
        float sum  = co[i]*g_twid3[3 + i] + si[i]*g_twid3[0 + i];
        overlap[i] = co[i]*g_twid3[0 + i] - si[i]*g_twid3[3 + i];
        dst[i]     = ovl*g_twid3[2 - i] - sum*g_twid3[5 - i];
        dst[5 - i] = ovl*g_twid3[5 - i] + sum*g_twid3[2 - i];
    }
}

static void L3_imdct_short(float *grbuf, float *overlap, int nbands)
{
    for (;nbands > 0; nbands--, overlap += 9, grbuf += 18)
    {
        float tmp[18];
        memcpy(tmp, grbuf, sizeof(tmp));
        memcpy(grbuf, overlap, 6*sizeof(float));
        L3_imdct12(tmp, grbuf + 6, overlap + 6);
        L3_imdct12(tmp + 1, grbuf + 12, overlap + 6);
        L3_imdct12(tmp + 2, overlap, overlap + 6);
    }
}

static void L3_change_sign(float *grbuf)
{
    int b, i;
    for (b = 0, grbuf += 18; b < 32; b += 2, grbuf += 36)
        for (i = 1; i < 18; i += 2)
            grbuf[i] = -grbuf[i];
}

static void L3_imdct_gr(float *grbuf, float *overlap, unsigned block_type, unsigned n_long_bands)
{
    static const float g_mdct_window[2][18] = {
        { 0.99904822f,0.99144486f,0.97629601f,0.95371695f,0.92387953f,0.88701083f,0.84339145f,0.79335334f,0.73727734f,0.04361938f,0.13052619f,0.21643961f,0.30070580f,0.38268343f,0.46174861f,0.53729961f,0.60876143f,0.67559021f },
        { 1,1,1,1,1,1,0.99144486f,0.92387953f,0.79335334f,0,0,0,0,0,0,0.13052619f,0.38268343f,0.60876143f }
    };
    if (n_long_bands)
    {
        L3_imdct36(grbuf, overlap, g_mdct_window[0], n_long_bands);
        grbuf += 18*n_long_bands;
        overlap += 9*n_long_bands;
    }
    if (block_type == SHORT_BLOCK_TYPE)
        L3_imdct_short(grbuf, overlap, 32 - n_long_bands);
    else
        L3_imdct36(grbuf, overlap, g_mdct_window[block_type == STOP_BLOCK_TYPE], 32 - n_long_bands);
}

static void L3_save_reservoir(mp3dec_t *h, mp3dec_scratch_t *s)
{
    int pos = (s->bs.pos + 7)/8u;
    int remains = s->bs.limit/8u - pos;
    if (remains > MAX_BITRESERVOIR_BYTES)
    {
        pos += remains - MAX_BITRESERVOIR_BYTES;
        remains = MAX_BITRESERVOIR_BYTES;
    }
    if (remains > 0)
    {
        memmove(h->reserv_buf, s->maindata + pos, remains);
    }
    h->reserv = remains;
}

static int L3_restore_reservoir(mp3dec_t *h, bs_t *bs, mp3dec_scratch_t *s, int main_data_begin)
{
    int frame_bytes = (bs->limit - bs->pos)/8;
    int bytes_have = MINIMP3_MIN(h->reserv, main_data_begin);
    memcpy(s->maindata, h->reserv_buf + MINIMP3_MAX(0, h->reserv - main_data_begin), MINIMP3_MIN(h->reserv, main_data_begin));
    memcpy(s->maindata + bytes_have, bs->buf + bs->pos/8, frame_bytes);
    bs_init(&s->bs, s->maindata, bytes_have + frame_bytes);
    return h->reserv >= main_data_begin;
}

static void L3_decode(mp3dec_t *h, mp3dec_scratch_t *s, L3_gr_info_t *gr_info, int nch)
{
    int ch;

    for (ch = 0; ch < nch; ch++)
    {
        int layer3gr_limit = s->bs.pos + gr_info[ch].part_23_length;
        L3_decode_scalefactors(h->header, s->ist_pos[ch], &s->bs, gr_info + ch, s->scf, ch);
        L3_huffman(s->grbuf[ch], &s->bs, gr_info + ch, s->scf, layer3gr_limit);
    }

    if (HDR_TEST_I_STEREO(h->header))
    {
        L3_intensity_stereo(s->grbuf[0], s->ist_pos[1], gr_info, h->header);
    } else if (HDR_IS_MS_STEREO(h->header))
    {
        L3_midside_stereo(s->grbuf[0], 576);
    }

    for (ch = 0; ch < nch; ch++, gr_info++)
    {
        int aa_bands = 31;
        int n_long_bands = (gr_info->mixed_block_flag ? 2 : 0) << (int)(HDR_GET_MY_SAMPLE_RATE(h->header) == 2);

        if (gr_info->n_short_sfb)
        {
            aa_bands = n_long_bands - 1;
            L3_reorder(s->grbuf[ch] + n_long_bands*18, s->syn[0], gr_info->sfbtab + gr_info->n_long_sfb);
        }

        L3_antialias(s->grbuf[ch], aa_bands);
        L3_imdct_gr(s->grbuf[ch], h->mdct_overlap[ch], gr_info->block_type, n_long_bands);
        L3_change_sign(s->grbuf[ch]);
    }
}

static void mp3d_DCT_II(float *grbuf, int n)
{
    static const float g_sec[24] = {
        10.19000816f,0.50060302f,0.50241929f,3.40760851f,0.50547093f,0.52249861f,2.05778098f,0.51544732f,0.56694406f,1.48416460f,0.53104258f,0.64682180f,1.16943991f,0.55310392f,0.78815460f,0.97256821f,0.58293498f,1.06067765f,0.83934963f,0.62250412f,1.72244716f,0.74453628f,0.67480832f,5.10114861f
    };
    int i, k = 0;
#if HAVE_SIMD
    if (have_simd()) for (; k < n; k += 4)
    {
        f4 t[4][8], *x;
        float *y = grbuf + k;

        for (x = t[0], i = 0; i < 8; i++, x++)
        {
            f4 x0 = VLD(&y[i*18]);
            f4 x1 = VLD(&y[(15 - i)*18]);
            f4 x2 = VLD(&y[(16 + i)*18]);
            f4 x3 = VLD(&y[(31 - i)*18]);
            f4 t0 = VADD(x0, x3);
            f4 t1 = VADD(x1, x2);
            f4 t2 = VMUL_S(VSUB(x1, x2), g_sec[3*i + 0]);
            f4 t3 = VMUL_S(VSUB(x0, x3), g_sec[3*i + 1]);
            x[0] = VADD(t0, t1);
            x[8] = VMUL_S(VSUB(t0, t1), g_sec[3*i + 2]);
            x[16] = VADD(t3, t2);
            x[24] = VMUL_S(VSUB(t3, t2), g_sec[3*i + 2]);
        }
        for (x = t[0], i = 0; i < 4; i++, x += 8)
        {
            f4 x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3], x4 = x[4], x5 = x[5], x6 = x[6], x7 = x[7], xt;
            xt = VSUB(x0, x7); x0 = VADD(x0, x7);
            x7 = VSUB(x1, x6); x1 = VADD(x1, x6);
            x6 = VSUB(x2, x5); x2 = VADD(x2, x5);
            x5 = VSUB(x3, x4); x3 = VADD(x3, x4);
            x4 = VSUB(x0, x3); x0 = VADD(x0, x3);
            x3 = VSUB(x1, x2); x1 = VADD(x1, x2);
            x[0] = VADD(x0, x1);
            x[4] = VMUL_S(VSUB(x0, x1), 0.70710677f);
            x5 = VADD(x5, x6);
            x6 = VMUL_S(VADD(x6, x7), 0.70710677f);
            x7 = VADD(x7, xt);
            x3 = VMUL_S(VADD(x3, x4), 0.70710677f);
            x5 = VSUB(x5, VMUL_S(x7, 0.198912367f)); /* rotate by PI/8 */
            x7 = VADD(x7, VMUL_S(x5, 0.382683432f));
            x5 = VSUB(x5, VMUL_S(x7, 0.198912367f));
            x0 = VSUB(xt, x6); xt = VADD(xt, x6);
            x[1] = VMUL_S(VADD(xt, x7), 0.50979561f);
            x[2] = VMUL_S(VADD(x4, x3), 0.54119611f);
            x[3] = VMUL_S(VSUB(x0, x5), 0.60134488f);
            x[5] = VMUL_S(VADD(x0, x5), 0.89997619f);
            x[6] = VMUL_S(VSUB(x4, x3), 1.30656302f);
            x[7] = VMUL_S(VSUB(xt, x7), 2.56291556f);
        }

        if (k > n - 3)
        {
#if HAVE_SSE
#define VSAVE2(i, v) _mm_storel_pi((__m64 *)(void*)&y[i*18], v)
#else /* HAVE_SSE */
#define VSAVE2(i, v) vst1_f32((float32_t *)&y[i*18],  vget_low_f32(v))
#endif /* HAVE_SSE */
            for (i = 0; i < 7; i++, y += 4*18)
            {
                f4 s = VADD(t[3][i], t[3][i + 1]);
                VSAVE2(0, t[0][i]);
                VSAVE2(1, VADD(t[2][i], s));
                VSAVE2(2, VADD(t[1][i], t[1][i + 1]));
                VSAVE2(3, VADD(t[2][1 + i], s));
            }
            VSAVE2(0, t[0][7]);
            VSAVE2(1, VADD(t[2][7], t[3][7]));
            VSAVE2(2, t[1][7]);
            VSAVE2(3, t[3][7]);
        } else
        {
#define VSAVE4(i, v) VSTORE(&y[i*18], v)
            for (i = 0; i < 7; i++, y += 4*18)
            {
                f4 s = VADD(t[3][i], t[3][i + 1]);
                VSAVE4(0, t[0][i]);
                VSAVE4(1, VADD(t[2][i], s));
                VSAVE4(2, VADD(t[1][i], t[1][i + 1]));
                VSAVE4(3, VADD(t[2][1 + i], s));
            }
            VSAVE4(0, t[0][7]);
            VSAVE4(1, VADD(t[2][7], t[3][7]));
            VSAVE4(2, t[1][7]);
            VSAVE4(3, t[3][7]);
        }
    } else
#endif /* HAVE_SIMD */
#ifdef MINIMP3_ONLY_SIMD
    {} /* for HAVE_SIMD=1, MINIMP3_ONLY_SIMD=1 case we do not need non-intrinsic "else" branch */
#else /* MINIMP3_ONLY_SIMD */
    for (; k < n; k++)
    {
        float t[4][8], *x, *y = grbuf + k;

        for (x = t[0], i = 0; i < 8; i++, x++)
        {
            float x0 = y[i*18];
            float x1 = y[(15 - i)*18];
            float x2 = y[(16 + i)*18];
            float x3 = y[(31 - i)*18];
            float t0 = x0 + x3;
            float t1 = x1 + x2;
            float t2 = (x1 - x2)*g_sec[3*i + 0];
            float t3 = (x0 - x3)*g_sec[3*i + 1];
            x[0] = t0 + t1;
            x[8] = (t0 - t1)*g_sec[3*i + 2];
            x[16] = t3 + t2;
            x[24] = (t3 - t2)*g_sec[3*i + 2];
        }
        for (x = t[0], i = 0; i < 4; i++, x += 8)
        {
            float x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3], x4 = x[4], x5 = x[5], x6 = x[6], x7 = x[7], xt;
            xt = x0 - x7; x0 += x7;
            x7 = x1 - x6; x1 += x6;
            x6 = x2 - x5; x2 += x5;
            x5 = x3 - x4; x3 += x4;
            x4 = x0 - x3; x0 += x3;
            x3 = x1 - x2; x1 += x2;
            x[0] = x0 + x1;
            x[4] = (x0 - x1)*0.70710677f;
            x5 =  x5 + x6;
            x6 = (x6 + x7)*0.70710677f;
            x7 =  x7 + xt;
            x3 = (x3 + x4)*0.70710677f;
            x5 -= x7*0.198912367f;  /* rotate by PI/8 */
            x7 += x5*0.382683432f;
            x5 -= x7*0.198912367f;
            x0 = xt - x6; xt += x6;
            x[1] = (xt + x7)*0.50979561f;
            x[2] = (x4 + x3)*0.54119611f;
            x[3] = (x0 - x5)*0.60134488f;
            x[5] = (x0 + x5)*0.89997619f;
            x[6] = (x4 - x3)*1.30656302f;
            x[7] = (xt - x7)*2.56291556f;

        }
        for (i = 0; i < 7; i++, y += 4*18)
        {
            y[0*18] = t[0][i];
            y[1*18] = t[2][i] + t[3][i] + t[3][i + 1];
            y[2*18] = t[1][i] + t[1][i + 1];
            y[3*18] = t[2][i + 1] + t[3][i] + t[3][i + 1];
        }
        y[0*18] = t[0][7];
        y[1*18] = t[2][7] + t[3][7];
        y[2*18] = t[1][7];
        y[3*18] = t[3][7];
    }
#endif /* MINIMP3_ONLY_SIMD */
}

#ifndef MINIMP3_FLOAT_OUTPUT
static int16_t mp3d_scale_pcm(float sample)
{
#if HAVE_ARMV6
    int32_t s32 = (int32_t)(sample + .5f);
    s32 -= (s32 < 0);
    int16_t s = (int16_t)minimp3_clip_int16_arm(s32);
#else
    if (sample >=  32766.5) return (int16_t) 32767;
    if (sample <= -32767.5) return (int16_t)-32768;
    int16_t s = (int16_t)(sample + .5f);
    s -= (s < 0);   /* away from zero, to be compliant */
#endif
    return s;
}
#else /* MINIMP3_FLOAT_OUTPUT */
static float mp3d_scale_pcm(float sample)
{
    return sample*(1.f/32768.f);
}
#endif /* MINIMP3_FLOAT_OUTPUT */

static void mp3d_synth_pair(mp3d_sample_t *pcm, int nch, const float *z)
{
    float a;
    a  = (z[14*64] - z[    0]) * 29;
    a += (z[ 1*64] + z[13*64]) * 213;
    a += (z[12*64] - z[ 2*64]) * 459;
    a += (z[ 3*64] + z[11*64]) * 2037;
    a += (z[10*64] - z[ 4*64]) * 5153;
    a += (z[ 5*64] + z[ 9*64]) * 6574;
    a += (z[ 8*64] - z[ 6*64]) * 37489;
    a +=  z[ 7*64]             * 75038;
    pcm[0] = mp3d_scale_pcm(a);

    z += 2;
    a  = z[14*64] * 104;
    a += z[12*64] * 1567;
    a += z[10*64] * 9727;
    a += z[ 8*64] * 64019;
    a += z[ 6*64] * -9975;
    a += z[ 4*64] * -45;
    a += z[ 2*64] * 146;
    a += z[ 0*64] * -5;
    pcm[16*nch] = mp3d_scale_pcm(a);
}

static void mp3d_synth(float *xl, mp3d_sample_t *dstl, int nch, float *lins)
{
    int i;
    float *xr = xl + 576*(nch - 1);
    mp3d_sample_t *dstr = dstl + (nch - 1);

    static const float g_win[] = {
        -1,26,-31,208,218,401,-519,2063,2000,4788,-5517,7134,5959,35640,-39336,74992,
        -1,24,-35,202,222,347,-581,2080,1952,4425,-5879,7640,5288,33791,-41176,74856,
        -1,21,-38,196,225,294,-645,2087,1893,4063,-6237,8092,4561,31947,-43006,74630,
        -1,19,-41,190,227,244,-711,2085,1822,3705,-6589,8492,3776,30112,-44821,74313,
        -1,17,-45,183,228,197,-779,2075,1739,3351,-6935,8840,2935,28289,-46617,73908,
        -1,16,-49,176,228,153,-848,2057,1644,3004,-7271,9139,2037,26482,-48390,73415,
        -2,14,-53,169,227,111,-919,2032,1535,2663,-7597,9389,1082,24694,-50137,72835,
        -2,13,-58,161,224,72,-991,2001,1414,2330,-7910,9592,70,22929,-51853,72169,
        -2,11,-63,154,221,36,-1064,1962,1280,2006,-8209,9750,-998,21189,-53534,71420,
        -2,10,-68,147,215,2,-1137,1919,1131,1692,-8491,9863,-2122,19478,-55178,70590,
        -3,9,-73,139,208,-29,-1210,1870,970,1388,-8755,9935,-3300,17799,-56778,69679,
        -3,8,-79,132,200,-57,-1283,1817,794,1095,-8998,9966,-4533,16155,-58333,68692,
        -4,7,-85,125,189,-83,-1356,1759,605,814,-9219,9959,-5818,14548,-59838,67629,
        -4,7,-91,117,177,-106,-1428,1698,402,545,-9416,9916,-7154,12980,-61289,66494,
        -5,6,-97,111,163,-127,-1498,1634,185,288,-9585,9838,-8540,11455,-62684,65290
    };
    float *zlin = lins + 15*64;
    const float *w = g_win;

    zlin[4*15]     = xl[18*16];
    zlin[4*15 + 1] = xr[18*16];
    zlin[4*15 + 2] = xl[0];
    zlin[4*15 + 3] = xr[0];

    zlin[4*31]     = xl[1 + 18*16];
    zlin[4*31 + 1] = xr[1 + 18*16];
    zlin[4*31 + 2] = xl[1];
    zlin[4*31 + 3] = xr[1];

    mp3d_synth_pair(dstr, nch, lins + 4*15 + 1);
    mp3d_synth_pair(dstr + 32*nch, nch, lins + 4*15 + 64 + 1);
    mp3d_synth_pair(dstl, nch, lins + 4*15);
    mp3d_synth_pair(dstl + 32*nch, nch, lins + 4*15 + 64);

#if HAVE_SIMD
    if (have_simd()) for (i = 14; i >= 0; i--)
    {
#define VLOAD(k) f4 w0 = VSET(*w++); f4 w1 = VSET(*w++); f4 vz = VLD(&zlin[4*i - 64*k]); f4 vy = VLD(&zlin[4*i - 64*(15 - k)]);
#define V0(k) { VLOAD(k) b =         VADD(VMUL(vz, w1), VMUL(vy, w0)) ; a =         VSUB(VMUL(vz, w0), VMUL(vy, w1));  }
#define V1(k) { VLOAD(k) b = VADD(b, VADD(VMUL(vz, w1), VMUL(vy, w0))); a = VADD(a, VSUB(VMUL(vz, w0), VMUL(vy, w1))); }
#define V2(k) { VLOAD(k) b = VADD(b, VADD(VMUL(vz, w1), VMUL(vy, w0))); a = VADD(a, VSUB(VMUL(vy, w1), VMUL(vz, w0))); }
        f4 a, b;
        zlin[4*i]     = xl[18*(31 - i)];
        zlin[4*i + 1] = xr[18*(31 - i)];
        zlin[4*i + 2] = xl[1 + 18*(31 - i)];
        zlin[4*i + 3] = xr[1 + 18*(31 - i)];
        zlin[4*i + 64] = xl[1 + 18*(1 + i)];
        zlin[4*i + 64 + 1] = xr[1 + 18*(1 + i)];
        zlin[4*i - 64 + 2] = xl[18*(1 + i)];
        zlin[4*i - 64 + 3] = xr[18*(1 + i)];

        V0(0) V2(1) V1(2) V2(3) V1(4) V2(5) V1(6) V2(7)

        {
#ifndef MINIMP3_FLOAT_OUTPUT
#if HAVE_SSE
            static const f4 g_max = { 32767.0f, 32767.0f, 32767.0f, 32767.0f };
            static const f4 g_min = { -32768.0f, -32768.0f, -32768.0f, -32768.0f };
            __m128i pcm8 = _mm_packs_epi32(_mm_cvtps_epi32(_mm_max_ps(_mm_min_ps(a, g_max), g_min)),
                                           _mm_cvtps_epi32(_mm_max_ps(_mm_min_ps(b, g_max), g_min)));
            dstr[(15 - i)*nch] = _mm_extract_epi16(pcm8, 1);
            dstr[(17 + i)*nch] = _mm_extract_epi16(pcm8, 5);
            dstl[(15 - i)*nch] = _mm_extract_epi16(pcm8, 0);
            dstl[(17 + i)*nch] = _mm_extract_epi16(pcm8, 4);
            dstr[(47 - i)*nch] = _mm_extract_epi16(pcm8, 3);
            dstr[(49 + i)*nch] = _mm_extract_epi16(pcm8, 7);
            dstl[(47 - i)*nch] = _mm_extract_epi16(pcm8, 2);
            dstl[(49 + i)*nch] = _mm_extract_epi16(pcm8, 6);
#else /* HAVE_SSE */
            int16x4_t pcma, pcmb;
            a = VADD(a, VSET(0.5f));
            b = VADD(b, VSET(0.5f));
            pcma = vqmovn_s32(vqaddq_s32(vcvtq_s32_f32(a), vreinterpretq_s32_u32(vcltq_f32(a, VSET(0)))));
            pcmb = vqmovn_s32(vqaddq_s32(vcvtq_s32_f32(b), vreinterpretq_s32_u32(vcltq_f32(b, VSET(0)))));
            vst1_lane_s16(dstr + (15 - i)*nch, pcma, 1);
            vst1_lane_s16(dstr + (17 + i)*nch, pcmb, 1);
            vst1_lane_s16(dstl + (15 - i)*nch, pcma, 0);
            vst1_lane_s16(dstl + (17 + i)*nch, pcmb, 0);
            vst1_lane_s16(dstr + (47 - i)*nch, pcma, 3);
            vst1_lane_s16(dstr + (49 + i)*nch, pcmb, 3);
            vst1_lane_s16(dstl + (47 - i)*nch, pcma, 2);
            vst1_lane_s16(dstl + (49 + i)*nch, pcmb, 2);
#endif /* HAVE_SSE */

#else /* MINIMP3_FLOAT_OUTPUT */

            static const f4 g_scale = { 1.0f/32768.0f, 1.0f/32768.0f, 1.0f/32768.0f, 1.0f/32768.0f };
            a = VMUL(a, g_scale);
            b = VMUL(b, g_scale);
#if HAVE_SSE
            _mm_store_ss(dstr + (15 - i)*nch, _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 1, 1, 1)));
            _mm_store_ss(dstr + (17 + i)*nch, _mm_shuffle_ps(b, b, _MM_SHUFFLE(1, 1, 1, 1)));
            _mm_store_ss(dstl + (15 - i)*nch, _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 0, 0, 0)));
            _mm_store_ss(dstl + (17 + i)*nch, _mm_shuffle_ps(b, b, _MM_SHUFFLE(0, 0, 0, 0)));
            _mm_store_ss(dstr + (47 - i)*nch, _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 3, 3, 3)));
            _mm_store_ss(dstr + (49 + i)*nch, _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 3, 3, 3)));
            _mm_store_ss(dstl + (47 - i)*nch, _mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 2, 2, 2)));
            _mm_store_ss(dstl + (49 + i)*nch, _mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 2, 2, 2)));
#else /* HAVE_SSE */
            vst1q_lane_f32(dstr + (15 - i)*nch, a, 1);
            vst1q_lane_f32(dstr + (17 + i)*nch, b, 1);
            vst1q_lane_f32(dstl + (15 - i)*nch, a, 0);
            vst1q_lane_f32(dstl + (17 + i)*nch, b, 0);
            vst1q_lane_f32(dstr + (47 - i)*nch, a, 3);
            vst1q_lane_f32(dstr + (49 + i)*nch, b, 3);
            vst1q_lane_f32(dstl + (47 - i)*nch, a, 2);
            vst1q_lane_f32(dstl + (49 + i)*nch, b, 2);
#endif /* HAVE_SSE */
#endif /* MINIMP3_FLOAT_OUTPUT */
        }
    } else
#endif /* HAVE_SIMD */
#ifdef MINIMP3_ONLY_SIMD
    {} /* for HAVE_SIMD=1, MINIMP3_ONLY_SIMD=1 case we do not need non-intrinsic "else" branch */
#else /* MINIMP3_ONLY_SIMD */
    for (i = 14; i >= 0; i--)
    {
#define LOAD(k) float w0 = *w++; float w1 = *w++; float *vz = &zlin[4*i - k*64]; float *vy = &zlin[4*i - (15 - k)*64];
#define S0(k) { int j; LOAD(k); for (j = 0; j < 4; j++) b[j]  = vz[j]*w1 + vy[j]*w0, a[j]  = vz[j]*w0 - vy[j]*w1; }
#define S1(k) { int j; LOAD(k); for (j = 0; j < 4; j++) b[j] += vz[j]*w1 + vy[j]*w0, a[j] += vz[j]*w0 - vy[j]*w1; }
#define S2(k) { int j; LOAD(k); for (j = 0; j < 4; j++) b[j] += vz[j]*w1 + vy[j]*w0, a[j] += vy[j]*w1 - vz[j]*w0; }
        float a[4], b[4];

        zlin[4*i]     = xl[18*(31 - i)];
        zlin[4*i + 1] = xr[18*(31 - i)];
        zlin[4*i + 2] = xl[1 + 18*(31 - i)];
        zlin[4*i + 3] = xr[1 + 18*(31 - i)];
        zlin[4*(i + 16)]   = xl[1 + 18*(1 + i)];
        zlin[4*(i + 16) + 1] = xr[1 + 18*(1 + i)];
        zlin[4*(i - 16) + 2] = xl[18*(1 + i)];
        zlin[4*(i - 16) + 3] = xr[18*(1 + i)];

        S0(0) S2(1) S1(2) S2(3) S1(4) S2(5) S1(6) S2(7)

        dstr[(15 - i)*nch] = mp3d_scale_pcm(a[1]);
        dstr[(17 + i)*nch] = mp3d_scale_pcm(b[1]);
        dstl[(15 - i)*nch] = mp3d_scale_pcm(a[0]);
        dstl[(17 + i)*nch] = mp3d_scale_pcm(b[0]);
        dstr[(47 - i)*nch] = mp3d_scale_pcm(a[3]);
        dstr[(49 + i)*nch] = mp3d_scale_pcm(b[3]);
        dstl[(47 - i)*nch] = mp3d_scale_pcm(a[2]);
        dstl[(49 + i)*nch] = mp3d_scale_pcm(b[2]);
    }
#endif /* MINIMP3_ONLY_SIMD */
}

static void mp3d_synth_granule(float *qmf_state, float *grbuf, int nbands, int nch, mp3d_sample_t *pcm, float *lins)
{
    int i;
    for (i = 0; i < nch; i++)
    {
        mp3d_DCT_II(grbuf + 576*i, nbands);
    }

    memcpy(lins, qmf_state, sizeof(float)*15*64);

    for (i = 0; i < nbands; i += 2)
    {
        mp3d_synth(grbuf + i, pcm + 32*nch*i, nch, lins + i*64);
    }
#ifndef MINIMP3_NONSTANDARD_BUT_LOGICAL
    if (nch == 1)
    {
        for (i = 0; i < 15*64; i += 2)
        {
            qmf_state[i] = lins[nbands*64 + i];
        }
    } else
#endif /* MINIMP3_NONSTANDARD_BUT_LOGICAL */
    {
        memcpy(qmf_state, lins + nbands*64, sizeof(float)*15*64);
    }
}

static int mp3d_match_frame(const uint8_t *hdr, int mp3_bytes, int frame_bytes)
{
    int i, nmatch;
    for (i = 0, nmatch = 0; nmatch < MAX_FRAME_SYNC_MATCHES; nmatch++)
    {
        i += hdr_frame_bytes(hdr + i, frame_bytes) + hdr_padding(hdr + i);
        if (i + HDR_SIZE > mp3_bytes)
            return nmatch > 0;
        if (!hdr_compare(hdr, hdr + i))
            return 0;
    }
    return 1;
}

static int mp3d_find_frame(const uint8_t *mp3, int mp3_bytes, int *free_format_bytes, int *ptr_frame_bytes)
{
    int i, k;
    for (i = 0; i < mp3_bytes - HDR_SIZE; i++, mp3++)
    {
        if (hdr_valid(mp3))
        {
            int frame_bytes = hdr_frame_bytes(mp3, *free_format_bytes);
            int frame_and_padding = frame_bytes + hdr_padding(mp3);

            for (k = HDR_SIZE; !frame_bytes && k < MAX_FREE_FORMAT_FRAME_SIZE && i + 2*k < mp3_bytes - HDR_SIZE; k++)
            {
                if (hdr_compare(mp3, mp3 + k))
                {
                    int fb = k - hdr_padding(mp3);
                    int nextfb = fb + hdr_padding(mp3 + k);
                    if (i + k + nextfb + HDR_SIZE > mp3_bytes || !hdr_compare(mp3, mp3 + k + nextfb))
                        continue;
                    frame_and_padding = k;
                    frame_bytes = fb;
                    *free_format_bytes = fb;
                }
            }
            if ((frame_bytes && i + frame_and_padding <= mp3_bytes &&
                mp3d_match_frame(mp3, mp3_bytes - i, frame_bytes)) ||
                (!i && frame_and_padding == mp3_bytes))
            {
                *ptr_frame_bytes = frame_and_padding;
                return i;
            }
            *free_format_bytes = 0;
        }
    }
    *ptr_frame_bytes = 0;
    return mp3_bytes;
}

void mp3dec_init(mp3dec_t *dec)
{
    dec->header[0] = 0;
}

int mp3dec_decode_frame(mp3dec_t *dec, const uint8_t *mp3, int mp3_bytes, mp3d_sample_t *pcm, mp3dec_frame_info_t *info)
{
    int i = 0, igr, frame_size = 0, success = 1;
    const uint8_t *hdr;
    bs_t bs_frame[1];
    mp3dec_scratch_t scratch;

    if (mp3_bytes > 4 && dec->header[0] == 0xff && hdr_compare(dec->header, mp3))
    {
        frame_size = hdr_frame_bytes(mp3, dec->free_format_bytes) + hdr_padding(mp3);
        if (frame_size != mp3_bytes && (frame_size + HDR_SIZE > mp3_bytes || !hdr_compare(mp3, mp3 + frame_size)))
        {
            frame_size = 0;
        }
    }
    if (!frame_size)
    {
        memset(dec, 0, sizeof(mp3dec_t));
        i = mp3d_find_frame(mp3, mp3_bytes, &dec->free_format_bytes, &frame_size);
        if (!frame_size || i + frame_size > mp3_bytes)
        {
            info->frame_bytes = i;
            return 0;
        }
    }

    hdr = mp3 + i;
    memcpy(dec->header, hdr, HDR_SIZE);
    info->frame_bytes = i + frame_size;
    info->frame_offset = i;
    info->channels = HDR_IS_MONO(hdr) ? 1 : 2;
    info->hz = hdr_sample_rate_hz(hdr);
    info->layer = 4 - HDR_GET_LAYER(hdr);
    info->bitrate_kbps = hdr_bitrate_kbps(hdr);

    if (!pcm)
    {
        return hdr_frame_samples(hdr);
    }

    bs_init(bs_frame, hdr + HDR_SIZE, frame_size - HDR_SIZE);
    if (HDR_IS_CRC(hdr))
    {
        get_bits(bs_frame, 16);
    }

    if (info->layer == 3)
    {
        int main_data_begin = L3_read_side_info(bs_frame, scratch.gr_info, hdr);
        if (main_data_begin < 0 || bs_frame->pos > bs_frame->limit)
        {
            mp3dec_init(dec);
            return 0;
        }
        success = L3_restore_reservoir(dec, bs_frame, &scratch, main_data_begin);
        if (success)
        {
            for (igr = 0; igr < (HDR_TEST_MPEG1(hdr) ? 2 : 1); igr++, pcm += 576*info->channels)
            {
                memset(scratch.grbuf[0], 0, 576*2*sizeof(float));
                L3_decode(dec, &scratch, scratch.gr_info + igr*info->channels, info->channels);
                mp3d_synth_granule(dec->qmf_state, scratch.grbuf[0], 18, info->channels, pcm, scratch.syn[0]);
            }
        }
        L3_save_reservoir(dec, &scratch);
    } else
    {
#ifdef MINIMP3_ONLY_MP3
        return 0;
#else /* MINIMP3_ONLY_MP3 */
        L12_scale_info sci[1];
        L12_read_scale_info(hdr, bs_frame, sci);

        memset(scratch.grbuf[0], 0, 576*2*sizeof(float));
        for (i = 0, igr = 0; igr < 3; igr++)
        {
            if (12 == (i += L12_dequantize_granule(scratch.grbuf[0] + i, bs_frame, sci, info->layer | 1)))
            {
                i = 0;
                L12_apply_scf_384(sci, sci->scf + igr, scratch.grbuf[0]);
                mp3d_synth_granule(dec->qmf_state, scratch.grbuf[0], 12, info->channels, pcm, scratch.syn[0]);
                memset(scratch.grbuf[0], 0, 576*2*sizeof(float));
                pcm += 384*info->channels;
            }
            if (bs_frame->pos > bs_frame->limit)
            {
                mp3dec_init(dec);
                return 0;
            }
        }
#endif /* MINIMP3_ONLY_MP3 */
    }
    return success*hdr_frame_samples(dec->header);
}

#ifdef MINIMP3_FLOAT_OUTPUT
void mp3dec_f32_to_s16(const float *in, int16_t *out, int num_samples)
{
    int i = 0;
#if HAVE_SIMD
    int aligned_count = num_samples & ~7;
    for(; i < aligned_count; i += 8)
    {
        static const f4 g_scale = { 32768.0f, 32768.0f, 32768.0f, 32768.0f };
        f4 a = VMUL(VLD(&in[i  ]), g_scale);
        f4 b = VMUL(VLD(&in[i+4]), g_scale);
#if HAVE_SSE
        static const f4 g_max = { 32767.0f, 32767.0f, 32767.0f, 32767.0f };
        static const f4 g_min = { -32768.0f, -32768.0f, -32768.0f, -32768.0f };
        __m128i pcm8 = _mm_packs_epi32(_mm_cvtps_epi32(_mm_max_ps(_mm_min_ps(a, g_max), g_min)),
                                       _mm_cvtps_epi32(_mm_max_ps(_mm_min_ps(b, g_max), g_min)));
        out[i  ] = _mm_extract_epi16(pcm8, 0);
        out[i+1] = _mm_extract_epi16(pcm8, 1);
        out[i+2] = _mm_extract_epi16(pcm8, 2);
        out[i+3] = _mm_extract_epi16(pcm8, 3);
        out[i+4] = _mm_extract_epi16(pcm8, 4);
        out[i+5] = _mm_extract_epi16(pcm8, 5);
        out[i+6] = _mm_extract_epi16(pcm8, 6);
        out[i+7] = _mm_extract_epi16(pcm8, 7);
#else /* HAVE_SSE */
        int16x4_t pcma, pcmb;
        a = VADD(a, VSET(0.5f));
        b = VADD(b, VSET(0.5f));
        pcma = vqmovn_s32(vqaddq_s32(vcvtq_s32_f32(a), vreinterpretq_s32_u32(vcltq_f32(a, VSET(0)))));
        pcmb = vqmovn_s32(vqaddq_s32(vcvtq_s32_f32(b), vreinterpretq_s32_u32(vcltq_f32(b, VSET(0)))));
        vst1_lane_s16(out+i  , pcma, 0);
        vst1_lane_s16(out+i+1, pcma, 1);
        vst1_lane_s16(out+i+2, pcma, 2);
        vst1_lane_s16(out+i+3, pcma, 3);
        vst1_lane_s16(out+i+4, pcmb, 0);
        vst1_lane_s16(out+i+5, pcmb, 1);
        vst1_lane_s16(out+i+6, pcmb, 2);
        vst1_lane_s16(out+i+7, pcmb, 3);
#endif /* HAVE_SSE */
    }
#endif /* HAVE_SIMD */
    for(; i < num_samples; i++)
    {
        float sample = in[i] * 32768.0f;
        if (sample >=  32766.5)
            out[i] = (int16_t) 32767;
        else if (sample <= -32767.5)
            out[i] = (int16_t)-32768;
        else
        {
            int16_t s = (int16_t)(sample + .5f);
            s -= (s < 0);   /* away from zero, to be compliant */
            out[i] = s;
        }
    }
}
#endif /* MINIMP3_FLOAT_OUTPUT */
#endif /* MINIMP3_IMPLEMENTATION && !_MINIMP3_IMPLEMENTATION_GUARD */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <mmreg.h>
#include <commctrl.h>
#include <shellapi.h>
#include <mmsystem.h>
#include <mmdeviceapi.h>
#include <functiondiscoverykeys_devpkey.h>
#include <Shlwapi.h>
#include "playback.h"
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "oleaut32.lib")
#pragma comment(lib, "shlwapi.lib")
#include <array>
#include <algorithm>
#include <cstdint>
#include <cwctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _MSC_VER
#pragma comment(lib, "winmm.lib")
#pragma comment(lib, "comctl32.lib")
#endif

constexpr int SLOT_COUNT = 6;
constexpr int BUTTON_ID_BASE = 100;
constexpr int RESET_BUTTON_ID_BASE = 200;
constexpr int KEYBIND_EDIT_ID_BASE = 300;
constexpr int KEYBIND_SET_BUTTON_ID_BASE = 400;
constexpr int BUTTON_WIDTH = 260;
constexpr int RESET_BUTTON_WIDTH = 90;
constexpr int SLIDER_WIDTH = 260;
constexpr int PERCENT_LABEL_WIDTH = 90;
constexpr int BUTTON_HEIGHT = 40;
constexpr int TRACK_HEIGHT = 25;
constexpr int CONTROL_LEFT = 20;
constexpr int CONTROL_SPACING = 10;
constexpr int CONTROL_SPACING_Y = 115;
constexpr int TAB_CONTROL_HEIGHT = 30;
constexpr int MAX_MCI_VOLUME = 1000;
constexpr int MIC_DEVICE_COMBO_ID = 500;
constexpr int MIC_REFRESH_BUTTON_ID = 600;
constexpr int MIC_DIALOG_NEXT_BUTTON_ID = 601;
constexpr int MIC_DIALOG_CANCEL_BUTTON_ID = 602;
constexpr int OUTPUT_DEVICE_COMBO_ID = 610;
constexpr int MIC_CONFIRM_BUTTON_ID = 620;
constexpr int MIC_NOTE_LABEL_HEIGHT = 60;
constexpr size_t STREAM_THRESHOLD = 1024 * 1024;

std::array<HWND, SLOT_COUNT> gButtons{};
std::array<HWND, SLOT_COUNT> gClearButtons{};
std::array<HWND, SLOT_COUNT> gSliders{};
std::array<HWND, SLOT_COUNT> gPercentLabels{};
std::array<HWND, SLOT_COUNT> gKeybindLabels{};
std::array<HWND, SLOT_COUNT> gKeybindEdits{};
std::array<HWND, SLOT_COUNT> gKeybindSetButtons{};
HWND gTabControl = nullptr;
HBRUSH gBackgroundBrush = nullptr;
HWND gMicLabel = nullptr;
HWND gMicDeviceCombo = nullptr;
HWND gMicSelectedLabel = nullptr;
HWND gMicRefreshButton = nullptr;
HWND gOutputLabel = nullptr;
HWND gOutputDeviceCombo = nullptr;
HWND gOutputSelectedLabel = nullptr;
HWND gMicNoteLabel = nullptr;
HWND gConfirmButton = nullptr;
int gSelectedMicDevice = -1;
int gSelectedOutputDevice = -1;
std::wstring gSelectedMicName;
std::wstring gVirtualMicName;

struct SoundSlot
{
    std::wstring path;
    int volume = 100;
    bool usingPlaySound = false;
    std::wstring keybind;
    HWAVEOUT hWaveOut = nullptr;
    std::vector<BYTE> wavData;
    WAVEFORMATEX wf{};
    WAVEHDR waveHdr{};
    HANDLE hFile = INVALID_HANDLE_VALUE;
    bool streaming = false;
    BYTE streamBuf[16384]{};
    WAVEHDR streamHdr{};
};

struct SelectMicData
{
    HWND combo = nullptr;
    bool completed = false;
};

SoundSlot gSlots[SLOT_COUNT];
static const PROPERTYKEY PKEY_Device_ListenToThisDevice = {
    {0x24dbb0fc, 0x9311, 0x4b3d, {0x9c, 0xf0, 0x18, 0xff, 0x15, 0x56, 0x39, 0xd4}}, 1
};
static const PROPERTYKEY PKEY_Device_PlaybackThrough = {
    {0x24dbb0fc, 0x9311, 0x4b3d, {0x9c, 0xf0, 0x18, 0xff, 0x15, 0x56, 0x39, 0xd4}}, 0
};

struct SavedMicProperty
{
    std::wstring deviceName;
    PROPVARIANT listenValue{};
    PROPVARIANT playbackValue{};
    bool hasListen = false;
    bool hasPlayback = false;
};
static std::vector<SavedMicProperty> gSavedMicProps;

static std::wstring GetDeviceName(IMMDevice* device)
{
    IPropertyStore* store = nullptr;
    if (FAILED(device->OpenPropertyStore(STGM_READ, &store)))
        return L"";
    PROPVARIANT nameProp;
    PropVariantInit(&nameProp);
    store->GetValue(PKEY_Device_FriendlyName, &nameProp);
    std::wstring result;
    if (nameProp.vt == VT_LPWSTR && nameProp.pwszVal)
        result = nameProp.pwszVal;
    PropVariantClear(&nameProp);
    store->Release();
    return result;
}

static std::wstring GetDeviceId(IMMDevice* device)
{
    LPWSTR id = nullptr;
    if (FAILED(device->GetId(&id)))
        return L"";
    std::wstring result(id);
    CoTaskMemFree(id);
    return result;
}

static bool ConfigureVBCableRouting()
{
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);

    IMMDeviceEnumerator* enumerator = nullptr;
    if (FAILED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_INPROC_SERVER,
                               __uuidof(IMMDeviceEnumerator), reinterpret_cast<void**>(&enumerator))))
        return false;
    IMMDeviceCollection* renderDevices = nullptr;
    if (FAILED(enumerator->EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE, &renderDevices)))
    {
        enumerator->Release();
        return false;
    }

    IMMDevice* vbCableDevice = nullptr;
    UINT renderCount = 0;
    renderDevices->GetCount(&renderCount);
    for (UINT i = 0; i < renderCount; ++i)
    {
        IMMDevice* dev = nullptr;
        renderDevices->Item(i, &dev);
        std::wstring name = GetDeviceName(dev);
        if (name.find(L"CABLE") != std::wstring::npos || name.find(L"VB-Audio") != std::wstring::npos)
        {
            vbCableDevice = dev;
            break;
        }
        dev->Release();
    }

    if (!vbCableDevice)
    {
        renderDevices->Release();
        enumerator->Release();
        return false;
    }

    std::wstring vbCableId = GetDeviceId(vbCableDevice);
    IMMDeviceCollection* captureDevices = nullptr;
    if (FAILED(enumerator->EnumAudioEndpoints(eCapture, DEVICE_STATE_ACTIVE, &captureDevices)))
    {
        vbCableDevice->Release();
        renderDevices->Release();
        enumerator->Release();
        return false;
    }

    UINT captureCount = 0;
    captureDevices->GetCount(&captureCount);
    bool configured = false;

    for (UINT i = 0; i < captureCount; ++i)
    {
        IMMDevice* dev = nullptr;
        captureDevices->Item(i, &dev);
        std::wstring name = GetDeviceName(dev);
        if (name.find(L"CABLE") != std::wstring::npos ||
            name.find(L"VB-Audio") != std::wstring::npos ||
            name.find(L"Virtual") != std::wstring::npos)
        {
            dev->Release();
            continue;
        }

        IPropertyStore* store = nullptr;
        if (SUCCEEDED(dev->OpenPropertyStore(STGM_WRITE, &store)))
        {
            SavedMicProperty saved;
            saved.deviceName = name;
            PropVariantInit(&saved.listenValue);
            PropVariantInit(&saved.playbackValue);

            if (SUCCEEDED(store->GetValue(PKEY_Device_ListenToThisDevice, &saved.listenValue)))
                saved.hasListen = true;
            if (SUCCEEDED(store->GetValue(PKEY_Device_PlaybackThrough, &saved.playbackValue)))
                saved.hasPlayback = true;
            gSavedMicProps.push_back(saved);
            PROPVARIANT listenProp;
            listenProp.vt = VT_BOOL;
            listenProp.boolVal = -1;
            store->SetValue(PKEY_Device_ListenToThisDevice, listenProp);
            std::wstring playbackStr = L"{0.0.0.00000000}." + vbCableId;
            PROPVARIANT playbackProp;
            playbackProp.vt = VT_LPWSTR;
            playbackProp.pwszVal = const_cast<LPWSTR>(playbackStr.c_str());
            store->SetValue(PKEY_Device_PlaybackThrough, playbackProp);

            configured = true;
            store->Release();
        }

        dev->Release();
        if (configured) break;
    }

    captureDevices->Release();
    vbCableDevice->Release();
    renderDevices->Release();
    enumerator->Release();
    return configured;
}

static void RestoreMicRouting()
{
    if (gSavedMicProps.empty()) return;

    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    IMMDeviceEnumerator* enumerator = nullptr;
    if (FAILED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_INPROC_SERVER,
                               __uuidof(IMMDeviceEnumerator), reinterpret_cast<void**>(&enumerator))))
        return;

    IMMDeviceCollection* captureDevices = nullptr;
    enumerator->EnumAudioEndpoints(eCapture, DEVICE_STATE_ACTIVE, &captureDevices);

    UINT captureCount = 0;
    captureDevices->GetCount(&captureCount);

    for (auto& saved : gSavedMicProps)
    {
        for (UINT i = 0; i < captureCount; ++i)
        {
            IMMDevice* dev = nullptr;
            captureDevices->Item(i, &dev);
            if (GetDeviceName(dev) == saved.deviceName)
            {
                IPropertyStore* store = nullptr;
                if (SUCCEEDED(dev->OpenPropertyStore(STGM_WRITE, &store)))
                {
                    if (saved.hasListen)
                        store->SetValue(PKEY_Device_ListenToThisDevice, saved.listenValue);
                    else
                    {
                        PROPVARIANT v;
                        PropVariantInit(&v);
                        v.vt = VT_BOOL;
                        v.boolVal = 0;
                        store->SetValue(PKEY_Device_ListenToThisDevice, v);
                    }
                    if (saved.hasPlayback)
                        store->SetValue(PKEY_Device_PlaybackThrough, saved.playbackValue);
                    store->Release();
                }
                dev->Release();
                break;
            }
            dev->Release();
        }
    }

    captureDevices->Release();
    enumerator->Release();
    gSavedMicProps.clear();
}

constexpr int MIC_BUF_COUNT = 4;
constexpr DWORD MIC_BUF_SIZE = 4096;

struct MicPassthrough
{
    HWAVEIN hWaveIn = nullptr;
    HWAVEOUT hWaveOut = nullptr;
    WAVEFORMATEX wf{};
    WAVEHDR hdrs[MIC_BUF_COUNT]{};
    BYTE bufs[MIC_BUF_COUNT][MIC_BUF_SIZE]{};
    bool running = false;
    HWND hwnd = nullptr;
    WAVEHDR outHdr{};
    bool outInProgress = false;
    std::vector<BYTE> sbData;
    volatile LONG sbPos = 0;
    volatile bool sbPlaying = false;
    BYTE outBuf[MIC_BUF_SIZE]{};
};

MicPassthrough gMic{};

void CALLBACK MicInProc(HWAVEIN, UINT uMsg, DWORD_PTR dwInstance, DWORD_PTR dwParam1, DWORD_PTR)
{
    if (uMsg != MM_WIM_DATA) return;
    auto* mic = reinterpret_cast<MicPassthrough*>(dwInstance);
    if (!mic->running || mic->hWaveOut == nullptr) return;

    WAVEHDR* hdr = reinterpret_cast<WAVEHDR*>(dwParam1);
    if (!(hdr->dwFlags & WHDR_DONE)) return;

    if (!mic->outInProgress)
    {
        if (mic->sbPlaying && mic->sbPos < static_cast<LONG>(mic->sbData.size()))
        {
            memcpy(mic->outBuf, hdr->lpData, hdr->dwBytesRecorded);
            DWORD micCount = hdr->dwBytesRecorded / sizeof(SHORT);
            SHORT* out = reinterpret_cast<SHORT*>(mic->outBuf);
            LONG pos = mic->sbPos;
            for (DWORD i = 0; i < micCount; ++i)
            {
                if (pos + static_cast<LONG>(sizeof(SHORT)) <= static_cast<LONG>(mic->sbData.size()))
                {
                    SHORT sb = *reinterpret_cast<const SHORT*>(mic->sbData.data() + pos);
                    LONG mixed = static_cast<LONG>(out[i]) + static_cast<LONG>(sb);
                    out[i] = static_cast<SHORT>(std::clamp(mixed, -32768L, 32767L));
                    pos += sizeof(SHORT);
                }
            }
            mic->sbPos = pos;
            if (pos >= static_cast<LONG>(mic->sbData.size()))
                mic->sbPlaying = false;

            ZeroMemory(&mic->outHdr, sizeof(WAVEHDR));
            mic->outHdr.lpData = reinterpret_cast<LPSTR>(mic->outBuf);
            mic->outHdr.dwBufferLength = hdr->dwBytesRecorded;
        }
        else
        {
            ZeroMemory(&mic->outHdr, sizeof(WAVEHDR));
            mic->outHdr.lpData = hdr->lpData;
            mic->outHdr.dwBufferLength = hdr->dwBytesRecorded;
        }
        mic->outInProgress = true;
        waveOutPrepareHeader(mic->hWaveOut, &mic->outHdr, sizeof(WAVEHDR));
        waveOutWrite(mic->hWaveOut, &mic->outHdr, sizeof(WAVEHDR));
    }

    waveInAddBuffer(mic->hWaveIn, hdr, sizeof(WAVEHDR));
}

void CALLBACK MicOutProc(HWAVEOUT hwo, UINT uMsg, DWORD_PTR, DWORD_PTR dwParam1, DWORD_PTR)
{
    if (uMsg != MM_WOM_DONE) return;
    WAVEHDR* hdr = reinterpret_cast<WAVEHDR*>(dwParam1);
    waveOutUnprepareHeader(hwo, hdr, sizeof(WAVEHDR));
    gMic.outInProgress = false;
}

bool StartMicPassthrough(HWND hwnd)
{
    if (gMic.running) return true;
    if (gSelectedMicDevice < 0 || gSelectedOutputDevice < 0) return false;

    gMic.wf.wFormatTag = WAVE_FORMAT_PCM;
    gMic.wf.nChannels = 1;
    gMic.wf.nSamplesPerSec = 44100;
    gMic.wf.wBitsPerSample = 16;
    gMic.wf.nBlockAlign = gMic.wf.nChannels * gMic.wf.wBitsPerSample / 8;
    gMic.wf.nAvgBytesPerSec = gMic.wf.nSamplesPerSec * gMic.wf.nBlockAlign;
    gMic.wf.cbSize = 0;
    gMic.hwnd = hwnd;

    UINT outDev = static_cast<UINT>(gSelectedOutputDevice);
    MMRESULT mr = waveOutOpen(&gMic.hWaveOut, outDev, &gMic.wf, reinterpret_cast<DWORD_PTR>(MicOutProc), reinterpret_cast<DWORD_PTR>(&gMic), CALLBACK_FUNCTION);
    if (mr != MMSYSERR_NOERROR) return false;

    mr = waveInOpen(&gMic.hWaveIn, gSelectedMicDevice, &gMic.wf, reinterpret_cast<DWORD_PTR>(MicInProc), reinterpret_cast<DWORD_PTR>(&gMic), CALLBACK_FUNCTION);
    if (mr != MMSYSERR_NOERROR)
    {
        waveOutClose(gMic.hWaveOut);
        gMic.hWaveOut = nullptr;
        return false;
    }

    for (int i = 0; i < MIC_BUF_COUNT; ++i)
    {
        ZeroMemory(&gMic.hdrs[i], sizeof(WAVEHDR));
        gMic.hdrs[i].lpData = reinterpret_cast<LPSTR>(gMic.bufs[i]);
        gMic.hdrs[i].dwBufferLength = MIC_BUF_SIZE;
        waveInPrepareHeader(gMic.hWaveIn, &gMic.hdrs[i], sizeof(WAVEHDR));
        waveInAddBuffer(gMic.hWaveIn, &gMic.hdrs[i], sizeof(WAVEHDR));
    }

    gMic.running = true;
    waveInStart(gMic.hWaveIn);
    return true;
}

void StopMicPassthrough()
{
    if (!gMic.running) return;
    gMic.running = false;

    if (gMic.hWaveIn)
    {
        waveInStop(gMic.hWaveIn);
        waveInReset(gMic.hWaveIn);
        for (int i = 0; i < MIC_BUF_COUNT; ++i)
        {
            if (gMic.hdrs[i].dwFlags & WHDR_PREPARED)
                waveInUnprepareHeader(gMic.hWaveIn, &gMic.hdrs[i], sizeof(WAVEHDR));
        }
        waveInClose(gMic.hWaveIn);
        gMic.hWaveIn = nullptr;
    }

    if (gMic.hWaveOut)
    {
        waveOutReset(gMic.hWaveOut);
        gMic.outInProgress = false;
        waveOutClose(gMic.hWaveOut);
        gMic.hWaveOut = nullptr;
    }
}

namespace fs = std::filesystem;

fs::path GetConfigPath()
{
    wchar_t exePath[MAX_PATH];
    GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    fs::path configPath(exePath);
    configPath.replace_filename(L"SoundBoardpp.config");
    return configPath;
}

std::wstring GetFileName(const std::wstring& path)
{
    size_t pos = path.find_last_of(L"\\/");

    if (pos == std::wstring::npos)
        return path;

    return path.substr(pos + 1);
}

bool IsSupportedAudioFile(const std::wstring& path)
{
    fs::path filePath(path);
    std::wstring ext = filePath.extension().wstring();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::towlower);
    return ext == L".wav" || ext == L".mp3";
}

std::wstring GetMciFileType(const std::wstring& path)
{
    fs::path filePath(path);
    std::wstring ext = filePath.extension().wstring();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::towlower);

    if (ext == L".wav")
        return L"waveaudio";
    if (ext == L".mp3")
        return L"mpegvideo";

    return L"";
}

int ScaleMciVolume(int volume)
{
    return std::clamp(volume * 10, 0, MAX_MCI_VOLUME);
}

bool IsPlainTypingHotkey(UINT modifiers, UINT vk)
{
    if (modifiers != 0)
        return false;

    return (vk >= 0x41 && vk <= 0x5A) ||
           (vk >= 0x30 && vk <= 0x39) ||
           vk == VK_SPACE;
}

bool IsModifierKey(UINT vk)
{
    return vk == VK_CONTROL || vk == VK_LCONTROL || vk == VK_RCONTROL ||
           vk == VK_MENU || vk == VK_LMENU || vk == VK_RMENU ||
           vk == VK_SHIFT || vk == VK_LSHIFT || vk == VK_RSHIFT;
}

DWORD MakeWaveVolume(int volume)
{
    DWORD channelVolume = static_cast<DWORD>(0xFFFF * std::clamp(volume, 0, 100) / 100);
    return (channelVolume & 0xFFFF) | (channelVolume << 16);
}

void ApplySlotVolume(int index)
{
    if (index < 0 || index >= SLOT_COUNT)
        return;

    if (gSlots[index].hWaveOut != nullptr)
    {
        waveOutSetVolume(gSlots[index].hWaveOut, MakeWaveVolume(gSlots[index].volume));
        return;
    }

    if (gSlots[index].usingPlaySound)
    {
        waveOutSetVolume(nullptr, MakeWaveVolume(gSlots[index].volume));
        return;
    }

    std::wstring alias = L"slot" + std::to_wstring(index);
    std::wstring volumeCommand = L"setaudio " + alias + L" volume to " + std::to_wstring(ScaleMciVolume(gSlots[index].volume));
    mciSendStringW(volumeCommand.c_str(), nullptr, 0, nullptr);
}

void ApplyMciWaveOutputDevice(const std::wstring& alias, const std::wstring& fileType)
{
    if (gSelectedOutputDevice < 0 || fileType != L"waveaudio")
        return;

    MCIDEVICEID deviceId = mciGetDeviceIDW(alias.c_str());
    if (deviceId == 0)
        return;

    MCI_WAVE_SET_PARMS waveSet{};
    waveSet.wOutput = static_cast<UINT>(gSelectedOutputDevice);
    mciSendCommandW(deviceId, MCI_SET, MCI_WAVE_OUTPUT, reinterpret_cast<DWORD_PTR>(&waveSet));
}

bool ParseHotkey(const std::wstring& keyStringIn, UINT &outModifiers, UINT &outVk)
{
    std::wstring s = keyStringIn;
    s.erase(std::remove_if(s.begin(), s.end(), ::iswspace), s.end());
    std::transform(s.begin(), s.end(), s.begin(), ::towupper);

    outModifiers = 0;
    outVk = 0;
    size_t pos = 0;
    while (true)
    {
        if (s.rfind(L"CTRL+", pos) == 0) { outModifiers |= MOD_CONTROL; s.erase(0,5); continue; }
        if (s.rfind(L"ALT+", pos) == 0) { outModifiers |= MOD_ALT; s.erase(0,4); continue; }
        if (s.rfind(L"SHIFT+", pos) == 0) { outModifiers |= MOD_SHIFT; s.erase(0,6); continue; }
        break;
    }

    if (s.size() >= 2 && s[0] == L'F')
    {
        try {
            int fnum = std::stoi(std::wstring(s.begin() + 1, s.end()));
            if (fnum >= 1 && fnum <= 24) { outVk = VK_F1 + fnum - 1; return true; }
        } catch(...) { return false; }
    }

    if (s.size() == 1)
    {
        wchar_t c = s[0];
        if (c >= L'A' && c <= L'Z') { outVk = 0x41 + (c - L'A'); return true; }
        if (c >= L'0' && c <= L'9') { outVk = 0x30 + (c - L'0'); return true; }
    }

    if (s == L"SPACE") { outVk = VK_SPACE; return true; }

    if (s.size() > 2 && s.rfind(L"VK", 0) == 0)
    {
        try
        {
            int vk = std::stoi(s.substr(2));
            if (vk > 0 && vk <= 0xFE)
            {
                outVk = static_cast<UINT>(vk);
                return true;
            }
        }
        catch (...)
        {
            return false;
        }
    }

    return false;
}

bool RegisterSlotHotkey(HWND hwnd, int slot, const std::wstring& keyString)
{
    if (slot < 0 || slot >= SLOT_COUNT)
        return false;

    UINT modifiers = 0;
    UINT vk = 0;
    if (!ParseHotkey(keyString, modifiers, vk) || vk == 0)
        return false;
    if (IsPlainTypingHotkey(modifiers, vk))
        return false;

    int hotkeyId = slot + 1;
    std::wstring previousKeybind = gSlots[slot].keybind;
    UnregisterHotKey(hwnd, hotkeyId);
    if (!RegisterHotKey(hwnd, hotkeyId, modifiers, vk))
    {
        UINT previousModifiers = 0;
        UINT previousVk = 0;
        if (!previousKeybind.empty() &&
            ParseHotkey(previousKeybind, previousModifiers, previousVk) &&
            !IsPlainTypingHotkey(previousModifiers, previousVk))
        {
            RegisterHotKey(hwnd, hotkeyId, previousModifiers, previousVk);
        }
        return false;
    }

    gSlots[slot].keybind = keyString;
    return true;
}

void UnregisterAllHotkeys(HWND hwnd)
{
    for (int i = 0; i < SLOT_COUNT; ++i)
    {
        UnregisterHotKey(hwnd, i + 1);
    }
}
void InjectKeyForSlot(HWND hwnd, int slot)
{
    if (slot < 0 || slot >= SLOT_COUNT) return;
    std::wstring key = gSlots[slot].keybind;
    if (key.empty()) return;

    UINT modifiers = 0, vk = 0;
    if (!ParseHotkey(key, modifiers, vk) || vk == 0) return;
    UnregisterHotKey(hwnd, slot + 1);

    INPUT inputs[2];
    ZeroMemory(inputs, sizeof(inputs));
    inputs[0].type = INPUT_KEYBOARD;
    inputs[0].ki.wVk = static_cast<WORD>(vk);
    inputs[1] = inputs[0];
    inputs[1].ki.dwFlags = KEYEVENTF_KEYUP;

    SendInput(2, inputs, sizeof(INPUT));
    RegisterSlotHotkey(hwnd, slot, key);
}

void SaveSlots()
{
    fs::path configPath = GetConfigPath();
    std::wofstream ofs(configPath, std::ios::trunc);
    if (!ofs)
        return;

    for (int i = 0; i < SLOT_COUNT; ++i)
    {
        ofs << i << L'\t' << gSlots[i].volume << L'\t' << gSlots[i].path << L'\t' << gSlots[i].keybind << L"\n";
    }
}

void LoadSlots()
{
    fs::path configPath = GetConfigPath();
    std::wifstream ifs(configPath);
    if (!ifs)
        return;

    std::wstring line;
    while (std::getline(ifs, line))
    {
        if (line.empty())
            continue;

        std::wistringstream iss(line);
        int index = 0;
        int volume = 100;
        if (!(iss >> index >> volume))
            continue;

        if (index < 0 || index >= SLOT_COUNT)
            continue;

        wchar_t separator;
        iss.get(separator);
        std::wstring path;
        std::getline(iss, path, L'\t');

        std::wstring keybind;
        std::getline(iss, keybind);

        if (!path.empty() && fs::exists(path) && IsSupportedAudioFile(path))
        {
            gSlots[index].path = std::move(path);
        }

        gSlots[index].volume = std::clamp(volume, 0, 100);
        if (!keybind.empty())
        {
            gSlots[index].keybind = std::move(keybind);
        }
    }
}

void UpdateMicStatus();
void PopulateMicDeviceList();
void PopulateOutputDeviceList();
void UpdateOutputStatus();
void StopSlotSound(int index);
void CloseMciAliasIfIdle(int index);
bool StartMicPassthrough(HWND hwnd);
void StopMicPassthrough();
bool ShowMicrophoneSelectionDialog(HINSTANCE hInst);
void PopulateInitialMicDeviceList(HWND combo);

void UpdateSlotControls(int index)
{
    std::wstring buttonText = gSlots[index].path.empty()
        ? L"Slot " + std::to_wstring(index + 1)
        : GetFileName(gSlots[index].path);

    SetWindowTextW(gButtons[index], buttonText.c_str());
    SendMessageW(gSliders[index], TBM_SETPOS, TRUE, gSlots[index].volume);

    std::wstring percentText = std::to_wstring(gSlots[index].volume) + L"%";
    SetWindowTextW(gPercentLabels[index], percentText.c_str());
    SetWindowTextW(gKeybindEdits[index], gSlots[index].keybind.c_str());
}

void UpdateTabVisibility(HWND /*hwnd*/)
{
    int sel = TabCtrl_GetCurSel(gTabControl);
    bool showSlotPage = sel == 0;
    bool showKeybindPage = sel == 1;
    bool showMicPage = sel == 2;

    for (int i = 0; i < SLOT_COUNT; ++i)
    {
        ShowWindow(gButtons[i], showSlotPage ? SW_SHOW : SW_HIDE);
        ShowWindow(gClearButtons[i], showSlotPage ? SW_SHOW : SW_HIDE);
        ShowWindow(gSliders[i], showSlotPage ? SW_SHOW : SW_HIDE);
        ShowWindow(gPercentLabels[i], showSlotPage ? SW_SHOW : SW_HIDE);
        ShowWindow(gKeybindLabels[i], showKeybindPage ? SW_SHOW : SW_HIDE);
        ShowWindow(gKeybindEdits[i], showKeybindPage ? SW_SHOW : SW_HIDE);
        ShowWindow(gKeybindSetButtons[i], showKeybindPage ? SW_SHOW : SW_HIDE);
    }

    ShowWindow(gMicLabel, showMicPage ? SW_SHOW : SW_HIDE);
    ShowWindow(gMicDeviceCombo, showMicPage ? SW_SHOW : SW_HIDE);
    ShowWindow(gMicSelectedLabel, showMicPage ? SW_SHOW : SW_HIDE);
    ShowWindow(gMicRefreshButton, showMicPage ? SW_SHOW : SW_HIDE);
    ShowWindow(gOutputLabel, showMicPage ? SW_SHOW : SW_HIDE);
    ShowWindow(gOutputDeviceCombo, showMicPage ? SW_SHOW : SW_HIDE);
    ShowWindow(gOutputSelectedLabel, showMicPage ? SW_SHOW : SW_HIDE);
    ShowWindow(gConfirmButton, showMicPage ? SW_SHOW : SW_HIDE);
    ShowWindow(gMicNoteLabel, showMicPage ? SW_SHOW : SW_HIDE);
}

void PopulateMicDeviceList()
{
    if (!gMicDeviceCombo)
        return;

    SendMessageW(gMicDeviceCombo, CB_RESETCONTENT, 0, 0);
    int deviceCount = waveInGetNumDevs();
    if (deviceCount <= 0)
    {
        SendMessageW(gMicDeviceCombo, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"No input devices found"));
        gSelectedMicDevice = -1;
        if (gMicSelectedLabel)
            SetWindowTextW(gMicSelectedLabel, L"No microphone devices available.");
        return;
    }

    for (int i = 0; i < deviceCount; ++i)
    {
        WAVEINCAPSW caps{};
        if (waveInGetDevCapsW(i, &caps, sizeof(caps)) == MMSYSERR_NOERROR)
        {
            int index = static_cast<int>(SendMessageW(gMicDeviceCombo, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(caps.szPname)));
            SendMessageW(gMicDeviceCombo, CB_SETITEMDATA, index, i);
        }
    }

    if (!gSelectedMicName.empty())
    {
        int count = static_cast<int>(SendMessageW(gMicDeviceCombo, CB_GETCOUNT, 0, 0));
        for (int i = 0; i < count; ++i)
        {
            wchar_t name[256] = {};
            SendMessageW(gMicDeviceCombo, CB_GETLBTEXT, i, reinterpret_cast<LPARAM>(name));
            if (gSelectedMicName == name)
            {
                SendMessageW(gMicDeviceCombo, CB_SETCURSEL, i, 0);
                UpdateMicStatus();
                return;
            }
        }
    }

    SendMessageW(gMicDeviceCombo, CB_SETCURSEL, 0, 0);
    UpdateMicStatus();
}

void PopulateOutputDeviceList()
{
    if (!gOutputDeviceCombo)
        return;

    SendMessageW(gOutputDeviceCombo, CB_RESETCONTENT, 0, 0);
    int deviceCount = waveOutGetNumDevs();
    if (deviceCount <= 0)
    {
        SendMessageW(gOutputDeviceCombo, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"No output devices found"));
        gSelectedOutputDevice = -1;
        if (gOutputSelectedLabel)
            SetWindowTextW(gOutputSelectedLabel, L"No output devices available.");
        return;
    }

    int vbCableIndex = -1;
    for (int i = 0; i < deviceCount; ++i)
    {
        WAVEOUTCAPSW caps{};
        if (waveOutGetDevCapsW(i, &caps, sizeof(caps)) == MMSYSERR_NOERROR)
        {
            int index = static_cast<int>(SendMessageW(gOutputDeviceCombo, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(caps.szPname)));
            SendMessageW(gOutputDeviceCombo, CB_SETITEMDATA, index, i);
            if (wcsstr(caps.szPname, L"CABLE") || wcsstr(caps.szPname, L"Virtual"))
            {
                vbCableIndex = index;
            }
        }
    }
    if (vbCableIndex >= 0)
    {
        SendMessageW(gOutputDeviceCombo, CB_SETCURSEL, vbCableIndex, 0);
    }
    else
    {
        SendMessageW(gOutputDeviceCombo, CB_SETCURSEL, 0, 0);
        int result = MessageBoxW(
            nullptr,
            L"VB-Cable Virtual Audio Device not found.\n\n"
            L"To use SoundBoardpp with Discord/Zoom/Teams, you need to install VB-Cable.\n\n"
            L"Download link (copy and paste in your browser):\n"
            L"https://vb-audio.com/Cable/\n\n"
            L"Click OK to continue or Cancel to download now.",
            L"VB-Cable Not Installed",
            MB_OKCANCEL | MB_ICONINFORMATION);
        
        if (result == IDCANCEL)
        {
            ShellExecuteW(nullptr, L"open", L"https://vb-audio.com/Cable/", nullptr, nullptr, SW_SHOW);
        }
    }
    UpdateOutputStatus();
}

void UpdateOutputStatus()
{
    if (!gOutputDeviceCombo)
        return;

    int sel = static_cast<int>(SendMessageW(gOutputDeviceCombo, CB_GETCURSEL, 0, 0));
    if (sel == CB_ERR)
    {
        gSelectedOutputDevice = -1;
        if (gOutputSelectedLabel)
            SetWindowTextW(gOutputSelectedLabel, L"No output device selected.");
        return;
    }

    gSelectedOutputDevice = static_cast<int>(SendMessageW(gOutputDeviceCombo, CB_GETITEMDATA, sel, 0));
    if (gOutputSelectedLabel)
    {
        wchar_t name[256] = {};
        SendMessageW(gOutputDeviceCombo, CB_GETLBTEXT, sel, reinterpret_cast<LPARAM>(name));
        std::wstring text = L"Playing to: ";
        text += name;
        SetWindowTextW(gOutputSelectedLabel, text.c_str());
    }

    StopMicPassthrough();
    if (gSelectedMicDevice >= 0 && gSelectedOutputDevice >= 0)
        StartMicPassthrough(gOutputDeviceCombo ? GetParent(gOutputDeviceCombo) : nullptr);
}

void UpdateMicStatus()
{
    if (!gMicDeviceCombo)
        return;

    int sel = static_cast<int>(SendMessageW(gMicDeviceCombo, CB_GETCURSEL, 0, 0));
    if (sel == CB_ERR)
    {
        gSelectedMicDevice = -1;
        if (gMicSelectedLabel)
            SetWindowTextW(gMicSelectedLabel, L"No microphone selected.");
        return;
    }

    gSelectedMicDevice = static_cast<int>(SendMessageW(gMicDeviceCombo, CB_GETITEMDATA, sel, 0));
    if (gMicSelectedLabel)
    {
        wchar_t name[256] = {};
        SendMessageW(gMicDeviceCombo, CB_GETLBTEXT, sel, reinterpret_cast<LPARAM>(name));
        gSelectedMicName = name;
        std::wstring text = L"Selected device: ";
        text += gSelectedMicName;
        SetWindowTextW(gMicSelectedLabel, text.c_str());
    }

    StopMicPassthrough();
    if (gSelectedMicDevice >= 0 && gSelectedOutputDevice >= 0)
        StartMicPassthrough(gMicDeviceCombo ? GetParent(gMicDeviceCombo) : nullptr);
}

void PopulateInitialMicDeviceList(HWND combo)
{
    if (!combo)
        return;

    SendMessageW(combo, CB_RESETCONTENT, 0, 0);
    int deviceCount = waveInGetNumDevs();
    if (deviceCount <= 0)
        return;

    for (int i = 0; i < deviceCount; ++i)
    {
        WAVEINCAPSW caps{};
        if (waveInGetDevCapsW(i, &caps, sizeof(caps)) == MMSYSERR_NOERROR)
        {
            int index = static_cast<int>(SendMessageW(combo, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(caps.szPname)));
            SendMessageW(combo, CB_SETITEMDATA, index, i);
        }
    }

    if (!gSelectedMicName.empty())
    {
        int count = static_cast<int>(SendMessageW(combo, CB_GETCOUNT, 0, 0));
        for (int i = 0; i < count; ++i)
        {
            wchar_t name[256] = {};
            SendMessageW(combo, CB_GETLBTEXT, i, reinterpret_cast<LPARAM>(name));
            if (gSelectedMicName == name)
            {
                SendMessageW(combo, CB_SETCURSEL, i, 0);
                return;
            }
        }
    }

    SendMessageW(combo, CB_SETCURSEL, 0, 0);
}

LRESULT CALLBACK SelectMicDialogProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    SelectMicData* data = reinterpret_cast<SelectMicData*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));

    switch (msg)
    {
    case WM_CREATE:
    {
        CREATESTRUCTW* cs = reinterpret_cast<CREATESTRUCTW*>(lParam);
        data = reinterpret_cast<SelectMicData*>(cs->lpCreateParams);
        if (!data)
            return -1;

        data->combo = nullptr;
        data->completed = false;
        SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(data));

        CreateWindowW(
            L"STATIC",
            L"Choose your microphone:",
            WS_CHILD | WS_VISIBLE | SS_LEFT,
            20,
            20,
            360,
            20,
            hwnd,
            nullptr,
            reinterpret_cast<HINSTANCE>(GetWindowLongPtr(hwnd, GWLP_HINSTANCE)),
            nullptr);

        data->combo = CreateWindowW(
            L"COMBOBOX",
            nullptr,
            WS_CHILD | WS_VISIBLE | WS_BORDER | WS_VSCROLL | CBS_DROPDOWNLIST | WS_TABSTOP,
            20,
            50,
            360,
            120,
            hwnd,
            reinterpret_cast<HMENU>(static_cast<INT_PTR>(MIC_DEVICE_COMBO_ID)),
            reinterpret_cast<HINSTANCE>(GetWindowLongPtr(hwnd, GWLP_HINSTANCE)),
            nullptr);

        CreateWindowW(
            L"BUTTON",
            L"Refresh",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            20,
            90,
            80,
            30,
            hwnd,
            reinterpret_cast<HMENU>(static_cast<INT_PTR>(MIC_REFRESH_BUTTON_ID)),
            reinterpret_cast<HINSTANCE>(GetWindowLongPtr(hwnd, GWLP_HINSTANCE)),
            nullptr);

        CreateWindowW(
            L"BUTTON",
            L"Next",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            210,
            90,
            80,
            30,
            hwnd,
            reinterpret_cast<HMENU>(static_cast<INT_PTR>(MIC_DIALOG_NEXT_BUTTON_ID)),
            reinterpret_cast<HINSTANCE>(GetWindowLongPtr(hwnd, GWLP_HINSTANCE)),
            nullptr);

        CreateWindowW(
            L"BUTTON",
            L"Cancel",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            300,
            90,
            80,
            30,
            hwnd,
            reinterpret_cast<HMENU>(static_cast<INT_PTR>(MIC_DIALOG_CANCEL_BUTTON_ID)),
            reinterpret_cast<HINSTANCE>(GetWindowLongPtr(hwnd, GWLP_HINSTANCE)),
            nullptr);

        PopulateInitialMicDeviceList(data->combo);
        break;
    }

    case WM_COMMAND:
    {
        int id = LOWORD(wParam);
        if (id == MIC_REFRESH_BUTTON_ID)
        {
            PopulateInitialMicDeviceList(data->combo);
            break;
        }
        if (id == MIC_DIALOG_CANCEL_BUTTON_ID)
        {
            data->completed = false;
            DestroyWindow(hwnd);
            break;
        }
        if (id == MIC_DIALOG_NEXT_BUTTON_ID)
        {
            int sel = static_cast<int>(SendMessageW(data->combo, CB_GETCURSEL, 0, 0));
            if (sel == CB_ERR)
            {
                MessageBoxW(hwnd, L"Please choose a microphone before continuing.", L"SoundBoardpp", MB_OK | MB_ICONINFORMATION);
                break;
            }

            wchar_t name[256] = {};
            SendMessageW(data->combo, CB_GETLBTEXT, sel, reinterpret_cast<LPARAM>(name));
            gSelectedMicName = name;
            data->completed = true;
            DestroyWindow(hwnd);
            break;
        }
        break;
    }

    case WM_DESTROY:
    {
        return 0;
    }
    }

    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

bool ShowMicrophoneSelectionDialog(HINSTANCE hInst)
{
    const wchar_t DIALOG_CLASS[] = L"SoundBoardppMicrophoneSelector";

    SelectMicData* context = new SelectMicData();

    WNDCLASSW wc{};
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = SelectMicDialogProc;
    wc.hInstance = hInst;
    wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
    wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_BTNFACE + 1);
    wc.lpszClassName = DIALOG_CLASS;

    RegisterClassW(&wc);

    HWND dlg = CreateWindowExW(
        0,
        DIALOG_CLASS,
        L"Setup Microphone",
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        420,
        170,
        nullptr,
        nullptr,
        hInst,
        reinterpret_cast<LPVOID>(context));

    if (!dlg)
    {
        delete context;
        return false;
    }

    ShowWindow(dlg, SW_SHOW);

    MSG msg{};
    while (IsWindow(dlg) && GetMessage(&msg, nullptr, 0, 0))
    {
        if (!IsDialogMessage(dlg, &msg))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    bool result = context->completed;
    delete context;
    return result;
}

void CloseMciAliasIfIdle(int index)
{
    std::wstring alias = L"slot" + std::to_wstring(index);
    std::wstring statusCmd = L"status " + alias + L" mode";
    wchar_t buf[64] = {};
    mciSendStringW(statusCmd.c_str(), buf, 64, nullptr);
    if (wcscmp(buf, L"stopped") == 0 || buf[0] == L'\0')
    {
        mciSendStringW((L"close " + alias).c_str(), nullptr, 0, nullptr);
    }
}

void StopSlotSound(int index)
{
    std::wstring alias = L"slot" + std::to_wstring(index);
    if (gSlots[index].usingPlaySound)
    {
        PlaySoundW(nullptr, nullptr, 0);
        gSlots[index].usingPlaySound = false;
        return;
    }
    if (gSlots[index].hWaveOut != nullptr)
    {
        waveOutReset(gSlots[index].hWaveOut);
        if (gSlots[index].streaming)
        {
            waveOutUnprepareHeader(gSlots[index].hWaveOut, &gSlots[index].streamHdr, sizeof(WAVEHDR));
        }
        else
        {
            waveOutUnprepareHeader(gSlots[index].hWaveOut, &gSlots[index].waveHdr, sizeof(WAVEHDR));
        }
        waveOutClose(gSlots[index].hWaveOut);
        gSlots[index].hWaveOut = nullptr;
        std::vector<BYTE>().swap(gSlots[index].wavData);
        ZeroMemory(&gSlots[index].waveHdr, sizeof(WAVEHDR));
        ZeroMemory(&gSlots[index].streamHdr, sizeof(WAVEHDR));
        ZeroMemory(&gSlots[index].wf, sizeof(WAVEFORMATEX));

        if (gSlots[index].hFile != INVALID_HANDLE_VALUE)
        {
            CloseHandle(gSlots[index].hFile);
            gSlots[index].hFile = INVALID_HANDLE_VALUE;
        }
        gSlots[index].streaming = false;
        return;
    }

    mciSendStringW((L"stop " + alias).c_str(), nullptr, 0, nullptr);
    mciSendStringW((L"close " + alias).c_str(), nullptr, 0, nullptr);
}

bool PlayPcmData(int index, const WAVEFORMATEX& wf, std::vector<BYTE>&& data)
{
    if (index < 0 || index >= SLOT_COUNT || data.empty())
        return false;

    gSlots[index].usingPlaySound = false;
    UINT outputDevice = WAVE_MAPPER;
    HWAVEOUT hwo = nullptr;
    MMRESULT result = waveOutOpen(&hwo, outputDevice, &wf, 0, 0, CALLBACK_NULL);
    if (result != MMSYSERR_NOERROR || hwo == nullptr)
        return false;

    gSlots[index].hWaveOut = hwo;
    gSlots[index].wavData = std::move(data);
    gSlots[index].wf = wf;
    gSlots[index].streaming = false;
    gSlots[index].hFile = INVALID_HANDLE_VALUE;
    ZeroMemory(&gSlots[index].waveHdr, sizeof(WAVEHDR));
    gSlots[index].waveHdr.dwBufferLength = static_cast<DWORD>(gSlots[index].wavData.size());
    gSlots[index].waveHdr.lpData = reinterpret_cast<LPSTR>(gSlots[index].wavData.data());

    result = waveOutPrepareHeader(gSlots[index].hWaveOut, &gSlots[index].waveHdr, sizeof(WAVEHDR));
    if (result != MMSYSERR_NOERROR) { StopSlotSound(index); return false; }

    ApplySlotVolume(index);
    result = waveOutWrite(gSlots[index].hWaveOut, &gSlots[index].waveHdr, sizeof(WAVEHDR));
    if (result != MMSYSERR_NOERROR) { StopSlotSound(index); return false; }

    return true;
}

constexpr size_t STREAM_BUF_SIZE = 16384;

bool ParseWavHeaders(const std::wstring& path, WAVEFORMATEX& outWf, uint32_t& outDataSize, uint64_t& outDataOffset)
{
    HANDLE h = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE)
        return false;

    LARGE_INTEGER fileSize = {};
    if (!GetFileSizeEx(h, &fileSize) || fileSize.QuadPart < 12)
    {
        CloseHandle(h);
        return false;
    }

    BYTE header[12] = {};
    DWORD read = 0;
    if (!ReadFile(h, header, 12, &read, nullptr) || read != 12)
    {
        CloseHandle(h);
        return false;
    }

    if (memcmp(header, "RIFF", 4) != 0 || memcmp(header + 8, "WAVE", 4) != 0)
    {
        CloseHandle(h);
        return false;
    }

    bool fmtFound = false, dataFound = false;
    WAVEFORMATEX tmpWf{};
    uint64_t remaining = static_cast<uint64_t>(fileSize.QuadPart) - 12;

    while (remaining >= 8 && (!fmtFound || !dataFound))
    {
        BYTE chunkId[4] = {};
        uint32_t chunkSize = 0;
        DWORD r = 0;

        if (!ReadFile(h, chunkId, 4, &r, nullptr) || r != 4) break;
        if (!ReadFile(h, &chunkSize, 4, &r, nullptr) || r != 4) break;
        remaining -= 8;

        if (chunkSize > remaining) break;

        if (memcmp(chunkId, "fmt ", 4) == 0)
        {
            BYTE fmtBuf[40] = {};
            DWORD toRead = (chunkSize < 40) ? chunkSize : 40;
            if (!ReadFile(h, fmtBuf, toRead, &r, nullptr) || r != toRead) break;
            if (toRead >= 16)
            {
                tmpWf.wFormatTag = *reinterpret_cast<const WORD*>(fmtBuf);
                tmpWf.nChannels = *reinterpret_cast<const WORD*>(fmtBuf + 2);
                tmpWf.nSamplesPerSec = *reinterpret_cast<const DWORD*>(fmtBuf + 4);
                tmpWf.nAvgBytesPerSec = *reinterpret_cast<const DWORD*>(fmtBuf + 8);
                tmpWf.nBlockAlign = *reinterpret_cast<const WORD*>(fmtBuf + 12);
                tmpWf.wBitsPerSample = *reinterpret_cast<const WORD*>(fmtBuf + 14);
                tmpWf.cbSize = 0;
                fmtFound = true;
            }
            DWORD skip = chunkSize - toRead;
            if (skip > 0)
            {
                LARGE_INTEGER li{};
                li.QuadPart = skip;
                SetFilePointerEx(h, li, nullptr, FILE_CURRENT);
            }
            remaining -= chunkSize;
        }
        else if (memcmp(chunkId, "data", 4) == 0)
        {
            outDataSize = chunkSize;
            outDataOffset = static_cast<uint64_t>(fileSize.QuadPart) - remaining;
            dataFound = true;
            break;
        }
        else
        {
            LARGE_INTEGER li{};
            li.QuadPart = chunkSize;
            SetFilePointerEx(h, li, nullptr, FILE_CURRENT);
            remaining -= chunkSize;
        }
    }

    CloseHandle(h);

    if (!fmtFound || !dataFound) return false;
    if (tmpWf.wFormatTag != WAVE_FORMAT_PCM) return false;
    outWf = tmpWf;
    return true;
}

bool PlayWavStream(int index)
{
    if (index < 0 || index >= SLOT_COUNT) return false;
    if (gSlots[index].path.empty()) return false;

    LARGE_INTEGER fileSize = {};
    HANDLE hCheck = CreateFileW(gSlots[index].path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hCheck == INVALID_HANDLE_VALUE) return false;
    GetFileSizeEx(hCheck, &fileSize);
    CloseHandle(hCheck);

    if (static_cast<uint64_t>(fileSize.QuadPart) <= STREAM_THRESHOLD)
        return false;

    WAVEFORMATEX wf{};
    uint32_t dataSize = 0;
    uint64_t dataOffset = 0;
    if (!ParseWavHeaders(gSlots[index].path, wf, dataSize, dataOffset))
        return false;

    HANDLE h = CreateFileW(gSlots[index].path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;

    LARGE_INTEGER li{};
    li.QuadPart = dataOffset;
    SetFilePointerEx(h, li, nullptr, FILE_BEGIN);

    UINT outputDevice = (gSelectedOutputDevice >= 0) ? static_cast<UINT>(gSelectedOutputDevice) : WAVE_MAPPER;
    HWAVEOUT hwo = nullptr;
    MMRESULT result = waveOutOpen(&hwo, outputDevice, &wf, 0, 0, CALLBACK_NULL);
    if (result != MMSYSERR_NOERROR || hwo == nullptr)
    {
        CloseHandle(h);
        return false;
    }

    gSlots[index].hWaveOut = hwo;
    gSlots[index].hFile = h;
    gSlots[index].wf = wf;
    gSlots[index].streaming = true;
    gSlots[index].usingPlaySound = false;

    DWORD toRead = (dataSize < STREAM_BUF_SIZE) ? dataSize : STREAM_BUF_SIZE;
    DWORD bytesRead = 0;
    if (!ReadFile(h, gSlots[index].streamBuf, toRead, &bytesRead, nullptr) || bytesRead == 0)
    {
        waveOutClose(hwo);
        CloseHandle(h);
        gSlots[index].hWaveOut = nullptr;
        gSlots[index].hFile = INVALID_HANDLE_VALUE;
        gSlots[index].streaming = false;
        return false;
    }

    ZeroMemory(&gSlots[index].streamHdr, sizeof(WAVEHDR));
    gSlots[index].streamHdr.dwBufferLength = bytesRead;
    gSlots[index].streamHdr.lpData = reinterpret_cast<LPSTR>(gSlots[index].streamBuf);

    result = waveOutPrepareHeader(hwo, &gSlots[index].streamHdr, sizeof(WAVEHDR));
    if (result != MMSYSERR_NOERROR)
    {
        waveOutClose(hwo);
        CloseHandle(h);
        gSlots[index].hWaveOut = nullptr;
        gSlots[index].hFile = INVALID_HANDLE_VALUE;
        gSlots[index].streaming = false;
        return false;
    }

    ApplySlotVolume(index);
    result = waveOutWrite(hwo, &gSlots[index].streamHdr, sizeof(WAVEHDR));
    if (result != MMSYSERR_NOERROR)
    {
        waveOutUnprepareHeader(hwo, &gSlots[index].streamHdr, sizeof(WAVEHDR));
        waveOutClose(hwo);
        CloseHandle(h);
        gSlots[index].hWaveOut = nullptr;
        gSlots[index].hFile = INVALID_HANDLE_VALUE;
        gSlots[index].streaming = false;
        return false;
    }

    return true;
}

void PlaySlotSound(int index)
{
    if (gSlots[index].path.empty())
        return;

    if (!IsSupportedAudioFile(gSlots[index].path))
        return;

    CloseMciAliasIfIdle(index);
    StopSlotSound(index);

    std::wstring alias = L"slot" + std::to_wstring(index);
    std::wstring fileType = GetMciFileType(gSlots[index].path);
    if (fileType == L"waveaudio")
    {
        if (PlayWavStream(index))
            return;

        WAVEFORMATEX wf{};
        std::vector<BYTE> data;
        auto LoadWavPCM = [&](const std::wstring &p, WAVEFORMATEX &outWf, std::vector<BYTE> &outData)->bool
        {
            HANDLE h = CreateFileW(p.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (h == INVALID_HANDLE_VALUE)
                return false;

            LARGE_INTEGER fileSize = {};
            if (!GetFileSizeEx(h, &fileSize) || fileSize.QuadPart < 12)
            {
                CloseHandle(h);
                return false;
            }

            BYTE header[12] = {};
            DWORD read = 0;
            if (!ReadFile(h, header, 12, &read, nullptr) || read != 12)
            {
                CloseHandle(h);
                return false;
            }

            if (memcmp(header, "RIFF", 4) != 0 || memcmp(header + 8, "WAVE", 4) != 0)
            {
                CloseHandle(h);
                return false;
            }

            bool fmtFound = false, dataFound = false;
            WAVEFORMATEX tmpWf{};
            uint32_t dataSize = 0;
            uint64_t remaining = static_cast<uint64_t>(fileSize.QuadPart) - 12;

            while (remaining >= 8 && (!fmtFound || !dataFound))
            {
                BYTE chunkId[4] = {};
                uint32_t chunkSize = 0;
                DWORD r = 0;

                if (!ReadFile(h, chunkId, 4, &r, nullptr) || r != 4) break;
                if (!ReadFile(h, &chunkSize, 4, &r, nullptr) || r != 4) break;
                remaining -= 8;

                if (chunkSize > remaining) break;

                if (memcmp(chunkId, "fmt ", 4) == 0)
                {
                    BYTE fmtBuf[40] = {};
                    DWORD toRead = (chunkSize < 40) ? chunkSize : 40;
                    if (!ReadFile(h, fmtBuf, toRead, &r, nullptr) || r != toRead) break;
                    if (toRead >= 16)
                    {
                        tmpWf.wFormatTag = *reinterpret_cast<const WORD*>(fmtBuf);
                        tmpWf.nChannels = *reinterpret_cast<const WORD*>(fmtBuf + 2);
                        tmpWf.nSamplesPerSec = *reinterpret_cast<const DWORD*>(fmtBuf + 4);
                        tmpWf.nAvgBytesPerSec = *reinterpret_cast<const DWORD*>(fmtBuf + 8);
                        tmpWf.nBlockAlign = *reinterpret_cast<const WORD*>(fmtBuf + 12);
                        tmpWf.wBitsPerSample = *reinterpret_cast<const WORD*>(fmtBuf + 14);
                        tmpWf.cbSize = 0;
                        fmtFound = true;
                    }
                    DWORD skip = chunkSize - toRead;
                    if (skip > 0)
                    {
                        LARGE_INTEGER li{};
                        li.QuadPart = skip;
                        SetFilePointerEx(h, li, nullptr, FILE_CURRENT);
                    }
                    remaining -= chunkSize;
                }
                else if (memcmp(chunkId, "data", 4) == 0)
                {
                    outData.resize(chunkSize);
                    if (!ReadFile(h, outData.data(), chunkSize, &r, nullptr) || r != chunkSize)
                    {
                        std::vector<BYTE>().swap(outData);
                        CloseHandle(h);
                        return false;
                    }
                    dataSize = chunkSize;
                    dataFound = true;
                    remaining -= chunkSize;
                }
                else
                {
                    LARGE_INTEGER li{};
                    li.QuadPart = chunkSize;
                    SetFilePointerEx(h, li, nullptr, FILE_CURRENT);
                    remaining -= chunkSize;
                }
            }

            CloseHandle(h);

            if (!fmtFound || !dataFound) return false;
            if (tmpWf.wFormatTag != WAVE_FORMAT_PCM) return false;
            outWf = tmpWf;
            return true;
        };

        if (LoadWavPCM(gSlots[index].path, wf, data) &&
            PlayPcmData(index, wf, std::move(data)))
        {
            return;
        }
    }
    if (fileType == L"mpegvideo")
    {
        bool decoded = false;
        WAVEFORMATEX pcmWf{};
        std::vector<BYTE> pcmData;

        HANDLE hFile = CreateFileW(gSlots[index].path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (hFile != INVALID_HANDLE_VALUE)
        {
            LARGE_INTEGER fSize{};
            GetFileSizeEx(hFile, &fSize);
            DWORD mp3Size = static_cast<DWORD>(fSize.QuadPart);
            std::vector<BYTE> mp3Buf(mp3Size);
            DWORD rd = 0;
            ReadFile(hFile, mp3Buf.data(), mp3Size, &rd, nullptr);
            CloseHandle(hFile);

            mp3dec_t dec;
            mp3dec_init(&dec);

            std::vector<short> pcmBuf(MINIMP3_MAX_SAMPLES_PER_FRAME * 1024);
            int totalSamples = 0;
            int channels = 0;
            int sampleRate = 0;
            DWORD consumed = 0;

            while (consumed < mp3Size)
            {
                mp3dec_frame_info_t info{};
                int samples = mp3dec_decode_frame(&dec, mp3Buf.data() + consumed, mp3Size - consumed, pcmBuf.data() + totalSamples, &info);

                if (info.frame_bytes == 0)
                    break;

                consumed += info.frame_bytes;

                if (samples > 0)
                {
                    if (channels == 0)
                    {
                        channels = info.channels;
                        sampleRate = info.hz;
                    }
                    totalSamples += samples * info.channels;

                    if (totalSamples + MINIMP3_MAX_SAMPLES_PER_FRAME * 2 > (int)pcmBuf.size())
                        pcmBuf.resize(pcmBuf.size() + MINIMP3_MAX_SAMPLES_PER_FRAME * 2048);
                }
            }

            if (totalSamples > 0 && channels > 0 && sampleRate > 0)
            {
                pcmWf.wFormatTag = WAVE_FORMAT_PCM;
                pcmWf.nChannels = static_cast<WORD>(channels);
                pcmWf.nSamplesPerSec = sampleRate;
                pcmWf.wBitsPerSample = 16;
                pcmWf.nBlockAlign = pcmWf.nChannels * pcmWf.wBitsPerSample / 8;
                pcmWf.nAvgBytesPerSec = pcmWf.nSamplesPerSec * pcmWf.nBlockAlign;
                pcmWf.cbSize = 0;

                DWORD dataBytes = totalSamples * sizeof(short);
                pcmData.resize(dataBytes);
                memcpy(pcmData.data(), pcmBuf.data(), dataBytes);
                decoded = true;
            }
        }
        std::wstring mciOpen = L"open \"" + gSlots[index].path + L"\" type mpegvideo alias " + alias;
        MCIERROR mciErr = mciSendStringW(mciOpen.c_str(), nullptr, 0, nullptr);
        if (mciErr != 0)
        {
            mciOpen = L"open \"" + gSlots[index].path + L"\" alias " + alias;
            mciErr = mciSendStringW(mciOpen.c_str(), nullptr, 0, nullptr);
        }
        if (mciErr == 0)
        {
            gSlots[index].usingPlaySound = false;
            ApplySlotVolume(index);
            mciSendStringW((L"play " + alias + L" from 0").c_str(), nullptr, 0, nullptr);
        }
        PlaySoundToVBCable(gSlots[index].path.c_str());

        return;
    }
    std::wstring openCommand = L"open \"" + gSlots[index].path + L"\"";
    if (!fileType.empty())
    {
        openCommand += L" type " + fileType;
    }
    openCommand += L" alias " + alias;

    MCIERROR mciErr = mciSendStringW(openCommand.c_str(), nullptr, 0, nullptr);
    if (mciErr != 0)
    {
        std::wstring openAuto = L"open \"" + gSlots[index].path + L"\" alias " + alias;
        mciErr = mciSendStringW(openAuto.c_str(), nullptr, 0, nullptr);
    }

    if (mciErr == 0)
    {
        gSlots[index].usingPlaySound = false;
        ApplySlotVolume(index);
        std::wstring playCommand = L"play " + alias + L" from 0";
        mciSendStringW(playCommand.c_str(), nullptr, 0, nullptr);
        return;
    }
    if (PlaySoundW(gSlots[index].path.c_str(), nullptr, SND_FILENAME | SND_ASYNC | SND_NODEFAULT))
    {
        gSlots[index].usingPlaySound = true;
        ApplySlotVolume(index);
    }
}
std::wstring FormatHotkeyString(UINT modifiers, UINT vk);
void CaptureHotkeyInteractive(HWND hwnd, int slot);
void RefillStreamBuffer(int index);

constexpr UINT_PTR STREAM_TIMER_ID = 1;
constexpr UINT STREAM_TIMER_MS = 50;

void RefillStreamBuffer(int index)
{
    if (index < 0 || index >= SLOT_COUNT) return;
    if (!gSlots[index].streaming || gSlots[index].hWaveOut == nullptr) return;
    if (gSlots[index].hFile == INVALID_HANDLE_VALUE) return;

    if ((gSlots[index].streamHdr.dwFlags & WHDR_DONE) == 0)
        return;

    waveOutUnprepareHeader(gSlots[index].hWaveOut, &gSlots[index].streamHdr, sizeof(WAVEHDR));

    DWORD bytesRead = 0;
    if (!ReadFile(gSlots[index].hFile, gSlots[index].streamBuf, STREAM_BUF_SIZE, &bytesRead, nullptr) || bytesRead == 0)
    {
        waveOutClose(gSlots[index].hWaveOut);
        CloseHandle(gSlots[index].hFile);
        gSlots[index].hWaveOut = nullptr;
        gSlots[index].hFile = INVALID_HANDLE_VALUE;
        gSlots[index].streaming = false;
        return;
    }

    ZeroMemory(&gSlots[index].streamHdr, sizeof(WAVEHDR));
    gSlots[index].streamHdr.dwBufferLength = bytesRead;
    gSlots[index].streamHdr.lpData = reinterpret_cast<LPSTR>(gSlots[index].streamBuf);

    waveOutPrepareHeader(gSlots[index].hWaveOut, &gSlots[index].streamHdr, sizeof(WAVEHDR));
    waveOutWrite(gSlots[index].hWaveOut, &gSlots[index].streamHdr, sizeof(WAVEHDR));
}

void CheckStreamingDone()
{
    for (int i = 0; i < SLOT_COUNT; ++i)
    {
        if (gSlots[i].streaming && gSlots[i].hWaveOut != nullptr)
        {
            if ((gSlots[i].streamHdr.dwFlags & WHDR_DONE) != 0 && (gSlots[i].streamHdr.dwBytesRecorded == 0))
            {
                waveOutUnprepareHeader(gSlots[i].hWaveOut, &gSlots[i].streamHdr, sizeof(WAVEHDR));
                waveOutClose(gSlots[i].hWaveOut);
                CloseHandle(gSlots[i].hFile);
                gSlots[i].hWaveOut = nullptr;
                gSlots[i].hFile = INVALID_HANDLE_VALUE;
                gSlots[i].streaming = false;
            }
            else
            {
                RefillStreamBuffer(i);
            }
        }
    }
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT msg,
                            WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_HSCROLL:
    {
        HWND slider = reinterpret_cast<HWND>(lParam);

        for (int i = 0; i < SLOT_COUNT; ++i)
        {
            if (slider == gSliders[i])
            {
                int volume = static_cast<int>(
                    SendMessageW(gSliders[i],
                                 TBM_GETPOS,
                                 0,
                                 0));

                gSlots[i].volume = volume;
                UpdateSlotControls(i);

                ApplySlotVolume(i);

                SaveSlots();
                break;
            }
        }
        return 0;
    }

    case WM_TIMER:
    {
        if (wParam == STREAM_TIMER_ID)
        {
            CheckStreamingDone();
            return 0;
        }
        break;
    }

    case WM_COMMAND:
    {
        int id = LOWORD(wParam);

        if (id >= BUTTON_ID_BASE && id < BUTTON_ID_BASE + SLOT_COUNT)
        {
            int slot = id - BUTTON_ID_BASE;

            if (!gSlots[slot].path.empty())
            {
                PlaySlotSound(slot);
            }
            else
            {
                MessageBoxW(hwnd,
                            L"No sound loaded",
                            L"SoundBoard",
                            MB_OK);
            }
        }
        else if (id >= RESET_BUTTON_ID_BASE && id < RESET_BUTTON_ID_BASE + SLOT_COUNT)
        {
            int slot = id - RESET_BUTTON_ID_BASE;
            gSlots[slot].path.clear();
            gSlots[slot].volume = 100;
            gSlots[slot].keybind.clear();
            UnregisterHotKey(hwnd, slot + 1);
            StopSlotSound(slot);
            UpdateSlotControls(slot);
            SaveSlots();
        }
        else if (id >= KEYBIND_SET_BUTTON_ID_BASE && id < KEYBIND_SET_BUTTON_ID_BASE + SLOT_COUNT)
        {
            int slot = id - KEYBIND_SET_BUTTON_ID_BASE;
            CaptureHotkeyInteractive(hwnd, slot);
        }
        else if (id == MIC_REFRESH_BUTTON_ID)
        {
            PopulateMicDeviceList();
        }
        else if (id == MIC_CONFIRM_BUTTON_ID)
        {
            if (gSelectedMicDevice >= 0 && gSelectedOutputDevice >= 0)
            {
                if (!gMic.running)
                    StartMicPassthrough(hwnd);
                MessageBoxW(hwnd, L"Mic and output configured!\n\nIn Discord, set Input Device to:\n'CABLE Output (VB-Audio Virtual Cable)'", L"SoundBoard", MB_OK | MB_ICONINFORMATION);
            }
            else
            {
                MessageBoxW(hwnd, L"Please select a microphone first.", L"SoundBoard", MB_OK | MB_ICONWARNING);
            }
        }
        else if (id == MIC_DEVICE_COMBO_ID && HIWORD(wParam) == CBN_SELCHANGE)
        {
            UpdateMicStatus();
        }
        else if (id == OUTPUT_DEVICE_COMBO_ID && HIWORD(wParam) == CBN_SELCHANGE)
        {
            UpdateOutputStatus();
        }

        return 0;
    }

    case WM_HOTKEY:
    {
        int slot = static_cast<int>(wParam) - 1;
        if (slot >= 0 && slot < SLOT_COUNT)
        {
            PlaySlotSound(slot);
            InjectKeyForSlot(hwnd, slot);
            return 0;
        }
        break;
    }

    case WM_NOTIFY:
    {
        LPNMHDR hdr = reinterpret_cast<LPNMHDR>(lParam);
        if (hdr->hwndFrom == gTabControl && hdr->code == TCN_SELCHANGE)
        {
            UpdateTabVisibility(hwnd);
            return 0;
        }
        break;
    }

    case WM_CTLCOLORBTN:
    {
        HDC hdc = reinterpret_cast<HDC>(wParam);

        SetBkColor(hdc, RGB(47,49,54));
        SetTextColor(hdc, RGB(255,255,255));
        return reinterpret_cast<INT_PTR>(gBackgroundBrush);
    }

    case WM_ERASEBKGND:
    {
        RECT rc;
        GetClientRect(hwnd, &rc);
        FillRect(reinterpret_cast<HDC>(wParam), &rc, gBackgroundBrush);
        return 1;
    }

    case WM_DROPFILES:
    {
        HDROP hDrop = reinterpret_cast<HDROP>(wParam);
        wchar_t filePath[MAX_PATH] = {};
        POINT pt = {};

        DragQueryFileW(hDrop, 0, filePath, MAX_PATH);
        DragQueryPoint(hDrop, &pt);

        for (int i = 0; i < SLOT_COUNT; ++i)
        {
            RECT rect;
            GetWindowRect(gButtons[i], &rect);

            POINT topLeft = { rect.left, rect.top };
            POINT bottomRight = { rect.right, rect.bottom };
            ScreenToClient(hwnd, &topLeft);
            ScreenToClient(hwnd, &bottomRight);

            rect.left = topLeft.x;
            rect.top = topLeft.y;
            rect.right = bottomRight.x;
            rect.bottom = bottomRight.y;

            if (PtInRect(&rect, pt))
            {
                if (!IsSupportedAudioFile(filePath))
                {
                    MessageBoxW(hwnd,
                                L"Only .wav and .mp3 files are supported.",
                                L"Unsupported file",
                                MB_OK | MB_ICONWARNING);
                    break;
                }

                gSlots[i].path = filePath;
                UpdateSlotControls(i);
                SaveSlots();
                break;
            }
        }

        DragFinish(hDrop);
        return 0;
    }

    case WM_DESTROY:
        KillTimer(hwnd, STREAM_TIMER_ID);
        StopMicPassthrough();
        DragAcceptFiles(hwnd, FALSE);
        UnregisterAllHotkeys(hwnd);
        for (int i = 0; i < SLOT_COUNT; ++i)
        {
            StopSlotSound(i);
        }

        if (gBackgroundBrush != nullptr)
        {
            DeleteObject(gBackgroundBrush);
            gBackgroundBrush = nullptr;
        }
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(
        hwnd,
        msg,
        wParam,
        lParam
    );
}

int WINAPI wWinMain(HINSTANCE hInst,
                    HINSTANCE,
                    LPWSTR,
                    int nCmdShow)
{
    INITCOMMONCONTROLSEX icc{};
    icc.dwSize = sizeof(icc);
    icc.dwICC = ICC_TAB_CLASSES | ICC_BAR_CLASSES;

    InitCommonControlsEx(&icc);
    SetUnhandledExceptionFilter([](PEXCEPTION_POINTERS ep) -> LONG {
        wchar_t msg[512];
        swprintf_s(msg, L"Crash! Code: 0x%08X\nAddr: 0x%p\n\nThe application will close.",
            ep->ExceptionRecord->ExceptionCode, ep->ExceptionRecord->ExceptionAddress);
        MessageBoxW(nullptr, msg, L"SoundBoard - Crash", MB_OK | MB_ICONERROR);
        return EXCEPTION_EXECUTE_HANDLER;
    });

    const wchar_t CLASS_NAME[] =
        L"DiscordSoundBoard";

    gBackgroundBrush = CreateSolidBrush(RGB(47,49,54));

    WNDCLASSEXW wcex{};
    wcex.cbSize = sizeof(wcex);
    wcex.style = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc = WindowProc;
    wcex.hInstance = hInst;
    wcex.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
    wcex.hbrBackground = gBackgroundBrush;
    wcex.lpszClassName = CLASS_NAME;

    RegisterClassExW(&wcex);

    const int startY = 20 + TAB_CONTROL_HEIGHT + CONTROL_SPACING;
    const int windowClientWidth = 400;
    const int windowClientHeight = 720;
    RECT windowRect = {0, 0, windowClientWidth, windowClientHeight};
    AdjustWindowRect(&windowRect, WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX, FALSE);
    int windowWidth = windowRect.right - windowRect.left;
    int windowHeight = windowRect.bottom - windowRect.top;

    if (!ShowMicrophoneSelectionDialog(hInst))
    {
        return 0;
    }

    HWND hwnd = CreateWindowExW(
            0,
            CLASS_NAME,
            L"SoundBoardpp",
            WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            windowWidth,
            windowHeight,
            nullptr,
            nullptr,
            hInst,
            nullptr
        );

    if (!hwnd)
        return 0;

    DragAcceptFiles(hwnd, TRUE);

    TCITEMW item{};
    item.mask = TCIF_TEXT;
    item.pszText = const_cast<LPWSTR>(L"Slots");

    gTabControl = CreateWindowExW(
        0,
        WC_TABCONTROLW,
        nullptr,
        WS_VISIBLE | WS_CHILD | WS_TABSTOP,
        CONTROL_LEFT,
        20,
        windowClientWidth - CONTROL_LEFT * 2,
        TAB_CONTROL_HEIGHT,
        hwnd,
        nullptr,
        hInst,
        nullptr);

    TabCtrl_InsertItem(gTabControl, 0, &item);
    item.pszText = const_cast<LPWSTR>(L"Keybinds");
    TabCtrl_InsertItem(gTabControl, 1, &item);
    item.pszText = const_cast<LPWSTR>(L"Microphone");
    TabCtrl_InsertItem(gTabControl, 2, &item);

    for (int i = 0; i < SLOT_COUNT; i++)
    {
        int y = startY + i * CONTROL_SPACING_Y;
        std::wstring buttonText = L"Slot " + std::to_wstring(i + 1);

        gButtons[i] = CreateWindowW(
            L"BUTTON",
            buttonText.c_str(),
            WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
            CONTROL_LEFT,
            y,
            BUTTON_WIDTH,
            BUTTON_HEIGHT,
            hwnd,
            reinterpret_cast<HMENU>(static_cast<INT_PTR>(BUTTON_ID_BASE + i)),
            hInst,
            nullptr);

        gClearButtons[i] = CreateWindowW(
            L"BUTTON",
            L"Clear",
            WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
            CONTROL_LEFT + BUTTON_WIDTH + CONTROL_SPACING,
            y,
            RESET_BUTTON_WIDTH,
            BUTTON_HEIGHT,
            hwnd,
            reinterpret_cast<HMENU>(static_cast<INT_PTR>(RESET_BUTTON_ID_BASE + i)),
            hInst,
            nullptr);

        gSliders[i] = CreateWindowExW(
            0,
            TRACKBAR_CLASSW,
            L"",
            WS_VISIBLE | WS_CHILD | TBS_AUTOTICKS,
            CONTROL_LEFT,
            y + BUTTON_HEIGHT + CONTROL_SPACING,
            SLIDER_WIDTH,
            TRACK_HEIGHT,
            hwnd,
            nullptr,
            hInst,
            nullptr);

        gPercentLabels[i] = CreateWindowW(
            L"STATIC",
            L"100%",
            WS_VISIBLE | WS_CHILD | SS_CENTER,
            CONTROL_LEFT + SLIDER_WIDTH + CONTROL_SPACING,
            y + BUTTON_HEIGHT + CONTROL_SPACING,
            PERCENT_LABEL_WIDTH,
            TRACK_HEIGHT,
            hwnd,
            nullptr,
            hInst,
            nullptr);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   

        std::wstring slotLabel = L"Slot " + std::to_wstring(i + 1) + L":";
        gKeybindLabels[i] = CreateWindowW(
            L"STATIC",
            slotLabel.c_str(),
            WS_CHILD | WS_VISIBLE | SS_LEFT,
            CONTROL_LEFT,
            y + BUTTON_HEIGHT + CONTROL_SPACING,
            60,
            BUTTON_HEIGHT,
            hwnd,
            nullptr,
            hInst,
            nullptr);

        gKeybindEdits[i] = CreateWindowW(
            L"EDIT",
            nullptr,
            WS_CHILD | WS_BORDER | ES_LEFT | ES_AUTOHSCROLL,
            CONTROL_LEFT + 60 + CONTROL_SPACING,
            y + BUTTON_HEIGHT + CONTROL_SPACING,
            BUTTON_WIDTH - 70,
            BUTTON_HEIGHT,
            hwnd,
            reinterpret_cast<HMENU>(static_cast<INT_PTR>(KEYBIND_EDIT_ID_BASE + i)),
            hInst,
            nullptr);

        gKeybindSetButtons[i] = CreateWindowW(
            L"BUTTON",
            L"Set",
            WS_CHILD | BS_PUSHBUTTON,
            CONTROL_LEFT + 60 + CONTROL_SPACING + (BUTTON_WIDTH - 70) + CONTROL_SPACING,
            y + BUTTON_HEIGHT + CONTROL_SPACING,
            RESET_BUTTON_WIDTH,
            BUTTON_HEIGHT,
            hwnd,
            reinterpret_cast<HMENU>(static_cast<INT_PTR>(KEYBIND_SET_BUTTON_ID_BASE + i)),
            hInst,
            nullptr);

        SendMessageW(gSliders[i], TBM_SETRANGE, TRUE, MAKELPARAM(0, 100));
        SendMessageW(gSliders[i], TBM_SETPOS, TRUE, gSlots[i].volume);
        ShowWindow(gKeybindLabels[i], SW_HIDE);
        ShowWindow(gKeybindEdits[i], SW_HIDE);
        ShowWindow(gKeybindSetButtons[i], SW_HIDE);
    }

    const int micTop = startY;
    gMicLabel = CreateWindowW(
        L"STATIC",
        L"Input device (your microphone):",
        WS_CHILD | WS_VISIBLE | SS_LEFT,
        CONTROL_LEFT,
        micTop,
        BUTTON_WIDTH,
        BUTTON_HEIGHT,
        hwnd,
        nullptr,
        hInst,
        nullptr);
    gMicDeviceCombo = CreateWindowW(
        L"COMBOBOX",
        nullptr,
        WS_CHILD | WS_BORDER | WS_VSCROLL | CBS_DROPDOWNLIST | WS_TABSTOP,
        CONTROL_LEFT,
        micTop + BUTTON_HEIGHT + CONTROL_SPACING,
        BUTTON_WIDTH,
        BUTTON_HEIGHT * 6,
        hwnd,
        reinterpret_cast<HMENU>(static_cast<INT_PTR>(MIC_DEVICE_COMBO_ID)),
        hInst,
        nullptr);

    gMicSelectedLabel = CreateWindowW(
        L"STATIC",
        L"Selected: none",
        WS_CHILD | WS_VISIBLE | SS_LEFT,
        CONTROL_LEFT,
        micTop + BUTTON_HEIGHT * 2 + CONTROL_SPACING * 2 + 4,
        BUTTON_WIDTH + RESET_BUTTON_WIDTH + CONTROL_SPACING,
        BUTTON_HEIGHT,
        hwnd,
        nullptr,
        hInst,
        nullptr);

    gMicRefreshButton = CreateWindowW(
        L"BUTTON",
        L"Refresh",
        WS_CHILD | BS_PUSHBUTTON,
        CONTROL_LEFT + BUTTON_WIDTH + CONTROL_SPACING,
        micTop + BUTTON_HEIGHT + CONTROL_SPACING,
        RESET_BUTTON_WIDTH,
        BUTTON_HEIGHT,
        hwnd,
        reinterpret_cast<HMENU>(static_cast<INT_PTR>(MIC_REFRESH_BUTTON_ID)),
        hInst,
        nullptr);
    gOutputLabel = CreateWindowW(
        L"STATIC",
        L"Output device:",
        WS_CHILD | WS_VISIBLE | SS_LEFT,
        CONTROL_LEFT,
        micTop + BUTTON_HEIGHT * 3 + CONTROL_SPACING * 3,
        BUTTON_WIDTH,
        BUTTON_HEIGHT,
        hwnd,
        nullptr,
        hInst,
        nullptr);

    gOutputDeviceCombo = CreateWindowW(
        L"COMBOBOX",
        nullptr,
        WS_CHILD | WS_BORDER | WS_VSCROLL | CBS_DROPDOWNLIST | WS_TABSTOP,
        CONTROL_LEFT,
        micTop + BUTTON_HEIGHT * 4 + CONTROL_SPACING * 4,
        BUTTON_WIDTH,
        BUTTON_HEIGHT * 6,
        hwnd,
        reinterpret_cast<HMENU>(static_cast<INT_PTR>(OUTPUT_DEVICE_COMBO_ID)),
        hInst,
        nullptr);

    gOutputSelectedLabel = CreateWindowW(
        L"STATIC",
        L"Playing to: none",
        WS_CHILD | WS_VISIBLE | SS_LEFT,
        CONTROL_LEFT,
        micTop + BUTTON_HEIGHT * 5 + CONTROL_SPACING * 5 + 4,
        BUTTON_WIDTH + RESET_BUTTON_WIDTH + CONTROL_SPACING,
        BUTTON_HEIGHT,
        hwnd,
        nullptr,
        hInst,
        nullptr);
    gConfirmButton = CreateWindowW(
        L"BUTTON",
        L"Confirm",
        WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
        CONTROL_LEFT,
        micTop + BUTTON_HEIGHT * 6 + CONTROL_SPACING * 6,
        BUTTON_WIDTH,
        BUTTON_HEIGHT,
        hwnd,
        reinterpret_cast<HMENU>(static_cast<INT_PTR>(MIC_CONFIRM_BUTTON_ID)),
        hInst,
        nullptr);

    gMicNoteLabel = CreateWindowW(
        L"STATIC",
        L"In Discord, set Input Device to:\n'CABLE Output (VB-Audio Virtual Cable)'",
        WS_CHILD | WS_VISIBLE | SS_LEFTNOWORDWRAP,
        CONTROL_LEFT,
        micTop + BUTTON_HEIGHT * 7 + CONTROL_SPACING * 7,
        BUTTON_WIDTH + RESET_BUTTON_WIDTH + CONTROL_SPACING,
        MIC_NOTE_LABEL_HEIGHT,
        hwnd,
        nullptr,
        hInst,
        nullptr);

    PopulateOutputDeviceList();

    PopulateMicDeviceList();

    ShowWindow(gMicDeviceCombo, SW_HIDE);
    ShowWindow(gMicRefreshButton, SW_HIDE);
    ShowWindow(gMicNoteLabel, SW_HIDE);
    ShowWindow(gOutputLabel, SW_HIDE);
    ShowWindow(gOutputDeviceCombo, SW_HIDE);
    ShowWindow(gOutputSelectedLabel, SW_HIDE);
    ShowWindow(gConfirmButton, SW_HIDE);

    LoadSlots();

    for (int i = 0; i < SLOT_COUNT; ++i)
    {
        if (!gSlots[i].keybind.empty())
            RegisterSlotHotkey(hwnd, i, gSlots[i].keybind);
        UpdateSlotControls(i);
    }

    TabCtrl_SetCurSel(gTabControl, 0);
    UpdateTabVisibility(hwnd);

    ShowWindow(hwnd, nCmdShow);

    SetTimer(hwnd, STREAM_TIMER_ID, STREAM_TIMER_MS, nullptr);

    MSG msg{};

    while (GetMessage(&msg,
                      nullptr,
                      0,
                      0))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}
std::wstring FormatHotkeyString(UINT modifiers, UINT vk)
{
    std::wstring out;
    if (modifiers & MOD_CONTROL) out += L"Ctrl+";
    if (modifiers & MOD_ALT) out += L"Alt+";
    if (modifiers & MOD_SHIFT) out += L"Shift+";

    if (vk >= VK_F1 && vk <= VK_F24)
    {
        out += L"F" + std::to_wstring(vk - VK_F1 + 1);
    }
    else if (vk == VK_SPACE)
    {
        out += L"SPACE";
    }
    else if (vk >= 0x30 && vk <= 0x39)
    {
        out += wchar_t(L'0' + (vk - 0x30));
    }
    else if (vk >= 0x41 && vk <= 0x5A)
    {
        out += wchar_t(L'A' + (vk - 0x41));
    }
    else
    {
        out += L"VK" + std::to_wstring(vk);
    }
    return out;
}
void CaptureHotkeyInteractive(HWND hwnd, int slot)
{
    SetWindowTextW(gKeybindEdits[slot], L"Press key combination... (Esc to cancel)");
    SetFocus(hwnd);

    MSG msg{};
    BOOL messageResult = 0;
    while ((messageResult = GetMessage(&msg, nullptr, 0, 0)) > 0)
    {
        if (msg.message == WM_KEYDOWN || msg.message == WM_SYSKEYDOWN)
        {
            UINT vk = static_cast<UINT>(msg.wParam);
            if (vk == VK_ESCAPE)
            {
                SetWindowTextW(gKeybindEdits[slot], gSlots[slot].keybind.c_str());
                break;
            }
        }

        if (msg.message == WM_KEYUP || msg.message == WM_SYSKEYUP)
        {
            UINT vk = static_cast<UINT>(msg.wParam);
            if (vk == VK_ESCAPE)
            {
                SetWindowTextW(gKeybindEdits[slot], gSlots[slot].keybind.c_str());
                break;
            }
            if (IsModifierKey(vk))
            {
            }
            else
            {
                UINT modifiers = 0;
                if (GetKeyState(VK_CONTROL) & 0x8000) modifiers |= MOD_CONTROL;
                if (GetKeyState(VK_MENU) & 0x8000) modifiers |= MOD_ALT;
                if (GetKeyState(VK_SHIFT) & 0x8000) modifiers |= MOD_SHIFT;
                if (IsPlainTypingHotkey(modifiers, vk))
                {
                    MessageBoxW(hwnd, L"Please include a modifier (Alt/Ctrl/Shift) with letters or numbers, e.g. Alt+1 or Ctrl+A. Plain keys would prevent typing.", L"Hotkey requirement", MB_OK | MB_ICONINFORMATION);
                    SetWindowTextW(gKeybindEdits[slot], gSlots[slot].keybind.c_str());
                    break;
                }

                std::wstring keyStr = FormatHotkeyString(modifiers, vk);
                if (!keyStr.empty() && RegisterSlotHotkey(hwnd, slot, keyStr))
                {
                    SetWindowTextW(gKeybindEdits[slot], keyStr.c_str());
                    SaveSlots();
                }
                else
                {
                    MessageBoxW(hwnd, L"Failed to register that hotkey. It may be in use.", L"Hotkey error", MB_OK | MB_ICONWARNING);
                    SetWindowTextW(gKeybindEdits[slot], gSlots[slot].keybind.c_str());
                }
                break;
            }
        }

        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    if (messageResult == 0)
    {
        PostQuitMessage(static_cast<int>(msg.wParam));
    }
}

