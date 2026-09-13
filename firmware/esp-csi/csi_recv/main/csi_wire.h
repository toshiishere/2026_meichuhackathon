/* CSI binary transport v1. All integer/float fields are little endian.
 * ESP targets are little endian; packed metadata is independent of IDF bitfields.
 * Samples are the original signed int8 bytes, imaginary then real. */
#pragma once
#include <stdint.h>
#include <stddef.h>
#define CSI_MAX_SAMPLES 1024
#define CSI_QUEUE_DEPTH 32
#define CSI_WIRE_VERSION 1
static const uint8_t CSI_MAGIC[4] = {0xa5, 'C', 'S', 'I'};
typedef struct __attribute__((packed)) {
    uint32_t tx_seq, local_timestamp, received_total, queue_drops, invalid_total;
    uint8_t mac[6];
    int8_t rssi, noise_floor, fft_gain;
    uint8_t agc_gain;
    float compensate_gain;
    uint16_t sig_len, len;
    uint8_t rate, sig_mode, mcs, bandwidth, smoothing, not_sounding, aggregation;
    uint8_t stbc, fec_coding, sgi, ampdu_cnt, channel, secondary_channel, ant;
    uint8_t rx_format, first_word, layout; /* 0=legacy PHY, 1=compact PHY */
} csi_wire_meta_t;
_Static_assert(sizeof(csi_wire_meta_t) == 55, "CSI v1 metadata layout changed");
static inline uint32_t csi_crc32(const uint8_t *data, size_t length)
{
    uint32_t crc = UINT32_MAX;
    for (size_t i = 0; i < length; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) crc = (crc >> 1) ^ (0xedb88320U & (0U - (crc & 1)));
    }
    return ~crc;
}
