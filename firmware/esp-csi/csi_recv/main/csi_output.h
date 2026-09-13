/* Whole-frame transport helper. Keep the slot until every byte is accepted.
 * This deliberately retries backpressure; the bounded callback queue accounts
 * for loss while a host is disconnected, without truncating frames silently. */
#pragma once
#include <stddef.h>
#include <stdint.h>
typedef int (*csi_write_fn)(const uint8_t *data, size_t size);
typedef void (*csi_wait_fn)(void);
static inline void csi_write_complete(const uint8_t *data, size_t size,
                                      csi_write_fn write_bytes, csi_wait_fn wait_ready)
{
    size_t sent = 0;
    while (sent < size) {
        int n = write_bytes(data + sent, size - sent);
        if (n > 0 && (size_t)n <= size - sent) sent += (size_t)n;
        else wait_ready();
    }
}
