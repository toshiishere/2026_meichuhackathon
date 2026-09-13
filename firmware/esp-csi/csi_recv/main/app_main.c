/*
 * SPDX-FileCopyrightText: 2025-2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */
/* Get Start Example

   This example code is in the Public Domain (or CC0 licensed, at your option.)

   Unless required by applicable law or agreed to in writing, this
   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied.
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "nvs_flash.h"

#include "esp_mac.h"
#include "rom/ets_sys.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_now.h"
#include "esp_csi_gain_ctrl.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "driver/uart.h"
#if CONFIG_CSI_OUTPUT_USB
#include "driver/usb_serial_jtag.h"
#endif
#include "csi_wire.h"
#include "csi_output.h"

#define CONFIG_LESS_INTERFERENCE_CHANNEL   11
#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C61 || (CONFIG_IDF_TARGET_ESP32C6 && ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 4, 0))
#define CONFIG_WIFI_BAND_MODE               WIFI_BAND_MODE_2G_ONLY
#define CONFIG_WIFI_2G_BANDWIDTHS           WIFI_BW_HT40
#define CONFIG_WIFI_5G_BANDWIDTHS           WIFI_BW_HT40
#define CONFIG_WIFI_2G_PROTOCOL             WIFI_PROTOCOL_11N
#define CONFIG_WIFI_5G_PROTOCOL             WIFI_PROTOCOL_11N
#else
#define CONFIG_WIFI_BANDWIDTH           WIFI_BW_HT40
#endif

#define CONFIG_ESP_NOW_PHYMODE           WIFI_PHY_MODE_HT40
#define CONFIG_ESP_NOW_RATE             WIFI_PHY_RATE_MCS0_LGI
#define CONFIG_FORCE_GAIN                   0

#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C61
#define CSI_FORCE_LLTF                      0
#endif

#if CONFIG_IDF_TARGET_ESP32S3 || CONFIG_IDF_TARGET_ESP32C3 || CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6 || CONFIG_IDF_TARGET_ESP32C61
#define CONFIG_GAIN_CONTROL                 1
#endif

#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
#define ESP_IF_WIFI_STA ESP_MAC_WIFI_STA
#endif

static const uint8_t CONFIG_CSI_SEND_MAC[] = {0x1a, 0x00, 0x00, 0x00, 0x00, 0x00};
static const char *TAG = "csi_recv";

static void wifi_init()
{
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    ESP_ERROR_CHECK(esp_netif_init());
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));

#if CONFIG_IDF_TARGET_ESP32C5
    ESP_ERROR_CHECK(esp_wifi_start());
    esp_wifi_set_band_mode(CONFIG_WIFI_BAND_MODE);
    wifi_protocols_t protocols = {
        .ghz_2g = CONFIG_WIFI_2G_PROTOCOL,
        .ghz_5g = CONFIG_WIFI_5G_PROTOCOL
    };
    ESP_ERROR_CHECK(esp_wifi_set_protocols(ESP_IF_WIFI_STA, &protocols));
    wifi_bandwidths_t bandwidth = {
        .ghz_2g = CONFIG_WIFI_2G_BANDWIDTHS,
        .ghz_5g = CONFIG_WIFI_5G_BANDWIDTHS
    };
    ESP_ERROR_CHECK(esp_wifi_set_bandwidths(ESP_IF_WIFI_STA, &bandwidth));
#elif (CONFIG_IDF_TARGET_ESP32C6 && ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 4, 0)) || CONFIG_IDF_TARGET_ESP32C61
    ESP_ERROR_CHECK(esp_wifi_start());
    esp_wifi_set_band_mode(CONFIG_WIFI_BAND_MODE);
    wifi_protocols_t protocols = {
        .ghz_2g = CONFIG_WIFI_2G_PROTOCOL,
    };
    ESP_ERROR_CHECK(esp_wifi_set_protocols(ESP_IF_WIFI_STA, &protocols));
    wifi_bandwidths_t bandwidth = {
        .ghz_2g = CONFIG_WIFI_2G_BANDWIDTHS,
    };
    ESP_ERROR_CHECK(esp_wifi_set_bandwidths(ESP_IF_WIFI_STA, &bandwidth));
#else
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(ESP_IF_WIFI_STA, CONFIG_WIFI_BANDWIDTH));
    ESP_ERROR_CHECK(esp_wifi_start());
#endif

    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
#if CONFIG_IDF_TARGET_ESP32C5
    if ((CONFIG_WIFI_BAND_MODE == WIFI_BAND_MODE_2G_ONLY && CONFIG_WIFI_2G_BANDWIDTHS == WIFI_BW_HT20)
            || (CONFIG_WIFI_BAND_MODE == WIFI_BAND_MODE_5G_ONLY && CONFIG_WIFI_5G_BANDWIDTHS == WIFI_BW_HT20)) {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_NONE));
    } else {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_BELOW));
    }
#elif (CONFIG_IDF_TARGET_ESP32C6 && ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 4, 0)) || CONFIG_IDF_TARGET_ESP32C61
    if (CONFIG_WIFI_BAND_MODE == WIFI_BAND_MODE_2G_ONLY && CONFIG_WIFI_2G_BANDWIDTHS == WIFI_BW_HT20) {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_NONE));
    } else {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_BELOW));
    }
#else
    if (CONFIG_WIFI_BANDWIDTH == WIFI_BW_HT20) {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_NONE));
    } else {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_BELOW));
    }
#endif

    ESP_ERROR_CHECK(esp_wifi_set_mac(WIFI_IF_STA, CONFIG_CSI_SEND_MAC));
}

static void wifi_esp_now_init(esp_now_peer_info_t peer)
{
    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_set_pmk((uint8_t *)"pmk1234567890123"));
    esp_now_rate_config_t rate_config = {
        .phymode = CONFIG_ESP_NOW_PHYMODE,
        .rate = CONFIG_ESP_NOW_RATE,//  WIFI_PHY_RATE_MCS0_LGI,
        .ersu = false,
        .dcm = false
    };
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));
    ESP_ERROR_CHECK(esp_now_set_peer_rate_config(peer.peer_addr, &rate_config));

}

typedef struct {
    wifi_pkt_rx_ctrl_t rx_ctrl;
    csi_wire_meta_t meta;
    int8_t samples[CSI_MAX_SAMPLES];
} queued_csi_t;
static queued_csi_t *slots;
static QueueHandle_t free_slots, ready_slots;

static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    /* Wi-Fi task: no printing, allocation, blocking, or sample conversion. */
    static uint32_t received_total, queue_drops, invalid_total;
    if (!info || memcmp(info->mac, CONFIG_CSI_SEND_MAC, 6)) return;
    ++received_total;
    if (!info->buf || !info->payload || info->payload_len < 19 ||
        !info->len || info->len > CSI_MAX_SAMPLES || (info->len & 1)) {
        ++invalid_total;
        return;
    }
    uint8_t index;
    if (xQueueReceive(free_slots, &index, 0) != pdTRUE) {
        ++queue_drops;
        return;
    }
    queued_csi_t *item = &slots[index];
    memset(&item->meta, 0, sizeof(item->meta));
    item->rx_ctrl = info->rx_ctrl;
    memcpy(item->meta.mac, info->mac, 6);
    /* ESP-NOW vendor payload offset used by the original firmware; avoid an
       unaligned uint32 dereference and reject short payloads before accessing. */
    memcpy(&item->meta.tx_seq, info->payload + 15, sizeof(uint32_t));
    item->meta.received_total = received_total;
    item->meta.queue_drops = queue_drops;
    item->meta.invalid_total = invalid_total;
    item->meta.len = info->len;
    item->meta.first_word = info->first_word_invalid;
    memcpy(item->samples, info->buf, info->len);
    if (xQueueSend(ready_slots, &index, 0) != pdTRUE) {
        ++queue_drops;
        xQueueSend(free_slots, &index, 0);
    }
}

/* Bulk driver writes follow the direct USB output path used by
 * TryTwoTop/esp32c5-csi-keystroke (8999ce8379266d5efb8b7cb3e73e6bdcd1db323d).
 * Retain our CRC protocol and handle short/zero writes instead of discarding them. */
static int csi_transport_write(const uint8_t *data, size_t size)
{
#if CONFIG_CSI_OUTPUT_USB
    return usb_serial_jtag_write_bytes(data, size, pdMS_TO_TICKS(20));
#else
    return uart_write_bytes(CONFIG_ESP_CONSOLE_UART_NUM, data, size);
#endif
}

static void csi_transport_wait(void)
{
    vTaskDelay(1);
}

static void csi_output_task(void *arg)
{
    uint8_t wire[7 + sizeof(csi_wire_meta_t) + CSI_MAX_SAMPLES + 4];
    uint32_t calibrated = 0;
    while (true) {
        uint8_t index;
        xQueueReceive(ready_slots, &index, portMAX_DELAY);
        queued_csi_t *item = &slots[index];
        csi_wire_meta_t *m = &item->meta;
        const wifi_pkt_rx_ctrl_t *rx = &item->rx_ctrl;
        m->rssi = rx->rssi;
        m->rate = rx->rate;
        m->noise_floor = rx->noise_floor;
        m->channel = rx->channel;
        m->local_timestamp = rx->timestamp;
        m->sig_len = rx->sig_len;
        m->compensate_gain = 1.0f;
#if CONFIG_GAIN_CONTROL
        uint8_t agc = 0;
        int8_t fft = 0;
        float compensation = 1.0f;
        esp_csi_gain_ctrl_get_rx_gain(rx, &agc, &fft);
        if (calibrated < 100) esp_csi_gain_ctrl_record_rx_gain(agc, fft);
        else if (calibrated == 100) {
            uint8_t agc_baseline;
            int8_t fft_baseline;
            esp_csi_gain_ctrl_get_rx_gain_baseline(&agc_baseline, &fft_baseline);
#if CONFIG_FORCE_GAIN
            esp_csi_gain_ctrl_set_rx_force_gain(agc_baseline, fft_baseline);
#endif
        }
        esp_csi_gain_ctrl_get_gain_compensation(&compensation, agc, fft);
        m->agc_gain = agc;
        m->fft_gain = fft;
        m->compensate_gain = compensation;
#endif
        ++calibrated;
#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6 || CONFIG_IDF_TARGET_ESP32C61
        m->layout = 1;
        m->rx_format = rx->cur_bb_format;
#else
        m->sig_mode = rx->sig_mode;
        m->mcs = rx->mcs;
        m->bandwidth = rx->cwb;
        m->smoothing = rx->smoothing;
        m->not_sounding = rx->not_sounding;
        m->aggregation = rx->aggregation;
        m->stbc = rx->stbc;
        m->fec_coding = rx->fec_coding;
        m->sgi = rx->sgi;
        m->ampdu_cnt = rx->ampdu_cnt;
        m->secondary_channel = rx->secondary_channel;
        m->ant = rx->ant;
        m->rx_format = rx->sig_mode;
#endif
        uint16_t body_len = sizeof(*m) + m->len;
        memcpy(wire, CSI_MAGIC, 4);
        wire[4] = CSI_WIRE_VERSION;
        memcpy(wire + 5, &body_len, 2);
        memcpy(wire + 7, m, sizeof(*m));
        memcpy(wire + 7 + sizeof(*m), item->samples, m->len);
        uint32_t crc = csi_crc32(wire + 4, 3 + body_len);
        memcpy(wire + 7 + body_len, &crc, 4);
        csi_write_complete(wire, 11 + body_len, csi_transport_write, csi_transport_wait);
        xQueueSend(free_slots, &index, portMAX_DELAY);
    }
}

static void csi_output_init(void)
{
    slots = calloc(CSI_QUEUE_DEPTH, sizeof(*slots));
    free_slots = xQueueCreate(CSI_QUEUE_DEPTH, sizeof(uint8_t));
    ready_slots = xQueueCreate(CSI_QUEUE_DEPTH, sizeof(uint8_t));
    ESP_ERROR_CHECK(slots && free_slots && ready_slots ? ESP_OK : ESP_ERR_NO_MEM);
    for (uint8_t i = 0; i < CSI_QUEUE_DEPTH; ++i) xQueueSend(free_slots, &i, 0);
#if CONFIG_CSI_OUTPUT_USB
    if (!usb_serial_jtag_is_driver_installed()) {
        usb_serial_jtag_driver_config_t usb_config = {.tx_buffer_size = 8192, .rx_buffer_size = 256};
        ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&usb_config));
    }
    ESP_LOGI(TAG, "CSI binary v1: direct USB Serial/JTAG, %d slots", CSI_QUEUE_DEPTH);
#else
    if (!uart_is_driver_installed(CONFIG_ESP_CONSOLE_UART_NUM)) {
        ESP_ERROR_CHECK(uart_driver_install(CONFIG_ESP_CONSOLE_UART_NUM, 256, 8192, 0, NULL, 0));
    }
    ESP_LOGI(TAG, "CSI binary v1: direct UART%d at %d baud, %d slots",
             CONFIG_ESP_CONSOLE_UART_NUM, CONFIG_ESP_CONSOLE_UART_BAUDRATE, CSI_QUEUE_DEPTH);
#endif
    /* Runtime console logs must not interleave with direct binary writes.
       Boot/startup logs remain visible; per-frame metadata carries diagnostics. */
    esp_log_level_set("*", ESP_LOG_NONE);
    ESP_ERROR_CHECK(xTaskCreate(csi_output_task, "csi_output", 4096, NULL, 5, NULL) == pdPASS ? ESP_OK : ESP_ERR_NO_MEM);
}

static void wifi_csi_init()
{
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));

    /**< default config */
#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C61
    wifi_csi_config_t csi_config = {
        .enable                   = true,
        .acquire_csi_legacy       = false,
        .acquire_csi_force_lltf   = CSI_FORCE_LLTF,
        .acquire_csi_ht20         = true,
        .acquire_csi_ht40         = true,
        .acquire_csi_vht          = false,
        .acquire_csi_su           = false,
        .acquire_csi_mu           = false,
        .acquire_csi_dcm          = false,
        .acquire_csi_beamformed   = false,
        .acquire_csi_he_stbc_mode = 2,
        .val_scale_cfg            = 0,
        .dump_ack_en              = false,
        .reserved                 = false
    };
#elif CONFIG_IDF_TARGET_ESP32C6
    wifi_csi_config_t csi_config = {
        .enable                 = true,
        .acquire_csi_legacy     = false,
        .acquire_csi_ht20       = true,
        .acquire_csi_ht40       = true,
        .acquire_csi_su         = true,
        .acquire_csi_mu         = true,
        .acquire_csi_dcm        = true,
        .acquire_csi_beamformed = true,
        .acquire_csi_he_stbc    = 2,
        .val_scale_cfg          = false,
        .dump_ack_en            = false,
        .reserved               = false
    };
#else
    wifi_csi_config_t csi_config = {
        .lltf_en           = true,
        .htltf_en          = true,
        .stbc_htltf2_en    = true,
        .ltf_merge_en      = true,
        .channel_filter_en = true,
        .manu_scale        = false,
        .shift             = false,
    };
#endif
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
}

void app_main()
{
    /**
     * @brief Initialize NVS
     */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    /**
     * @brief Initialize Wi-Fi
     */
    wifi_init();

    /**
     * @brief Initialize ESP-NOW
     *        ESP-NOW protocol see: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/network/esp_now.html
     */

    esp_now_peer_info_t peer = {
        .channel   = CONFIG_LESS_INTERFERENCE_CHANNEL,
        .ifidx     = WIFI_IF_STA,
        .encrypt   = false,
        .peer_addr = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
    };

    wifi_esp_now_init(peer);

    csi_output_init();
    wifi_csi_init();
}
