#include <stdbool.h>

#include "driver/gpio.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"

#if CONFIG_BLINK_LED_RGB
#include "led_strip.h"
static led_strip_handle_t led;
#endif

static const char *TAG = "blink";

static void set_led(bool on)
{
#if CONFIG_BLINK_LED_RGB
    if (on) {
        ESP_ERROR_CHECK(led_strip_set_pixel(led, 0, 0, 16, 0));
        ESP_ERROR_CHECK(led_strip_refresh(led));
    } else {
        ESP_ERROR_CHECK(led_strip_clear(led));
    }
#else
#if CONFIG_BLINK_ACTIVE_LOW
    on = !on;
#endif
    ESP_ERROR_CHECK(gpio_set_level(CONFIG_BLINK_GPIO, on));
#endif
}

void app_main(void)
{
    uint8_t mac[6];
    ESP_ERROR_CHECK(esp_efuse_mac_get_default(mac));
    ESP_LOGI(TAG, "Board identification: target=%s, base MAC=" MACSTR,
             CONFIG_IDF_TARGET, MAC2STR(mac));
    ESP_LOGI(TAG, "LED on GPIO%d, toggling every %d ms",
             CONFIG_BLINK_GPIO, CONFIG_BLINK_INTERVAL_MS);

    if (!GPIO_IS_VALID_OUTPUT_GPIO(CONFIG_BLINK_GPIO)) {
        ESP_LOGE(TAG, "GPIO%d cannot drive an LED; change Blink Configuration",
                 CONFIG_BLINK_GPIO);
        return;
    }

#if CONFIG_BLINK_LED_RGB
    const led_strip_config_t strip_config = {
        .strip_gpio_num = CONFIG_BLINK_GPIO,
        .max_leds = 1,
        .led_model = LED_MODEL_WS2812,
    };
    const led_strip_rmt_config_t rmt_config = {
        .resolution_hz = 10 * 1000 * 1000,
    };
    ESP_ERROR_CHECK(led_strip_new_rmt_device(&strip_config, &rmt_config, &led));
    ESP_LOGI(TAG, "LED type: addressable WS2812 (green)");
#else
    ESP_ERROR_CHECK(gpio_reset_pin(CONFIG_BLINK_GPIO));
    set_led(false);
    ESP_ERROR_CHECK(gpio_set_direction(CONFIG_BLINK_GPIO, GPIO_MODE_OUTPUT));
    ESP_LOGI(TAG, "LED type: simple GPIO");
#endif
    set_led(false);

    bool on = false;
    while (true) {
        on = !on;
        set_led(on);
        ESP_LOGI(TAG, "%s " MACSTR " LED %s", CONFIG_IDF_TARGET,
                 MAC2STR(mac), on ? "ON" : "OFF");
        vTaskDelay(pdMS_TO_TICKS(CONFIG_BLINK_INTERVAL_MS));
    }
}
