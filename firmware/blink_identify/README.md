# Blink and identify a board

The default build targets ESP32 and blinks a simple LED on GPIO2: 500 ms
on, then 500 ms off. Each toggle also logs the chip target and the board's
factory base MAC address at 115200 baud, so you can identify the board
from its serial output even if its LED wiring differs.

The LED pin and type depend on the **board**, not just the chip. GPIO2 is
a starting assumption for ESP32 boards. Some boards only have a power LED,
which software cannot blink. Check the board's pinout if the LED stays on
or does not light.

## Build and configure

```bash
cd /home/toshi/esp-csi/blink
source /opt/esp/activate-v5.5.sh
idf.py set-target esp32
idf.py menuconfig
idf.py build
```

In **Blink Configuration**, select the LED type, GPIO number, polarity
(active low for LEDs that turn on with a low output), and toggle interval.
Addressable WS2812 LEDs blink green; they require the RGB option rather
than the simple GPIO option.

The firmware, bootloader, and partition table are generated in `build/`.

## Find the port and flash

List ports before and after plugging in a board:

```bash
python -m serial.tools.list_ports -v
ls -l /dev/serial/by-id/
```

The newly appearing port is the connected board. If multiple boards are
already connected, unplug and reconnect one to see which entry changes.
Linux ports often look like `/dev/ttyUSB0` or `/dev/ttyACM0`; the names
can change when you reconnect devices. A `/dev/serial/by-id/...` path,
when available, gives a more stable name.

You can identify the chip on a selected port without replacing its firmware:

```bash
python -m esptool --port /dev/ttyUSB0 chip_id
```

This reports the detected chip and resets the board. Replace the example
port with the one you found. To load blink and watch its identification logs:

```bash
idf.py -p /dev/ttyUSB0 flash monitor
```

Flashing replaces the existing application on that board. Exit the monitor
with **Ctrl+]**. If a connection stalls, hold BOOT while connecting and
release it once the tool identifies the chip. If the port reports permission
denied on Ubuntu, add your user to `dialout` with
`sudo usermod -aG dialout "$USER"`, then log out and back in.

## Switch between chip targets

Run these commands in the project you want to change:

```bash
idf.py set-target esp32       # ESP32
idf.py set-target esp32c3     # ESP32-C3
idf.py set-target esp32s3     # ESP32-S3
idf.py --list-targets         # Show all targets supported by this IDF release
```

Choose **one** `set-target` command, then run `idf.py menuconfig` and
`idf.py build`. Target names have no hyphen. `set-target` clears the build
directory and regenerates `sdkconfig`, saving the old settings as
`sdkconfig.old`. Recheck the board's LED settings after switching targets.

For ESP32-C3, this project's starting pin is GPIO8, with a simple GPIO LED.
If your board has an addressable RGB LED, select **Addressable WS2812 RGB
LED**. If it has a simple LED wired active low, enable **LED is active low**.
Confirm the GPIO against the board's documentation.

`menuconfig` saves the current settings in `sdkconfig`. To keep chosen
settings as defaults for future clean configurations, run
`idf.py save-defconfig` and review `sdkconfig.defaults`. Its
`CONFIG_IDF_TARGET="esp32"` setting supplies the initial target; an explicit
`set-target` overrides it. Editing defaults does not override settings
already present in `sdkconfig`.

The same target commands work in `../csi_recv` and `../csi_send`, but the
**Blink Configuration** menu belongs only to this blink project. The shared
installation currently has tools for ESP32 and ESP32-C3; additional chip
targets may require an administrator to install their tools.

References: [ESP-IDF target and build commands](https://docs.espressif.com/projects/esp-idf/en/v5.5/esp32/api-guides/tools/idf-py.html),
[serial connections](https://docs.espressif.com/projects/esp-idf/en/v5.5/esp32/get-started/establish-serial-connection.html).
