import time
import board
import neopixel_spi as neopixel

NUM_PIXELS = 8
PIXEL_ORDER = neopixel.GRB
COLORS = [0x5564eb, 0xff7e00]
DELAY = 0.1

spi = board.SPI()

pixels = neopixel.NeoPixel_SPI(spi,
                                NUM_PIXELS,
                                pixel_order=PIXEL_ORDER,
                                auto_write=False)

while True:
    for i in range(NUM_PIXELS):
        for color in COLORS:
            pixels[i] = color
            pixels.show()
            time.sleep(DELAY)
            pixels.fill(0)
            i += 1
except KeyboardInterrupt:
    # Turn off all LEDs when the program is interrupted
    pixels.fill(0)
    pixels.show()