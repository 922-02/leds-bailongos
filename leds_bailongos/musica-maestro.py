"""
Audio-Reactive NeoPixel Strip (SPI Version with File Support)

Captures audio input from an audio file, analyzes it using FFT,
and maps frequency bands to LED colors.

Requirements:
- numpy
- neopixel_spi
- board
- pydub (for audio file decoding)
- ffmpeg (system dependency for decoding .mp3)
"""

import time
import struct
import argparse
from typing import Tuple

import numpy as np
from pydub import AudioSegment

import board
import neopixel_spi as neopixel

# ========== Configuration ==========
# NeoPixel setup
NUM_PIXELS = 8
PIXEL_ORDER = neopixel.GRB
BRIGHTNESS_SCALE = 255

# Audio setup
CHANNELS = 2
RATE = 44100
N_FFT = 512
BUFFER_SIZE = 4 * N_FFT

CALIBRATION_SAMPLES = 300
FFT_BRIGHTNESS_MAX = 1000
# ===================================


class AudioReactiveLEDs:
    def __init__(self, file_path: str = "music_test/test_01.mp3"):
        self.pixels = neopixel.NeoPixel_SPI(
            spi=board.SPI(),
            n=NUM_PIXELS,
            pixel_order=PIXEL_ORDER,
            auto_write=False
        )

        self.file_path = file_path
        self.audio_data = AudioSegment.from_file(file_path).set_channels(CHANNELS).set_frame_rate(RATE)
        self.raw_data = self.audio_data.raw_data
        self.max_y = float(2 ** (self.audio_data.sample_width * 8 - 1))
        self.play_pos = 0

        self.offset_r = 0.0
        self.offset_g = 0.0
        self.offset_b = 0.0

    def _scale(self, value: float, min_in: float, max_in: float, min_out: float, max_out: float) -> float:
        return np.interp(value, [min_in, max_in], [min_out, max_out])

    def _constrain(self, value: float, min_val: float, max_val: float) -> int:
        return int(min(max(value, min_val), max_val))

    def _get_fft_magnitudes(self, calibrate: bool = False) -> Tuple[float, float, float]:
        try:
            end = self.play_pos + BUFFER_SIZE * CHANNELS * 2
            chunk = self.raw_data[self.play_pos:end]
            self.play_pos += len(chunk)
            if not chunk:
                raise EOFError
        except (IOError, EOFError):
            return 0.0, 0.0, 0.0

        count = len(chunk) // 2
        y = np.array(struct.unpack(f"{count}h", chunk)) / self.max_y
        y_l = y[::2]
        y_r = y[1::2]

        fft_l = np.fft.fft(y_l, N_FFT)
        fft_r = np.fft.fft(y_r, N_FFT)
        fft_combined = abs(np.hstack((fft_l[-N_FFT // 2:-1], fft_r[:N_FFT // 2])))

        d1, d2 = 4, 20
        c01, c02 = 255 - d1, 256 + d1
        c11, c12 = c01 - d2, c02 + d2
        c21, c22 = 0, 511

        low = sum(fft_combined[c01:c02])
        mid = sum(fft_combined[c11:c01]) + sum(fft_combined[c02:c12])
        high = sum(fft_combined[c21:c11]) + sum(fft_combined[c12:c22])

        if calibrate:
            return low, mid, high

        return (
            max(0.0, low - self.offset_r),
            max(0.0, mid - self.offset_g),
            max(0.0, high - self.offset_b)
        )

    def calibrate_noise(self, samples: int = CALIBRATION_SAMPLES) -> None:
        total_r = total_g = total_b = 0.0
        for _ in range(samples):
            r, g, b = self._get_fft_magnitudes(calibrate=True)
            total_r += r
            total_g += g
            total_b += b

        self.offset_r = (total_r / samples) * 3
        self.offset_g = (total_g / samples) * 3
        self.offset_b = (total_b / samples) * 3

    def _map_audio_to_color(self, r: float, g: float, b: float) -> Tuple[int, int, int]:
        red = self._constrain(self._scale(r, 0, FFT_BRIGHTNESS_MAX, 0, BRIGHTNESS_SCALE), 0, BRIGHTNESS_SCALE)
        green = self._constrain(self._scale(g, 0, FFT_BRIGHTNESS_MAX, 0, BRIGHTNESS_SCALE), 0, BRIGHTNESS_SCALE)
        blue = self._constrain(self._scale(b, 0, FFT_BRIGHTNESS_MAX, 0, BRIGHTNESS_SCALE), 0, BRIGHTNESS_SCALE)
        return red, green, blue

    def run(self) -> None:
        self.calibrate_noise()

        try:
            while self.play_pos < len(self.raw_data):
                r, g, b = self._get_fft_magnitudes()
                color = self._map_audio_to_color(r, g, b)
                self.pixels.fill(color)
                self.pixels.show()
        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self) -> None:
        self.pixels.fill((0, 0, 0))
        self.pixels.show()
        print("Audio reactive LEDs stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio-reactive NeoPixel controller.")
    parser.add_argument("--file", type=str, help="Path to audio file (e.g., music.mp3)")
    args = parser.parse_args()

    leds = AudioReactiveLEDs(file_path=args.file)
    leds.run()
