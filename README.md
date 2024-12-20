# morse

Morse code decoder for the [BF1 easter eggs](https://wiki.gamedetectives.net/index.php?title=Battlefield_1#Headphones_and_Morse_code).

## Dependencies

* `numpy`
* `scipy`
* `numba`

## Usage

This code is intended for `.wav` files, with Morse transmitted at 20 wpm. To use, run `python main.py [PATH_TO_AUDIO]` in the command line. A few characters may be wrong due to noise in the recording.

## Features

* Pitch detection
* Timing detection, for time units 40-120 ms (10-30 wpm) [TODO]
* Denoising [TODO]
