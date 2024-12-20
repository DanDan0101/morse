# morse

Morse code decoder for the [BF1 easter eggs](https://wiki.gamedetectives.net/index.php?title=Battlefield_1#Headphones_and_Morse_code).

## Dependencies

* `numpy`
* `scipy`
* `numba`

## Usage

This code is intended for `.wav` files. The Morse code should be pitched around $790\,\mathrm{Hz}$, with a base time unit of around $60\,\mathrm{ms}$, though these can easily be tweaked by chaging the appropriate global variables in `main.py`.

To use, run `python main.py [PATH_TO_AUDIO]` in the command line.
