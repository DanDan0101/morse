# morse

Morse code decoder for the [BF1 easter eggs](https://wiki.gamedetectives.net/index.php?title=Battlefield_1#Headphones_and_Morse_code).

## Dependencies

* `numpy`
* `scipy`
* `numba`

## Usage

This code is intended for `.wav` files, with Morse transmitted at 20 wpm. To use, run `python main.py [PATH_TO_AUDIO]` in the command line. A few characters may be wrong due to noise in the recording. The first run of the program will be slower than usual due to the creation of a `numba` JIT cache for the spectrogram subroutine.

### Example

```bash
> python main.py ~/Videos/B2.wav
Detected frequency: 791.002443208847 Hz
Detected WPM: 20.0 wpm
 HCRUHC REMEMBER FIRST RULE. IF COMPROMISED. L PILL. SNEIMANIURHCRUHC REMEMBER FIRST RULE. IF COMPROMISED. L PILL. SNEIMANIURHCRUHC REMEMBER FI
```

## Algorithm

1. Stereo audio is preprocessed to mono by taking the mean of all tracks.
2. Frequency is detected using the peak of the [periodogram](https://en.wikipedia.org/wiki/Periodogram) (power spectral density) of the audio data.
3. The [spectrogram](https://en.wikipedia.org/wiki/Spectrogram) is computed every 64 samples as the norm square of the [STFT](https://en.wikipedia.org/wiki/Short-time_Fourier_transform), using a [Hann window](https://en.wikipedia.org/wiki/Hann_function) with width 2048. This is evaluated at the frequency detected in step 2.
4. The spectrogram is thresholded at half of its mean to convert it to a binary signal.
5. The times at which the signal changes are computed.
6. The wpm of the Morse code is detected using a least-squares fit to $\mathop{\mathrm{min}}\left(\left|t-\delta t\right|,\left|t-3\delta t\right|,\left|t-7\delta t\right|\right)$, where $t$ is the time between signal changes. The wpm is computed as $\frac{1200}{\delta t}$.
7. Discarding any $t<\frac{\delta t}{2}$ as noise, the signal is parsed into a Morse code string.
8. The Morse code string is decoded to plaintext.

## TODO

1. Add automatic decryption for BF1 easter egg ciphers, based on input stage
2. Port everything to a web app
3. Improve sensitivity for low S/N ratio data
4. Add support for more file types
