import numpy as np
from scipy.signal.windows import hann
from scipy.signal import periodogram
from scipy.optimize import least_squares
from numba import njit, prange

MORSE = {
    ".-":      'A',
    "-...":    'B',
    "-.-.":    'C',
    "-..":     'D',
    ".":       'E',
    "..-.":    'F',
    "--.":     'G',
    "....":    'H',
    "..":      'I',
    ".---":    'J',
    "-.-":     'K',
    ".-..":    'L',
    "--":      'M',
    "-.":      'N',
    "---":     'O',
    ".--.":    'P',
    "--.-":    'Q',
    ".-.":     'R',
    "...":     'S',
    "-":       'T',
    "..-":     'U',
    "...-":    'V',
    ".--":     'W',
    "-..-":    'X',
    "-.--":    'Y',
    "--..":    'Z',
    ".----":   '1',
    "..---":   '2',
    "...--":   '3',
    "....-":   '4',
    ".....":   '5',
    "-....":   '6',
    "--...":   '7',
    "---..":   '8',
    "----.":   '9',
    "-----":   '0',
    ".-.-.-":  '.',
    "--..--":  ',',
    "..--..":  '?',
    ".---.":   '\'',
    "-.-.--":  '!',
    "-..-.":   '/',
    "-.--.":   '(',
    "-.--.-":  ')',
    ".-...":   '&',
    "---...":  ':',
    "-.-.-.":  ';',
    "-...-":   '=',
    ".-.-.":   '+',
    "-....-":  '-',
    "..--.-":  '_',
    ".-..-.":  '"',
    "...-..-": '$',
    ".--.-.":  '@',
    "/":       ' ',
}

W = 2048 # window size
window = hann(W, False)
HOP = 64

def detect_frequency(data: np.ndarray, rate: float) -> float:
    f, Pxx = periodogram(data, rate)
    low = np.argmin(np.abs(f - 700))
    high = np.argmin(np.abs(f - 1100))
    f = f[low:high]
    Pxx = Pxx[low:high]
    return f[np.argmax(Pxx)]

@njit(cache = True, parallel = True)
def spectrogram_i(i: int, data: np.ndarray, rate: float, f: float) -> np.ndarray:
    """
    Compute the spectrogram (norm squared STFT) at time i / rate, for frequency f.

    Parameters:
    i (int): The time index, corresponding to time i / rate.
    data (np.ndarray): The audio data.
    rate (float): The sample rate of the audio.
    f (float): The frequency to compute the spectrogram at.

    Returns:
    np.ndarray: The spectrogram at time i.
    """
    stft = np.sum(data[i:i + W] * window * np.exp(-2j * np.pi * f * np.arange(W) / rate)) / rate
    return np.abs(stft) ** 2

@njit(cache = True, parallel = True)
def spectrogram(data: np.ndarray, rate: float, f: float, hop: int = HOP) -> np.ndarray:
    """
    Compute the spectrogram timeseries of the audio data, every hop samples.

    Parameters:
    data (np.ndarray): The audio data.
    rate (float): The sample rate of the audio.
    f (float): The frequency to compute the spectrogram at.
    hop (int): The number of samples to skip between each spectrogram. Default is 64, which is 1.333 ms for a sample rate of 48 kHz.

    Returns:
    np.ndarray: The spectrogram timeseries.
    """
    N = data.shape[0]
    M = (N - W + 1) // hop
    S = [spectrogram_i(i * hop, data, rate, f) for i in prange(M)]
    return np.array(S)

def parse(signal: np.ndarray, rate: float, detect_wpm: bool = False) -> tuple[str, float]:
    """
    Parse the Morse code signal.

    Parameters:
    signal (np.ndarray): The spectrogram timeseries.
    rate (float): The sample rate of the audio.
    detect_wpm (bool): Whether to detect the WPM. Default is False.

    Returns:
    str: The parsed Morse code.
    float: The detected wpm.
    """
    grad = np.nonzero(signal[1:] ^ signal[:-1])[0]
    lasti = 0
    morse = ""
    diff = np.ediff1d(grad) / rate * HOP
    if detect_wpm:
        def residual(x):
            diffs = np.stack([np.abs(diff - x[0]), np.abs(diff - 3*x[0]), np.abs(diff - 7*x[0])])
            return np.min(diffs, axis = 1)
        dt = least_squares(residual, 0.06).x[0]
    else:
        dt = 0.06

    for i in grad:
        diff = (i - lasti) / rate * HOP
        if diff < dt / 2:
            continue
        if diff < 2 * dt:
            if signal[i] == 1:
                morse += '.'
        elif diff < 5 * dt:
            if signal[i] == 1:
                morse += '-'
            else:
                morse += ' '
        else:
            morse += " / "
        lasti = i
    return morse, 1.2 / dt

def decode(morse: str) -> str:
    """
    Decode the Morse code string.

    Parameters:
    morse (str): The Morse code string.

    Returns:
    str: The decoded string.
    """
    message = ""
    for c in morse.split():
        try:
            message += MORSE[c]
        except KeyError:
            message += '?'
    return message