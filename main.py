import numpy as np
from scipy.io.wavfile import read
from scipy.signal.windows import hann
from numba import njit, prange
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(
    prog = "Morse",
    description = "Decode Morse code from an audio file.",
    epilog = "Hacked together by Daniel Sun"
)
parser.add_argument("filename", type = Path)
parser.add_argument("pitch", type = float, default = 790.5)
args = parser.parse_args()
file = args.filename.expanduser().resolve()
F = args.pitch

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

DT = 0.06 # s, base time unit for the Morse code audio
W = 256 # window size
window = hann(W, False)

@njit(cache = True, parallel = True)
def spectrogram_i(i: int, data: np.ndarray, rate: float, f: float = F):
    """
    Compute the spectrogram (norm squared STFT) at time i / rate, for frequency f.

    Parameters:
    i (int): The time index, corresponding to time i / rate.
    data (np.ndarray): The audio data.
    rate (float): The sample rate of the audio.
    f (float): The frequency to compute the spectrogram at. Default is F = 790.5 Hz.

    Returns:
    np.ndarray: The spectrogram at time i.
    """
    stft = np.sum(data[i:i + W] * window * np.exp(-2j * np.pi * f * np.arange(W) / rate)) / rate
    return np.abs(stft) ** 2

@njit(cache = True, parallel = True)
def spectrogram(data: np.ndarray, rate: float, hop: int = 64):
    """
    Compute the spectrogram timeseries of the audio data, every hop samples.

    Parameters:
    data (np.ndarray): The audio data.
    rate (float): The sample rate of the audio.
    hop (int): The number of samples to skip between each spectrogram. Default is 64, which is 1.333 ms.

    Returns:
    np.ndarray: The spectrogram timeseries.
    """
    N = data.shape[0]
    M = (N - W + 1) // hop
    S = [spectrogram_i(i * hop, data, rate) for i in prange(M)]
    return np.array(S)

def parse(signal: np.ndarray, rate: float):
    """
    Parse the Morse code signal.

    Parameters:
    signal (np.ndarray): The spectrogram timeseries.
    rate (float): The sample rate of the audio.

    Returns:
    str: The parsed Morse code.
    """
    grad = np.nonzero(signal[1:] ^ signal[:-1])[0]
    lasti = 0
    outstr = ""
    for i in grad:
        diff = (i - lasti) / rate * 64 # hop
        if diff < 2 * DT:
            if signal[i] == 1:
                outstr += '.'
        elif diff < 5 * DT:
            if signal[i] == 1:
                outstr += '-'
            else:
                outstr += ' '
        else:
            outstr += " / "
        lasti = i
    return outstr

def decode(s: str):
    """
    Decode the Morse code string.

    Parameters:
    s (str): The Morse code string.

    Returns:
    str: The decoded string.
    """
    return "".join(MORSE[c] for c in s.split())

rate, data = read(file)
data = data.mean(axis = 1) # Convert stereo to mono
signal = spectrogram(data, rate) > 0.05
morse = parse(signal, rate)
message = decode(morse)

print(message)
