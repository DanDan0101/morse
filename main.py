from morse import detect_frequency, spectrogram, parse, decode
from scipy.io.wavfile import read
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(
    prog = "Morse",
    description = "Decode Morse code from an audio file.",
    epilog = "Hacked together by Daniel Sun"
)
parser.add_argument("filename", type = Path)
args = parser.parse_args()
file = args.filename.expanduser().resolve()

rate, data = read(file)
if data.ndim >= 2:
    data = data.mean(axis = 1) # Convert stereo to mono
F = detect_frequency(data, rate)
print(f"Detected frequency: {F} Hz")

spec = spectrogram(data, rate, F)
signal = spec > spec.mean() / 2
morse, wpm = parse(signal, rate)
print(f"Detected WPM: {wpm} wpm")

message = decode(morse)

print(message)