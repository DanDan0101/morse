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
parser.add_argument("-w", "--wpm", type = bool, default = False, help = "Whether to trigger WPM detection.")
parser.add_argument("-f", "--freq", type = bool, default = True, help = "Whether to trigger frequency detection.")
args = parser.parse_args()
file = args.filename.expanduser().resolve()
detect_wpm = args.wpm
detect_freq = args.freq

rate, data = read(file)
if data.ndim >= 2:
    data = data.mean(axis = 1) # Convert stereo to mono
if detect_freq:
    F = detect_frequency(data, rate)
    print(f"Detected frequency: {F} Hz")
else:
    F = 800

spec = spectrogram(data, rate, F)
signal = spec > spec.mean() / 2
morse, wpm = parse(signal, rate, detect_wpm)
if detect_wpm:
    print(f"Detected WPM: {wpm} wpm")

message = decode(morse)

print(message)