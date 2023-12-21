import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

input_folder = "W:\\for-norm\\for-norm\\training\\real"
output_folder = "W:\workdir\\train\\real"


def pad(x, max_len=48000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]

    return padded_x


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Parameters for mel spectrogram calculation
hop_length = 512
factor = 1  # To adjust the size of the saved images
sr = 16000


def save_mel_spectrogram(input_file, output_file):
    try:
        signal, _ = librosa.load(input_file)

        if len(signal) > 0:
            audiolength = librosa.get_duration(y=signal, sr=sr, hop_length=hop_length)
            signal = pad(signal)
            mel_signal = librosa.feature.melspectrogram(y=signal, sr=sr, hop_length=hop_length)
            spectrogram = librosa.power_to_db(mel_signal, ref=np.max)

            plt.figure(figsize=(factor * audiolength, 1))
            plt.axis('off')
            plt.imshow(spectrogram, cmap='magma', aspect='auto', extent=[0, 1, 0, 1])
            plt.savefig(output_file, dpi=224, bbox_inches="tight", pad_inches=0)
            plt.close()
        else:
            print(f"Skipping file: {input_file} - Empty audio file.")
    except Exception as e:
        print(f"Error processing file: {input_file} - {e}")


# Process each audio file in the input folder
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith(".wav"):  # Consider only WAV files, adjust if needed
            input_file = os.path.join(root, file)
            output_file = os.path.join(output_folder, f"{os.path.splitext(file)[0]}.png")

            try:
                signal, _ = librosa.load(input_file)

                if len(signal) > 0:
                    save_mel_spectrogram(input_file, output_file)
                else:
                    print(f"Skipping file: {input_file} - Empty audio file.")
            except Exception as e:
                print(f"Error processing file: {input_file} - {e}")
