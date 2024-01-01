import os
import librosa
import numpy as np

input_folder = "W:\\for-norm\\for-norm\\testing\\fake"
output_folder = "W:\\workdir\\1\\test\\fake\\mel_spectrograms"  # Modify the output folder accordingly

hop_length = 512
sr = 16000
max_time_steps = 200  # Define the maximum time steps for your model

def pad_spectrogram(spec):
    spec_len = spec.shape[1]
    if spec_len >= max_time_steps:
        return spec[:, :max_time_steps]
    else:
        return np.pad(spec, ((0, 0), (0, max_time_steps - spec_len)), mode='constant')

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

def save_mel_spectrogram(input_file, output_file):
    print(f"Processing file: {input_file}")
    try:
        signal, _ = librosa.load(input_file)
        if len(signal) > 0:
            signal = pad(signal)
            mel_signal = librosa.feature.melspectrogram(y=signal, sr=sr, hop_length=hop_length)
            spectrogram = librosa.power_to_db(mel_signal, ref=np.max)
            padded_spectrogram = pad_spectrogram(spectrogram)
            # Store the padded Mel spectrogram in the output folder as a NumPy file
            np.save(output_file, padded_spectrogram)
        else:
            print(f"Skipping file: {input_file} - Empty audio file.")
    except Exception as e:
        print(f"Error processing file: {input_file} - {e}")

# Process each audio file in the input folder
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith(".wav"):  # Adjust the file extension if needed
            input_file = os.path.join(root, file)
            output_file = os.path.join(output_folder, f"{os.path.splitext(file)[0]}.npy")
            try:
                save_mel_spectrogram(input_file, output_file)
            except Exception as e:
                print(f"Error processing file: {input_file} - {e}")
