import tensorflow as tf
import numpy as np
from tensorflow import keras
import pathlib
import time
import matplotlib.pyplot as plt
import tensorflow_io as tfio  # Import tensorflow-io for signal processing
import seaborn as sns
# Load the pre-trained model
saved_model_path = "W:/workdir/Models/model5.h5"
loaded_model = keras.models.load_model(saved_model_path)

# Prepare the new dataset
training_dir = pathlib.Path("W:\\workdir3\\train")
test_dir = pathlib.Path("W:\\workdir3\\test")
train_count = len(list(training_dir.glob('*/*.png')))
test_count = len(list(test_dir.glob('*/*.png')))
validation_split = test_count / train_count
batch_size = 100
epochs=20
img_height = 224
img_width = 224

print("\033[1mCreating training and validation datasets:\033[0m")
training_ds = tf.keras.utils.image_dataset_from_directory(
    training_dir,
    validation_split=validation_split,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    crop_to_aspect_ratio=True,
    label_mode='binary',
    class_names=['fake', 'real']
)
validation_ds = tf.keras.utils.image_dataset_from_directory(
    training_dir,
    validation_split=validation_split,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    crop_to_aspect_ratio=True,
    label_mode='binary',
)
print("\n\033[1mCreating test dataset:\033[0m")
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    crop_to_aspect_ratio=True,
    label_mode='binary',
    class_names=['fake', 'real']
)

class_names = training_ds.class_names
print("\nNames of", str(len(class_names)), "classes:", class_names)
# Set the layers for fine-tuning
for layer in loaded_model.layers[:-1]:  # Unfreeze the last 30 layers for example
    layer.trainable = False

def apply_audio_augmentation(image, label):
    # Apply time masking
    spectrogram = tfio.audio.time_mask(image, param=10)  # Adjust 'param' as needed

    # Apply frequency masking
    spectrogram = tfio.audio.freq_mask(image, param=5)  # Adjust 'param' as needed

    return image, label


augmented_training_ds = training_ds.map(
    apply_audio_augmentation,
    num_parallel_calls=tf.data.AUTOTUNE
)

    # Compile the model
loaded_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Set your desired learning rate
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.BinaryAccuracy(),
        keras.metrics.Precision(),
        keras.metrics.Recall(),
        keras.metrics.TruePositives(),
        keras.metrics.FalsePositives(),
        keras.metrics.TrueNegatives(),
        keras.metrics.FalseNegatives(),
    ]
)
loaded_model.summary()

# Model training
steps_per_epoch = len(training_ds) / epochs

t0 = time.time()
# Train the model
history = loaded_model.fit(
    augmented_training_ds,  # New dataset for fine-tuning
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_ds,  # New validation set
    validation_steps=1,  # Define based on validation dataset size
    epochs=epochs
)
# Access training loss and accuracy
train_loss = history.history['loss']
train_accuracy = history.history['binary_accuracy']
val_loss = history.history['val_loss']  # Validation loss
val_accuracy = history.history['val_binary_accuracy']  # Validation accuracy

# Print or use the values as needed
print("Training Loss per epoch:", train_loss)
print("Training Accuracy per epoch:", train_accuracy)


t1 = time.time()
dt = (t1 - t0)

# Evaluate the fine-tuned model on the new dataset
results = loaded_model.evaluate(
    test_ds,
    return_dict=True
)
test_accuracy = results['binary_accuracy']
print("Test accuracy on new dataset:", test_accuracy)

def f1score(p, r):
    f1 = 2 / ((1 / p) + (1 / r))
    return f1

print("-" * 70)
print('\033[1m' + "Model metrics:" + '\033[0m')
for i in results:
    print(i + ": " + str(results[i]))
print("-" * 70)
print("F1 Score: " + str(f1score(results['precision'], results['recall'])))
print("Time to train: ", dt)
print("-" * 70)

tp, fp = results['true_positives'], results['false_positives']
fn, tn = results['false_negatives'], results['true_negatives']
cmx = np.array([[tp, fp], [fn, tn]], np.int32)

#model.save("W:/workdir/Models/model5.h5")

cmx_plot = sns.heatmap(
    cmx / np.sum(cmx),
    cmap='Blues',
    annot=True,
    fmt=".1%",
    linewidth=5,
    cbar=False,
    square=True,
    xticklabels=['Spoof (1)', 'Real (0)'],
    yticklabels=['Spoof (1)', 'Real (0)']
)
cmx_plot.set(xlabel="Actual", ylabel="Predicted")


plt.figure(figsize=(12, 4))

# Plotting training loss
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting training accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig("W:/workdir/Plots/plot5.png")
plt.show()