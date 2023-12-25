import numpy as np
import os
import PIL
import PIL.Image
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_io as tfio  # Import tensorflow-io for signal processing




saved_model_path = "W:/workdir/Models/model2.h5"
loaded_model = keras.models.load_model(saved_model_path)
#training_dir = pathlib.Path("W:\\workdir\\train")
test_dir = pathlib.Path("W:\\workdir3\\test")

test_count = len(list(test_dir.glob('*/*.png')))
#train_count = len(list(training_dir.glob('*/*.png')))
#validation_split = test_count / train_count
# Define parameters and create datasets
batch_size = 100
#epochs = 20
img_height = 224
img_width = 224



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


# Build the model
'''
base_model = keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    pooling="avg"
)

# Freeze layers except the last few
for layer in base_model.layers[:-55]:  # Unfreeze the last 7 layers for example
    layer.trainable = False

# Create your model on top of the base model
model = keras.Sequential([
    base_model,
    keras.layers.Dense(1, activation='sigmoid')
])
#model.add(keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l1(0.01)))
#model.layers[0].trainable = True

model.compile(
    optimizer="adam",
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

model.summary()

# Model training
steps_per_epoch = len(training_ds) / epochs

t0 = time.time()

saved_model_path = "W:/workdir/Models/model2.h5"
loaded_model = keras.models.load_model(saved_model_path)
# Model training
history = model.fit(
    augmented_training_ds,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_ds,
    validation_steps=1,
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
'''
# Model evaluation
results = loaded_model.evaluate(
    test_ds,
    return_dict=True
)
test_accuracy = results['binary_accuracy']
print("test accuracy on ADD dataset", test_accuracy )

def f1score(p, r):
    f1 = 2 / ((1 / p) + (1 / r))
    return f1

print("-" * 70)
print('\033[1m' + "Model metrics:" + '\033[0m')
for i in results:
    print(i + ": " + str(results[i]))
print("-" * 70)
print("F1 Score: " + str(f1score(results['precision'], results['recall'])))
#print("Time to train: ", dt)
print("-" * 70)

tp, fp = results['true_positives'], results['false_positives']
fn, tn = results['false_negatives'], results['true_negatives']
cmx = np.array([[tp, fp], [fn, tn]], np.int32)

#model.save("W:/workdir/Models/model4.h5")

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
'''
# Plotting training loss
plt.subplot(1, 2, 1)
#plt.plot(train_loss, label='Training Loss')
#plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting training accuracy
plt.subplot(1, 2, 2)
#plt.plot(train_accuracy, label='Training Accuracy')
#plt.plot(val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig("W:/workdir/Plots/plot4.png")
plt.show()
'''