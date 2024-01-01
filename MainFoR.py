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
from sklearn.model_selection import KFold
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model



training_dir = pathlib.Path("W:\\workdir\\train")
val_dir = pathlib.Path("W:\\workdir\\dev")
test_dir = pathlib.Path("W:\\workdir\\test")

train_count = len(list(training_dir.glob('*/*.png')))
val_count = len(list(val_dir.glob('*/*.png')))
test_count = len(list(test_dir.glob('*/*.png')))

#validation_split = test_count / train_count
# Define parameters and create datasets
batch_size = 100
epochs = 100
img_height = 224
img_width = 224

input_shape = (img_height, img_width, 3)

print("\033[1mCreating training dataset:\033[0m")

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest'  # You can explore more options in the documentation
)
training_ds = tf.keras.utils.image_dataset_from_directory(
    training_dir,
    validation_split=None,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    crop_to_aspect_ratio=True,
    label_mode='binary',
    class_names=['fake', 'real']
)
train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True,
    seed=123
)

print("\n\033[1mCreating val dataset:\033[0m")
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    crop_to_aspect_ratio=True,
    label_mode='binary',
    class_names=['fake', 'real']
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

# Define your class names explicitly
class_names = training_ds.class_names
print("\nNames of", str(len(class_names)), "classes:", class_names)

model_input = Input(shape=input_shape)
x = Conv2D(32, kernel_size=(3, 3), activation='relu')(model_input)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)

# Add Dense layers with Dropout for regularization
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
model_output = Dense(1, activation='sigmoid')(x)  # Output layer with a single neuron for binary classification

model = Model(inputs=model_input, outputs=model_output)



#model.add(keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l1(0.01)))
#model.layers[0].trainable = True

# Define the optimizer with a specific learning rate
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
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
validation_steps = len(val_ds) // batch_size

t0 = time.time()

checkpoint_path = "W:/workdir/FoR_Models/best_model2.h5"
checkpoint = ModelCheckpoint(checkpoint_path,
                             monitor='val_binary_accuracy',
                             verbose=1,
                             save_best_only=True,
                             mode='max')
# Model training
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    epochs=epochs,
    callbacks=[checkpoint]
)

# Access training loss and accuracy
train_loss = history.history['loss']
train_accuracy = history.history['binary_accuracy']
val_loss = history.history['val_loss']  # Validation loss
val_accuracy = history.history['val_binary_accuracy']  # Validation accuracy

# Print or use the values as needed
print("----------------------------------------")
print("Training Loss per epoch:", train_loss)
print("Training Accuracy per epoch:", train_accuracy)
print("Validation Loss per epoch:", val_loss)
print("----------------------------------------")

t1 = time.time()
dt = (t1 - t0)

best_model = tf.keras.models.load_model(checkpoint_path)
# Model evaluation
results = best_model.evaluate(
    test_ds,
    return_dict=True
)
test_accuracy = results['binary_accuracy']
print("Test Accuracy", test_accuracy )




plt.figure(figsize=(12, 6))

# Plotting training loss
plt.subplot(1, 2, 1)
plt.plot(train_loss, 'g', label='Training Loss')
plt.plot(val_loss, 'r', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting training accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracy, 'g', label='Training Accuracy')
plt.plot(val_accuracy, 'r', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


plot_title = 'Basic CNN Trained on FoR Dataset'  # Your desired plot title

plt.suptitle(plot_title, fontsize=14, y=0.98)  # Adjusted title
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjusted spacing for the title

# Generating the filename based on the plot title
plot_filename = f"W:/workdir/Plots/{plot_title.replace(' ', '_')}.png"

plt.savefig(plot_filename)
plt.show()


def f1score(p, r):
    epsilon = 1e-7  # A small value to avoid division by zero

    # Handling potential division by zero cases
    if p == 0 and r == 0:
        return 0.0  # Return 0 when both precision and recall are zero
    elif p + r == 0:
        return 0.0  # Return 0 when the sum of precision and recall is zero
    else:
        f1 = 2 * p * r / (p + r + epsilon)
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
cmx_plot.set_title('FoR')
cmx_plot.set(xlabel="Actual", ylabel="Predicted")




