import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, utils
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Define hyperparameters
batch_size = 32
epochs = 20
input_shape = (224, 224, 3)
num_classes = 40

# Define the data paths
train_path = r"C:\Users\User4\Desktop\ASL_new\training_set"
valid_path =  r"C:\Users\User4\Desktop\ASL_new\test_set"
test_path =  r"C:\Users\User4\Desktop\ASL_new\test_set"

# Define the VGG-19 model
base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
x = base_model.output
x = layers.Flatten()(x)
x = layers.Dense(4096, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(4096, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=output)

# Freeze the convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=optimizers.Nadam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define data generators with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical')

valid_datagen = ImageDataGenerator(rescale=1./255)
valid_generator = valid_datagen.flow_from_directory(
    directory=valid_path,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    directory=test_path,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical')

# Train the model with fine-tuning
history = model.fit(train_generator, epochs=epochs, validation_data=valid_generator, steps_per_epoch=len(train_generator)//batch_size, validation_steps=len(valid_generator)//batch_size)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator)//batch_size)
print('Test accuracy:', test_acc)

# Save the model
model.save('model.h5')

# Plot confusion matrix using Seaborn
y_true = test_generator.classes
y_pred = np.argmax(model.predict(test_generator), axis=-1)
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, cmap='Blues', annot=True, fmt='g', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')

# Plot learning curves using Seaborn
plt.figure(figsize=(8, 6))
sns.lineplot(x=range(epochs), y=history.history['accuracy'], label='Training Accuracy')
sns.lineplot(x=range(epochs), y=history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Plot loss curves using Seaborn
plt.figure(figsize=(8, 6))
sns.lineplot(x=range(epochs), y=history.history['loss'], label='Training Loss')
sns.lineplot(x=range(epochs), y=history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()