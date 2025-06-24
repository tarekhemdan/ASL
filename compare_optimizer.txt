import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, Dense, Dropout, BatchNormalization, Flatten, GlobalAveragePooling2D, Multiply, Reshape
from keras.layers import concatenate, Activation
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Nadam, Adam, SGD, RMSprop

# Load data
train_df = pd.read_csv(r'C:\Users\User4\Desktop\ASL_novel\sign_mnist_train.csv')
test_df = pd.read_csv(r'C:\Users\User4\Desktop\ASL_novel\sign_mnist_test.csv')
test = pd.read_csv(r'C:\Users\User4\Desktop\ASL_novel\sign_mnist_test.csv')
y = test['label']
y_train = train_df['label']
y_test = test_df['label']
del train_df['label']
del test_df['label']

# Label binarization
from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.fit_transform(y_test)

# Normalize and reshape data
x_train = train_df.values / 255
x_test = test_df.values / 255
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False
)
datagen.fit(x_train)

# Learning rate reduction callback
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)

# Define the Attention Mechanism
def attention_block(input_tensor):
    attention = Conv2D(1, (1, 1), activation='sigmoid')(input_tensor)
    return Multiply()([input_tensor, attention])

# Define the Nevestro Densenet Attention (SNDA) model
def build_snda(input_shape=(28, 28, 1)):
    inputs = Input(shape=input_shape)
    
    # Initial Conv Layer
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 2), strides=2, padding='same')(x)
    
    # Dense Block 1
    x1 = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x1 = BatchNormalization()(x1)
    x1 = attention_block(x1)  # Add attention
    x = concatenate([x, x1])
    
    # Dense Block 2
    x2 = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x2 = BatchNormalization()(x2)
    x2 = attention_block(x2)  # Add attention
    x = concatenate([x, x2])
    
    # Transition Layer
    x = Conv2D(256, (1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 2), strides=2, padding='same')(x)
    
    # Dense Block 3
    x3 = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x3 = BatchNormalization()(x3)
    x3 = attention_block(x3)  # Add attention
    x = concatenate([x, x3])
    
    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)
    
    # Fully Connected Layer
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Output Layer
    outputs = Dense(24, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

# List of optimizers to compare
optimizers = {
    'Nadam': Nadam(),
    'Adam': Adam(),
    'SGD': SGD(learning_rate=0.01, momentum=0.9),
    'RMSprop': RMSprop()
}

# Dictionary to store results
results = {}

# Train and evaluate the model with each optimizer
for optimizer_name, optimizer in optimizers.items():
    print(f"Training with {optimizer_name} optimizer...")
    
    # Build and compile the model
    model = build_snda()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    history = model.fit(datagen.flow(x_train, y_train, batch_size=128),
                        epochs=20,
                        validation_data=(x_test, y_test),
                        callbacks=[learning_rate_reduction],
                        verbose=0)
    
    # Evaluate the model
    test_accuracy = model.evaluate(x_test, y_test, verbose=0)[1]
    results[optimizer_name] = {
        'history': history,
        'test_accuracy': test_accuracy
    }
    print(f"Test accuracy with {optimizer_name}: {test_accuracy * 100:.2f}%")

# Plot training and validation accuracy for each optimizer
plt.figure(figsize=(14, 8))
for optimizer_name, result in results.items():
    plt.plot(result['history'].history['val_accuracy'], label=f'{optimizer_name} (Val)')
    plt.plot(result['history'].history['accuracy'], label=f'{optimizer_name} (Train)', linestyle='--')
plt.title('Training and Validation Accuracy for Different Optimizers')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss for each optimizer
plt.figure(figsize=(14, 8))
for optimizer_name, result in results.items():
    plt.plot(result['history'].history['val_loss'], label=f'{optimizer_name} (Val)')
    plt.plot(result['history'].history['loss'], label=f'{optimizer_name} (Train)', linestyle='--')
plt.title('Training and Validation Loss for Different Optimizers')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Compare test accuracy of all optimizers
optimizer_names = list(results.keys())
test_accuracies = [results[optimizer]['test_accuracy'] * 100 for optimizer in optimizer_names]

plt.figure(figsize=(8, 6))
sns.barplot(x=optimizer_names, y=test_accuracies)
plt.title('Test Accuracy for Different Optimizers')
plt.xlabel('Optimizer')
plt.ylabel('Test Accuracy (%)')
plt.show()