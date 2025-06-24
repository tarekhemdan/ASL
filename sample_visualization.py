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
from keras.optimizers import Nadam

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

# Visualize some samples
f, ax = plt.subplots(2, 5)
f.set_size_inches(10, 10)
k = 0
for i in range(2):
    for j in range(5):
        ax[i, j].imshow(x_train[k].reshape(28, 28), cmap="gray")
        k += 1
    plt.tight_layout()

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
from keras.utils import plot_model

# Build the SNDA model
model = build_snda(input_shape=(28, 28, 1))

# Compile the model
model.compile(optimizer=Nadam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Generate and save the model summary as an image
plot_model(model, to_file='model_summary.png', show_shapes=True, show_layer_names=True, dpi=96)

# Display the model summary in the console
model.summary()