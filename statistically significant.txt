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
from scipy.stats import ttest_rel  # For paired t-tests

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

# Build and compile the SNDA model
model_snda = build_snda()
model_snda.compile(optimizer=Nadam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the SNDA model
history_snda = model_snda.fit(datagen.flow(x_train, y_train, batch_size=128),
                              epochs=20,
                              validation_data=(x_test, y_test),
                              callbacks=[learning_rate_reduction],
                              verbose=0)

# Evaluate the SNDA model
snda_test_accuracy = model_snda.evaluate(x_test, y_test, verbose=0)[1]
print(f"SNDA Test Accuracy: {snda_test_accuracy * 100:.2f}%")

# Baseline Model (e.g., a simpler CNN)
def build_baseline_model(input_shape=(28, 28, 1)):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = MaxPool2D((2, 2), strides=2, padding='same')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPool2D((2, 2), strides=2, padding='same')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(24, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

# Build and compile the baseline model
model_baseline = build_baseline_model()
model_baseline.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the baseline model
history_baseline = model_baseline.fit(datagen.flow(x_train, y_train, batch_size=128),
                                      epochs=20,
                                      validation_data=(x_test, y_test),
                                      verbose=0)

# Evaluate the baseline model
baseline_test_accuracy = model_baseline.evaluate(x_test, y_test, verbose=0)[1]
print(f"Baseline Test Accuracy: {baseline_test_accuracy * 100:.2f}%")

# Statistical Significance Testing
def evaluate_model_multiple_times(model, x_test, y_test, n_runs=10):
    """
    Evaluate the model multiple times to account for variability.
    """
    accuracies = []
    for _ in range(n_runs):
        accuracy = model.evaluate(x_test, y_test, verbose=0)[1]
        accuracies.append(accuracy)
    return np.array(accuracies)

# Evaluate SNDA and baseline models multiple times
snda_accuracies = evaluate_model_multiple_times(model_snda, x_test, y_test, n_runs=10)
baseline_accuracies = evaluate_model_multiple_times(model_baseline, x_test, y_test, n_runs=10)

# Perform a paired t-test to compare the accuracies
t_statistic, p_value = ttest_rel(snda_accuracies, baseline_accuracies)
print(f"Paired t-test results: t = {t_statistic:.4f}, p = {p_value:.4f}")

# Interpret the results
if p_value < 0.05:
    print("The difference in accuracy between SNDA and the baseline model is statistically significant (p < 0.05).")
else:
    print("The difference in accuracy between SNDA and the baseline model is not statistically significant (p >= 0.05).")

import matplotlib.pyplot as plt
import numpy as np

means = [np.mean(baseline_accuracies), np.mean(snda_accuracies)]
labels = ['Baseline', 'SNDA']

plt.bar(labels, means, color=['blue', 'green'])
plt.title('Mean Test Accuracies (10 Runs)')
plt.ylabel('Accuracy')
plt.show()