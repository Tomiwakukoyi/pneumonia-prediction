import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam


# Define CNN model
model = Sequential([
     Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    BatchNormalization(),  # Normalize activations
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Prevent overfitting
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.0001),  # Lower LR for fine-tuning
    metrics=['accuracy']
)


# Define directories
train_dir = "chest_xray/train"
test_dir = "chest_xray/test"

# Image Preprocessing & Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,     # Normalize pixel values (0-255 -> 0-1)
    rotation_range=20,   # Randomly rotate images
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
     fill_mode='nearest'    
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load Data from Directory
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model_checkpoint = ModelCheckpoint(
    filepath="best_model.h5",
    monitor="val_loss",
    save_best_only=True
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    patience=3,  # Reduce LR if no improvement for 3 epochs
    factor=0.5,  # Reduce LR by half
    min_lr=1e-6
)
# Include callbacks in training
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=test_generator,
    callbacks=[early_stopping, model_checkpoint]  # Ensure callbacks are passed as a list

)

# Save the trained model
model.save("pneumonia_cnn_model.h5")

# Plot training history
import matplotlib.pyplot as plt

def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title("Model Accuracy")

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title("Model Loss")

    plt.show()

plot_training_history(history)

# Print model summary
model.summary()
