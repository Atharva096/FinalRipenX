import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- Configuration ---
# Update this path to where your 'dataset' folder is located
DATASET_DIR = "mango_data/train" 
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
NUM_CLASSES = 3

def create_datasets(dataset_dir):
    """Loads and prepares the training and validation datasets."""
    print("Loading training data...")
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=42, # Seed ensures no overlap between train/val sets
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int' # Outputs integers (0, 1, 2) for sparse_categorical_crossentropy
    )

    print("Loading validation data...")
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int'
    )

    # Store class names for reference before optimizing
    class_names = train_dataset.class_names
    print(f"Detected classes: {class_names}")

    # Optimize datasets for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    return train_dataset, val_dataset, class_names

def build_model():
    """Builds the MobileNetV2 model with custom classification layers."""
    # 1. Input layer
    inputs = Input(shape=(224, 224, 3))
    
    # 2. Preprocess input to match MobileNetV2 requirements (-1 to 1 scaling)
    x = preprocess_input(inputs)

    # 3. Load the base model
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False, 
        weights='imagenet'
    )
    
    # Freeze the base model
    base_model.trainable = False

    # 4. Pass inputs through the base model
    # training=False forces Batch Normalization layers to stay in inference mode
    x = base_model(x, training=False) 

    # 5. Add custom classification head
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    # 6. Assemble the final model
    model = Model(inputs, outputs)
    
    # 7. Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    # 1. Load Data
    train_dataset, val_dataset, class_names = create_datasets(DATASET_DIR)

    # 2. Build Model
    model = build_model()
    model.summary()

    # 3. Define Callbacks (Optional but highly recommended)
    # Stops training early if validation loss stops improving to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # 4. Train the Model
    print("\nStarting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[early_stop]
    )

    # 5. Save the final model
    model_path = "mango_ripeness_mobilenetv2.keras"
    model.save(model_path)
    print(f"\nModel successfully trained and saved to {model_path}")

if __name__ == "__main__":
    main()