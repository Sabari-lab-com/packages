
def start_train(path):
   

    # Now you can access the folder
    data_dir = path

    # Fixed, resume-capable training script (Colab-ready)
    import os, glob, re
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import ModelCheckpoint
    import matplotlib.pyplot as plt

    # === SETTINGS ===
    #data_dir = "/content/drive/MyDrive/Datasets/archive/Tomato Leaf Disease"     # your dataset folder (contains class subfolders)
    img_height, img_width = 224, 224
    batch_size = 32
    seed = 40
    total_epochs = 30      # total number of epochs you want to run (set > already-run epochs to resume)

    # === LOAD DATA (train/validation split inside same folder) ===
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    print("Detected classes:", class_names)

    # Data augmentation: improves generalization by creating random variations of input images
    # (random flip, rotation, zoom) to help the model learn more robust features
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    # Normalize and improve pipeline performance
    normalization_layer = layers.Rescaling(1./255)
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda x, y: (data_augmentation(normalization_layer(x)), y)).cache().shuffle(1000).prefetch(autotune)
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)).cache().prefetch(autotune)

    # === MIXED PRECISION (for speed on GPU) ===
    # Enables mixed precision (float16) for faster training on modern GPUs
    # Reduces memory usage and can speed up training without loss of accuracy
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled.")
    except Exception as e:
        print("Mixed precision not enabled:", e)

    # === BUILD MODEL ===
    # Model architecture: BatchNormalization layers are added after each Conv2D and Dense layer
    # to stabilize and speed up training by normalizing activations
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(len(class_names), activation='softmax', dtype='float32')  # output in float32 for stability
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    #model name
    model_name = input("Enter a name for this model (e.g., tomato_disease_v1): ").strip()

    # === CHECKPOINT SETUP ===
    checkpoint_dir = f"/content/drive/MyDrive/checkpoints/{model_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save every epoch with the epoch number in filename (enables auto-resume)
    checkpoint_template = os.path.join(checkpoint_dir, f"{model_name}_epoch_{{epoch:02d}}.keras")

    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_template,
        save_best_only=False,    # save every epoch (so we can resume)
        save_weights_only=False, # save full model (architecture + optimizer state)
        verbose=1
    )
    # Early stopping and learning rate reduction callbacks:
    # EarlyStopping: stops training if validation loss doesn't improve, preventing overfitting
    # ReduceLROnPlateau: reduces learning rate if validation loss plateaus, helping model converge
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    # === AUTO-RESUME: find latest checkpoint for this model ===
    pattern = os.path.join(checkpoint_dir, f"{model_name}_epoch_*.keras")
    checkpoint_files = glob.glob(pattern)

    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        print("Found checkpoint, loading:", latest_checkpoint)
        try:
            model = tf.keras.models.load_model(latest_checkpoint)
            m = re.search(rf"{model_name}_epoch_(\d+)\.keras$", latest_checkpoint)
            if m:
                initial_epoch = int(m.group(1))
                print(f"Resuming from epoch {initial_epoch}.")
            else:
                initial_epoch = 0
                print("Could not parse epoch from filename; starting from epoch 0.")
        except Exception as e:
            print("Failed to load checkpoint:", e)
            initial_epoch = 0
    else:
        print("No checkpoints found for this model; starting fresh.")
        initial_epoch = 0


    # === TRAIN (resume using initial_epoch) ===
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=initial_epoch,
        callbacks=[checkpoint_callback, early_stop, reduce_lr]
    )

    # === PLOT METRICS ===
    plt.plot(history.history.get('accuracy', []), label='train acc')
    plt.plot(history.history.get('val_accuracy', []), label='val acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    final_path = os.path.join(checkpoint_dir, f"{model_name}_epoch_{total_epochs:02d}.keras")
    return (checkpoint_dir,model_name,total_epochs,model)

