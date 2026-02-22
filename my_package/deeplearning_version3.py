# -*- coding: utf-8 -*-

def run():
    from google.colab import files
    import zipfile
    import os

    print("upload you dataset in zip file")
    # Upload a zip file
    uploaded = files.upload()

    # Assume you uploaded "myfolder.zip"
    zip_path = list(uploaded.keys())[0]

    # Extract it
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("/content/myfolder")

    # Now you can access the folder
    data_dir = "/content/myfolder"

    # Fixed, resume-capable training script (Colab-ready)
    import os, glob, re
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import ModelCheckpoint
    import matplotlib.pyplot as plt

    # === SETTINGS ===
    #data_dir = "/content/drive/MyDrive/Datasets/archive/Tomato Leaf Disease"     # your dataset folder (contains class subfolders)
    img_height, img_width = 128, 128
    batch_size = 32
    seed = 40
    total_epochs = 20      # total number of epochs you want to run (set > already-run epochs to resume)

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

    # Normalize and improve pipeline performance
    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)).cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)).cache().prefetch(tf.data.AUTOTUNE)

    # === BUILD MODEL ===
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(class_names), activation='softmax')
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
        callbacks=[checkpoint_callback]
    )

    # === PLOT METRICS ===
    plt.plot(history.history.get('accuracy', []), label='train acc')
    plt.plot(history.history.get('val_accuracy', []), label='val acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def save():
    #=== SAVE final model (optional) ===
    final_path = os.path.join(checkpoint_dir, f"{model_name}_epoch_{total_epochs:02d}.keras")
    #model.save(final_path)
    #print("Final model saved to:", final_path)

    # === ASK EXPORT FORMAT ===
    print("\nWhich format do you want to export?")
    print("1. TensorFlow Keras (new .keras)")
    print("2. Legacy Keras HDF5 (.h5)")
    print("3. ONNX (.onnx)")
    print("4. TensorFlow Lite (.tflite)")

    choice = input("Enter choice (1-4): ").strip()

    ext_map = {
        "1": "keras",
        "2": "h5",
        "3": "tflite"
    }

    if choice in ext_map:
        ext = ext_map[choice]
        final_path = os.path.join(checkpoint_dir, f"{model_name}_epoch_{total_epochs:02d}.{ext}")
    else:
        print("Invalid choice, defaulting to .keras")
        final_path = os.path.join(checkpoint_dir, f"{model_name}_epoch_{total_epochs:02d}.keras")

    # === EXPORT BASED ON CHOICE ===
    if choice == "1":
        model.save(final_path)

    elif choice == "2":
        model.save(final_path)

    elif choice == "3":
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(final_path, "wb") as f:
            f.write(tflite_model)



    # === EXPORT TO DRIVE AND LOCAL DOWNLOAD ===
    if final_path:
        print("Model exported to Google Drive at:", final_path)

        # Also make available for local download
        from google.colab import files
        files.download(final_path)