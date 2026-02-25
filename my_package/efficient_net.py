def start_train_efficient_net(epochs,path):

    import os, glob, re
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    from google.colab import drive


    # ================= SETTINGS =================
    data_dir = path
    img_height, img_width = 224, 224
    batch_size = 32
    seed = 40
    total_epochs = int(epochs)

    # ================= LOAD DATA =================
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
    num_classes = len(class_names)
    print("Detected classes:", class_names)

    # ================= PREPROCESS =================
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.08),
        layers.RandomContrast(0.08),
    ])

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)

    val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    # ================= MODEL =================
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(img_height, img_width, 3),
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])

    # ================= FIXED METRICS (NO ERROR) =================
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=metrics
    )

    model.summary()

    # ================= CALLBACKS =================
    model_name = input("Set model name: ").strip() or "efficientnet_model"

    checkpoint_dir = f"/content/drive/MyDrive/checkpoints/{model_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_template = os.path.join(
        checkpoint_dir,
        f"{model_name}_epoch_{{epoch:02d}}.keras"
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_template,
        save_best_only=False,
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=6,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=1e-7
    )

    # ================= AUTO RESUME =================
    pattern = os.path.join(checkpoint_dir, f"{model_name}_epoch_*.keras")
    checkpoint_files = glob.glob(pattern)

    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        print("Loading checkpoint:", latest_checkpoint)
        model = tf.keras.models.load_model(latest_checkpoint)
        match = re.search(rf"{re.escape(model_name)}_epoch_(\d+)\.keras$", latest_checkpoint)
        initial_epoch = int(match.group(1)) if match else 0
    else:
        initial_epoch = 0
        print("Starting fresh training.")

    # ================= TRAIN =================
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=initial_epoch,
        callbacks=[checkpoint_callback, early_stop, reduce_lr]
    )

    # ================= PLOTS =================
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.legend()
    plt.title("Accuracy")
    plt.show()

    # ================= EVALUATION =================
    y_true, y_pred = [], []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_names,
                yticklabels=class_names,
                cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

    return (checkpoint_dir, model_name, total_epochs, model)
