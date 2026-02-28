def start_resnet(epochs,path):

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
    total_epochs = epochs      # total number of epochs you want to run (set > already-run epochs to resume)

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
    # Added color jitter, random contrast, and brightness for more robust augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
        layers.RandomBrightness(0.1),
    ])

    # Mixup augmentation function
    def mixup(ds, alpha=0.2):
        import tensorflow as tf
        def _mixup(batch_x, batch_y):
            batch_size = tf.shape(batch_x)[0]
            l = tf.compat.v1.distributions.Beta(alpha, alpha).sample([batch_size])
            x_l = tf.reshape(l, (batch_size, 1, 1, 1))
            y_l = tf.reshape(l, (batch_size, 1))
            index = tf.random.shuffle(tf.range(batch_size))
            mixed_x = batch_x * x_l + tf.gather(batch_x, index) * (1 - x_l)
            mixed_y = batch_y * y_l + tf.gather(batch_y, index) * (1 - y_l)
            return mixed_x, mixed_y
        return ds.map(lambda x, y: _mixup(x, tf.one_hot(y, depth=len(class_names))), num_parallel_calls=tf.data.AUTOTUNE)

    # Preprocessing for ResNet
    preprocess_input = tf.keras.applications.resnet.preprocess_input
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda x, y: (data_augmentation(preprocess_input(x)), y)).cache().shuffle(1000).prefetch(autotune)
    val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y)).cache().prefetch(autotune)

    # Apply Mixup augmentation to training set
    train_ds = mixup(train_ds)

    # === MIXED PRECISION (for speed on GPU) ===
    # Enables mixed precision (float16) for faster training on modern GPUs
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled.")
    except Exception as e:
        print("Mixed precision not enabled:", e)

    # === BUILD MODEL: Transfer Learning with ResNet50 ===
    # Use pretrained ResNet50 as feature extractor, add custom classification head
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(img_height, img_width, 3),
        pooling='avg'
    )
    base_model.trainable = False  # Freeze base model for initial training

    model = models.Sequential([
        base_model,
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(len(class_names), activation='softmax', dtype='float32')  # output in float32 for stability
    ])

    # Use label smoothing for better generalization
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy'])

    # === CALLBACKS ===
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
    checkpoint_dir = "./resnet_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "resnet_epoch_{epoch:02d}.keras"),
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    tensorboard_cb = TensorBoard(log_dir="./logs", histogram_freq=1)

    # Compute class weights for imbalanced datasets
    from collections import Counter
    import numpy as np
    labels = []
    for _, y in train_ds.unbatch():
        labels.append(np.argmax(y.numpy()))
    class_counts = Counter(labels)
    total = sum(class_counts.values())
    class_weights = {i: total/(len(class_counts)*c) for i, c in class_counts.items()}

    # === TRAIN ===
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        callbacks=[checkpoint_callback, early_stop, reduce_lr, tensorboard_cb],
        class_weight=class_weights
    )

    # === FINE-TUNING: Unfreeze base model and retrain with low learning rate ===
    base_model.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print("Fine-tuning the entire ResNet50 model...")
    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,  # Additional epochs for fine-tuning
        callbacks=[checkpoint_callback, early_stop, reduce_lr]
    )

    # === PLOT METRICS ===
    plt.plot(history.history.get('accuracy', []), label='train acc')
    plt.plot(history.history.get('val_accuracy', []), label='val acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()