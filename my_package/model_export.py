def save(train):
        import os
        import os, glob, re
        import tensorflow as tf
        from tensorflow.keras import layers, models
        from tensorflow.keras.callbacks import ModelCheckpoint
        import matplotlib.pyplot as plt
        checkpoint_dir = train[0]
        model_name = train[1]
        total_epochs = train[2]
        model = train[3]
        # === ASK EXPORT FORMAT ===
        print("\nWhich format do you want to export?")
        print("1. TensorFlow Keras (new .keras)")
        print("2. Legacy Keras HDF5 (.h5)")
        print("3. TensorFlow Lite (.tflite)")

        choice = input("Enter choice (1-3): ").strip()

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