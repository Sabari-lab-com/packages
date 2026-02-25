def load_dataset():
    from google.colab import files, drive
    import zipfile
    import os

    print("Upload your dataset as a zip file")
    uploaded = files.upload()

    # Get uploaded file name
    zip_path = list(uploaded.keys())[0]

    # Define target folder inside Drive
    target_folder = "/content/drive/MyDrive/image_dataset"
    os.makedirs(target_folder, exist_ok=True)

    # Extract into Drive folder
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_folder)

    print("Check your drive folder name:image_dataset \n copy the path your Dataset")
    return target_folder

