import tarfile
import urllib.request
import os
import shutil

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def download_dataset():

    url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1.tar"

    tar_filename = "aircraft_damage_dataset_v1.tar"
    extracted_folder = "aircraft_damage_dataset_v1"

    urllib.request.urlretrieve(
        url,
        tar_filename
    )

    if os.path.exists(extracted_folder):
        shutil.rmtree(extracted_folder)

    with tarfile.open(
        tar_filename,
        "r"
    ) as tar_ref:

        tar_ref.extractall()

    return extracted_folder


def create_generators(
        train_dir,
        valid_dir,
        test_dir,
        img_rows=224,
        img_cols=224,
        batch_size=32,
        seed=42
):

    train_datagen = ImageDataGenerator(
        rescale=1./255
    )

    valid_datagen = ImageDataGenerator(
        rescale=1./255
    )

    test_datagen = ImageDataGenerator(
        rescale=1./255
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True,
        seed=seed
    )

    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False,
        seed=seed
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False,
        seed=seed
    )

    return train_generator, valid_generator, test_generator
