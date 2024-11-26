


"""
This script is used to preprocess the vehicle dataset. The preprocessing steps include:
1. Adding all the images to a pandas DataFrame
2. Adding the labels to the DataFrame
3. Encoding the labels
4. Resizing the images to a fixed size
5. Grayscaling the images
6. Splitting the dataset into training and testing sets
6. Normalizing the pixel values
7. Flattening the images
8. Performing Principal Component Analysis (PCA) on the images
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import PIL
import PIL.ImageOps
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def resize_image(image, size=(128, 128)):
    """
    Resize the given image to the specified size.

    Parameters:
    - image (np.ndarray): The image to resize.
    - size (tuple): The target size of the image.

    Returns:
    - resized_image (np.ndarray): The resized image.
    """
    return cv2.resize(image, size)

def grayscale_image(image):
    """
    Convert the given image to grayscale.

    Parameters:
    - image (np.ndarray): The image to convert to grayscale.

    Returns:
    - grayscale_image (np.ndarray): The grayscale image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def collect_data(data_dir):
    """
    Collect the images and labels from the given directory.

    Parameters:
    - data_dir (str): The directory containing the images.

    Returns:
    - df (pd.DataFrame): The DataFrame containing the images and labels.
    """
    images = []
    labels = []

    for label in os.listdir(data_dir):
        print(f"Collecting images for {label}...")
        label_dir = os.path.join(data_dir, label)

        for image_name in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_name)
            image = cv2.imread(image_path)
            images.append(image)
            # Encode the label
            if label == "hatchback":
                labels.append(0)
            elif label == "sedan":
                labels.append(1)
            elif label == "suv":
                labels.append(2)
            elif label == "pickup":
                labels.append(3)
            elif label == "motorcycle":
                labels.append(4)

    # Convert to pandas DataFrame
    df = pd.DataFrame({"image": images, "label": labels})

    return df

def perform_pca(images, variance_threshold=0.8, n_components=-1):
    """
    Perform Principal Component Analysis (PCA) on the given images.

    Parameters:
    - images (np.ndarray): The images to perform PCA on.
    - n_components (int): The number of principal components to keep.

    Returns:
    - pca_images (np.ndarray): The images transformed by PCA.
    - num_components (int): The number of components retained.
    """

    # Flatten the images
    flattened_images = [image.flatten() for image in images]

    # Normalize the pixel values
    images_normalized = np.array(flattened_images) / 255.0

    if n_components != -1:
        # Apply PCA with the specified number of components
        pca = PCA(n_components=n_components)
        pca_images = pca.fit_transform(images_normalized)
        return pca_images

    # Fit PCA to determine the number of components to retain the desired variance
    pca = PCA()
    pca.fit(images_normalized)
    cumulative_variance = pca.explained_variance_ratio_.cumsum()
    num_components = (cumulative_variance >= variance_threshold).argmax() + 1

    # Apply PCA with the determined number of components
    pca = PCA(n_components=num_components)
    pca_images = pca.fit_transform(images_normalized)

    return pca_images, num_components

def preprocess_dataset(data_dir):
    """
    Preprocess the dataset by resizing, grayscaling, normalizing, and flattening the images.

    Parameters:
    - data_dir (str): The directory containing the dataset.

    Returns:
    - X_train (np.ndarray): The training images.
    - X_test (np.ndarray): The testing images.
    - y_train (np.ndarray): The training labels.
    - y_test (np.ndarray): The testing labels.
    """
    # Collect the images and labels
    df = collect_data(data_dir)

    # Resize, grayscal, normalize, and flatten the images
    preprocessed_images = []
    for image in df['image']:
        resized_image = resize_image(image)
        grayscaled_image = grayscale_image(resized_image)
        preprocessed_images.append(grayscaled_image)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(preprocessed_images, df["label"], test_size=0.2, random_state=42)

    # Perform PCA on the images
    # X_train, num_components = perform_pca(X_train)
    # X_test = perform_pca(X_test, n_components=num_components)

    # Convert to numpy arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    new_X_train = []
    new_X_test = []
    for i in range(len(X_train)):
        new_X_train.append(X_train[i].flatten())
    for i in range(len(X_test)):
        new_X_test.append(X_test[i].flatten())

    X_train = np.array(new_X_train) / 255.0
    X_test = np.array(new_X_test) / 255.0

    return X_train, X_test, y_train, y_test
