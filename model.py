# Importowanie niezbędnych bibliotek
import numpy as np  # Importuje bibliotekę do obliczeń numerycznych
import pandas as pd  # Importuje bibliotekę do pracy z danymi w formie tabel
import os  # Importuje bibliotekę do pracy z systemem plików
import random  # Importuje bibliotekę do generowania liczb losowych
import matplotlib.pyplot as plt  # Importuje bibliotekę do rysowania wykresów
import tensorflow as tf  # Importuje TensorFlow, framework do głębokiego uczenia
from matplotlib.image import imread  # Importuje funkcję do wczytywania obrazów
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img  # Importuje funkcje do ładowania obrazów
from tensorflow.keras.models import Sequential, load_model  # Importuje klasy do tworzenia i ładowania modeli
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout  # Importuje warstwy do tworzenia sieci neuronowej
from tensorflow.keras.optimizers import Adam  # Importuje optymalizator Adam


# Katalog główny dla zbioru danych
dataset_path = './data/'  # Ścieżka do folderu, w którym znajdują się dane (zmień na odpowiednią)

train_folder = 'train'  # Folder z danymi treningowymi
test_folder = 'test'  # Folder z danymi testowymi

# Wysokość i szerokość obrazu
IMAGE_HEIGHT = 32  # Ustalamy wysokość obrazu na 32 piksele
IMAGE_WIDTH = 32  # Ustalamy szerokość obrazu na 32 piksele

# Generatory danych obrazu - normalizacja wartości pikseli
train_gen = ImageDataGenerator(rescale=1.0 / 255)  # Normalizuje obrazy treningowe, dzieląc piksele przez 255
test_gen = ImageDataGenerator(rescale=1.0 / 255)  # Normalizuje obrazy testowe, dzieląc piksele przez 255

# Generator danych treningowych
train_dataset = train_gen.flow_from_directory(
    batch_size=100,  # Rozmiar partii danych
    directory=os.path.join(dataset_path, train_folder),  # Ścieżka do folderu treningowego
    class_mode='sparse',  # Klasy są reprezentowane jako liczby całkowite
    shuffle=True,  # Losowe mieszanie danych
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH)  # Rozmiar obrazu, na który będą skalowane dane
)

# Generator danych walidacyjnych
test_dataset = test_gen.flow_from_directory(
    batch_size=100,  # Rozmiar partii danych
    directory=os.path.join(dataset_path, test_folder),  # Ścieżka do folderu testowego
    class_mode='sparse',  # Klasy są reprezentowane jako liczby całkowite
    shuffle=False,  # Dane testowe nie są mieszane
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH)  # Rozmiar obrazu, na który będą skalowane dane
)