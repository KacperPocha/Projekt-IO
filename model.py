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
