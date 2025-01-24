import numpy as np
import requests
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Zdefiniowanie wysokości i szerokości obrazu
IMG_HEIGHT = 32  # Wysokość obrazu, na jaką zostaną przeskalowane wszystkie obrazy
IMG_WIDTH = 32   # Szerokość obrazu, na jaką zostaną przeskalowane wszystkie obrazy

# Twój słownik z URL-ami do obrazów
dict_of_urls = {
    'znak': 'https://www.prawo-jazdy-360.pl/image/znaki-drogowe/D-6-przejscie-dla-pieszych.webp' 
}
