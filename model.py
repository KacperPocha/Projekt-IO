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

# Liczba klas w zbiorze danych
NUM_CLASSES = train_dataset.num_classes  # Liczba klas to liczba katalogów w zbiorze danych
print(f'Liczba klas: {NUM_CLASSES}')  # Wyświetlenie liczby klas

# Mapowanie nazw klas na ich opisy
label_descriptions = {  # Słownik z opisami poszczególnych klas
    'A-1': 'Niebezpieczny zakręt w prawo',
    'A-11': 'Nierówna droga',
    'A-11a': 'Próg zwalniający',
    'A-12a': 'Zwężenie jezdni - dwustronne',
    'A-14': 'Roboty drogowe',
    'A-15': 'Śliska jezdnia',
    'A-16': 'Przejście dla pieszych',
    'A-17': 'Dzieci',
    'A-18b': 'Zwierzęta dzikie',
    'A-2': 'Niebezpieczny zakręt w lewo',
    'A-20': 'Odcinek jezdni o ruchu dwukierunkowym',
    'A-21': 'Tramwaj',
    'A-24': 'Rowerzyści',
    'A-29': 'Sygnały świetlne',
    'A-3': 'Niebezpieczne zakręty, pierwszy w prawo',
    'A-30': 'Inne niebezpieczeństwo',
    'A-32': 'Oszronienie jezdni',
    'A-4': 'Niebezpieczne zakręty, pierwszy w lewo',
    'A-6a': 'Skrzyżowanie z drogą podporządkowaną występującą po obu stronach',
    'A-6b': 'Skrzyżowanie z drogą podporządkowaną występującą po prawej stronie',
    'A-6c': 'Skrzyżowanie z drogą podporządkowaną występującą po lewej stronie',
    'A-6d': 'Wlot drogi jednokierunkowej z prawej strony',
    'A-6e': 'Wlot drogi jednokierunkowej z lewej strony',
    'A-7': 'Ustąp pierwszeństwa',
    'A-8': 'Skrzyżowanie o ruchu okrężnym',
    'B-1': 'Zakaz ruchu w obu kierunkach',
    'B-18': 'Zakaz wjazdu pojazdów o rzeczywistej masie całkowitej ponad ... t.',
    'B-2': 'Zakaz wjazdu',
    'B-20': 'STOP',
    'B-21': 'Zakaz skręcania w lewo',
    'B-22': 'Zakaz skręcania w prawo',
    'B-25': 'Zakaz wyprzedzania',
    'B-26': 'Zakaz wyprzedzania przez samochody ciężarowe',
    'B-27': 'Koniec zakazu wyprzedzania',
    'B-33': 'Ograniczenie prędkości',
    'B-34': 'Koniec ograniczenia prędkości',
    'B-36': 'Zakaz zatrzymywania się',
    'B-41': 'Zakaz ruchu pieszych',
    'B-42': 'Koniec zakazów',
    'B-43': 'Strefa ograniczonej prędkości',
    'B-44': 'Koniec strefy ograniczonej prędkości',
    'B-5': 'Zakaz wjazdu samochodów ciężarowych',
    'B-6-B-8-B-9': 'Zakaz wjazdu pojazdów innych niż samochodowe',
    'B-8': 'Zakaz wjazdu pojazdów zaprzęgowych',
    'B-9': 'Zakaz wjazdu rowerów',
    'C-10': 'Nakaz jazdy z lewej strony znaku',
    'C-12': 'Ruch okrężny',
    'C-13': 'Droga dla rowerów',
    'C-13-C-16': 'Droga dla pieszych i rowerzystów',
    'C-13a': 'Koniec drogi dla rowerów',
    'C-13a-C-16a': 'Koniec drogi dla pieszych i rowerzystów',
    'C-16': 'Droga dla pieszych',
    'C-2': 'Nakaz jazdy w prawo za znakiem',
    'C-4': 'Nakaz jazdy w lewo za znakiem',
    'C-5': 'Nakaz jazdy prosto',
    'C-6': 'Nakaz jazdy prosto lub w prawo',
    'C-7': 'Nakaz jazdy prosto lub w lewo',
    'C-9': 'Nakaz jazdy z prawej strony znaku',
    'D-1': 'Droga z pierwszeństwem',
    'D-14': 'Koniec pasa ruchu',
    'D-15': 'Przystanek autobusowy',
    'D-18': 'Parking',
    'D-18b': 'Parking zadaszony',
    'D-2': 'Koniec drogi z pierwszeństwem',
    'D-21': 'Szpital',
    'D-23': 'Stacja paliwowa',
    'D-23a': 'Stacja paliwowa tylko z gazem do napędu pojazdów',
    'D-24': 'Telefon',
    'D-26': 'Stacja obsługi technicznej',
    'D-26b': 'Myjnia',
    'D-26c': 'Toaleta publiczna',
    'D-27': 'Bufet lub kawiarnia',
    'D-28': 'Restauracja',
    'D-29': 'Hotel (motel)',
    'D-4': 'Punkt widokowy',
    'D-5': 'Przystanek kolejowy',
    'D-6': 'Przystanek tramwajowy',
    'D-7': 'Stacja rowerów publicznych'
}