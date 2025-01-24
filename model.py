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
dataset_path = './data/data'  # Ścieżka do folderu, w którym znajdują się dane (zmień na odpowiednią)

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

# Budowanie modelu CNN (Convolutional Neural Network) do klasyfikacji obrazów
traffic_model = Sequential([  # Tworzymy model sekwencyjny
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),  # Pierwsza warstwa konwolucyjna
    MaxPooling2D((2, 2)),  # Warstwa max-pooling do redukcji wymiarów
    Conv2D(64, (3, 3), activation='relu'),  # Druga warstwa konwolucyjna
    MaxPooling2D((2, 2)),  # Warstwa max-pooling
    Flatten(),  # Spłaszczenie wyników do jednowymiarowej tablicy
    Dense(128, activation='relu'),  # Warstwa gęsta z 128 neuronami
    Dropout(0.5),  # Warstwa dropout w celu uniknięcia overfittingu
    Dense(NUM_CLASSES, activation='softmax')  # Warstwa wyjściowa z liczbą neuronów odpowiadającą liczbie klas
])

# Kompilacja modelu
traffic_model.compile(
    optimizer=Adam(learning_rate=0.001),  # Optymalizator Adam z określoną szybkością uczenia
    loss='sparse_categorical_crossentropy',  # Funkcja straty dla klasyfikacji wieloklasowej
    metrics=['accuracy']  # Metrika - dokładność
)

# Określanie liczby kroków na epokę i walidacji
steps_per_epoch = np.ceil(train_dataset.samples / train_dataset.batch_size).astype(int)  # Obliczamy liczbę kroków na epokę
val_steps = np.ceil(test_dataset.samples / test_dataset.batch_size).astype(int)  # Obliczamy liczbę kroków walidacyjnych

# Trening modelu
training_history = traffic_model.fit(
    train_dataset,  # Dane treningowe
    steps_per_epoch=steps_per_epoch,  # Liczba kroków na epokę
    epochs=10,  # Liczba epok
    validation_data=test_dataset,  # Dane walidacyjne
    validation_steps=val_steps,  # Liczba kroków walidacyjnych
    verbose=1  # Wyświetlanie postępu treningu
)

# Zapisanie wytrenowanego modelu
traffic_model.save('ReadyModel.h5')  # Zapisanie modelu do pliku



def plot_metrics(history):
    training_acc = history.history['accuracy']
    validation_acc = history.history['val_accuracy']
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    epochs_range = range(1, len(training_acc) + 1)

    plt.figure(figsize=(14, 6))  # Większy wykres dla lepszej widoczności

    # Wykres dokładności
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, training_acc, marker='o', linestyle='-', color='green', label='Train Accuracy')
    plt.plot(epochs_range, validation_acc, marker='o', linestyle='--', color='orange', label='Validation Accuracy')
    plt.title('Model Accuracy', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)  # Dodanie siatki

    # Wykres straty
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, training_loss, marker='s', linestyle='-', color='blue', label='Train Loss')
    plt.plot(epochs_range, validation_loss, marker='s', linestyle='--', color='red', label='Validation Loss')
    plt.title('Model Loss', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()  # Lepsze rozmieszczenie wykresów
    plt.show()

# Rysowanie wykresów z historii treningu
plot_metrics(training_history)
