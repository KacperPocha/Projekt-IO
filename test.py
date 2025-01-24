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
# Słownik mapujący kody znaków na ich pełne opisy
class_names = {
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
    'D-3': 'Droga jednokierunkowa',
    'D-40': 'Strefa zamieszkania',
    'D-41': 'Koniec strefy zamieszkania',
    'D-42': 'Obszar zabudowany',
    'D-43': 'Koniec obszaru zabudowanego',
    'D-4a': 'Droga bez przejazdu',
    'D-4b': 'Wjazd na drogę bez przejazdu',
    'D-51': 'Automatyczna kontrola prędkości',
    'D-52': 'Strefa ruchu',
    'D-53': 'Koniec strefy ruchu',
    'D-6': 'Przejście dla pieszych',
    'D-6b': 'Przejście dla pieszych i przejazd dla rowerzystów',
    'D-7': 'Droga ekspresowa',
    'D-8': 'Koniec drogi ekspresowej',
    'D-9': 'Autostrada',
    'D-tablica': 'Zbiorcza tablica informacyjna',
    'G-1a': 'Słupek wskaźnikowy z trzema kreskami umieszczany po prawej stronie jezdni',
    'G-3': 'Krzyż św. Andrzeja przed przejazdem kolejowym jednotorowym'
}

# Przygotowanie generatora danych (np. do uczenia modelu)
train_image_generator = ImageDataGenerator(rescale=1. / 255)  # Normalizacja obrazów do zakresu 0-1
train_data_gen = train_image_generator.flow_from_directory(
    batch_size=100,  # Rozmiar paczki danych
    directory='./data/train',  # Ustaw ścieżkę do folderu z danymi treningowymi
    class_mode='sparse',  # Tryb klasyfikacji, w tym przypadku "sparse" dla liczb całkowitych
    shuffle=True,  # Przemieszanie danych przed rozpoczęciem trenowania
    target_size=(IMG_HEIGHT, IMG_WIDTH)  # Zmiana rozmiaru obrazów do docelowych wymiarów
)