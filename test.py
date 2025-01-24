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
# Ładowanie wytrenowanego modelu
model = load_model('ReadyModel.h5')  # Ładowanie zapisanej wstępnie wytrenowanej sieci neuronowej

def fetch_and_preprocess_image(url, img_height, img_width):
    """
    Pobieranie obrazu z URL i wstępne przetwarzanie.
    """
    try:
        response = requests.get(url, stream=True)  # Wysłanie żądania HTTP GET, aby pobrać obraz
        response.raise_for_status()  # Sprawdzenie, czy odpowiedź jest poprawna (status 200)
        image = Image.open(response.raw)  # Otworzenie obrazu z odpowiedzi HTTP
        image = image.resize((img_height, img_width))  # Zmiana rozmiaru obrazu na docelowy wymiar
        image_arr = np.array(image.convert('RGB'))  # Konwersja obrazu do formatu RGB (jeśli to konieczne)
        image_arr = image_arr / 255.0  # Normalizacja obrazu (skalowanie wartości pikseli do zakresu [0, 1])
        return image_arr.reshape(1, img_height, img_width, 3), image  # Zwrócenie obrazu jako tablicy i oryginalnego obrazu
    except requests.exceptions.RequestException as e:
        print(f"Błąd pobierania obrazu z URL {url}: {e}")  # Obsługa błędów pobierania obrazu
        return None, None  # Zwrócenie None w przypadku błędu
    except Exception as e:
        print(f"Błąd przetwarzania obrazu z URL {url}: {e}")  # Obsługa innych błędów
        return None, None  # Zwrócenie None w przypadku błędu

def predict_image(model, image_arr, class_indices, class_names):
    """
    Predykcja klasy dla obrazu.
    """
    try:
        result = model.predict(image_arr, verbose=0)  # Wykonanie predykcji na obrazie
        predicted_index = np.argmax(result)  # Znalezienie indeksu klasy o najwyższej prawdopodobieństwie
        predicted_code = list(class_indices.keys())[predicted_index]  # Odczytanie kodu znaku
        predicted_description = class_names.get(predicted_code, "Nieznany znak")  # Odczytanie opisu znaku
        return predicted_code, predicted_description  # Zwrócenie kodu i opisu przewidywanego znaku
    except Exception as e:
        print(f"Błąd podczas przewidywania: {e}")  # Obsługa błędów podczas predykcji
        return None, "Błąd przewidywania"  # Zwrócenie błędu w przypadku problemu z predykcją

def testing_v2(model, dict_of_urls, class_names, img_height=IMG_HEIGHT, img_width=IMG_WIDTH):
 
    """
    Testowanie modelu na obrazach z URL-ów.
    """
    for actual_code, url in dict_of_urls.items():  # Iterowanie przez każdy znak i URL
        print(f"Przetwarzam znak: {actual_code} z URL: {url}")  # Wyświetlanie aktualnego znaku i URL
        
        # Pobranie i przetwarzanie obrazu
        image_arr, image = fetch_and_preprocess_image(url, img_height, img_width)
        
        if image_arr is not None:  # Jeśli obraz został poprawnie przetworzony
            # Predykcja
            predicted_code, predicted_description = predict_image(
                model, image_arr, train_data_gen.class_indices, class_names
            )
        
            # Pobranie rzeczywistej klasy
            actual_description = class_names.get(actual_code, "Nieznany znak")
            
            # Wyświetlenie wyników
            plt.imshow(image)  # Wyświetlenie obrazu
            plt.title(f'Rzeczywisty znak: "{actual_description}". Model przewidział: "{predicted_description}".')  # Tytuł wykresu z wynikami
            plt.axis('off')  # Ukrycie osi na wykresie
            plt.show()  # Pokazanie wykresu z obrazem
        else:
            print(f"Nie udało się pobrać lub przetworzyć obrazu z URL: {url}")  # Obsługa błędów podczas pobierania obrazu

# Testowanie modelu na wybranych obrazach
testing_v2(model, dict_of_urls, class_names)  # Uruchomienie funkcji testowej z przekazaniem modelu, słownika URL-i i nazw klas