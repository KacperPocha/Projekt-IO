# Projekt-IO
1. Opis projektu
  Projekt to aplikacja do rozpoznawania znaków drogowych za pomocą sieci neuronowej zbudowanej w TensorFlow/Keras. Celem projektu jest wykorzystanie obrazu (np. zdjęć z kamery samochodowej) do klasyfikacji znaków drogowych zgodnie z ich kategorią. Używane dane to zestaw      obrazów z odpowiednimi etykietami, a model umożliwia ocenę dokładności na zbiorach walidacyjnych i testowych.

2. Wymagania niefunkcjonalne
  - Wydajność: Aplikacja powinna być w stanie przetworzyć obraz w czasie rzeczywistym (maks. 1 sekunda na analizę).
  - Zgodność: Oprogramowanie musi działać na systemach z Pythonem 3.8+ i bibliotekami TensorFlow 2.x.
  - Skalowalność: Możliwość przeszkolenia modelu na większym zestawie danych.
  - Czytelność kodu: Kod powinien być dobrze udokumentowany i modularny.

3. Wymagania funkcjonalne
  - Wejście: Obsługa plików graficznych (JPEG, PNG) lub bezpośredniego wideo.
  - Wyjście: Zwrot etykiety z klasyfikacją znaku drogowego (np. „Stop”, „Zakaz wjazdu”).
  - Uczenie: Model ma możliwość dalszego trenowania na nowych danych.
  - Wizualizacja wyników: Wyświetlanie obrazu wejściowego wraz z przewidywaną etykietą.

4. Potencjalne ryzyka
  - Nasza ograniczona wiedza w szkoleniu sztucznej inteligencji
  - Problemy z jakością danych treningowych
  - Brak doświadczenia w wykorzystaniu TensorFlow i Keras
  - Problemy z konfiguracją środowiska
  - Ograniczenia sprzętowe

5. Prosty diagram aplikacji:
  Dane wejściowe (pickle i etykiety)  >  Wczytywanie i przygotowanie danych  >  Wizualizacja danych  >  Budowanie modelu  >    Trenowanie modelu  >  Ewaluacja modelu  >  Predykcja pojedynczego obrazu  >  Mapowanie predykcji na etykiety


  Aby uruchomić program sklonuj repozytorium z githuba, a następnie 
  - Utwórz wirtualne środowisko python -m venv myVenv
  - Aktywuj środwisko .\myVenv\Scripts\Activate
  - Zainstaluj biblioteki pip install -r requirements.txt

