# Praca Inżynierska Sieci Neuronowe
Praca inżynierska dotyczy systemu wykrywania emocji w głosie (radość, smutek, zaskoczenie, złość, neutralność) z użyciem konwolucyjnych sieci neuronowych. Przeanalizowano wyniki na bazie ESD i danych autorskich oraz wdrożono model w aplikacji webowej z funkcją nagrywania, odsłuchu i oceny predykcji.


## Wstęp

Repozytorium zawiera wszystkie pliki stworzone podczas pracy nad projektem realizującym wykrywanie emocji w głosie. Składa się z 3 folderów:

1. **Aplikacja_webowa** – zawiera wszystkie pliki związane z stworzoną aplikacją, pozwalającą użytkownikowi wykrywać emocje w próbce dźwiękowej.
2. **Autorska_baza_danych** – zawiera plik .zip z stworzoną bazą danych składającą się z 50 plików wraz z subskrypcją.
3. **Notebook** – folder zawierający notebook pobrany ze środowiska Kaggle, w którym został stworzony model rozpoznawania emocji.

## Jak uruchomić aplikację

Aby uruchomić aplikację EmoLens na swoim komputerze, wystarczy sklonować repozytorium do wcześniej przygotowanego folderu, używając komendy:

git clone "https://gitlab-stud.elka.pw.edu.pl/srojek/praca_inzynierska.git"


Następnie przy użyciu środowiska programistycznego, np. Visual Studio Code, uruchomić plik `app.py` i wpisać w przeglądarkę adres:

http://127.0.0.1:5000/


W folderze **Aplikacja_webowa** znajduje się folder **model_55epok**, w którym znajduje się wytrenowany model wraz z wagami oraz kodowaniem i dekodowaniem emocji (encoder, decoder).

