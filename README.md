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

# Bachelor’s Thesis – Neural Networks for Speech Emotion Recognition

This engineering thesis presents a speech emotion recognition system (happiness, sadness, surprise, anger, and neutrality) developed using Convolutional Neural Networks (CNNs).

The model was trained and evaluated on the ESD dataset as well as a custom-created dataset. The final solution was deployed as a web application featuring audio recording, playback, and real-time emotion prediction.

## Introduction

This repository contains all files created during the development of the speech emotion recognition system. It consists of three main directories:

### Web_Application
Contains all files related to the developed web application, which enables users to detect emotions in audio samples.

### Custom_Dataset
Includes a `.zip` file with the self-created dataset consisting of 50 labeled audio recordings along with transcription.

### Notebook
Contains the Kaggle notebook used to train and develop the emotion recognition model.

## Running the Application

To run the EmoLens application locally:

1. Clone the repository into your selected directory using:

git clone "https://gitlab-stud.elka.pw.edu.pl/srojek/praca_inzynierska.git"

2. Open the project in your preferred development environment (e.g., Visual Studio Code).
3. Run the `app.py` file.
4. Open your browser and navigate to:

http://127.0.0.1:5000/

## Model

Inside the `Web_Application` directory, the `model_55epochs` folder contains:

- The trained CNN model with weights  
- The emotion encoder and decoder used for label transformation

