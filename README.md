# 🎵 AI Music Generation using LSTM

## 📌 Project Overview

This project was developed as part of the **CodeAlpha AI Internship – Task 3: Music Generation with AI**.

The objective is to train a deep learning model on MIDI music files and generate new music by learning musical patterns from the dataset.

The project uses:

* Python
* TensorFlow / Keras
* LSTM (Long Short-Term Memory)
* Music21
* NumPy

---

## 🚀 Task Objectives

* Collect MIDI music data
* Preprocess MIDI files into note sequences
* Train an LSTM neural network
* Generate new music sequences
* Convert generated sequences back into MIDI format

---

## 📂 Project Structure

```text
Task3/
│
├── dataset/
│   ├── music1.mid
│   ├── music2.mid
│   └── ...
│
├── preprocess.py
├── train.py
├── generate.py
│
├── notes.pkl
├── model.h5
├── output.mid
│
└── README.md
```

---

## 🔹 preprocess.py

This script prepares the dataset for training.

### Responsibilities

* Reads MIDI files from the dataset folder
* Extracts notes and chords using Music21
* Converts music into note sequences
* Saves extracted notes into:

```text
notes.pkl
```

---

## 🔹 train.py

This script trains the AI model.

### Responsibilities

* Loads note sequences
* Converts notes into numerical values
* Creates training sequences
* Builds a multi-layer LSTM network
* Trains the model on musical patterns
* Saves the trained model as:

```text
model.h5
```

---

## 🔹 generate.py

This script generates new music.

### Responsibilities

* Loads the trained model
* Loads extracted notes
* Uses temperature-based sampling
* Predicts new note sequences
* Creates a MIDI file containing generated music

Output:

```text
output.mid
```

---

## 🧠 Model Architecture

The project uses a deep LSTM network:

* LSTM (512 units)
* Dropout (0.3)
* LSTM (512 units)
* Dropout (0.3)
* LSTM (256 units)
* Dense (256 units)
* Dense Softmax Output Layer

Loss Function:

```text
Categorical Crossentropy
```

Optimizer:

```text
Adam
```

---

## ▶️ Workflow

### Step 1

Run preprocessing:

```bash
python preprocess.py
```

### Step 2

Train the model:

```bash
python train.py
```

### Step 3

Generate music:

```bash
python generate.py
```

### Step 4

Play the generated file:

```text
output.mid
```

using any MIDI-compatible music player.

---

## 🎼 Output

The AI generates a completely new sequence of notes based on the musical patterns learned from the training dataset.

The generated music is saved as:

```text
output.mid
```

---

## 📚 Technologies Used

* Python
* TensorFlow
* Keras
* Music21
* NumPy
* Pickle

---

## 👨‍💻 Author

Hamza Kerdi

CodeAlpha AI Internship – Task 3
