import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

print("Loading dataset...")

with open("notes.pkl", "rb") as f:
    notes = pickle.load(f)

print("Total notes:", len(notes))

if len(notes) == 0:
    print("ERROR: No notes found")
    exit()

# Mapping notes → integers
pitchnames = sorted(set(notes))
note_to_int = {n: i for i, n in enumerate(pitchnames)}

print("Unique notes:", len(pitchnames))

sequence_length = 50

network_input = []
network_output = []

for i in range(len(notes) - sequence_length):

    seq_in = notes[i:i + sequence_length]
    seq_out = notes[i + sequence_length]

    network_input.append([note_to_int[n] for n in seq_in])
    network_output.append(note_to_int[seq_out])

print("Training samples:", len(network_input))

if len(network_input) == 0:
    print("Not enough data")
    exit()

X = np.reshape(network_input, (len(network_input), sequence_length, 1))
X = X / float(len(pitchnames))

y = to_categorical(network_output, num_classes=len(pitchnames))

print("Building model...")

model = Sequential()

model.add(LSTM(512, return_sequences=True, input_shape=(sequence_length, 1)))
model.add(Dropout(0.3))

model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(256))

model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(len(pitchnames), activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam")

print("Training started...")

model.fit(X, y, epochs=30, batch_size=32)

model.save("model.h5")

print("MODEL SAVED ✔")