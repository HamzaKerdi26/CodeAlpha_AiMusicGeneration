import numpy as np
import pickle
import random
from music21 import stream, note, chord
from tensorflow.keras.models import load_model

print("Loading model...")

model = load_model("model.h5")

print("Loading notes...")

with open("notes.pkl", "rb") as f:
    notes = pickle.load(f)

if len(notes) < 50:
    print("ERROR: Not enough notes in dataset.")
    exit()

pitchnames = sorted(set(notes))

note_to_int = {
    n: i for i, n in enumerate(pitchnames)
}

int_to_note = {
    i: n for n, i in note_to_int.items()
}

sequence_length = 50

start = random.randint(
    0,
    len(notes) - sequence_length
)

pattern = [
    note_to_int[n]
    for n in notes[start:start + sequence_length]
]

output_notes = []

# Improved temperature sampling
def sample(preds, temperature=0.8):

    preds = np.asarray(preds).astype("float64")

    preds = np.log(preds + 1e-8) / temperature

    exp_preds = np.exp(preds)

    preds = exp_preds / np.sum(exp_preds)

    preds = np.nan_to_num(preds)

    preds = preds / np.sum(preds)

    return np.random.choice(
        len(preds),
        p=preds
    )

print("Generating music...")

for _ in range(400):

    input_seq = np.reshape(
        pattern,
        (1, sequence_length, 1)
    )

    input_seq = input_seq / float(len(pitchnames))

    prediction = model.predict(
        input_seq,
        verbose=0
    )

    index = sample(
        prediction[0],
        temperature=0.8
    )

    result = int_to_note[index]

    output_notes.append(result)

    pattern.append(index)
    pattern = pattern[1:]

print("Generated notes:", len(output_notes))

# Create MIDI
offset = 0
output_stream = stream.Stream()

print("Creating MIDI file...")

for item in output_notes:

    try:

        # Chord
        if "." in str(item):

            notes_in_chord = str(item).split(".")

            chord_notes = []

            for current_note in notes_in_chord:

                if current_note.isdigit():
                    chord_notes.append(
                        note.Note(
                            int(current_note)
                        )
                    )

            if len(chord_notes) > 0:

                new_chord = chord.Chord(
                    chord_notes
                )

                new_chord.offset = offset

                output_stream.append(
                    new_chord
                )

        # Numeric note
        elif str(item).isdigit():

            new_note = note.Note(
                int(item)
            )

            new_note.offset = offset

            output_stream.append(
                new_note
            )

        # Normal note
        else:

            new_note = note.Note(
                str(item)
            )

            new_note.offset = offset

            output_stream.append(
                new_note
            )

        offset += 0.25

    except Exception as e:

        print(
            "Skipped invalid note:",
            item
        )

output_stream.write(
    "midi",
    fp="output.mid"
)

print("\nDONE!")
print("Saved: output.mid")