import os
import pickle
from music21 import converter, instrument, note, chord

notes = []

dataset_path = "dataset"

print("Starting preprocessing...")

for file in os.listdir(dataset_path):
    if file.endswith(".mid"):

        file_path = os.path.join(dataset_path, file)
        print(f"\nReading: {file}")

        try:
            midi = converter.parse(file_path)

            parts = instrument.partitionByInstrument(midi)
            notes_to_parse = parts.parts[0].recurse() if parts else midi.recurse()

            count = 0

            for element in notes_to_parse:

                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                    count += 1

                elif isinstance(element, chord.Chord):
                    notes.append(".".join(str(n) for n in element.normalOrder))
                    count += 1

            print("Notes found:", count)

        except Exception as e:
            print("Error:", file, e)

with open("notes.pkl", "wb") as f:
    pickle.dump(notes, f)

print("\nTOTAL NOTES EXTRACTED:", len(notes))