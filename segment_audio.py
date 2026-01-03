import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

# --- ΡΥΘΜΙΣΕΙΣ ---
INPUT_DIR = "data/raw_smp"        # Πού είναι τα ολόκληρα τραγούδια
OUTPUT_DIR = "data/processed_smp" # Πού θα μπουν τα κομμένα (Segments)
BEATS_PER_SEGMENT = 16            # 4 μέτρα * 4 beats = 16 beats (για 4/4 ρυθμό)
# -----------------

def segment_track(file_path, output_folder):
    """Διαβάζει ένα wav, βρίσκει beats και το κόβει σε segments."""
    try:
        # 1. Φόρτωση ήχου
        y, sr = librosa.load(file_path, sr=None) # sr=None για να κρατήσει το αρχικό
        
        # 2. Εντοπισμός Beats
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_samples = librosa.frames_to_samples(beat_frames)
        
        # Αν δεν βρήκε αρκετά beats, το αγνοούμε ή το κόβουμε σταθερά
        if len(beat_samples) < BEATS_PER_SEGMENT:
            # Fallback: Κόψιμο ανά 10 δευτερόλεπτα αν αποτύχει το beat tracking
            print(f"⚠️ Warning: Low beats detected in {file_path}. Skipping beat sync.")
            return

        # 3. Τεμαχισμός
        # Παίρνουμε τα beats ανά 16 (BEATS_PER_SEGMENT)
        num_segments = 0
        for i in range(0, len(beat_samples) - BEATS_PER_SEGMENT, BEATS_PER_SEGMENT):
            start_sample = beat_samples[i]
            end_sample = beat_samples[i + BEATS_PER_SEGMENT]
            
            # Κόβουμε τον ήχο
            segment = y[start_sample:end_sample]
            
            # Αγνοούμε πολύ μικρά κομμάτια (< 2 sec) που μπορεί να είναι λάθος
            if len(segment) / sr < 2.0:
                continue

            # 4. Αποθήκευση
            seg_filename = f"{num_segments}.wav"
            sf.write(os.path.join(output_folder, seg_filename), segment, sr)
            num_segments += 1
            
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Δεν βρέθηκε ο φάκελος {INPUT_DIR}")
        return

    # Διαβάζουμε τους φακέλους των ζευγαριών (pair_0, pair_1...)
    pairs = sorted([p for p in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, p))])
    
    print(f"🚀 Ξεκινάει το Segmentation για {len(pairs)} ζευγάρια...")

    for pair in tqdm(pairs):
        pair_input_path = os.path.join(INPUT_DIR, pair)
        pair_output_path = os.path.join(OUTPUT_DIR, pair)
        
        # Για κάθε αρχείο μέσα στο ζευγάρι (original.wav, suspicious.wav)
        for wav_file in os.listdir(pair_input_path):
            if not wav_file.endswith(".wav"):
                continue
                
            # Δημιουργία υποφακέλου: data/processed_smp/pair_0/original/
            # Το όνομα του φακέλου είναι το όνομα του αρχείου χωρίς το .wav
            version_name = os.path.splitext(wav_file)[0] 
            version_output_folder = os.path.join(pair_output_path, version_name)
            
            os.makedirs(version_output_folder, exist_ok=True)
            
            # Εκτέλεση του τεμαχισμού
            input_wav_path = os.path.join(pair_input_path, wav_file)
            segment_track(input_wav_path, version_output_folder)

    print(f"\n✅ Ολοκληρώθηκε! Τα segments είναι στο: {OUTPUT_DIR}")
    print("Τώρα μπορείς να ανεβάσεις ΑΥΤΟΝ τον φάκελο στο Drive για το MERT.")

if __name__ == "__main__":
    main()