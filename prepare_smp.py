import os
import pandas as pd
import yt_dlp
from tqdm import tqdm
from typing import Dict, Any

# --- ΡΥΘΜΙΣΕΙΣ ---
CSV_FILE = "Final_dataset_pairs.csv"
OUTPUT_DIR = "data/raw_smp"

# ✅ ΤΑ ΣΩΣΤΑ ΟΝΟΜΑΤΑ ΣΤΗΛΩΝ (από το αρχείο που ανέβασες)
COL_ORIGINAL = "ori_link"
COL_SUSPICIOUS = "comp_link"
# -----------------

def download_audio(url, output_path):
    """Κατεβάζει το audio από YouTube και το σώζει ως .wav"""
    
    # Δηλώνουμε ρητά τον τύπο για να μην παραπονιέται ο linter
    ydl_opts: Dict[str, Any] = {
        'format': 'bestaudio/best',
        # Το yt-dlp βάζει μόνο του την επέκταση, οπότε την αφαιρούμε από το path
        'outtmpl': output_path.replace('.wav', ''), 
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        # Το type: ignore λέει στο VS Code να αγνοήσει το συγκεκριμένο λάθος τύπου
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: # type: ignore
            ydl.download([url])
        return True
    except Exception as e:
        print(f"\n❌ Error downloading {url}: {e}")
        return False

def main():
    if not os.path.exists(CSV_FILE):
        print(f"Δεν βρέθηκε το αρχείο {CSV_FILE}. Βεβαιώσου ότι είναι στον ίδιο φάκελο.")
        return

    # Διάβασμα του CSV
    try:
        df = pd.read_csv(CSV_FILE)
        print(f"Βρέθηκαν {len(df)} ζευγάρια στο CSV.")
    except Exception as e:
        print(f"Σφάλμα κατά την ανάγνωση του CSV: {e}")
        return

    # Δημιουργία κεντρικού φακέλου
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    success_count = 0

    # Iteration σε κάθε γραμμή του CSV
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading Dataset"):
        # Δημιουργία φακέλου για το ζευγάρι (π.χ. data/raw_smp/pair_0)
        pair_folder = os.path.join(OUTPUT_DIR, f"pair_{idx}")
        os.makedirs(pair_folder, exist_ok=True)

        # 1. Κατέβασμα Original
        orig_url = row[COL_ORIGINAL]
        orig_path = os.path.join(pair_folder, "original.wav")
        
        if not os.path.exists(orig_path):
            ok1 = download_audio(orig_url, orig_path)
        else:
            ok1 = True

        # 2. Κατέβασμα Suspicious (Plagiarized)
        susp_url = row[COL_SUSPICIOUS]
        susp_path = os.path.join(pair_folder, "suspicious.wav")
        
        if not os.path.exists(susp_path):
            ok2 = download_audio(susp_url, susp_path)
        else:
            ok2 = True

        if ok1 and ok2:
            success_count += 1
    
    print(f"\n✅ Ολοκληρώθηκε! Κατέβηκαν επιτυχώς {success_count}/{len(df)} ζευγάρια.")
    print(f"Τα αρχεία βρίσκονται στο: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()