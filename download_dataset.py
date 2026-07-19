"""Download the Quick, Draw! datasets used by the Zoodle project."""

from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlretrieve

from settings import ANIMAL_CLASSES, DATASET_URL, RAW_DATA_DIR


# ============================================================
# 1. LADDA NER DATASETET
# ============================================================
# Varje djurklass laddas ner som en egen NumPy-fil.
# Filer som redan finns hoppas över så att nedladdningen kan fortsätta senare.
def download_dataset():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = []

    print("\nNedladdningen startar.")

    for number, animal in enumerate(ANIMAL_CLASSES, start=1):
        file_path = RAW_DATA_DIR / f"{animal}.npy"

        if file_path.exists() and file_path.stat().st_size > 0:
            print(f"[{number}/{len(ANIMAL_CLASSES)}] {animal} finns redan och hoppas över.")
            skipped += 1
            continue

        # Mellanslag i klassnamn måste anpassas för att fungera i webbadressen.
        file_url = f"{DATASET_URL}/{quote(animal)}.npy"

        try:
            print(f"[{number}/{len(ANIMAL_CLASSES)}] Laddar ner {animal}...")
            urlretrieve(file_url, file_path)
            downloaded += 1
        except (HTTPError, URLError, OSError) as error:
            print(f"Kunde inte ladda ner {animal}: {error}")
            failed.append(animal)

    # --------------------------------------------------------
    # 1.1 Visa en sammanfattning
    # --------------------------------------------------------
    print("\n========================================")
    print("NEDLADDNINGEN ÄR KLAR")
    print("========================================")
    print(f"Nedladdade filer: {downloaded}")
    print(f"Redan befintliga filer: {skipped}")
    print(f"Misslyckade filer: {len(failed)}")

    if failed:
        print("\nFiler som inte kunde laddas ner:")
        for animal in failed:
            print(f"- {animal}")


# ============================================================
# 2. STARTA NEDLADDNINGEN
# ============================================================
# Funktionen kan köras direkt eller genom projektets pipeline-meny.
if __name__ == "__main__":
    download_dataset()
