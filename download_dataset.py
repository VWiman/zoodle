"""Download the Quick, Draw! datasets used by the Zoodle project."""

from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlretrieve

from settings import ANIMAL_CLASSES, DATASET_URL, RAW_DATA_DIR


# ============================================================
# 1. LADDA NED DATASETET
# ============================================================
# Varje djurklass laddas ned som en egen NumPy-fil.
# Filer som redan finns hoppas över så att nedladdningen kan fortsätta senare.
def download_dataset() -> bool:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = []

    print("\nNedladdningen startar.")

    for number, animal in enumerate(ANIMAL_CLASSES, start=1):
        file_path = RAW_DATA_DIR / f"{animal}.npy"
        temporary_path = file_path.with_suffix(".npy.part")

        if file_path.exists() and file_path.stat().st_size > 0:
            temporary_path.unlink(missing_ok=True)
            print(f"[{number}/{len(ANIMAL_CLASSES)}] {animal} finns redan och hoppas över.")
            skipped += 1
            continue

        # Mellanslag i klassnamn måste anpassas för att fungera i webbadressen.
        file_url = f"{DATASET_URL}/{quote(animal)}.npy"

        try:
            print(f"[{number}/{len(ANIMAL_CLASSES)}] Laddar ned {animal}...")
            temporary_path.unlink(missing_ok=True)
            urlretrieve(file_url, temporary_path)

            if temporary_path.stat().st_size == 0:
                raise OSError("Den nedladdade filen är tom.")

            temporary_path.replace(file_path)
            downloaded += 1
        except (HTTPError, URLError, OSError) as error:
            temporary_path.unlink(missing_ok=True)
            print(f"Kunde inte ladda ned {animal}: {error}")
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
        print("\nFiler som inte kunde laddas ned:")
        for animal in failed:
            print(f"- {animal}")

    return not failed


# ============================================================
# 2. STARTA NEDLADDNINGEN
# ============================================================
# Funktionen kan köras direkt eller genom projektets pipeline-meny.
if __name__ == "__main__":
    download_dataset()
