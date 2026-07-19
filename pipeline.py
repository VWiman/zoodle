"""
Main pipeline for the Zoodle project.

Provides a command-line menu for running each pipeline step
separately or running the complete pipeline in order.
"""

from download_dataset import download_dataset
from eda import run_eda
from prepare_data import prepare_dataset


# ============================================================
# 1. LADDA NER DATA
# ============================================================
# Datasetet laddas ner och sparas i projektets datamapp.
def download_data():
    download_dataset()


# ============================================================
# 2. FÖRBERED OCH RENGÖR DATA
# ============================================================
# Bilderna kontrolleras, rengörs och delas upp i dataset.
def prepare_data():
    prepare_dataset()


# ============================================================
# 3. UTFORSKA DATA
# ============================================================
# Träningsdatan undersöks med statistik och visualiseringar.
def explore_data():
    run_eda()


# ============================================================
# 4. ANALYSERA MED PCA OCH UMAP
# ============================================================
# Här ska PCA och UMAP användas för att undersöka mönster i bilderna.
def analyze_dimensions():
    print("\nPCA och UMAP är inte implementerade än.")


# ============================================================
# 5. TRÄNA MODELLEN
# ============================================================
# Här ska CNN-modellen byggas och tränas.
def train_model():
    print("\nModellträning är inte implementerad än.")


# ============================================================
# 6. UTVÄRDERA MODELLEN
# ============================================================
# Här ska den tränade modellen utvärderas och resultatet sparas.
def evaluate_model():
    print("\nModellutvärdering är inte implementerad än.")


# ============================================================
# 7. KÖR HELA FLÖDET
# ============================================================
# Stegen körs i samma ordning som de visas i menyn.
def run_pipeline():
    download_data()
    prepare_data()
    explore_data()
    analyze_dimensions()
    train_model()
    evaluate_model()


# ============================================================
# 8. VISA MENYN
# ============================================================
# Menyn gör det möjligt att köra ett steg i taget eller hela flödet.
def show_menu():
    while True:
        print("\n========================================")
        print("ZOODLE - TRÄNINGSPIPELINE")
        print("========================================")
        print("1. Ladda ner data")
        print("2. Förbered och rengör data")
        print("3. Utforska data")
        print("4. Analysera med PCA och UMAP")
        print("5. Träna modellen")
        print("6. Utvärdera modellen")
        print("7. Kör hela flödet")
        print("0. Avsluta")

        choice = input("\nVälj ett alternativ: ").strip()

        if choice == "1":
            download_data()
        elif choice == "2":
            prepare_data()
        elif choice == "3":
            explore_data()
        elif choice == "4":
            analyze_dimensions()
        elif choice == "5":
            train_model()
        elif choice == "6":
            evaluate_model()
        elif choice == "7":
            run_pipeline()
        elif choice == "0":
            print("\nProgrammet avslutas.")
            break
        else:
            print("\nOgiltigt val. Försök igen.")


# ============================================================
# 9. STARTA PROGRAMMET
# ============================================================
# Menyn visas bara när filen körs direkt.
if __name__ == "__main__":
    show_menu()
