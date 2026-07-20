"""
Main pipeline for the Zoodle project.

Provides a command-line menu for running each pipeline step
separately or running the complete pipeline in order.
"""

from download_dataset import download_dataset
from eda import run_eda
from evaluation import evaluate_model as evaluate_cnn_model
from pca_umap import run_pca_umap
from prepare_data import prepare_dataset
from training import train_model as train_cnn_model


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
# PCA och UMAP används för att undersöka mönster i träningsbilderna.
def analyze_dimensions():
    run_pca_umap()


# ============================================================
# 5. TRÄNA MODELLEN
# ============================================================
# CNN-modellen tränas och varje körning sparas i en egen mapp.
def train_model():
    train_cnn_model()


# ============================================================
# 6. UTVÄRDERA MODELLEN
# ============================================================
# Användaren väljer en checkpoint som utvärderas mot testdatan.
def evaluate_model():
    evaluate_cnn_model()


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
