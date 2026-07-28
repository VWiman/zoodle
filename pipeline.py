"""
Main pipeline for the Zoodle project.

Provides a command-line menu for running each pipeline step
separately or running the complete pipeline in order.
"""

from download_dataset import download_dataset
from eda import run_eda
from evaluation import evaluate_model as evaluate_saved_model
from knn_training import train_knn_model as run_knn_training
from pca_umap import run_pca_umap
from prepare_data import prepare_dataset
from training import train_model as run_cnn_training


# ============================================================
# 1. LADDA NER DATA
# ============================================================
# Datasetet laddas ner och sparas i projektets datamapp.
def download_data() -> bool:
    return download_dataset()


# ============================================================
# 2. FÖRBERED OCH RENGÖR DATA
# ============================================================
# Bilderna kontrolleras, rengörs och delas upp i dataset.
def prepare_data() -> bool:
    return prepare_dataset()


# ============================================================
# 3. UTFORSKA DATA
# ============================================================
# Träningsdatan undersöks med statistik och visualiseringar.
def explore_data() -> bool:
    return run_eda()


# ============================================================
# 4. ANALYSERA MED PCA OCH UMAP
# ============================================================
# PCA och UMAP används för att undersöka mönster i träningsbilderna.
def analyze_dimensions() -> bool:
    return run_pca_umap()


# ============================================================
# 5. TRÄNA PCA + KNN-MODELLEN
# ============================================================
# PCA och KNN tränas på balanserade urval från träning och validering.
def train_knn_model() -> bool:
    return run_knn_training()


# ============================================================
# 6. TRÄNA CNN-MODELLEN
# ============================================================
# CNN-modellen tränas och varje körning sparas i en egen mapp.
def train_cnn_model() -> bool:
    return run_cnn_training()


# ============================================================
# 7. UTVÄRDERA EN MODELL
# ============================================================
# Användaren väljer en sparad CNN- eller KNN-modell som utvärderas mot testdatan.
def evaluate_model() -> bool:
    return evaluate_saved_model()


# ============================================================
# 8. KÖR HELA TRÄNINGSFLÖDET
# ============================================================
# Stegen körs i ordning och flödet stoppas direkt om ett steg misslyckas.
def run_pipeline() -> bool:
    pipeline_steps = [
        ("Ladda ner data", download_data),
        ("Förbered och rengör data", prepare_data),
        ("Utforska data", explore_data),
        ("Analysera med PCA och UMAP", analyze_dimensions),
        ("Träna PCA + KNN-modellen", train_knn_model),
        ("Träna CNN-modellen", train_cnn_model),
    ]

    for step_name, pipeline_step in pipeline_steps:
        if not pipeline_step():
            print(f"\nFlödet stoppades vid steget: {step_name}.")
            return False

    print("\nHela träningsflödet är klart.")
    return True


# ============================================================
# 9. VISA MENYN
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
        print("5. Träna PCA + KNN-modellen")
        print("6. Träna CNN-modellen")
        print("7. Utvärdera en modell")
        print("8. Kör hela träningsflödet")
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
            train_knn_model()
        elif choice == "6":
            train_cnn_model()
        elif choice == "7":
            evaluate_model()
        elif choice == "8":
            run_pipeline()
        elif choice == "0":
            print("\nProgrammet avslutas.")
            break
        else:
            print("\nOgiltigt val. Försök igen.")


# ============================================================
# 10. STARTA PROGRAMMET
# ============================================================
# Menyn visas bara när filen körs direkt.
if __name__ == "__main__":
    show_menu()
