# Zoodle

Zoodle är ett projekt där en CNN-modell ska tränas för att klassificera handritade djur. Tanken är att modellen senare ska användas i ett enkelt ritspel byggt med Streamlit.

Projektet använder bilder från Googles [Quick, Draw!-dataset](https://github.com/googlecreativelab/quickdraw-dataset). Datasetet består av gråskalebilder med storleken 28 x 28 pixlar. Zoodle använder för närvarande 48 djurklasser.

## Projektets flöde

Projektets terminalmeny är uppdelad i följande steg:

1. Ladda ner data
2. Förbered och rengör data
3. Utforska data
4. Analysera med PCA och UMAP
5. Träna modellen
6. Utvärdera modellen
7. Kör hela flödet

Nedladdningen är implementerad. Övriga steg är för närvarande tomma delar som kommer att byggas vidare under projektets gång.

## Projektstruktur

```text
zoodle/
├── data/
│   └── raw/                  # Nedladdade Quick, Draw!-filer
├── settings/
│   ├── __init__.py
│   └── settings.py          # Gemensamma sökvägar och inställningar
├── download_dataset.py          # Laddar ner datasetet
├── pipeline.py                  # Projektets terminalmeny
├── .gitignore
└── README.md
```

Mappen `data/` skapas automatiskt när datasetet laddas ner. Nedladdade datafiler sparas inte i Git.

## Utvecklingsmiljö

Projektet använder:

- Conda-miljön `Tensorflow_AI`
- Python 3.12.13
- TensorFlow 2.18.1

Aktivera miljön:

```bash
conda activate Tensorflow_AI
```

## Kör projektet

Starta projektets terminalmeny:

```bash
python pipeline.py
```

Datasetet kan även laddas ner direkt:

```bash
python download_dataset.py
```

Filer som redan har laddats ner hoppas över, vilket gör att nedladdningen kan fortsätta om den avbryts.

## Status

Projektet är under utveckling och byggs stegvis. Nästa delar blir dataförberedelse, analys och modellträning.
