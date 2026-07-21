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

Nedladdningen, dataförberedelsen, EDA, PCA, UMAP, modellträningen och modellutvärderingen är implementerade.

## Projektstruktur

```text
zoodle/
├── artifacts/
│   └── training/             # Bästa modellen från varje träningskörning
├── data/
│   ├── raw/                  # Nedladdade Quick, Draw!-filer
│   └── processed/            # Träning, validering och test
├── output/
│   ├── eda/                  # Statistik och figurer från EDA
│   ├── evaluation/           # Resultat från modellutvärderingar
│   ├── pca/                  # Resultat från PCA
│   ├── training/             # Historik från varje träningskörning
│   └── umap/                 # Resultat från UMAP
├── settings/
│   ├── __init__.py
│   └── settings.py          # Gemensamma sökvägar och inställningar
├── download_dataset.py          # Laddar ner datasetet
├── eda.py                       # Utforskar träningsdatan
├── evaluation.py                # Utvärderar en vald modell på testdatan
├── model.py                     # Bygger och kompilerar CNN-modellen
├── pca_umap.py                  # Analyserar bilder med PCA och UMAP
├── prepare_data.py              # Förbereder och delar upp datasetet
├── pipeline.py                  # Projektets terminalmeny
├── training.py                  # Tränar och sparar CNN-modellen
├── requirements.txt            # Projektets Python-bibliotek
├── .gitignore
└── README.md
```

Mappen `data/` skapas automatiskt när datasetet laddas ner. Nedladdade datafiler sparas inte i Git.

## Installera bibliotek

Installera projektets bibliotek med:

```text
python -m pip install -r requirements.txt
```

Filen `requirements.txt` uppdateras när projektet börjar använda nya bibliotek.

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

När råfilerna har laddats ner kan datasetet förberedas direkt med:

```bash
python prepare_data.py
```

Det förberedda datasetet sparas som `train.npz`, `validation.npz` och `test.npz` i `data/processed`.

EDA kan köras direkt när dataförberedelsen är klar:

```bash
python eda.py
```

EDA använder endast träningsdatan. Statistik, tabeller och figurer sparas i `output/eda`. Tidigare EDA-filer skrivs över när steget körs igen.

PCA och UMAP kan köras direkt efter dataförberedelsen:

```bash
python pca_umap.py
```

PCA-resultatet sparas i `output/pca` och UMAP-resultatet sparas i `output/umap`. Båda analyserna använder balanserade urval från träningsdatan och skriver över tidigare filer.

CNN-modellen kan tränas direkt med:

```bash
python training.py
```

Varje träningskörning får ett eget ID baserat på starttiden. Den bästa modellen sparas i en egen mapp under `artifacts/training` och träningshistoriken sparas i motsvarande mapp under `output/training`. När träningen är klar skrivs en classification report för valideringsdatan ut i terminalen och sparas som text och CSV.

Dataaugmentering, dropout, early stopping och automatisk sänkning av learning rate kan justeras i `settings/settings.py` inför en ny jämförelsekörning. Learning rate för varje epok sparas i träningshistoriken och visas i resultatfiguren.

En tränad modell kan utvärderas direkt med:

```bash
python evaluation.py
```

Programmet visar tillgängliga checkpoints tillsammans med kort träningsinformation. Den valda modellen utvärderas mot testdatan och varje utvärdering sparas i en unik mapp under `output/evaluation/<checkpoint_id>`.

Resultatet innehåller sammanfattande testmått, förväxlingsmatris, klassernas F1-resultat, ROC-AUC, vanliga förväxlingar och tydliga exempel på felklassificeringar. ROC visas som mikro- och makrokurvor samt ett sorterat AUC-diagram för alla djurklasser.

## Git-konventioner

### Branch-namn

- Skriv branch-namn på engelska med små bokstäver.
- Använd bindestreck mellan orden.
- Börja namnet med typen av ändring.
- Använd ett kort och beskrivande namn för uppgiften.

Vanliga prefix:

- `feature-` för ny funktionalitet
- `fix-` för felrättningar
- `refactor-` för omstrukturering av kod
- `docs-` för dokumentation
- `test-` för tester
- `chore-` för underhåll

Exempel:

```text
feature-data-download
feature-model-training
fix-download-error
docs-update-readme
```

### Commit-meddelanden

Skriv korta och tydliga commit-meddelanden på engelska. Använd ett prefix som visar vilken typ av ändring commiten innehåller.

Exempel:

```text
feat: add dataset download
fix: handle missing data file
refactor: simplify pipeline menu
docs: update project instructions
test: add settings tests
chore: update gitignore
```

## Status

Projektet är under utveckling och byggs stegvis. Pipeline-steg 1–6 är implementerade.
