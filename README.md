# Zoodle

Zoodle är ett projekt där modeller tränas för att klassificera handritade djur. En CNN är projektets huvudmodell och PCA + KNN används som en klassisk baslinje. Tanken är att CNN-modellen senare ska användas i ett enkelt ritspel byggt med Streamlit.

Projektet använder bilder från Googles [Quick, Draw!-dataset](https://github.com/googlecreativelab/quickdraw-dataset). Datasetet består av gråskalebilder med storleken 28 x 28 pixlar. Zoodle använder för närvarande 48 djurklasser.

## Projektets flöde

Projektets terminalmeny är uppdelad i följande steg:

1. Ladda ner data
2. Förbered och rengör data
3. Utforska data
4. Analysera med PCA och UMAP
5. Träna PCA + KNN-modellen
6. Träna CNN-modellen
7. Utvärdera en modell
8. Kör hela träningsflödet

`0. Avsluta`

Det fullständiga träningsflödet kör steg 1–6 och stannar om ett steg misslyckas. Testutvärderingen startas separat med menyval 7.

## Projektstruktur

```text
zoodle/
├── artifacts/
│   ├── knn/                  # Sparade PCA- och KNN-modeller
│   └── training/             # Bästa CNN-modellen från varje körning
├── data/
│   ├── raw/                  # Nedladdade Quick, Draw!-filer
│   └── processed/            # Träning, validering och test
├── output/
│   ├── eda/                  # Statistik och figurer från EDA
│   ├── evaluation/
│   │   ├── cnn/              # CNN-resultat per checkpoint
│   │   └── knn/              # KNN-resultat per checkpoint
│   ├── knn/                  # Resultat från KNN-träning
│   ├── pca/                  # Resultat från PCA
│   ├── training/             # Historik från CNN-träning
│   └── umap/                 # Resultat från UMAP
├── settings/
│   ├── __init__.py
│   └── settings.py          # Gemensamma sökvägar och inställningar
├── download_dataset.py          # Laddar ner datasetet
├── eda.py                       # Utforskar träningsdatan
├── evaluation.py                # Utvärderar en vald CNN- eller KNN-modell
├── cnn_model.py                 # Bygger och kompilerar CNN-modellen
├── knn_model.py                 # Bygger PCA- och KNN-modellen
├── knn_training.py              # Tränar och sparar PCA + KNN
├── pca_umap.py                  # Analyserar bilder med PCA och UMAP
├── prepare_data.py              # Förbereder och delar upp datasetet
├── pipeline.py                  # Projektets terminalmeny
├── training.py                  # Tränar och sparar CNN-modellen
├── zoodle_project.ipynb         # Presenterar projektets analyser och resultat
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

Filer som redan har laddats ner hoppas över. Pågående nedladdningar sparas tillfälligt med ändelsen `.part` och ersätter den slutliga filen först när nedladdningen är klar.

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

PCA + KNN-modellen kan tränas direkt med:

```bash
python knn_training.py
```

KNN använder hela den balanserade tränings- och valideringsdatan. Modellen tränar en egen PCA på träningsdatan och är därför inte tekniskt beroende av att analyssteget för PCA och UMAP har körts. Under valideringsprediktionerna visas en progressbar. Varje körning sparar modellen under `artifacts/knn` och valideringsresultatet under `output/knn`.

CNN-modellen kan tränas direkt med:

```bash
python training.py
```

Varje träningskörning får ett eget ID baserat på starttiden. Den bästa modellen sparas i en egen mapp under `artifacts/training` och träningshistoriken sparas i motsvarande mapp under `output/training`. När träningen är klar skrivs en classification report för valideringsdatan ut i terminalen och sparas som text och CSV.

CNN-inställningar och KNN-inställningar ligger i separata avsnitt i `settings/settings.py`. Dataaugmentering, dropout, early stopping och automatisk sänkning av learning rate kan justeras inför en ny CNN-körning. CNN-träningen använder eager mode och CNN-prediktioner skapas i eager-batchar så att samma säkra beräkningssätt används vid valideringsrapport och utvärdering. Learning rate för varje epok sparas i träningshistoriken och visas i resultatfiguren.

En tränad modell kan utvärderas direkt med:

```bash
python evaluation.py
```

Programmet visar tillgängliga CNN- och KNN-körningar tillsammans med modelltyp och kort träningsinformation. Båda modellerna utvärderas mot hela testmängden. Under KNN-utvärderingen visas en progressbar. CNN-resultat sparas under `output/evaluation/cnn/<checkpoint_id>` och KNN-resultat under `output/evaluation/knn/<checkpoint_id>`.

Resultatet innehåller sammanfattande testmått, förväxlingsmatris, klassernas F1-resultat, ROC-AUC, vanliga förväxlingar och tydliga exempel på felklassificeringar. ROC visas som mikro- och makrokurvor samt ett sorterat AUC-diagram för alla djurklasser.

Varje checkpoint kan utvärderas en gång. Om en resultatmapp redan finns stoppas en ny utvärdering innan testdata eller modell läses in. För en avsiktlig omkörning behöver den befintliga resultatmappen först tas bort manuellt.

## Projektets notebook

`zoodle_project.ipynb` sammanfattar projektidén, dataförberedelsen, EDA, PCA, UMAP, modellträningen och den slutliga utvärderingen. Notebooken importerar gemensamma inställningar och funktioner från projektet, visar resultat som redan har skapats av pipeline-stegen och jämför den klassiska PCA + KNN-baselinen med den vinnande CNN-arkitekturen.

Starta notebooken från projektroten med det notebook-verktyg som finns installerat lokalt. De pipeline-steg som ska presenteras behöver ha körts minst en gång så att motsvarande filer finns i `output/`.

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

Projektet är under utveckling och byggs stegvis. Träningsflöden för PCA + KNN och CNN samt gemensam modellutvärdering är implementerade. Notebooken redovisar de aktuella modellresultaten sida vid sida, medan ett separat automatiserat jämförelsesteg i pipeline kan läggas till senare.
