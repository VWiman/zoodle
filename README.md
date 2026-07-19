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

Nedladdningen och dataförberedelsen är implementerade. Övriga steg är för närvarande tomma delar som kommer att byggas vidare under projektets gång.

## Projektstruktur

```text
zoodle/
├── data/
│   ├── raw/                  # Nedladdade Quick, Draw!-filer
│   └── processed/            # Träning, validering och test
├── settings/
│   ├── __init__.py
│   └── settings.py          # Gemensamma sökvägar och inställningar
├── download_dataset.py          # Laddar ner datasetet
├── prepare_data.py              # Förbereder och delar upp datasetet
├── pipeline.py                  # Projektets terminalmeny
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

Projektet är under utveckling och byggs stegvis. Nästa delar blir dataanalys och modellträning.
