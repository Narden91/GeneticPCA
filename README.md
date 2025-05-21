# Progetto di Classificazione Multiclasse con Struttura Modulare Avanzata

Questo progetto fornisce una struttura modulare per un task di classificazione multiclasse utilizzando Python e diverse librerie di machine learning. Include moduli per il caricamento dei dati, preprocessing, classificazione con XGBoost, CatBoost e Random Forest, calibrazione dei modelli, interpretabilità, algoritmi genetici (esempio) e metodi bayesiani (esempio).

Il file `main.py` si trova al primo livello (root) del repository e funge da punto di ingresso principale per l'esecuzione del pipeline.

## Struttura del Progetto

```
genetic_ml_project/
├── data/                     # Cartella per i dataset (es. file CSV)
│   └── esempio.csv           # File CSV di esempio creato da main.py per test
├── models/                   # Cartella per i modelli addestrati salvati (es. shap_summary_plot.png)
├── notebooks/                # Cartella per Jupyter notebooks di analisi esplorativa o sperimentazione
├── data_loader/              # Modulo per il caricamento dei dati
│   ├── __init__.py
│   └── data_loader.py
├── preprocessor/             # Modulo per il preprocessing e la pulizia dei dati
│   ├── __init__.py
│   └── preprocessor.py
├── classifier/               # Modulo per l'addestramento e la valutazione dei modelli
│   ├── __init__.py
│   └── classifier.py
├── calibration/              # Modulo per la calibrazione dei modelli
│   ├── __init__.py
│   └── calibration.py
├── explainer/                # Modulo per l'interpretabilità dei modelli
│   ├── __init__.py
│   └── explainer.py
├── genetic_algorithm/        # Modulo per algoritmi genetici (esempio)
│   ├── __init__.py
│   └── genetic_algorithm.py
├── bayesian_methods/         # Modulo per metodi bayesiani (esempio)
│   ├── __init__.py
│   └── bayesian_methods.py
├── tests/                    # Cartella per i test unitari e di integrazione (da implementare)
├── main.py                   # Script principale per eseguire il pipeline
├── requirements.txt          # File con le dipendenze Python del progetto
└── README.md                 # Questo file
```

## Funzionalità Principali

- **Punto di Ingresso Unico**: `main.py` nella root del progetto orchestra l'intero flusso di lavoro.
- **Moduli Indipendenti**: Ogni funzionalità chiave (caricamento dati, preprocessing, classificazione, ecc.) è incapsulata nel proprio modulo Python (package), facilitando la manutenibilità e l'espandibilità.
    - `data_loader`: Legge dati da file CSV.
    - `preprocessor`: Gestisce la pulizia base e la suddivisione dei dati.
    - `classifier`: Addestra e valuta modelli XGBoost, CatBoost, Random Forest.
    - `calibration`: Calibra le probabilità dei modelli.
    - `explainer`: Fornisce interpretabilità tramite SHAP.
    - `genetic_algorithm`: Contiene una classe di esempio per un algoritmo genetico.
    - `bayesian_methods`: Contiene una classe di esempio per metodi bayesiani.
- **Esempio Funzionante**: `main.py` include la creazione di un dataset di esempio e l'esecuzione di un pipeline completo con tutti i moduli.

## Come Iniziare

1.  **Clonare il repository (o scaricare i file).**
2.  **Creare un ambiente virtuale (consigliato):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Su Windows: venv\Scripts\activate
    ```
3.  **Installare le dipendenze:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Preparare i dati (Opzionale, se non si usa l'esempio):**
    -   Se si desidera utilizzare un proprio dataset, inserire il file CSV nella cartella `data/`.
    -   Aggiornare il percorso del file (`file_path`) e il nome della colonna target (`target_column`) in `main.py`.
5.  **Eseguire il pipeline:**
    ```bash
    python main.py
    ```
    Questo eseguirà il pipeline utilizzando il file `data/esempio.csv` creato automaticamente se non esiste.

## Moduli Dettagliati

Ciascun modulo nella struttura (es. `data_loader`, `preprocessor`, ecc.) è un package Python contenente:
-   `__init__.py`: Rende la cartella un package e gestisce le importazioni principali dal modulo (es. `from .data_loader import load_csv`).
-   Un file `.py` con l'implementazione logica (es. `data_loader.py`).

Consultare il codice sorgente all'interno di ciascun modulo e il file `main.py` per i dettagli di implementazione e di interazione tra i moduli.

## TODO / Prossimi Passi

-   Espandere le funzionalità dei moduli `genetic_algorithm` e `bayesian_methods` con logiche più complesse.
-   Implementare la logica per salvare e caricare i modelli addestrati nella cartella `models/`.
-   Aggiungere la gestione degli argomenti da riga di comando (es. con `argparse`) in `main.py`.
-   Sviluppare test unitari e di integrazione nella cartella `tests/`.
-   Migliorare le strategie di preprocessing, validazione incrociata e ricerca degli iperparametri.

## Contributi

Sentitevi liberi di contribuire al progetto aprendo issue o pull request.

