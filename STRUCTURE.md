# BCI_Seminar_2026 — Proposed Directory Structure

This document describes the recommended layout for the redesigned project.

```
BCI_Seminar_2026/
│
├── data/
│   └── raw/                        # Original, read-only input files (never overwritten)
│       ├── ecogStruct1.pkl         # Primary ECoG recording struct
│       ├── ecogStruct2.pkl         # Second recording block (if used)
│       ├── ecogStruct3.pkl         # Third recording block (if used)
│       ├── rawEcog.pkl             # Raw ECoG before structuring
│       ├── gloveResamp.pkl         # Resampled glove sensor data
│       ├── gloveResamp_Dav_2011_9_7_14_28_21.pkl  # Timestamped glove file
│       └── ecogAnalog.pkl          # Analog reference channel
│
├── docs/                           # Reference materials (read-only)
│   ├── Experimental_paradigm_and_data_structure.pdf
│   ├── GP33_anatomy_40_electrodes.png
│   ├── Session0_Python Tutorial (installation).pdf
│   ├── Session0_Tutorial_Notebook_install_dev.docx
│   ├── Build a Launcher for yourselves.docx
│   └── python-cheatsheets.pdf
│
├── plots/                          # All figures saved by notebooks and scripts
│   └── (generated at runtime)
│
├── results/                        # Computed / intermediate .pkl outputs
│   ├── data_for_own_epoch.pkl      # Combined glove + analog, input to epoch labeler
│   ├── ecogStruct1_processed.pkl   # After Session 02 preprocessing
│   ├── ecogStruct1_periodogram.pkl # After bad-channel removal (Session 02)
│   ├── ecogStruct1_badEpochs.pkl   # Marked bad epochs (ecog_gui_v3)
│   ├── epochs.pkl                  # Labeled gesture epochs (get_epochs)
│   ├── epoch.pkl / epoch2.pkl      # Per-session epoch files
│   ├── zScoredData.pkl             # Z-scored feature matrix (Session 05)
│   └── resultsPCA.pkl              # PCA output (Session 05–06)
│
├── src/                            # Reusable Python helper modules
│   ├── get_epochs.py               # Interactive epoch labeling (Session 01)
│   ├── plot_glove_data.py
│   ├── plot_own_labels.py
│   ├── plot_raw_gesture.py
│   ├── ecog_gui_v3.py              # ECoG time-series GUI viewer (Session 02)
│   ├── removeBadChannels_periodogram_V2.py
│   ├── ecogMkPeriodogramMultitaper.py
│   ├── multitaper_spectrum.py
│   ├── nearly.py
│   ├── plot_std.py
│   ├── remove_bad_epochs.py
│   ├── ecog_segment_ts.py          # (Sessions 03–04)
│   ├── plot_spectrum.py
│   ├── create_subsets.py           # (Session 05)
│   ├── plot_features.py
│   ├── classification_svm.py       # (Sessions 06–07)
│   ├── ecogGetDefC.py
│   ├── getBalancedTrainset.py
│   ├── train_bayes.py
│   ├── train_lda.py
│   ├── test_bayes.py
│   ├── test_lda.py
│   ├── gen_selector.py             # (Session 07)
│   └── plot_ROC.py
│
│
├── requirements.txt                # Python dependencies
│
├── Session_00.ipynb                # Python environment setup
├── Session_01.ipynb                # Data exploration & epoch labeling
├── Session_02.ipynb                # Preprocessing & bad-channel removal
├── Session_03.ipynb                # Spectral analysis
├── Session_04.ipynb                # Time-frequency analysis
├── Session_05.ipynb                # Feature extraction
├── Session_06.ipynb                # Classification (LDA / Bayes)
└── Session_07.ipynb                # Advanced classification & ROC
```

## Notes

* **`data/raw/`** — place all original data files here before running any notebook.
  These files are inputs only; no script should write into this folder.

* **`results/`** — everything written by the pipeline ends up here.
  Intermediate files produced by one session become inputs for the next.

* **`src/`** — helper functions.

* **`plots/`** — all `plt.savefig(...)` and similar calls should target this folder.

* **`docs/`** — static reference PDFs and images; not executed, just read.
