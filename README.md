# Formative 2: Human Activity Recognition with a Hidden Markov Model

Author: Inyange Larissa (ALU). Four hidden states (standing, walking, jumping, still) inferred from phone
accelerometer + gyroscope data (Sensor Logger, iPhone 11 Pro Max) with an HMM written from scratch in NumPy
(Baum-Welch with a log-likelihood convergence check, and Viterbi decoding).

## Repository layout
| Path | Content |
|---|---|
| `raw_zips/` | original Sensor Logger exports (`<activity>_<NN>.zip`) |
| `extract_zips.py` | turns the exports into clean CSVs (uniform 50 Hz grid, 1 s trimmed at each end, `time_s` column) |
| `data/train/` | `<activity>_01..12.csv`, training recordings |
| `data/test/` | `<activity>_13..20.csv`, unseen recordings |
| `hmm_activity_recognition.ipynb` | full pipeline: QC, harmonization, features, HMM, evaluation |
| `HMM_Activity_Recognition_Report.docx` | short report (4 pages) |
| `figures/` | every figure used in the report |
| `results_summary.json` | all numbers quoted in the report |

## Run it
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
jupyter notebook hmm_activity_recognition.ipynb   # Run All
```
Optional: `python extract_zips.py` rebuilds `data/` from `raw_zips/`.

## Headline result (unseen recordings, 177 windows)
Final model (supervised estimates, variance floor 0.1, chosen by cross-validation on the training data) + Viterbi:
**94.9 %** accuracy; Baum-Welch-refined model 97.2 %; per-window Gaussian without transitions 90.4 %.
Walking and jumping are recognized perfectly; all errors are standing vs still.
