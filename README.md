# Repair Recommendation System

Suggests repair actions for an impacted area from two short descriptions: the **negative side** (where the damage shows)
and the **positive side** (the suspected source). A surveyor reviews and edits the suggestion, and approved or edited
versions are saved so the model can be retrained on confirmed answers.

Text only. Built from past waterproofing inspection reports. Suggestions are a draft for the surveyor, not a final
recommendation.

## How well does it work?

Measured on 813 held-out areas (no report appears in both training and test data):

| Measure | Result |
|---|---|
| Macro F1 over 12 repair actions | **0.54** (train 0.66) |
| Best actions (F1) | plumbing repair 0.69, tile joint grouting 0.70, epoxy grouting 0.66 |
| Weakest actions (F1) | sealant 0.38, painting 0.36 |
| Exactly the right set of actions | 14% of areas |

This is modest. When two different reports have the same description, their repair actions agree only partly (overlap
0.47), because inspectors word repairs differently. The training labels come from keyword rules on those texts, not from
surveyor review, so scores show how well the model reproduces those rules. Tuning the model did not help; better labels
would. Full details: [docs/recommendation_system_doc.html](docs/recommendation_system_doc.html).

## Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. Without `APP_PASSWORD` there is no sign-in, so use that only on your own machine.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io) choose **Create app**: repository, branch `main`, main file `app.py`.
3. In **Advanced settings** set **Python 3.13** and paste the secrets from `.streamlit/secrets.toml.example`
   (`APP_PASSWORD` and `ADMIN_PASSWORD`).
4. Deploy, then restrict who can view the app in its sharing settings.

On Streamlit Cloud:
- The app **refuses to open without `APP_PASSWORD`**.
- The correction log is **temporary** and is erased when the app restarts. Open **Administrator** at the bottom of the
  page (needs `ADMIN_PASSWORD`) and download the CSV regularly.
- Free apps sleep after a few days without visitors and take about a minute to wake.

## Run with Docker

```bash
cp .env.example .env        # then set the passwords
docker compose up -d --build
```

Corrections are kept in a Docker volume. Serve it on an internal network or behind HTTPS; the password is sent in clear
text over plain http.

## Repository layout

```
app.py                    Streamlit page (entry point)
recsys/                   code shared by the app and the pipeline
  cleaning.py             text cleaning and privacy masking (one function used for training and for the app)
  actions.py              the 12 actions: keyword rules for labels, display names, standard step sentences
  classifier.py           loads the model and scores text
  storage.py              correction log (SQLite) and CSV export
  config.py               paths and settings (environment variables or Streamlit secrets)
models/                   trained model, spelling fixes, vocabulary_review.txt (the word list, for privacy review)
data/examples.json        written sample inputs for the "Load example" button
pipeline/                 rebuild the dataset and model from inspection reports (needs the reports, see below)
  parse_reports.py        raw report text and JSON -> impacted areas
  clean_split.py          clean, remove duplicates, split train/val/test by report
  fix_labels.py           keep only the action part of each recommendation
  action_classifier.py    compare models, per-action scores
  build_app_model.py      train the model used by the app -> models/
  experiments/            keyword search, semantic search and tuning tests that shaped the design
tests/                    18 tests (cleaning, actions, model, log, whole page)
docs/                     technical reference with results and diagrams
pyproject.toml            lint (ruff) and test settings
requirements*.txt         pinned packages: app, development, and the optional semantic-search experiment
```

## Rebuild the model

The inspection reports are **not** in this repository (they hold customer data). To retrain you need the reports folder:

```bash
pip install -r requirements-dev.txt
export INSPECTION_DATASET="/path/to/inspection/folders"     # Windows: set INSPECTION_DATASET=...
export BUILD_DIR="/path/for/outputs"                        # optional; default <INSPECTION_DATASET>/_built
python -m pipeline.parse_reports
python -m pipeline.clean_split
python -m pipeline.fix_labels
python -m pipeline.build_app_model          # writes models/model.joblib
```

Run every command from the repository root. Intermediate files (datasets, review sheets) go to `BUILD_DIR`, never into the
repository.

Keep scikit-learn at the version in `requirements.txt`, or retrain with the new one: a saved model can fail to load on a
different version.

## Tests and code checks

```bash
pip install -r requirements-dev.txt
python -m pytest          # 18 tests
python -m ruff check .     # lint (settings in pyproject.toml)
python -m ruff format --check .
```

The same three commands run on every push (`.github/workflows/tests.yml`).

## Privacy

- No reports, review sheets or datasets are stored here. The sample inputs are written examples.
- The model keeps only plain English words that appear in at least 25 training areas (1,203 terms). No names, unit
  numbers or phone numbers. `models/vocabulary_review.txt` lists every single word for checking.
- The log masks phone numbers, e-mail addresses and links before saving. It cannot detect names or addresses, so the page
  asks users not to type them.
- One shared password; use a company login proxy if individual accounts are needed.
