# FuzzyShield

A binary NSL-KDD intrusion detection research project with a scikit-learn MLP classifier, a trainable adaptive Sugeno ANFIS, a Streamlit dashboard, and a Next.js dashboard with FastAPI inference endpoints.

## Python 3.12 compatibility

The ANN uses scikit-learn `MLPClassifier`. The ANFIS is implemented in `fuzzyshield/src/anfis.py` and does not rely on the unmaintained `anfis` package: it has Gaussian membership functions, product antecedents, normalized rule firing strengths, zero-order Sugeno consequents, and minibatch gradient updates for its centers, widths, and consequent weights. This is compatible with Python 3.12 and modern NumPy/scikit-learn.

## Train and evaluate

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python fuzzyshield/src/train.py
```

Training downloads the original `KDDTrain+.txt` and `KDDTest+.txt` partitions from the [NSL-KDD Zenodo mirror](https://zenodo.org/records/17424143) of the UNB/CIC release. The loader verifies the published MD5 digests and official row counts before accepting data. The final difficulty column is ignored; the original attack labels are mapped to Normal/Attack. Preprocessing is fitted on training data only and shared by both classifiers. ANFIS uses a fixed-seed 24,000-record training subsample to keep gradient optimization laptop-friendly. ANN architecture and regularization are tuned with stratified two-fold cross-validation on a training subset, then refit on all training records.

Artifacts:

- `fuzzyshield/models/ann.joblib`, `anfis.joblib`
- `fuzzyshield/results/metrics.json`, `metrics.csv`
- `fuzzyshield/results/plots/model_evaluation.png`
- Vercel copies in `fuzzyshield-vercel/backend/models/` and `fuzzyshield-vercel/public/`

All reported metrics are produced from the held-out official test set, including timing measured on the training machine. The checked-in workspace results are from a completed run; retraining overwrites them with a new measured run.

## Streamlit dashboard

```bash
streamlit run fuzzyshield/app.py
```

The dashboard supports editable normal/attack presets, paired inference, probabilities, and measured model comparisons.

## Next.js + Python API

Training copies the trained artifacts into the FastAPI service's `backend/models/` directory and writes its pinned runtime dependencies there.

```bash
cd fuzzyshield-vercel
npm install
cp .env.example .env.local
# Leave NEXT_PUBLIC_API_URL empty for a same-project Vercel API; set it only for an external Python host.
npm run dev
```

The FastAPI application is `fuzzyshield-vercel/backend/main.py`. Run it locally from the repository root with:

```bash
uvicorn --app-dir fuzzyshield-vercel/backend main:app --reload --port 8000
```

For local development, run the API on port 8000 and set `NEXT_PUBLIC_API_URL=http://localhost:8000` in `.env.local`. In a same-project Vercel deployment, leave the variable empty; the Services routing table sends `/api/*` requests to FastAPI on the same origin. Request bodies contain the 41 NSL-KDD features. For a separately hosted API, configure `CORS_ORIGINS` there to include the dashboard's origin. The Next app's demo presets send the complete feature schema to those real inference endpoints; the visual activity sparkline is clearly marked illustrative, not live telemetry.

## Vercel deployment and size constraints

`vercel.json` configures two Vercel Services: the Next.js frontend and a Python 3.12 FastAPI backend. Services currently run in beta; use `vercel dev -L` locally to exercise the same `/api/*` routing. Vercel's standard Python function bundle limit is 500 MB uncompressed. The two model artifacts are about 136 KB total; NumPy/SciPy/scikit-learn/pandas make up most of the runtime size, so confirm the service build stays within the current limit. If Services are unavailable for the account or the Python bundle exceeds its limits, deploy `backend/` as a separate Python service and set `NEXT_PUBLIC_API_URL` to that service's URL. The frontend still uses the same real prediction endpoints.

## Dataset citation

NSL-KDD: M. Tavallaee et al., “A Detailed Analysis of the KDD CUP 99 Data Set,” 2009. Dataset access and citation notes are available on the [UNB/CIC NSL-KDD page](https://www.unb.ca/cic/datasets/nsl.html). Automated downloading uses the checksum-verified [Zenodo mirror](https://zenodo.org/records/17424143).
