"""NSL-KDD loading and shared, train-fitted preprocessing."""
from pathlib import Path
import hashlib
import urllib.request
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
BASE = "https://zenodo.org/records/17424143/files/"
FILES = {"train": ("KDDTrain%2B.txt?download=1", "KDDTrain+.txt"), "test": ("KDDTest%2B.txt?download=1", "KDDTest+.txt")}
MD5 = {"train":"120b13c9092ab72d4902f2079143cc2c", "test":"1e53a506c102628114ca1b1fa509b6a8"}
FEATURES = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]
CATEGORICAL = ["protocol_type", "service", "flag"]
COLUMNS = FEATURES + ["label", "difficulty"]

def download_dataset():
    RAW.mkdir(parents=True, exist_ok=True)
    paths = {}
    for part, (url_filename, local_filename) in FILES.items():
        path = RAW / local_filename
        if not path.exists() or path.stat().st_size < 1_000_000:
            request = urllib.request.Request(BASE + url_filename, headers={"User-Agent": "FuzzyShield research project"})
            with urllib.request.urlopen(request, timeout=90) as response, path.open("wb") as out:
                out.write(response.read())
        paths[part] = path
    expected={"train":125_973,"test":22_544}
    for part,path in paths.items():
        with path.open("rb") as source:
            digest=hashlib.md5(source.read()).hexdigest()
        if digest != MD5[part] or sum(1 for _ in path.open("rb")) != expected[part]:
            path.unlink(missing_ok=True)
            raise ValueError(f"Downloaded {path.name} failed the official NSL-KDD checksum or row-count check")
    return paths

def load_part(path):
    with path.open("r", encoding="utf-8") as source:
        delimiter="\t" if "\t" in source.readline() else ","
    frame = pd.read_csv(path, header=None, names=COLUMNS, sep=delimiter)
    normalized=frame.label.astype(str).str.strip().str.rstrip(".").str.lower()
    frame["label"] = (normalized != "normal").astype("int8")
    return frame[FEATURES], frame.label

def build_preprocessor():
    numeric = [f for f in FEATURES if f not in CATEGORICAL]
    return ColumnTransformer([("numeric", StandardScaler(), numeric), ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)], remainder="drop", verbose_feature_names_out=False)
