# data_loader.py
"""
NSL-KDD dataset loader for Pandas.

Functions:
- download_nsl_kdd(base_dir): downloads train/test files if missing
- load_nsl_kdd(split='train'/'test'/'both', base_dir, ...): loads dataset as Pandas DataFrame(s)
"""

import os
from pathlib import Path
import pandas as pd

# Filenames
TRAIN_FILE = "KDDTrain+.txt"
TEST_FILE  = "KDDTest+.txt"

# Column names
NSL_KDD_COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment",
    "urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted",
    "num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

# 5-class mapping
LABEL_TO_5CLASS = {
    "normal":"Normal",
    "back":"DoS","land":"DoS","neptune":"DoS","pod":"DoS","smurf":"DoS","teardrop":"DoS",
    "apache2":"DoS","udpstorm":"DoS","processtable":"DoS","worm":"DoS",
    "satan":"Probe","ipsweep":"Probe","nmap":"Probe","portsweep":"Probe","mscan":"Probe","saint":"Probe",
    "guess_passwd":"R2L","ftp_write":"R2L","imap":"R2L","phf":"R2L","multihop":"R2L","warezmaster":"R2L",
    "warezclient":"R2L","spy":"R2L","xlock":"R2L","xsnoop":"R2L","snmpguess":"R2L","snmpgetattack":"R2L",
    "httptunnel":"R2L","sendmail":"R2L","named":"R2L",
    "buffer_overflow":"U2R","loadmodule":"U2R","rootkit":"U2R","perl":"U2R","sqlattack":"U2R",
    "xterm":"U2R","ps":"U2R"
}

# URLs
MIRRORS = {
    "github": {
        "train": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt",
        "test":  "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt",
    }
}

# Helper to create folder
def _expand_base_dir(base_dir):
    p = Path(os.path.expanduser(base_dir)).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p

# Download dataset
def download_nsl_kdd(base_dir, overwrite=False, source="github", verbose=True):
    base = _expand_base_dir(base_dir)
    urls = MIRRORS[source]
    train_path = base / TRAIN_FILE
    test_path  = base / TEST_FILE

    def _maybe_download(url, path):
        if path.exists() and not overwrite:
            if verbose: print(f"Found existing: {path}")
            return
        if verbose: print(f"Downloading: {url} -> {path}")
        import urllib.request
        with urllib.request.urlopen(url) as resp, open(path, "wb") as f:
            f.write(resp.read())
        if verbose: print(f"Saved: {path}")

    _maybe_download(urls["train"], train_path)
    _maybe_download(urls["test"],  test_path)
    return train_path, test_path

# Read dataset
def _read_split(path, drop_difficulty=True):
    df = pd.read_csv(path, header=None, names=NSL_KDD_COLUMNS)
    for col in CATEGORICAL_COLS + ["label"]:
        df[col] = df[col].astype("category")
    if drop_difficulty:
        df = df.drop(columns=["difficulty"])
    return df

# Load dataset
def load_nsl_kdd(split="train", base_dir="~/datasets/nsl-kdd",
                 download_if_missing=True, drop_difficulty=True,
                 map_to_5class=False, one_hot=False):
    base = _expand_base_dir(base_dir)
    train_path = base / TRAIN_FILE
    test_path  = base / TEST_FILE

    if download_if_missing and (not train_path.exists() or not test_path.exists()):
        download_nsl_kdd(base)

    if split in ("train", "both"):
        train_df = _read_split(train_path, drop_difficulty)
    if split in ("test", "both"):
        test_df = _read_split(test_path, drop_difficulty)

    def _map_labels(df):
        if map_to_5class:
            df = df.copy()
            df["label_5class"] = df["label"].astype(str).map(lambda x: LABEL_TO_5CLASS.get(x, "Unknown")).astype("category")
        return df

    def _encode(df):
        if one_hot:
            df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)
        return df

    if split == "train":
        train_df = _map_labels(train_df)
        return _encode(train_df)
    elif split == "test":
        test_df = _map_labels(test_df)
        return _encode(test_df)
    else:  # both
        train_df = _map_labels(train_df)
        test_df  = _map_labels(test_df)
        if one_hot:
            combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
            combined = pd.get_dummies(combined, columns=CATEGORICAL_COLS, drop_first=False)
            train_df = combined.iloc[:len(train_df)].reset_index(drop=True)
            test_df  = combined.iloc[len(train_df):].reset_index(drop=True)
        return train_df, test_df

# Self-test
if __name__ == "__main__":
    try:
        tr = load_nsl_kdd(split="train", base_dir="~/datasets/nsl-kdd", download_if_missing=False)
        print(tr.head())
        print("Train shape:", tr.shape)
    except Exception as e:
        print("Self-test failed:", e)
