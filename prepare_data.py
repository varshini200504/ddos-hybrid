# prepare_data.py
from src.data_loader import load_nsl_kdd  # ✅ make sure this import is correct
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 1️⃣ Check if dataset folder exists
dataset_path = os.path.expanduser("~/datasets/nsl-kdd")
print("Dataset folder exists:", os.path.exists(dataset_path))

# 2️⃣ Load the data
train_df, test_df = load_nsl_kdd(
    split="both",
    base_dir=dataset_path,
    download_if_missing=False,
    drop_difficulty=True,
    map_to_5class=True,
    one_hot=True
)

# 3️⃣ Print shapes
print("Train shape:", None if train_df is None else train_df.shape)
print("Test shape:", None if test_df is None else test_df.shape)

# 4️⃣ Features: drop label columns
X_train = train_df.drop(columns=['label', 'label_5class'])
X_test  = test_df.drop(columns=['label', 'label_5class'])

# 5️⃣ Labels: use the 5-class mapping
y_train = train_df['label_5class']
y_test  = test_df['label_5class']

# 6️⃣ Convert to NumPy arrays
X_train_np = X_train.to_numpy()
X_test_np  = X_test.to_numpy()
y_train_np = y_train.to_numpy()
y_test_np  = y_test.to_numpy()

print("X_train shape:", X_train_np.shape)
print("y_train shape:", y_train_np.shape)

# 7️⃣ Convert labels to integers
le = LabelEncoder()
y_train_int = le.fit_transform(y_train_np)
y_test_int  = le.transform(y_test_np)

# 8️⃣ Optional: one-hot encoding for deep learning
y_train_oh = np.eye(len(le.classes_))[y_train_int]
y_test_oh  = np.eye(len(le.classes_))[y_test_int]

print("y_train one-hot shape:", y_train_oh.shape)
