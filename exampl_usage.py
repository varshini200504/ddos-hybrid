from src.data_loader import download_nsl_kdd, load_nsl_kdd

# Step 1: Make sure dataset is present (run this once if dataset not downloaded yet)
# download_nsl_kdd("~/datasets/nsl-kdd")

# Step 2: Load both train and test datasets
train_df, test_df = load_nsl_kdd(
    split="both",
    base_dir="~/datasets/nsl-kdd",
    download_if_missing=False,   # set True if you want auto-download
    drop_difficulty=True,
    map_to_5class=True,          # Normal, DoS, Probe, R2L, U2R
    one_hot=False                # set True if you want categorical → numbers
)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print(train_df.head(10))
