import os
import sys
import numpy as np
import torch
from voxelwise_tutorials.io import load_hdf5_array, get_data_home
from voxelwise_tutorials.utils import zscore_runs, explainable_variance

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
N_RUNS = 1
N_TRS_PER_RUN = 300
T_WIN = 2
TOP_K = 6016
SUBJECT = "S01"
def preprocessed_fmri_dir() -> str:
    return os.path.join(data_root(), "test")
def data_root() -> str:
    return get_data_home(dataset="shortclips")
responses_hdf = os.path.join(data_root(), "responses", f"{SUBJECT}_responses.hdf")

print("Loading fMRI responses...")
Y_train_raw = load_hdf5_array(responses_hdf, key="Y_train")   # [T_train, V]
Y_test_raw  = load_hdf5_array(responses_hdf, key="Y_test")    # [T_test, V]
run_onsets  = load_hdf5_array(responses_hdf, key="run_onsets")

V_full = Y_train_raw.shape[1]
print(f"  Y_train : {Y_train_raw.shape}") # (3600, 84038)
print(f"  Y_test  : {Y_test_raw.shape}") # (10, 270, 84038)

# ── 1. Select top-k voxels by explainable variance (from test split) ───────
print(f"\nComputing explainable variance on test set; keeping top {TOP_K} voxels...")
ev             = explainable_variance(Y_test_raw)
top_k_indices  = np.argsort(ev)[-TOP_K:]   # highest EV last
top_k_indices  = np.sort(top_k_indices)    # preserve spatial order
min_ev         = ev[top_k_indices].min()
print(f"  EV range of selected voxels: [{min_ev:.4f}, {ev.max():.4f}]") # [0.1469, 0.7088]

Y_train_filtered = Y_train_raw[:, top_k_indices].astype("float32")  # [T_train, V_k]

# ── 2. Z-score by run ──────────────────────────────────────────────────────
print("\nZ-scoring by run...")
X_zscored = zscore_runs(np.nan_to_num(Y_train_filtered), run_onsets)

# ── 3. Sliding windows within each run ────────────────────────────────────
print(f"\nBuilding sliding windows (T_WIN={T_WIN})...")
windows    = []
global_trs = []

for run_i in range(N_RUNS):
    start_tr = run_i * N_TRS_PER_RUN
    run_data = X_zscored[start_tr : start_tr + N_TRS_PER_RUN]   # [300, V_k]
    n_valid  = N_TRS_PER_RUN - T_WIN + 1
    for t in range(n_valid):
        windows.append(run_data[t : t + T_WIN])   # [T_WIN, V_k]
        global_trs.append(start_tr + t)

X_windows  = np.stack(windows,    axis=0).astype("float32")   # [N, T_WIN, V_k]
global_trs = np.array(global_trs, dtype=np.int64)             # [N]

print(f"  X_windows   : {X_windows.shape}") # (3552, 2, 5000)
print(f"  global_trs  : {global_trs.shape}") # (3552,)

# ── 4. Save ────────────────────────────────────────────────────────────────
out_dir = preprocessed_fmri_dir()
os.makedirs(out_dir, exist_ok=True)

torch.save(torch.from_numpy(X_windows), os.path.join(out_dir, "X_fmri_windows.pt"))
np.save(os.path.join(out_dir, "valid_tr_global.npy"), global_trs)
np.save(os.path.join(out_dir, "voxel_indices.npy"),   top_k_indices)
torch.save(
    {
        "N_RUNS":        N_RUNS,
        "N_TRS_PER_RUN": N_TRS_PER_RUN,
        "T_WIN":         T_WIN,
        "top_k":         TOP_K,
        "V_full":        V_full,
        "subject":       SUBJECT,
    },
    os.path.join(out_dir, "metadata.pt"),
)

print(f"\nAll outputs saved to: {out_dir}")
