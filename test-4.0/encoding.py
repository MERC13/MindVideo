import os
import sys
import torch
from MindVideo import fMRIEncoder

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

# Load fMRI windows from voxelwise_tutorials
file_path = os.path.expanduser("~/voxelwise_tutorials_data/shortclips/test/X_fmri_windows.pt")
fmri_sample = torch.load(file_path)  # (299, 2, 5000)
print(f"fMRI tensor shape: {fmri_sample.shape}")

# Load pretrained encoder
checkpoint_path = os.path.join(REPO_ROOT, "pretrains", "sub1")
num_voxels = fmri_sample.shape[-1]  # 5000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

print("\nLoading fMRI encoder...")
fmri_encoder = fMRIEncoder.from_pretrained(checkpoint_path, subfolder='fmri_encoder', num_voxels=num_voxels)
fmri_encoder = fmri_encoder.to(device)
fmri_encoder.eval()

# Encode all samples
print(f"\nEncoding {fmri_sample.shape[0]} samples...")
with torch.no_grad():
    fmri_embeddings = fmri_encoder(fmri_sample.to(device))

print(f"\nfMRI embeddings shape: {fmri_embeddings.shape}")
print(f"Expected: (299, 77, 768)")

# Optionally save embeddings to same directory as input
output_dir = os.path.expanduser("~/voxelwise_tutorials_data/shortclips/test")
torch.save(fmri_embeddings.cpu(), os.path.join(output_dir, "fmri_embeddings.pt"))
print(f"\nEmbeddings saved to: {os.path.join(output_dir, 'fmri_embeddings.pt')}")
