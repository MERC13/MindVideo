import argparse
import os
import sys

import torch
from diffusers import AutoencoderKL, PNDMScheduler
from einops import rearrange
from tqdm.auto import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from MindVideo import UNet3DConditionModel, save_videos_grid

EMBEDDINGS_PATH = os.path.expanduser("~/voxelwise_tutorials_data/shortclips/test/fmri_embeddings.pt")


def decode_latents(vae, latents):
    video_length = latents.shape[2]
    latents = latents.to(dtype=vae.dtype)
    latents = 1.0 / 0.18215 * latents
    latents = rearrange(latents, "b c f h w -> (b f) c h w")
    video = vae.decode(latents).sample
    video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
    return (video / 2 + 0.5).clamp(0, 1)


@torch.no_grad()
def main(args):
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(
            f"Missing {EMBEDDINGS_PATH}. Run test-4.0/encoding.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.half_precision else torch.float32

    embeddings = torch.load(EMBEDDINGS_PATH, map_location="cpu").float()
    if embeddings.ndim != 3:
        raise ValueError(f"Expected embeddings shape [B, 77, 768], got {tuple(embeddings.shape)}")

    start = args.start_index
    end = min(start + args.batch_size, embeddings.shape[0])
    if start >= end:
        raise ValueError(f"Invalid slice [{start}:{end}] for embeddings with batch {embeddings.shape[0]}")

    embeddings = embeddings[start:end].to(device=device, dtype=dtype)
    batch_size = embeddings.shape[0]

    print(f"Device: {device}")
    print(f"Embeddings: {EMBEDDINGS_PATH}")
    print(f"Using samples [{start}:{end}] with shape {tuple(embeddings.shape)}")

    checkpoint_path = args.checkpoint_path
    unet = UNet3DConditionModel.from_pretrained_2d(checkpoint_path, subfolder="unet").to(device, dtype=dtype)
    vae = AutoencoderKL.from_pretrained(checkpoint_path, subfolder="vae").to(device, dtype=dtype)
    scheduler = PNDMScheduler.from_pretrained(checkpoint_path, subfolder="scheduler")

    unet.eval()
    vae.eval()

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    latent_shape = (
        batch_size,
        unet.config.in_channels,
        args.video_length,
        args.height // vae_scale_factor,
        args.width // vae_scale_factor,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma

    scheduler.set_timesteps(args.num_inference_steps, device=device)
    print(f"Denoising steps: {len(scheduler.timesteps)}")

    for t in tqdm(scheduler.timesteps, desc="Denoising"):
        latent_model_input = scheduler.scale_model_input(latents, t)
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=embeddings).sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    videos = decode_latents(vae, latents).cpu()

    os.makedirs(args.output_dir, exist_ok=True)
    tensor_path = os.path.join(args.output_dir, "generated_videos.pt")
    torch.save(videos, tensor_path)
    print(f"Saved tensor: {tensor_path}")

    fps = max(1, args.video_length // 2)
    for i in range(videos.shape[0]):
        sample_idx = start + i
        gif_path = os.path.join(args.output_dir, f"sample_{sample_idx:04d}.gif")
        save_videos_grid(videos[i : i + 1], gif_path, fps=fps)
        print(f"Saved gif: {gif_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Minimal UNet pipeline from fmri_embeddings.pt")
    parser.add_argument("--checkpoint_path", type=str, default="pretrains/sub1")
    parser.add_argument("--output_dir", type=str, default="results/test-4.0")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--video_length", type=int, default=6)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--half_precision", action="store_true")
    main(parser.parse_args())
