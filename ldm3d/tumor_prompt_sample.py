#!/usr/bin/env python3
"""
tumor_prompt_sample.py

Prompt-driven sampling wrapper on TOP of your existing trained models (no retraining).
Uses your LDM (UNet + DDPM) + VAE to generate 3D MRI volumes while injecting a
"tumor prompt" (location, grade, size) by synthesizing a control tensor
(mask/edge/dist) in latent space (control_channels=3).

Inputs:
  --location  (string, e.g. pons/brainstem/thalamus/temporal/frontal/parietal/occipital/cerebellum)
  --grade     (low/high)
  --size      (float 0..1, normalized tumor volume proxy)

Outputs:
  outdir/prompt_samples/<location>_<grade>_s<size>/sample_##/{t1,t2,flair}_synth.nii.gz
"""

from __future__ import annotations
import os
import math
import argparse
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
import nibabel as nib

from ldm3d.config import DEVICE
from ldm3d.vae import VAE3D
from ldm3d.unet import UNet3DLatentCond
from ldm3d.diffusion import LatentDDPM
from ldm3d.ema import EMA
from ldm3d.latent_stats import maybe_load_latent_stats


from pathlib import Path
import torch
from torch.utils.data import DataLoader

from ldm3d.data import VolFolder
from ldm3d.latent_stats import estimate_latent_stats

def recompute_and_save_latent_stats(outdir: str, gbm_root: str, vae, batches: int,
                                   batch_size: int, num_workers: int, use_posterior_noise: bool):
    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    # Recompute latent mean/std from raw volumes to match current VAE.
    print(f"[STATS] Recomputing latent stats from: {gbm_root}")
    ds = VolFolder(gbm_root)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    stats = estimate_latent_stats(
        dl, vae,
        max_batches=batches,
        use_posterior_noise=use_posterior_noise,
    )

    # Force tensors to CPU so saved file is portable across devices.
    # ensure CPU tensors for saving + portability
    stats = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in stats.items()}

    stats_path = outdir_p / "latent_stats.pt"
    torch.save(stats, stats_path)
    print(f"[STATS] Saved recomputed latent stats -> {stats_path}")
    return stats

# ---------------------------
# Prompt -> latent-space "control" synthesis
# ---------------------------
LOCATION_TO_CENTER = {
    # centers are (x,y,z) in [0,1] of latent grid
    "pons": (0.52, 0.55, 0.45),
    "brainstem": (0.52, 0.58, 0.45),
    "thalamus": (0.50, 0.45, 0.55),
    "frontal": (0.50, 0.25, 0.70),
    "temporal": (0.25, 0.50, 0.55),
    "parietal": (0.50, 0.55, 0.70),
    "occipital": (0.50, 0.80, 0.65),
    "cerebellum": (0.50, 0.80, 0.35),
}


def _make_3d_gaussian_mask(
    latent_size: int,
    center_xyz01: Tuple[float, float, float],
    size01: float,
    grade: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns mask: (1, 1, L, L, L) in [0,1]
    size01 controls sigma; grade controls amplitude/sharpness.
    """
    L = latent_size
    cx01, cy01, cz01 = center_xyz01
    # Convert normalized [0,1] center to voxel coordinates in latent grid.
    cx = cx01 * (L - 1)
    cy = cy01 * (L - 1)
    cz = cz01 * (L - 1)

    # sigma: small size -> tight; big size -> wide
    # clamp so it doesn't become degenerate
    size01 = float(np.clip(size01, 0.02, 0.95))
    sigma = 1.0 + 5.5 * size01  # ~[1..6.2] for L=28

    # grade affects sharpness and amplitude
    if grade.lower() == "high":
        amp = 1.0
        sharp = 1.0
        thresh = 0.28
    else:
        amp = 0.75
        sharp = 0.8
        thresh = 0.36

    xs = torch.arange(L, device=device).float()
    ys = torch.arange(L, device=device).float()
    zs = torch.arange(L, device=device).float()
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")

    # Gaussian blob in 3D, clipped to [0,1].
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
    g = amp * torch.exp(-0.5 * d2 / (sigma**2))
    g = g.clamp(0.0, 1.0)

    # make a soft mask (not binary), but accent tumor core
    mask = (g**(1.0 / max(1e-6, sharp))).clamp(0.0, 1.0)
    # add a gentle floor so controls aren’t all zeros for tiny sizes
    mask = torch.where(mask > thresh, mask, 0.0)

    return mask.unsqueeze(0).unsqueeze(0)  # (1,1,L,L,L)


def _edge_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Cheap gradient-magnitude edge: (1,1,L,L,L)
    """
    # mask is (B,1,D,H,W)
    # Forward finite differences, then pad back to original size.
    dz = torch.abs(mask[:, :, 1:, :, :] - mask[:, :, :-1, :, :])
    dy = torch.abs(mask[:, :, :, 1:, :] - mask[:, :, :, :-1, :])
    dx = torch.abs(mask[:, :, :, :, 1:] - mask[:, :, :, :, :-1])

    # pad back to same size
    dz = torch.nn.functional.pad(dz, (0, 0, 0, 0, 0, 1))
    dy = torch.nn.functional.pad(dy, (0, 0, 0, 1, 0, 0))
    dx = torch.nn.functional.pad(dx, (0, 1, 0, 0, 0, 0))

    edge = (dx + dy + dz).clamp(0.0, 1.0)
    return edge


def _dist_like(mask: torch.Tensor) -> torch.Tensor:
    """
    Simple "distance-ish" proxy: 1 - blurred mask, normalized to [0,1].
    This is NOT a true distance transform, but it provides a smooth spatial cue.
    """
    x = mask
    # A few rounds of average pooling acts as a cheap low-pass filter.
    # 3D average pooling blur
    x = torch.nn.functional.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
    x = torch.nn.functional.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
    dist = (1.0 - x).clamp(0.0, 1.0)
    return dist


def build_control_from_prompt(
    latent_size: int,
    location: str,
    grade: str,
    size01: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns control tensor (B=1, C=3, L, L, L): [mask, edge, dist]
    """
    # Map location to a canonical center; fall back to thalamus.
    loc = location.lower().strip()
    center = LOCATION_TO_CENTER.get(loc, LOCATION_TO_CENTER["thalamus"])
    mask = _make_3d_gaussian_mask(latent_size, center, size01, grade, device)
    edge = _edge_from_mask(mask)
    dist = _dist_like(mask)
    control = torch.cat([mask, edge, dist], dim=1)
    return control


# ---------------------------
# EMA loading / applying
# ---------------------------
@torch.no_grad()
def apply_ema_shadow(unet: torch.nn.Module, ema_shadow: Dict[str, torch.Tensor]) -> None:
    """
    Copies EMA weights into unet params (in-place).
    Handles CPU-loaded shadow by moving each tensor to the param's device/dtype.
    """
    for name, p in unet.named_parameters():
        if not p.requires_grad:
            continue
        if name not in ema_shadow:
            continue
        w = ema_shadow[name].to(device=p.device, dtype=p.dtype)
        p.data.copy_(w.data)


# ---------------------------
# Latent helpers
# ---------------------------
def _broadcast_lat_stats(lat_mean: torch.Tensor, lat_std: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    lat_mean/std usually shape (z_channels,). Broadcast to (1,C,1,1,1).
    """
    # Ensure mean/std are channel-only and match z's device/dtype.
    if lat_mean.ndim == 1:
        lat_mean = lat_mean.view(1, -1, 1, 1, 1)
    if lat_std.ndim == 1:
        lat_std = lat_std.view(1, -1, 1, 1, 1)
    lat_mean = lat_mean.to(device=z.device, dtype=z.dtype)
    lat_std = lat_std.to(device=z.device, dtype=z.dtype)
    return lat_mean, lat_std


@torch.no_grad()
def vae_decode_any(vae: torch.nn.Module, z: torch.Tensor) -> torch.Tensor:
    """
    Try common VAE decode entrypoints.
    Returns (B,3,D,H,W) float (expected in [0,1] in your pipeline).
    """
    # Prefer explicit decode() when available.
    if hasattr(vae, "decode"):
        out = vae.decode(z)
        return out
    # fallback: try forward signatures
    try:
        out = vae(z, decode=True)
        return out
    except TypeError:
        out = vae(z)
        return out


# ---------------------------
# Diffusion sampling (robust-ish)
# ---------------------------
@torch.no_grad()
def sample_latents(
    unet: torch.nn.Module,
    ddpm: LatentDDPM,
    shape: Tuple[int, int, int, int, int],
    cond_mask: torch.Tensor,
    control: torch.Tensor,
    guidance_scale: float,
    cond_drop_p: float,
    sample_seed: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns z_norm (B,C,L,L,L).
    This tries to use your ddpm implementation if it exposes a p_sample_loop-like method.
    """
    if sample_seed >= 0:
        g = torch.Generator(device=device)
        g.manual_seed(sample_seed)
        z = torch.randn(shape, device=device, generator=g)
    else:
        z = torch.randn(shape, device=device)

    # Prefer ddpm-native sampling helpers if available.
    # Best-case: your LatentDDPM already has a sampling helper.
    # We try a few common names; otherwise we fall back to a minimal DDPM loop
    # ONLY if ddpm exposes needed buffers.
    for fn_name in ["p_sample_loop", "sample", "ddim_sample_loop"]:
        if hasattr(ddpm, fn_name):
            fn = getattr(ddpm, fn_name)
            try:
                return fn(
                    unet=unet,
                    shape=shape,
                    z_init=z,
                    cond_mask=cond_mask,
                    control=control,
                    guidance_scale=guidance_scale,
                    cond_drop_p=cond_drop_p,
                )
            except TypeError:
                # try without named args
                try:
                    return fn(unet, z, cond_mask, control, guidance_scale, cond_drop_p)
                except Exception:
                    pass

    # Fallback minimal loop (assumes ddpm has betas/alphas like typical DDPM)
    if not hasattr(ddpm, "betas"):
        raise RuntimeError("Your LatentDDPM has no known sampling method (p_sample_loop/sample) and no betas buffer.")

    # Minimal DDPM sampler using betas/alphas (kept for robustness).
    betas = ddpm.betas.to(device)
    alphas = 1.0 - betas
    alphas_cum = torch.cumprod(alphas, dim=0)

    T = int(getattr(ddpm, "T", betas.numel()))
    x = z

    def _eps_pred(x_t, t_idx: int, drop: float) -> torch.Tensor:
        # try unet signatures
        t = torch.full((x_t.shape[0],), t_idx, device=device, dtype=torch.long)
        try:
            return unet(x_t, t, cond_mask, control, cond_drop_p=drop)
        except TypeError:
            try:
                return unet(x_t, t, cond_mask, control)
            except TypeError:
                return unet(x_t, t)

    for t in reversed(range(T)):
        a_t = alphas[t]
        ac_t = alphas_cum[t]
        b_t = betas[t]

        # Classifier-free guidance via conditional vs dropped conditions.
        # classifier-free-ish: conditional vs "dropped"
        eps_c = _eps_pred(x, t, drop=cond_drop_p)
        eps_u = _eps_pred(x, t, drop=1.0)  # fully drop condition
        eps = eps_u + guidance_scale * (eps_c - eps_u)

        # DDPM reverse step
        # x0_pred = (x_t - sqrt(1-ac_t)*eps) / sqrt(ac_t)
        sqrt_ac = torch.sqrt(ac_t)
        sqrt_one_minus_ac = torch.sqrt(1.0 - ac_t)
        x0 = (x - sqrt_one_minus_ac * eps) / (sqrt_ac + 1e-8)

        # mean of p(x_{t-1} | x_t)
        coef1 = torch.sqrt(alphas_cum[t - 1]) if t > 0 else torch.tensor(1.0, device=device)
        coef2 = torch.sqrt(1.0 - alphas_cum[t - 1]) if t > 0 else torch.tensor(0.0, device=device)

        # DDPM posterior mean approximation
        mean = coef1 * x0 + coef2 * eps

        if t > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(b_t)
            x = mean + sigma * noise
        else:
            x = mean

    return x


# ---------------------------
# Save helpers
# ---------------------------
def save_modalities_as_nifti(vol: torch.Tensor, out_dir: Path, prefix: str = "") -> None:
    """
    vol: (1,3,D,H,W) torch float
    Saves t1,t2,flair.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # NIfTI expects (H,W,D); transpose from (D,H,W).
    vol = vol.detach().cpu().float().numpy()[0]  # (3,D,H,W)

    # Your volumes are usually (H,W,D) in NIfTI; here we have (D,H,W) likely.
    # We'll transpose to (H,W,D) for writing.
    t1 = np.transpose(vol[0], (1, 2, 0))
    t2 = np.transpose(vol[1], (1, 2, 0))
    fl = np.transpose(vol[2], (1, 2, 0))

    affine = np.eye(4, dtype=np.float32)
    nib.save(nib.Nifti1Image(t1, affine), str(out_dir / f"{prefix}t1_synth.nii.gz"))
    nib.save(nib.Nifti1Image(t2, affine), str(out_dir / f"{prefix}t2_synth.nii.gz"))
    nib.save(nib.Nifti1Image(fl, affine), str(out_dir / f"{prefix}flair_synth.nii.gz"))


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()

    # Required prompt inputs
    ap.add_argument("--location", type=str, required=True, help="e.g., pons/brainstem/thalamus/temporal/frontal/...")
    ap.add_argument("--grade", type=str, required=True, choices=["low", "high"], help="low or high")
    ap.add_argument("--size", type=float, required=True, help="0..1 normalized tumor size proxy")

    # Model paths (your exact ones)
    ap.add_argument("--unet_ckpt", type=str, default="/home/j98my/models/runs/ldm_3d_diffuse_glioma/new_LAT28_VAE200_50_LDM300_200_TUNE2_UNET96/unet_final.pt")
    ap.add_argument("--vae_ckpt", type=str, default="/home/j98my/models/runs/ldm_3d_diffuse_glioma/new_LAT28_VAE200_50_LDM300_200_TUNE2_UNET96/vae_final.pt")
    ap.add_argument("--ema_ckpt", type=str, default="/home/j98my/models/runs/ldm_3d_diffuse_glioma/new_LAT28_VAE200_50_LDM300_200_TUNE2_UNET96/ema_final.pt")
    ap.add_argument("--latent_stats", type=str, default="/home/j98my/models/runs/ldm_3d_diffuse_glioma/new_LAT28_VAE200_50_LDM300_200_TUNE2_UNET96/latent_stats.pt")

    # Output
    ap.add_argument("--outdir", type=str, default="/home/j98my/models/runs/ldm_3d_diffuse_glioma/new_LAT28_VAE200_50_LDM300_200_TUNE2_UNET96")
    ap.add_argument("--n", type=int, default=16, help="number of samples to generate")

    # Latent/model hyperparams must match your training
    ap.add_argument("--latent_size", type=int, default=28)
    ap.add_argument("--z_channels", type=int, default=8)
    ap.add_argument("--vae_base", type=int, default=64)
    ap.add_argument("--unet_base", type=int, default=96)  # matches TUNE2_UNET96
    ap.add_argument("--t_dim", type=int, default=256)

    # Diffusion schedule (match training)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--beta_start", type=float, default=1e-4)
    ap.add_argument("--beta_end", type=float, default=2e-2)

    # Sampling knobs
    ap.add_argument("--guidance_scale", type=float, default=4.0)
    ap.add_argument("--cond_drop_p", type=float, default=0.1)
    ap.add_argument("--sample_seed", type=int, default=42, help="seed for reproducible sampling (-1 disables)")

    ap.add_argument("--gbm_root", type=str, default="/home/j98my/Pre-Processing/prep/gbm_all_aligned",
                help="Used only when recomputing latent stats.")
    ap.add_argument("--latent_stat_batches", type=int, default=400)
    ap.add_argument("--use_posterior_noise", action="store_true")
    ap.add_argument("--recompute_latent_stats", action="store_true",
                    help="If set: recompute latent mean/std using gbm_root + current VAE, then overwrite latent_stats.pt")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)


    args = ap.parse_args()

    outroot = Path(args.outdir)
    prompt_tag = f"{args.location.lower()}_{args.grade.lower()}_s{args.size:.2f}"
    save_root = outroot / "prompt_samples" / prompt_tag

    # --- Load models
    device = torch.device(DEVICE)

    vae = VAE3D(z_channels=args.z_channels, base=args.vae_base).to(device)
    unet = UNet3DLatentCond(
        z_channels=args.z_channels,
        cond_channels=1,
        control_channels=3,
        base=args.unet_base,
        t_dim=args.t_dim,
        use_controlnet=True,
    ).to(device)

    ddpm = LatentDDPM(T=args.timesteps, beta_start=args.beta_start, beta_end=args.beta_end)

    vae_sd = torch.load(args.vae_ckpt, map_location="cpu")
    unet_sd = torch.load(args.unet_ckpt, map_location="cpu")
    ema_shadow = torch.load(args.ema_ckpt, map_location="cpu")
    stats = torch.load(args.latent_stats, map_location="cpu")

    vae.load_state_dict(vae_sd, strict=True)
    unet.load_state_dict(unet_sd, strict=True)
    apply_ema_shadow(unet, ema_shadow)

    vae.eval()
    unet.eval()

    # Latent normalization stats for z -> x decode.
    lat_mean = stats["mean"]
    lat_std = stats["std"]

    # --- Build prompt control
    control = build_control_from_prompt(
        latent_size=args.latent_size,
        location=args.location,
        grade=args.grade,
        size01=args.size,
        device=device,
    )  # (1,3,L,L,L)

    # cond_mask is required by your UNet as cond_channels=1
    # We feed the prompt mask channel as the "mask condition" too.
    cond_mask = control[:, :1, ...].contiguous()  # (1,1,L,L,L)

    print(f"[PROMPT] location={args.location} grade={args.grade} size={args.size:.3f}")
    print(f"[SAVE] -> {save_root}")

    # --- Sample N volumes
    for i in range(args.n):
        # vary seed per sample for diversity but still reproducible
        seed_i = (args.sample_seed + i) if args.sample_seed >= 0 else -1

        z_norm = sample_latents(
            unet=unet,
            ddpm=ddpm,
            shape=(1, args.z_channels, args.latent_size, args.latent_size, args.latent_size),
            cond_mask=cond_mask,
            control=control,
            guidance_scale=args.guidance_scale,
            cond_drop_p=args.cond_drop_p,
            sample_seed=seed_i,
            device=device,
        )

        # unnormalize latents: z = z_norm * std + mean
        lat_mean_b, lat_std_b = _broadcast_lat_stats(lat_mean, lat_std, z_norm)
        z = z_norm * (lat_std_b + 1e-8) + lat_mean_b

        # Decode to 3-channel MRI volume and save to disk.
        vol = vae_decode_any(vae, z)  # expected (1,3,D,H,W)

        sample_dir = save_root / f"sample_{i:02d}"
        save_modalities_as_nifti(vol, sample_dir)

        print(f"[OK] saved sample {i:02d} -> {sample_dir}")

    print("--- DONE ---")


if __name__ == "__main__":
    main()
