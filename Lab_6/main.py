#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convolutional VAE (latent_dim default=2) trained on MNIST resized to 128x128.
- Uses data augmentation on the training set.
- Focuses on reconstruction quality (beta small by default).
- At the end: generates N images and saves a few examples into ./example_gen (configurable).

Run example:
  python vae_mnist_128.py --epochs 20 --latent-dim 2 --image-size 128 --n-gen 64 --beta 0.001

Notes:
- For latent_dim==2, it also saves a nice 2D latent manifold grid.
"""

import os
import math
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.utils import save_image, make_grid

import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Model: Conv VAE for 128x128
# -----------------------------
class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int = 2, in_channels: int = 1, base_ch: int = 32, image_size: int = 128):
        super().__init__()
        assert image_size == 128, "This architecture is set for 128x128. Adjust strides if you change image_size."

        # Encoder: 128 -> 64 -> 32 -> 16 -> 8
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 4, 2, 1),      # (B, 32, 64, 64)
            nn.BatchNorm2d(base_ch),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1),      # (B, 64, 32, 32)
            nn.BatchNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1),  # (B, 128, 16, 16)
            nn.BatchNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch * 4, base_ch * 8, 4, 2, 1),  # (B, 256, 8, 8)
            nn.BatchNorm2d(base_ch * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.enc_out_ch = base_ch * 8
        self.enc_out_hw = 8
        enc_feat = self.enc_out_ch * self.enc_out_hw * self.enc_out_hw

        self.fc_mu = nn.Linear(enc_feat, latent_dim)
        self.fc_logvar = nn.Linear(enc_feat, latent_dim)

        # Decoder: latent -> (256, 8, 8) -> 16 -> 32 -> 64 -> 128
        self.fc_dec = nn.Linear(latent_dim, enc_feat)

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(self.enc_out_ch, base_ch * 4, 4, 2, 1),  # (B, 128, 16, 16)
            nn.BatchNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1),      # (B, 64, 32, 32)
            nn.BatchNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1),          # (B, 32, 64, 64)
            nn.BatchNorm2d(base_ch),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(base_ch, in_channels, 4, 2, 1),          # (B, 1, 128, 128)
            nn.Sigmoid(),  # output in [0,1]
        )

    def encode(self, x):
        h = self.enc(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(z.size(0), self.enc_out_ch, self.enc_out_hw, self.enc_out_hw)
        xhat = self.dec(h)
        return xhat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xhat = self.decode(z)
        return xhat, mu, logvar


# -----------------------------
# Loss
# -----------------------------
def vae_loss(x, xhat, mu, logvar, beta: float):
    # Reconstruction: BCE is typical for MNIST-like data (pixels in [0,1])
    # sum over all pixels, then mean over batch for stable reporting
    recon = F.binary_cross_entropy(xhat, x, reduction="sum") / x.size(0)

    # KL divergence per batch element, averaged
    # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / x.size(0)

    return recon + beta * kl, recon.detach(), kl.detach()


# -----------------------------
# Train / Eval
# -----------------------------
@dataclass
class TrainStats:
    train_total: list
    train_recon: list
    train_kl: list
    val_total: list
    val_recon: list
    val_kl: list


def train_one_epoch(model, loader, opt, device, beta):
    model.train()
    total_loss, total_recon, total_kl = 0.0, 0.0, 0.0
    n = 0

    for x, _ in loader:
        x = x.to(device)

        opt.zero_grad(set_to_none=True)
        xhat, mu, logvar = model(x)
        loss, recon, kl = vae_loss(x, xhat, mu, logvar, beta)
        loss.backward()
        opt.step()

        bs = x.size(0)
        total_loss += float(loss) * bs
        total_recon += float(recon) * bs
        total_kl += float(kl) * bs
        n += bs

    return total_loss / n, total_recon / n, total_kl / n


@torch.no_grad()
def eval_one_epoch(model, loader, device, beta):
    model.eval()
    total_loss, total_recon, total_kl = 0.0, 0.0, 0.0
    n = 0

    for x, _ in loader:
        x = x.to(device)
        xhat, mu, logvar = model(x)
        loss, recon, kl = vae_loss(x, xhat, mu, logvar, beta)

        bs = x.size(0)
        total_loss += float(loss) * bs
        total_recon += float(recon) * bs
        total_kl += float(kl) * bs
        n += bs

    return total_loss / n, total_recon / n, total_kl / n


# -----------------------------
# Sampling / Saving
# -----------------------------
@torch.no_grad()
def save_recon_examples(model, loader, device, out_dir, max_items=16, fname="recon_grid.png"):
    model.eval()
    x, _ = next(iter(loader))
    x = x.to(device)
    xhat, _, _ = model(x)

    x = x[:max_items].cpu()
    xhat = xhat[:max_items].cpu()

    # Interleave original and reconstruction
    interleaved = torch.cat([x, xhat], dim=0)  # (2*B,1,H,W)
    grid = make_grid(interleaved, nrow=max_items, padding=2)  # one row orig, one row recon (visually)
    save_image(grid, os.path.join(out_dir, fname))


@torch.no_grad()
def generate_and_save(model, device, latent_dim, image_size, out_dir, n_gen=64, grid_nrow=8, prefix="gen"):
    model.eval()

    z = torch.randn(n_gen, latent_dim, device=device)
    xg = model.decode(z).cpu()

    grid = make_grid(xg, nrow=grid_nrow, padding=2)
    grid_path = os.path.join(out_dir, f"{prefix}_grid.png")
    save_image(grid, grid_path)

    # Save a few individual samples
    for i in range(min(10, n_gen)):
        save_image(xg[i], os.path.join(out_dir, f"{prefix}_{i:02d}.png"))

    return grid_path


@torch.no_grad()
def generate_latent_manifold_2d(model, device, out_dir, grid_size=20, span=2.5, fname="latent_manifold.png"):
    """
    For latent_dim == 2: decode a 2D grid (linspace) -> nice visualization.
    """
    model.eval()
    lin = torch.linspace(-span, span, grid_size, device=device)
    zz = torch.cartesian_prod(lin, lin)  # (grid_size^2, 2)

    xg = model.decode(zz).cpu()  # (N,1,128,128)
    grid = make_grid(xg, nrow=grid_size, padding=1)
    path = os.path.join(out_dir, fname)
    save_image(grid, path)
    return path


def plot_curves(stats: TrainStats, out_dir: str):
    epochs = range(1, len(stats.train_total) + 1)

    plt.figure()
    plt.plot(list(epochs), stats.train_total, label="train_total")
    plt.plot(list(epochs), stats.val_total, label="val_total")
    plt.legend()
    plt.title("VAE Loss (total)")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_total.png"))
    plt.close()

    plt.figure()
    plt.plot(list(epochs), stats.train_recon, label="train_recon")
    plt.plot(list(epochs), stats.val_recon, label="val_recon")
    plt.legend()
    plt.title("Reconstruction Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_recon.png"))
    plt.close()

    plt.figure()
    plt.plot(list(epochs), stats.train_kl, label="train_kl")
    plt.plot(list(epochs), stats.val_kl, label="val_kl")
    plt.legend()
    plt.title("KL Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_kl.png"))
    plt.close()


# -----------------------------
# Main
# -----------------------------
def build_transforms(image_size: int, augment: bool):
    # MNIST -> 1x28x28, we resize to 128x128.
    # Augmentations are mild but useful even for MNIST: affine/rotation/perspective/erasing.
    if augment:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=20,
                    translate=(0.10, 0.10),
                    scale=(0.85, 1.15),
                    shear=10
                )
            ], p=0.9),
            transforms.RandomApply([transforms.RandomRotation(20)], p=0.7),
            transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.2, p=1.0)], p=0.3),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25, scale=(0.01, 0.08), ratio=(0.3, 3.3), value=0.0),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])


def main():
    parser = argparse.ArgumentParser(description="Conv VAE on MNIST (128x128), latent_dim small (default=2).")
    parser.add_argument("--data-dir", type=str, default="./data", help="Where to download/store MNIST.")
    parser.add_argument("--out-dir", type=str, default="./example_gen", help="Output dir for samples/plots.")
    parser.add_argument("--image-size", type=int, default=128, help="Image size (use 128 as requested).")
    parser.add_argument("--latent-dim", type=int, default=2, help="Latent dimension (small, e.g., 2).")
    parser.add_argument("--base-ch", type=int, default=32, help="Base channels for conv net width.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--beta", type=float, default=1e-3, help="KL weight (small -> better recon).")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--no-aug", action="store_true", help="Disable training augmentation.")
    parser.add_argument("--n-gen", type=int, default=64, help="How many images to generate at the end.")
    parser.add_argument("--show", action="store_true", help="If set, show generated grid via matplotlib.")
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")

    # Data
    train_tf = build_transforms(args.image_size, augment=(not args.no_aug))
    test_tf = build_transforms(args.image_size, augment=False)

    train_ds = torchvision.datasets.MNIST(root=args.data_dir, train=True, download=True, transform=train_tf)
    test_ds = torchvision.datasets.MNIST(root=args.data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # Model
    model = ConvVAE(latent_dim=args.latent_dim, in_channels=1, base_ch=args.base_ch, image_size=args.image_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)

    stats = TrainStats(train_total=[], train_recon=[], train_kl=[],
                       val_total=[], val_recon=[], val_kl=[])

    # Train
    for epoch in range(1, args.epochs + 1):
        tr_tot, tr_rec, tr_kl = train_one_epoch(model, train_loader, opt, device, args.beta)
        va_tot, va_rec, va_kl = eval_one_epoch(model, test_loader, device, args.beta)

        stats.train_total.append(tr_tot)
        stats.train_recon.append(tr_rec)
        stats.train_kl.append(tr_kl)

        stats.val_total.append(va_tot)
        stats.val_recon.append(va_rec)
        stats.val_kl.append(va_kl)

        print(f"[epoch {epoch:03d}/{args.epochs}] "
              f"train: total={tr_tot:.4f} recon={tr_rec:.4f} kl={tr_kl:.4f} | "
              f"val: total={va_tot:.4f} recon={va_rec:.4f} kl={va_kl:.4f}")

        # Save recon examples each epoch (optional but useful)
        save_recon_examples(model, test_loader, device, args.out_dir, max_items=16, fname=f"recon_epoch_{epoch:03d}.png")

    # Save curves
    plot_curves(stats, args.out_dir)

    # Final recon grid
    save_recon_examples(model, test_loader, device, args.out_dir, max_items=16, fname="recon_final_grid.png")

    # Generate and save images
    grid_path = generate_and_save(
        model=model,
        device=device,
        latent_dim=args.latent_dim,
        image_size=args.image_size,
        out_dir=args.out_dir,
        n_gen=args.n_gen,
        grid_nrow=int(math.sqrt(args.n_gen)) if int(math.sqrt(args.n_gen))**2 == args.n_gen else 8,
        prefix="generated"
    )
    print(f"[info] saved generated grid: {grid_path}")

    # If latent_dim == 2, also generate a manifold grid
    if args.latent_dim == 2:
        manifold_path = generate_latent_manifold_2d(model, device, args.out_dir, grid_size=20, span=2.5)
        print(f"[info] saved latent manifold: {manifold_path}")

    # Optionally show the generated grid
    if args.show:
        img = torchvision.io.read_image(grid_path).permute(1, 2, 0).numpy()
        plt.figure()
        plt.imshow(img)
        plt.axis("off")
        plt.title("Generated samples")
        plt.show()

    # Save model weights
    ckpt_path = os.path.join(args.out_dir, "vae_mnist_128.pt")
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, ckpt_path)
    print(f"[info] saved checkpoint: {ckpt_path}")
    print(f"[done] outputs in: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()


#vae mnist python vae_mnist_128.py --epochs 20 --latent-dim 2 --image-size 128 --batch-size 128 --lr 2e-4 --beta 1e-3 --n-gen 64 --out-dir ./example_gen --show
