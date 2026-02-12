import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import argparse
import os
import math
import kagglehub
from pathlib import Path

# Import metric libraries
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

# ==========================================
# 1. Configuration
# ==========================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using device: Apple Silicon MPS")
else:
    device = torch.device("cpu")
    print("⚠️ Using device: CPU")

IMG_SIZE = 64
BATCH_SIZE = 64
LATENT_DIM = 128
EPOCHS = 50
NUM_CLASSES = 2
MODEL_PATH = "cvae_strict.pth"

# ==========================================
# 2. Data Preparation
# ==========================================
def get_dataloaders():
    """
    Functionality:
    1) Download/locate the Chest X-ray dataset directory
    2) Construct Training and Testing DataLoaders

    Returns:
    - train_loader: For training, shuffle=True
    - test_loader: For evaluation, shuffle=False
    """
    print("📥 Downloading/Loading dataset...")
    root_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    
    # Path compatibility handling
    possible_train_dirs = [
        os.path.join(root_path, "chest_xray", "train"),
        os.path.join(root_path, "chest_xray", "chest_xray", "train"),
        os.path.join(root_path, "train")
    ]
    train_dir = next((d for d in possible_train_dirs if os.path.exists(d)), None)
    if train_dir is None: raise FileNotFoundError("❌ 'train' folder not found")

    test_dir = train_dir.replace("train", "test")
    if not os.path.exists(test_dir): test_dir = train_dir.replace("train", "val")

    # Unify input specifications:
    # - Resize to 64x64
    # - Force grayscale (single channel)
    # - Convert to tensor, pixel range [0, 1]
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=1), 
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    
    return train_loader, test_loader


def parse_args():
    """
    Optional runtime arguments:
    - --target: Generation target category (normal / pneumonia / both)
    - --num-gen: Number of images to generate per category
    - --save-dir: Directory to save real/generated/comparison images
    - --no-show: Do not display images in pop-up (save only)
    """
    parser = argparse.ArgumentParser(description="CVAE for Chest X-ray generation and evaluation")
    parser.add_argument("--target", type=str, default="both", choices=["normal", "pneumonia", "both"])
    parser.add_argument("--num-gen", type=int, default=8)
    parser.add_argument("--save-dir", type=str, default="outputs")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def resolve_target_class_ids(class_to_idx, target):
    """
    Maps user-input target categories (normal/pneumonia/both) to dataset class_ids.
    """
    # Original class_to_idx is usually {"NORMAL": 0, "PNEUMONIA": 1}
    lower_map = {name.lower(): idx for name, idx in class_to_idx.items()}
    alias_map = dict(lower_map)
    for name, idx in lower_map.items():
        if "normal" in name:
            alias_map["normal"] = idx
        if "pneumonia" in name:
            alias_map["pneumonia"] = idx

    if target == "both":
        needed = ["normal", "pneumonia"]
    else:
        needed = [target]

    missing = [name for name in needed if name not in alias_map]
    if missing:
        raise ValueError(f"Category not found in dataset: {missing}. Available classes: {list(class_to_idx.keys())}")

    # Remove duplicates while maintaining order
    result = []
    for name in needed:
        idx = alias_map[name]
        if idx not in result:
            result.append(idx)
    return result


def collect_real_images_by_class(dataloader, class_id, num_images):
    """
    Extracts real images of a specific class from the dataloader (returns CPU tensor).
    """
    collected = []
    total = 0
    for imgs, labels in dataloader:
        mask = labels == class_id
        if mask.any():
            selected = imgs[mask]
            collected.append(selected)
            total += selected.size(0)
            if total >= num_images:
                break

    if not collected:
        raise RuntimeError(f"No real samples found in dataset for class_id={class_id}.")

    return torch.cat(collected, dim=0)[:num_images]


@torch.no_grad()
def generate_images_by_class(model, class_id, num_images, latent_dim, device):
    """
    Generates images based on specified class:
    z ~ N(0, I), labels fixed to class_id
    """
    z = torch.randn(num_images, latent_dim, device=device)
    labels = torch.full((num_images,), class_id, dtype=torch.long, device=device)
    fake = model.decode(z, labels).cpu()
    return fake


def save_image_tensors(images, out_dir, prefix):
    """
    Saves [N, 1, H, W] grayscale images as PNG files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(images.size(0)):
        out_path = out_dir / f"{prefix}_{i:03d}.png"
        plt.imsave(out_path, images[i].squeeze(0).numpy(), cmap="gray", vmin=0.0, vmax=1.0)


def plot_training_loss(loss_history, out_path, show_image=True):
    """
    Plots and saves the training loss curve.
    """
    if not loss_history:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("CVAE Training Loss Curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    if show_image:
        plt.show()
    else:
        plt.close(fig)


def plot_real_vs_generated(real_imgs, fake_imgs, class_name, out_path, show_image=True):
    """
    Draws a 2-row comparison plot with labels:
    - Row 1: REAL
    - Row 2: GENERATED (CVAE)
    """
    n = min(real_imgs.size(0), fake_imgs.size(0))
    if n <= 0:
        raise ValueError("No images available for comparison display.")

    fig, axs = plt.subplots(2, n, figsize=(1.8 * n, 4))
    if n == 1:
        axs = axs.reshape(2, 1)

    for i in range(n):
        axs[0, i].imshow(real_imgs[i].squeeze(), cmap="gray")
        axs[0, i].set_title("REAL", fontsize=9)
        axs[0, i].axis("off")

        axs[1, i].imshow(fake_imgs[i].squeeze(), cmap="gray")
        axs[1, i].set_title("GENERATED", fontsize=9)
        axs[1, i].axis("off")

    fig.suptitle(f"Class: {class_name} | Top: REAL | Bottom: GENERATED (CVAE)", fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    if show_image:
        plt.show()
    else:
        plt.close(fig)

# ==========================================
# 3. CVAE Model
# ==========================================
class CVAE(nn.Module):
    def __init__(self, latent_dim=128, num_classes=2):
        """
        Conditional VAE:
        - Condition is the class label (NORMAL / PNEUMONIA)
        - Encoder input: Image + Label one-hot plane
        - Decoder input: Latent variable z + Label one-hot vector
        """
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Encoder
        self.enc1 = nn.Conv2d(1 + num_classes, 32, 4, 2, 1)  
        self.enc2 = nn.Conv2d(32, 64, 4, 2, 1)               
        self.enc3 = nn.Conv2d(64, 128, 4, 2, 1)              
        self.enc4 = nn.Conv2d(128, 256, 4, 2, 1)             
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

        # Decoder
        self.dec_input = nn.Linear(latent_dim + num_classes, 256 * 4 * 4)
        self.dec1 = nn.ConvTranspose2d(256, 128, 4, 2, 1) 
        self.dec2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)  
        self.dec3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)   
        self.dec4 = nn.ConvTranspose2d(32, 1, 4, 2, 1)    

    def one_hot_image(self, labels, b, h, w):
        """
        Expands class labels into a condition map with the same height and width as the image:
        labels: [B] -> [B, num_classes, H, W]
        """
        y = F.one_hot(labels, self.num_classes).view(b, self.num_classes, 1, 1)
        return y.expand(b, self.num_classes, h, w).float().to(labels.device)

    def encode(self, x, labels):
        """
        Encoder:
        Input x: [B, 1, 64, 64], labels: [B]
        Output:
        - mu:     [B, latent_dim]
        - logvar: [B, latent_dim]
        """
        labels_img = self.one_hot_image(labels, x.size(0), x.size(2), x.size(3))
        x = torch.cat([x, labels_img], dim=1)
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick:
        z = mu + eps * std, eps ~ N(0, I)
        Enables the sampling step to be differentiable for backpropagation.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        """
        Decoder:
        Input:
        - z: [B, latent_dim]
        - labels: [B]
        Output:
        - Reconstructed image: [B, 1, 64, 64], pixel range [0, 1]
        """
        labels_vec = F.one_hot(labels, self.num_classes).float().to(z.device)
        z = torch.cat([z, labels_vec], dim=1)
        x = self.dec_input(z)
        x = x.view(-1, 256, 4, 4)
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        return torch.sigmoid(self.dec4(x))

    def forward(self, x, labels):
        """
        Forward pass:
        x -> (mu, logvar) -> z -> recon_x
        Returns recon_x, mu, logvar for the loss function.
        """
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    """
    CVAE Loss = Reconstruction Error + KL Divergence Regularization
    - BCE: Difference between reconstructed and original image
    - KLD: Distance between posterior q(z|x) and prior N(0, I)
    """
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# ==========================================
# 4. Main Program
# ==========================================
if __name__ == "__main__":
    args = parse_args()
    save_root = Path(args.save_dir)

    # Step A: Prepare data and model
    train_loader, test_loader = get_dataloaders()
    model = CVAE(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_history = []

    # Step B: Always train from scratch for 50 epochs, do not use old weights
    if os.path.exists(MODEL_PATH):
        print(f"ℹ️ Existing model detected, will be overwritten after training: {MODEL_PATH}")

    print(f"🚀 Starting training from scratch (Epochs={EPOCHS})...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(data, labels)
            loss = loss_function(recon, data, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Average loss by total samples in dataset to observe convergence
        avg_loss = train_loss / len(train_loader.dataset)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Avg Loss: {avg_loss:.2f}")

    torch.save(model.state_dict(), MODEL_PATH)
    loss_curve_path = save_root / "training_loss_curve.png"
    plot_training_loss(loss_history, loss_curve_path, show_image=not args.no_show)
    print(f"📉 Training loss curve saved: {loss_curve_path}")

    # --- Evaluation Phase ---
    model.eval()

    # 1) Generate images for specified classes and plot with real image comparisons
    print("🎨 Generating and labeling REAL / GENERATED images...")
    class_to_idx = train_loader.dataset.class_to_idx
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    target_ids = resolve_target_class_ids(class_to_idx, args.target)

    for class_id in target_ids:
        class_name = idx_to_class.get(class_id, f"class_{class_id}")
        real_imgs = collect_real_images_by_class(test_loader, class_id, args.num_gen)
        fake_imgs = generate_images_by_class(model, class_id, args.num_gen, LATENT_DIM, device)

        # Save original and generated images with clear real/fake prefixes
        save_image_tensors(real_imgs, save_root / "real" / class_name.lower(), "real")
        save_image_tensors(fake_imgs, save_root / "generated" / class_name.lower(), "fake")

        compare_path = save_root / "comparisons" / f"real_vs_generated_{class_name.lower()}.png"
        plot_real_vs_generated(
            real_imgs,
            fake_imgs,
            class_name=class_name,
            out_path=compare_path,
            show_image=not args.no_show,
        )

        print(f"   Class: {class_name}")
        print(f"   REAL save directory      : {save_root / 'real' / class_name.lower()}")
        print(f"   GENERATED save directory : {save_root / 'generated' / class_name.lower()}")
        print(f"   Comparison plot          : {compare_path}")

    # 2) Calculate FID:
    # - real: Real images from the test set
    # - fake: Reconstructed images from the same test set batch
    # Note: Using a simplified 64-dim version for speed; not directly comparable to standard 2048-dim FID.
    print("📊 Calculating FID Metric...")
    fid_metric = FrechetInceptionDistance(feature=64).to("cpu")
    
    with torch.no_grad():
        # Limit to 10 batches for quick demonstration
        for i, (real_data, labels) in enumerate(test_loader):
            if i >= 10: break 
            real_data, labels = real_data.to(device), labels.to(device)
            fake_data, _, _ = model(real_data, labels)
            
            # FID/IS expects uint8 RGB; replicate single-channel grayscale to 3 channels
            def preprocess(img):
                img = (img * 255).clamp(0, 255).to(torch.uint8)
                return img.repeat(1, 3, 1, 1).to("cpu")

            fid_metric.update(preprocess(real_data), real=True)
            fid_metric.update(preprocess(fake_data), real=False)
            
    fid_score = fid_metric.compute()
    print(f"✅ FID Score (64-dim): {fid_score:.4f}")

    # 3) Calculate Inception Score:
    # Sample from prior z~N(0, I) to measure sample diversity and classifiability
    print("📊 Calculating Inception Score...")
    is_metric = InceptionScore().to("cpu")
    with torch.no_grad():
        # Generate 1000 images for a more stable IS
        z_large = torch.randn(1000, LATENT_DIM).to(device)
        l_large = torch.randint(0, NUM_CLASSES, (1000,)).to(device)
        gen_imgs = model.decode(z_large, l_large)
        
        # Transform and update
        is_metric.update((gen_imgs * 255).clamp(0, 255).to(torch.uint8).repeat(1, 3, 1, 1).to("cpu"))
        
    is_mean, is_std = is_metric.compute()
    print(f"✅ Inception Score: {is_mean:.4f} ± {is_std:.4f}")