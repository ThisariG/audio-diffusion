import argparse
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from datasets import load_dataset, load_from_disk
from diffusers.pipelines.audio_diffusion import Mel
from ldm.util import instantiate_from_config
from librosa.util import normalize
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from audiodiffusion.utils import convert_ldm_to_hf_vae

class AudioDiffusion(Dataset):
    def _init_(self, model_id, channels=3):
        super()._init_()
        self.channels = channels
        if os.path.exists(model_id):
            self.hf_dataset = load_from_disk(model_id)["train"]
        else:
            self.hf_dataset = load_dataset(model_id)["train"]

    def _len_(self):
        return len(self.hf_dataset)

    def _getitem_(self, idx):
        image = self.hf_dataset[idx]["image"]
        if self.channels == 3:
            image = image.convert("RGB")
        image = np.frombuffer(image.tobytes(), dtype="uint8").reshape((image.height, image.width, self.channels))
        image = (image / 255) * 2 - 1
        return {"image": image}
        
class AudioDiffusionDataModule(pl.LightningDataModule):
    def __init__(self, model_id, batch_size, channels):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = AudioDiffusion(model_id=model_id, channels=channels)
        self.num_workers = 1

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)

class YourLightningModule(pl.LightningModule):
    def __init__(self, ldm_config, hf_checkpoint, args):
        super().__init__()
        self.ldm_config = ldm_config
        self.hf_checkpoint = hf_checkpoint
        self.args = args

        # Define your model here
        self.model = instantiate_from_config(ldm_config.model)
        self.model.learning_rate = ldm_config.model.base_learning_rate

        # Create AudioDiffusionDataModule
        self.data = AudioDiffusionDataModule(
            model_id=args.dataset_name,
            batch_size=args.batch_size,
            channels=ldm_config.model.params.ddconfig.in_channels,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Your training logic here
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.model.learning_rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE using ldm")
    parser.add_argument("-d", "--dataset_name", type=str, default=None)
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("-c", "--ldm_config_file", type=str, default="config/ldm_autoencoder_kl.yaml")
    parser.add_argument("--ldm_checkpoint_dir", type=str, default="models/ldm-autoencoder-kl")
    parser.add_argument("--hf_checkpoint_dir", type=str, default="models/autoencoder-kl")
    parser.add_argument("-r", "--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("-g", "--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--save_images_batches", type=int, default=1000)
    parser.add_argument("--max_epochs", type=int, default=100)
    args = parser.parse_args()

    config = OmegaConf.load(args.ldm_config_file)
    lightning_module = YourLightningModule(config, args.hf_checkpoint_dir, args)

    trainer = pl.Trainer(
        accelerator="ddp",  # Use Distributed Data Parallel for multi-GPU training
        max_epochs=args.max_epochs,
        resume_from_checkpoint=args.resume_from_checkpoint,
        gpus=2,  # Number of GPUs to use (or use gpus=1 for a single GPU)
        callbacks=[
            ImageLogger(
                every=args.save_images_batches,
                hop_length=args.hop_length,
                sample_rate=args.sample_rate,
                n_fft=args.n_fft,
            ),
            ModelCheckpoint(
                dirpath=args.ldm_checkpoint_dir,
                filename="{epoch:06}",
                verbose=True,
                save_last=True,
            ),
        ],
    )

    trainer.fit(lightning_module)
