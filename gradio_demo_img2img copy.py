import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import gradio as gr
from diffusers import AutoencoderKL
from pipeline_demofusion_sdxl1_copy_2 import DemoFusionSDXLPipeline
from gradio_imageslider import ImageSlider
import torch, gc
from torchvision import transforms
from PIL import Image
from diffusers.schedulers import KarrasDiffusionSchedulers

from diffusers.models import AutoencoderKL, UNet2DConditionModel

# Define the Dataset class
class ImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# Configuration
batch_size = 1
num_epochs = 5
learning_rate = 1e-4
output_type = "pil"
height = 2048
width = 2048
num_inference_steps = 50
guidance_scale = 5.0
view_batch_size = batch_size
stride = 64
sigma = 1.0
lowvram = True

# Device configuration
transform = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

import torch
from torchvision import models




# Now ResNet50 is ready for feature extraction


# Dataset and DataLoader
dataset = ImageDataset(image_folder=r"D:\Dataset-1\dataset\DIV2K_train_LR_bicubic\X2", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
# Initialize your pipeline with VAE and clip (passing both into the pipeline)
model = DemoFusionSDXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    vae=vae,
    torch_dtype=torch.float16)
    # unet= UNet2DConditionModel,
    # Scheduler = KarrasDiffusionSchedulers)
    # model.lowvram = True




# Initialize the optimizer with parameters from each component that requires training
optimizer = torch.optim.Adam(
    list(model.vae.parameters()) + list(model.unet.parameters()),  # Include other components as needed
    lr=learning_rate
)

# Define optimizer and loss function
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()
seed= 2013
generator = torch.Generator(device='cuda')
generator = generator.manual_seed(int(seed))
cosine_scale_1 = 3
cosine_scale_2 = 1
cosine_scale_3 = 1
sigma = 0.8

# Training loop
for epoch in range(num_epochs):
    # model.train()
    epoch_loss = 0

    for batch_idx, images in enumerate(dataloader):
        
       
        # Forward pass through the model
        outputs = model.train(
            # prompt=None,  # No prompt as per the model requirements
            image_lr = images.to(torch.float16),# Low-resolution images
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            # num_images_per_prompt=1,
            output_type=output_type,
            return_dict=False,
            view_batch_size=view_batch_size,
            stride=stride,
            sigma=sigma,
            lowvram=lowvram,
            generator = generator,
            cosine_scale_1 = cosine_scale_1,
            cosine_scale_2 = cosine_scale_2,
            cosine_scale_3= cosine_scale_3,


        )

        # Obtain the generated images from the model's output
        generated_images = outputs[-1]  # Assuming the final generated image is in the last position

        # Calculate the loss between generated images and input low-resolution images
        loss = loss_fn(generated_images, images)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Print progress
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item()}")

    print(f"Epoch [{epoch+1}/{num_epochs}] completed with average loss: {epoch_loss / len(dataloader)}")

print("Training finished.")
