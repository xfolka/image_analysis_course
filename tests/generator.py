import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from skimage import io
from datetime import datetime

import os

# Load dataset
#df = pd.read_csv("tests/bacteria_drug_experiment_separated.csv")
df = pd.read_csv("tests/bacterial_colonies_area_gaussian_radius_clustered_40rep.csv")

# Output folder
#output_dir = "student folder/1. exercises/data/bacteria"
# Get current date and time
now = datetime.now()

# Format it as a string
timestamp_str = now.strftime("%Y-%m-%d_%H_%M_%S")
output_dir = "tests/bacteria_images" + "_" + timestamp_str
os.makedirs(output_dir, exist_ok=True)

# Image size
img_size =720

# Function to draw an irregular colony
def draw_colony(img, center, base_radius, intensity=0.7):
    cx, cy = center
    angles = np.linspace(0, 2*np.pi, 100)
    for theta in angles:
        r = base_radius
        y, x = np.ogrid[:img.shape[0], :img.shape[1]]
        mask = (x - cx)**2 + (y - cy)**2 <= r**2
        img[mask] = intensity + np.random.normal(-0.02, 0.09)
        return img

def circle_array(img, center, radius, filled=False):
    # Define image size (make it large enough to contain the circle)
    size = radius * 2 + 1
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    
    # Equation of a circle
    mask = x**2 + y**2 <= radius**2 if filled else np.isclose(x**2 + y**2, radius**2, atol=radius*0.5)
    
    # Create an array (0 = black, 1 = white)
    img = np.zeros((size, size), dtype=np.uint8)
    img[mask] = 255
    
    return img

# Function to draw an irregular colony
def draw_irregular_colony(img, center, base_radius, intensity=0.7, irregularity=0.3):
    cx, cy = center
    angles = np.linspace(0, 2*np.pi, 100)
    for theta in angles:
        r = base_radius * (1 + irregularity * np.random.uniform(-1, 1))
        y, x = np.ogrid[:img.shape[0], :img.shape[1]]
        mask = (x - cx)**2 + (y - cy)**2 <= r**2
        img[mask] = intensity
    return img

# Function to place colonies without overlap
def place_colonies(radii, img_size, margin=40, max_attempts=2000):
    centers = []
    for r in radii:
        for _ in range(max_attempts):
            x, y = np.random.randint(margin, img_size - margin, size=2)
            if all((x - cx)**2 + (y - cy)**2 >= (r + cr + margin)**2 for cx, cy, cr in centers):
                centers.append((x, y, r))
                break
    return centers

# Function to generate uneven lighting
def uneven_lighting(img_size):
    gradient_x = np.tile(np.linspace(0.7, 0.92, img_size), (img_size, 1))
    gradient_y = np.tile(np.linspace(0.75, 0.92, img_size), (img_size, 1)).T
    return gradient_x * gradient_y

# Function to add salt & pepper noise
def add_salt_pepper(img, amount):
    noisy = img.copy()
    num_pixels = img.size
    num_salt = int(amount * num_pixels)
    num_pepper = int(amount * num_pixels)
    
    # Salt noise (bright spots)
    coords = (np.random.randint(0, img.shape[0], num_salt),
              np.random.randint(0, img.shape[1], num_salt))
    noisy[coords] = 1.0
    
    # Pepper noise (dark spots)
    coords = (np.random.randint(0, img.shape[0], num_pepper),
              np.random.randint(0, img.shape[1], num_pepper))
    noisy[coords] = 0.0
    
    return noisy

# Loop over each bacteria-drug group
for (b, d), group in df.groupby(["bacteria", "drug"]):

    # Start with uneven lighting background
    lighting = uneven_lighting(img_size)
    img = np.ones((img_size, img_size)) * lighting
    
    # Get radii for this group
    radii = [int(max(3, r)) for r in group["radius"]]
    
    # Place colonies without overlap
    colonies = place_colonies(radii, img_size)
    
    # Draw irregular colonies
    for x, y, r in colonies:
        img = draw_colony(img, (x, y), r, intensity=0.995)
    
    # Add Gaussian noise
    noise = np.random.normal(0, 0.05, img.shape)
    noisy_img = np.clip(img + noise, 0, 1)
    
    # Add salt & pepper noise
    final_img = add_salt_pepper(noisy_img, amount=0.001)
    
     # Convert to 8-bit RGB (strictly 3 channels)
    rgb_img = (final_img * 255).astype(np.uint8)
    rgb_img = np.stack([rgb_img]*3, axis=-1)  # shape (H, W, 3)
    
    # Embed metadata as JSON in TIFF tag 270 (ImageDescription) via the tifffile plugin
    desc_json = f'{{"bacteria":"{b}","drug":"{d}"}}'
    io.imsave(
        os.path.join(output_dir, f"{b}_{d}.tiff"),
        rgb_img,
        plugin="tifffile",
        photometric="rgb",       # ensure correct RGB interpretation
        description=desc_json    # goes into TIFF tag 270 (ImageDescription)    
    )
    
    # Save as RGB TIFF (no alpha)
    #io.imsave(os.path.join(output_dir, f"{b}_{d}.tiff"), rgb_img)
    
  
print(f"Images saved in folder: {output_dir}")
