import os
import random
import shutil

# Step 1: Define source and destination folders
source_folder = r"D:\\fashion_images"
destination_folder = "images"
os.makedirs(destination_folder, exist_ok=True)

# Step 2: List all image files in the source folder
all_images = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

# Step 3: Select a subset (e.g., 100 random images)
subset_size = 500
subset_images = random.sample(all_images, min(subset_size, len(all_images)))

# Step 4: Copy the subset to the destination folder
for image in subset_images:
    shutil.copy(os.path.join(source_folder, image), os.path.join(destination_folder, image))

print(f"Copied {len(subset_images)} images to {destination_folder}.")
