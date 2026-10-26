import os
import shutil

# Paths
real_folder = "dataset/real/Au"
fake_folder = "dataset/fake/Tp"

# Output folders
real_output = "dataset/processed/real"
fake_output = "dataset/processed/fake"

# Create output folders
os.makedirs(real_output, exist_ok=True)
os.makedirs(fake_output, exist_ok=True)

# Copy real images
for file in os.listdir(real_folder):
    source = os.path.join(real_folder, file)
    destination = os.path.join(real_output, file)

    if os.path.isfile(source):
        shutil.copy2(source, destination)

# Copy fake images
for file in os.listdir(fake_folder):
    source = os.path.join(fake_folder, file)
    destination = os.path.join(fake_output, file)

    if os.path.isfile(source):
        shutil.copy2(source, destination)

print("Dataset preparation completed!")
print("Real images:", len(os.listdir(real_output)))
print("Fake images:", len(os.listdir(fake_output)))