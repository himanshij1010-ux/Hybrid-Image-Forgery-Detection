
import os
import cv2

# Dataset paths
REAL_DIR = "dataset/real/Au"
FAKE_DIR = "dataset/fake/Tp"

# Supported image formats
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Check folders
print("Real folder exists:", os.path.exists(REAL_DIR))
print("Fake folder exists:", os.path.exists(FAKE_DIR))


# Get only image files
real_images = [
    f for f in os.listdir(REAL_DIR)
    if os.path.isfile(os.path.join(REAL_DIR, f))
    and f.lower().endswith(IMAGE_EXTENSIONS)
]

fake_images = [
    f for f in os.listdir(FAKE_DIR)
    if os.path.isfile(os.path.join(FAKE_DIR, f))
    and f.lower().endswith(IMAGE_EXTENSIONS)
]


# Print image counts
print("Number of real images:", len(real_images))
print("Number of fake images:", len(fake_images))


# Test one real image
if len(real_images) > 0:
    real_path = os.path.join(REAL_DIR, real_images[0])
    real_img = cv2.imread(real_path)

    if real_img is not None:
        print("Real image loaded successfully")
        print("Real image:", real_images[0])
        print("Real image size:", real_img.shape)
    else:
        print("Could not load real image")


# Test one fake image
if len(fake_images) > 0:
    fake_path = os.path.join(FAKE_DIR, fake_images[0])
    fake_img = cv2.imread(fake_path)

    if fake_img is not None:
        print("Fake image loaded successfully")
        print("Fake image:", fake_images[0])
        print("Fake image size:", fake_img.shape)
    else:
        print("Could not load fake image")
