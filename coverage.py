import cv2
import numpy as np
from ultralytics import YOLO

# ------------------------
# User settings
# ------------------------
MODEL_PATH = "best-seg.pt"
IMAGE_PATH = "quadrat.jpg"

# Quadrat real dimensions (cm)
QUADRAT_WIDTH_CM = 50
QUADRAT_HEIGHT_CM = 50

# ------------------------
# Load model & image
# ------------------------
model = YOLO(MODEL_PATH)
results = model(IMAGE_PATH)[0]

orig_img = results.orig_img
H, W = orig_img.shape[:2]

# ------------------------
# Quadrat scaling
# ------------------------
quadrat_area_cm2 = QUADRAT_WIDTH_CM * QUADRAT_HEIGHT_CM
total_pixels = H * W
cm2_per_pixel = quadrat_area_cm2 / total_pixels

# ------------------------
# Prepare species masks
# ------------------------
species_masks = {}   # {species_name: full_res_bool_mask}

# ------------------------
# Loop through YOLO detections
# ------------------------
for cls_id, mask in zip(results.boxes.cls, results.masks.data):
    species = model.names[int(cls_id)]

    raw_mask = mask.cpu().numpy()

    # Resize YOLO mask to original image resolution
    full_mask = cv2.resize(
        raw_mask,
        (W, H),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    if species not in species_masks:
        species_masks[species] = np.zeros((H, W), dtype=bool)

    # Merge mask into species total (pixel-wise OR)
    species_masks[species] |= full_mask

# ------------------------
# Calculate % cover per species
# ------------------------
print("\n=== Species % Cover ===")

for species, mask in species_masks.items():
    species_pixels = mask.sum()
    species_area_cm2 = species_pixels * cm2_per_pixel
    percent_cover = (species_area_cm2 / quadrat_area_cm2) * 100

    print(f"{species:15s}: {percent_cover:6.2f} %")

# ------------------------
# Optional: bare space / unclassified
# ------------------------
all_species_mask = np.zeros((H, W), dtype=bool)
for mask in species_masks.values():
    all_species_mask |= mask

bare_pixels = (~all_species_mask).sum()
bare_area_cm2 = bare_pixels * cm2_per_pixel
bare_percent = (bare_area_cm2 / quadrat_area_cm2) * 100

print(f"{'Bare/Other':15s}: {bare_percent:6.2f} %")
