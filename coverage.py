from ultralytics import YOLO
import numpy as np

PATH_TO_MODEL = r"C:\Users\dolap\OneDrive\Documents\DOLAPO\data-analysis\photoquadrats_analysis\model\customV8.pt"
PATH_TO_IMAGE = r"C:\Users\dolap\OneDrive\Documents\DOLAPO\data-analysis\photoquadrats_analysis\quadrat.jpg"

model = YOLO(PATH_TO_MODEL)
results = model(PATH_TO_IMAGE)[0]

H, W = results.orig_img.shape[:2]

# Quadrat metadata
quadrat_width_cm = 50
quadrat_height_cm = 50
quadrat_area_cm2 = quadrat_width_cm * quadrat_height_cm

pixels_per_cm2 = (H * W) / quadrat_area_cm2

species_masks = {}

for cls_id, mask in zip(results.boxes.cls, results.masks.data):
    species = model.names[int(cls_id)]

    if species not in species_masks:
        species_masks[species] = np.zeros((H, W), dtype=bool)

    species_masks[species] |= mask.cpu().numpy().astype(bool)

for species, mask in species_masks.items():
    species_pixels = mask.sum()
    species_area_cm2 = species_pixels / pixels_per_cm2
    percent_cover = (species_area_cm2 / quadrat_area_cm2) * 100

    print(species, f"{percent_cover:.2f}%")
