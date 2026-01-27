import cv2
import numpy as np
from ultralytics import YOLO

# Settings
MODEL_PATH = "C:\\Users\\dolap\\OneDrive\\Documents\\DOLAPO\\data-analysis\\photoquadrats_analysis\\model\\best.pt"
IMAGE_PATH = "C:\\Users\\dolap\\OneDrive\\Documents\\DOLAPO\\data-analysis\\photoquadrats_analysis\\img\\sample_img.jpg"
QUADRAT_SIZE_CM = 20  # 20x20 cm
OUTPUT_VIZ = "quadrat_analysis.png"

# Load model and image
model = YOLO(MODEL_PATH)
results = model(IMAGE_PATH, imgsz=640)[0]
orig_img = results.orig_img
H, W = orig_img.shape[:2]

# Calculate area conversion
quadrat_area_cm2 = QUADRAT_SIZE_CM * QUADRAT_SIZE_CM
cm2_per_pixel = quadrat_area_cm2 / (H * W)

# Process detections
species_data = {}

for cls_id, mask, conf in zip(results.boxes.cls, results.masks.data, results.boxes.conf):
    species = model.names[int(cls_id)]
    confidence = float(conf)
    
    # Resize mask to image resolution
    mask_resized = cv2.resize(
        mask.cpu().numpy(),
        (W, H),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    
    if species not in species_data:
        species_data[species] = {
            'mask': np.zeros((H, W), dtype=bool),
            'detections': []
        }
    
    species_data[species]['mask'] |= mask_resized
    species_data[species]['detections'].append(confidence)

# Calculate coverage and prepare results
results_dict = {}
print("\n=== Species Coverage ===")

for species, data in species_data.items():
    pixels = data['mask'].sum()
    area_cm2 = pixels * cm2_per_pixel
    coverage_pct = (area_cm2 / quadrat_area_cm2) * 100
    confidences = data['detections']
    
    results_dict[species] = {
        'coverage_percent': coverage_pct,
        'confidences': confidences,
        'area_cm2': area_cm2,
        'num_detections': len(confidences)
    }
    
    conf_str = ", ".join([f"{c:.3f}" for c in confidences])
    print(f"{species:20s}: {coverage_pct:6.2f} % | detections: {confidences} | count: {len(confidences)}")

# Bare space
total_coverage = sum(r['coverage_percent'] for r in results_dict.values())
bare_pct = 100 - total_coverage
print(f"{'Bare/Other':20s}: {bare_pct:6.2f} %\n")

# Create visualization
viz = orig_img.copy()
colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

for i, (species, data) in enumerate(species_data.items()):
    color = colors[i % len(colors)]
    colored = np.zeros_like(viz)
    colored[data['mask']] = color
    viz = cv2.addWeighted(viz, 0.7, colored, 0.3, 0)

# Add legend
y = 30
for i, species in enumerate(species_data.keys()):
    color = colors[i % len(colors)]
    cv2.rectangle(viz, (10, y - 15), (25, y), color, -1)
    cv2.putText(viz, f"{species} ({results_dict[species]['coverage_percent']:.1f}%)", 
               (35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y += 30

cv2.imwrite(OUTPUT_VIZ, viz)
print(f"Visualization saved to {OUTPUT_VIZ}")

# Return results
print("\n=== Results Summary ===")
for species, data in results_dict.items():
    print(f"\n{species}:")
    print(f"  Coverage: {data['coverage_percent']:.2f}%")
    print(f"  Area: {data['area_cm2']:.2f} cm²")
    print(f"  Detections: {data['num_detections']}")
    print(f"  Confidences: {data['confidences']}")