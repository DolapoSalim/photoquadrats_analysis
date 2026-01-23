import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# ========================
# User settings
# ========================
MODEL_PATH = "best-seg.pt"
IMAGE_PATH = "quadrat.jpg"
QUADRAT_WIDTH_CM = 20
QUADRAT_HEIGHT_CM = 20
CONFIDENCE_THRESHOLD = 0.35  # Filter detections below this confidence
OUTPUT_VISUALIZATION = "quadrat_analysis.png"  # Set to None to skip

# ========================
# Validation & Setup
# ========================
def validate_inputs(model_path, image_path, width_cm, height_cm):
    """Validate that all inputs exist and are valid."""
    errors = []
    
    if not Path(model_path).exists():
        errors.append(f"Model file not found: {model_path}")
    
    if not Path(image_path).exists():
        errors.append(f"Image file not found: {image_path}")
    
    if width_cm <= 0 or height_cm <= 0:
        errors.append(f"Quadrat dimensions must be positive (got {width_cm}x{height_cm} cm)")
    
    if errors:
        for error in errors:
            logger.error(error)
        return False
    
    return True

# ========================
# Load model & image
# ========================
def load_model_and_image(model_path, image_path):
    """Load YOLO model and run inference."""
    try:
        logger.info(f"Loading model from {model_path}...")
        model = YOLO(model_path)
        
        logger.info(f"Loading image from {image_path}...")
        results = model(image_path)[0]
        
        orig_img = results.orig_img
        H, W = orig_img.shape[:2]
        logger.info(f"Image dimensions: {W}x{H} pixels")
        
        return model, results, orig_img, H, W
    
    except Exception as e:
        logger.error(f"Failed to load model or image: {e}")
        sys.exit(1)

# ========================
# Quadrat scaling
# ========================
def calculate_scaling(quadrat_width_cm, quadrat_height_cm, H, W):
    """Calculate cm² per pixel based on quadrat dimensions."""
    quadrat_area_cm2 = quadrat_width_cm * quadrat_height_cm
    total_pixels = H * W
    cm2_per_pixel = quadrat_area_cm2 / total_pixels
    
    logger.info(f"Quadrat area: {quadrat_area_cm2} cm²")
    logger.info(f"Conversion: {cm2_per_pixel:.6f} cm²/pixel")
    
    return quadrat_area_cm2, cm2_per_pixel

# ========================
# Process detections
# ========================
def process_detections(results, model, H, W, confidence_threshold=0.5):
    """Process YOLO detections and create species masks."""
    species_masks = {}
    detection_count = 0
    filtered_count = 0
    
    if results.masks is None:
        logger.warning("No segmentation masks found in results!")
        return species_masks
    
    for cls_id, mask, conf in zip(results.boxes.cls, results.masks.data, results.boxes.conf):
        detection_count += 1
        confidence = float(conf)
        
        # Filter by confidence threshold
        if confidence < confidence_threshold:
            filtered_count += 1
            continue
        
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
    
    logger.info(f"Processed {detection_count} detections ({filtered_count} filtered by confidence)")
    
    return species_masks

# ========================
# Calculate coverage
# ========================
def calculate_coverage(species_masks, quadrat_area_cm2, cm2_per_pixel):
    """Calculate and return coverage percentages for each species."""
    results = {}
    
    if not species_masks:
        logger.warning("No species masks found!")
        return results
    
    for species, mask in species_masks.items():
        species_pixels = mask.sum()
        species_area_cm2 = species_pixels * cm2_per_pixel
        percent_cover = (species_area_cm2 / quadrat_area_cm2) * 100
        results[species] = {
            'pixels': species_pixels,
            'area_cm2': species_area_cm2,
            'percent': percent_cover
        }
    
    return results

# ========================
# Generate visualization
# ========================
def create_visualization(orig_img, species_masks, output_path):
    """Create and save a visualization of detected masks."""
    try:
        # Create a color-coded overlay
        overlay = orig_img.copy()
        colors = {}
        
        # Generate distinct colors for each species
        num_species = len(species_masks)
        for i, species in enumerate(species_masks.keys()):
            colors[species] = tuple(
                int(255 * c) for c in 
                [(i * 73 % 256) / 255, ((i * 137) % 256) / 255, ((i * 211) % 256) / 255]
            )
        
        # Apply colored masks
        for species, mask in species_masks.items():
            color = colors[species]
            overlay[mask] = cv2.addWeighted(
                overlay[mask], 0.6,
                np.full_like(overlay[mask], color), 0.4,
                0
            )
        
        # Blend with original
        result = cv2.addWeighted(orig_img, 0.5, overlay, 0.5, 0)
        
        # Save
        cv2.imwrite(output_path, result)
        logger.info(f"Visualization saved to {output_path}")
        
    except Exception as e:
        logger.warning(f"Failed to create visualization: {e}")

# ========================
# Main execution
# ========================
def main():
    logger.info("=== Quadrat Species Coverage Analysis ===\n")
    
    # Validate inputs
    if not validate_inputs(MODEL_PATH, IMAGE_PATH, QUADRAT_WIDTH_CM, QUADRAT_HEIGHT_CM):
        sys.exit(1)
    
    # Load model and image
    model, results, orig_img, H, W = load_model_and_image(MODEL_PATH, IMAGE_PATH)
    
    # Calculate scaling
    quadrat_area_cm2, cm2_per_pixel = calculate_scaling(
        QUADRAT_WIDTH_CM, QUADRAT_HEIGHT_CM, H, W
    )
    
    # Process detections
    species_masks = process_detections(results, model, H, W, CONFIDENCE_THRESHOLD)
    
    # Calculate coverage
    coverage_results = calculate_coverage(species_masks, quadrat_area_cm2, cm2_per_pixel)
    
    # Print results
    print("\n=== Species % Cover ===")
    total_identified = 0
    
    if coverage_results:
        for species in sorted(coverage_results.keys()):
            data = coverage_results[species]
            print(f"{species:20s}: {data['percent']:6.2f} %")
            total_identified += data['percent']
    else:
        print("No species detected!")
        sys.exit(1)
    
    # Bare space
    bare_percent = 100 - total_identified
    print(f"{'Bare/Other':20s}: {bare_percent:6.2f} %")
    print(f"{'Total':20s}: {100.0:6.2f} %\n")
    
    # Generate visualization
    if OUTPUT_VISUALIZATION:
        create_visualization(orig_img, species_masks, OUTPUT_VISUALIZATION)
    
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main()