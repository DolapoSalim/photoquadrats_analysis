import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from ultralytics import YOLO
import pandas as pd

# ========== CONFIGURATION ==========
IMAGE_PATH = r"C:\Users\dolap\OneDrive\Documents\DOLAPO\data-analysis\photoquadrats_analysis\img\C24169_S.jpg"
FRAME_MODEL_PATH = r"C:\Users\dolap\OneDrive\Documents\DOLAPO\data-analysis\photoquadrats_analysis\model\pauline_et_al\Frame_detection_model.pt"
SEGMENTATION_MODEL_PATH = r"C:\Users\dolap\OneDrive\Documents\DOLAPO\data-analysis\photoquadrats_analysis\model\pauline_et_al\Species_segmentation_model.pt"
CONFIDENCE_THRESHOLD = 0.25  # Lowered from 0.5 to detect more classes
FRAME_CONFIDENCE = 0.5  # Lower threshold for frame detection
GRID_SHAPE = (4, 4)
USE_GRID = True  # Toggle grid analysis
FRAME_SIZE_CM = 50  # Physical size of quadrat frame
OUTPUT_VIZ = "quadrat_analysis.png"
OUTPUT_EXCEL = "grid_analysis_report.xlsx"

# ========== FRAME DETECTION FUNCTIONS ==========
def detect_frame(image, frame_model, conf_threshold=FRAME_CONFIDENCE):
    """
    Detect the photoquadrat frame in the image.
    
    Args:
        image: Input image
        frame_model: YOLO model for frame detection
        conf_threshold: Confidence threshold for frame detection
    
    Returns:
        frame_coords: (x1, y1, x2, y2) bounding box of the frame
        frame_polygon: Polygon points if segmentation mask is available
        detection_confidence: Confidence score of the detection
    """
    print("Detecting frame...")
    results = frame_model(image, conf=conf_threshold)
    
    if results[0].boxes is None or len(results[0].boxes) == 0:
        print("WARNING: No frame detected! Using full image.")
        h, w = image.shape[:2]
        return (0, 0, w, h), None, 0.0
    
    # Get the highest confidence frame detection
    boxes = results[0].boxes
    confidences = boxes.conf.cpu().numpy()
    best_idx = np.argmax(confidences)
    
    frame_box = boxes[best_idx].xyxy[0].cpu().numpy()
    frame_coords = tuple(map(int, frame_box))
    detection_confidence = float(confidences[best_idx])
    
    print(f"Frame detected with confidence: {detection_confidence:.3f}")
    print(f"Frame coordinates: {frame_coords}")
    
    # Extract polygon if mask is available
    frame_polygon = None
    if results[0].masks is not None and len(results[0].masks) > best_idx:
        mask = results[0].masks[best_idx]
        mask_np = mask.data.cpu().numpy()
        
        # Find contours from mask
        if len(mask_np.shape) == 3:
            mask_np = mask_np[0]
            
        if mask_np.shape != image.shape[:2]:
            mask_resized = cv.resize(mask_np, (image.shape[1], image.shape[0]))
        else:
            mask_resized = mask_np
        
        mask_binary = (mask_resized * 255).astype(np.uint8)
        contours, _ = cv.findContours(mask_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour
            frame_polygon = max(contours, key=cv.contourArea)
            print(f"Frame polygon extracted with {len(frame_polygon)} points")
    
    return frame_coords, frame_polygon, detection_confidence


def crop_to_frame(image, frame_coords, frame_polygon=None, padding=0):
    """
    Crop image to the detected frame region.
    
    Args:
        image: Input image
        frame_coords: (x1, y1, x2, y2) bounding box
        frame_polygon: Optional polygon for more precise masking
        padding: Extra pixels to add around the frame
    
    Returns:
        cropped_img: Cropped image
        scale_factor: Pixels per cm (for real-world measurements)
        crop_offset: (x_offset, y_offset) to map back to original coordinates
    """
    x1, y1, x2, y2 = frame_coords
    
    # Add padding
    h, w = image.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    # Crop the image
    cropped_img = image[y1:y2, x1:x2].copy()
    
    # Calculate scale factor (pixels per cm)
    # Assuming the frame is roughly square
    frame_width_px = x2 - x1
    frame_height_px = y2 - y1
    avg_frame_size_px = (frame_width_px + frame_height_px) / 2
    scale_factor = avg_frame_size_px / FRAME_SIZE_CM
    
    print(f"Cropped to frame region: {cropped_img.shape}")
    print(f"Scale factor: {scale_factor:.2f} pixels/cm")
    
    crop_offset = (x1, y1)
    
    # Optional: Create a mask if polygon is available
    if frame_polygon is not None:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv.drawContours(mask, [frame_polygon], -1, 255, -1)
        mask_cropped = mask[y1:y2, x1:x2]
        
        # Apply mask to cropped image
        cropped_img = cv.bitwise_and(cropped_img, cropped_img, mask=mask_cropped)
    
    return cropped_img, scale_factor, crop_offset


# ========== SEGMENTATION FUNCTIONS ==========
def grid_lines(image, grid_shape, color=(0, 255, 0), thickness=1):
    """Draw grid on image."""
    img_height, img_width = image.shape[:2]
    rows, cols = grid_shape
    row_height = img_height // rows
    col_width = img_width // cols

    for i in range(1, rows):
        y = i * row_height
        cv.line(image, (0, y), (img_width, y), color, thickness)

    for j in range(1, cols):
        x = j * col_width
        cv.line(image, (x, 0), (x, img_height), color, thickness)

    return image


def get_class_colors(num_classes):
    """Generate distinct, bright colors for each class by ID."""
    colors = {
        0: (255, 0, 0),      # Bright Blue
        1: (0, 255, 0),      # Bright Green
        2: (0, 0, 255),      # Bright Red
        3: (0, 255, 255),    # Bright Yellow
        4: (255, 0, 255),    # Bright Magenta
        5: (255, 255, 0),    # Bright Cyan
        6: (255, 128, 0),    # Orange
        7: (128, 0, 255),    # Purple
        8: (0, 255, 128),    # Spring Green
        9: (255, 128, 128),  # Light Coral
        10: (128, 255, 128), # Light Green
        11: (128, 128, 255), # Light Blue
        12: (255, 255, 128), # Light Yellow
        13: (255, 128, 255), # Light Magenta
        14: (128, 255, 255), # Light Cyan
    }
    return {i: colors.get(i, colors[i % 15]) for i in range(num_classes)}


def calculate_segmentation_coverage(image, masks, results, use_grid=True, grid_shape=(4, 4)):
    """
    Calculate segmentation coverage - with or without grid.
    
    Args:
        image: Input image
        masks: Segmentation masks from YOLO
        results: YOLO results object
        use_grid: Boolean to enable/disable grid-based analysis
        grid_shape: Tuple (rows, cols) for grid division
    
    Returns:
        grid_areas or total_areas: Coverage data
        grid_detections or total_detections: Detection data
        class_masks: Per-class segmentation masks
    """
    img_height, img_width = image.shape[:2]
    class_names = results[0].names
    class_masks = {}
    
    # Initialize class masks
    for class_id in range(len(class_names)):
        class_masks[class_id] = np.zeros((img_height, img_width), dtype=np.uint8)
    
    # Process masks by class
    if results[0].masks is not None:
        print(f"\nProcessing {len(results[0].masks.data)} masks...")
        
        for mask_idx in range(len(results[0].masks.data)):
            mask = results[0].masks.data[mask_idx]
            class_id = int(results[0].boxes.cls[mask_idx].item())
            class_name = class_names[class_id]
            
            # Resize mask to image dimensions
            mask_np = mask.cpu().numpy().astype(np.float32)
            
            # Handle different mask shapes
            if len(mask_np.shape) == 3:
                mask_np = mask_np[0]
            
            if mask_np.shape != (img_height, img_width):
                mask_resized = cv.resize(mask_np, (img_width, img_height), interpolation=cv.INTER_LINEAR)
            else:
                mask_resized = mask_np
            
            # Normalize to 0-255 and binarize
            mask_resized = (mask_resized * 255).astype(np.uint8)
            _, mask_resized = cv.threshold(mask_resized, 127, 255, cv.THRESH_BINARY)
            
            # Add this mask to the class mask
            class_masks[class_id] = cv.bitwise_or(class_masks[class_id], mask_resized)
            
            # Debug output
            pixel_count = np.sum(mask_resized > 0)
            print(f"  Mask {mask_idx}: Class {class_id} ({class_name}) - {pixel_count} pixels")
    
    if use_grid:
        # Grid-based analysis
        return _calculate_grid_coverage(image, grid_shape, class_masks, class_names, results)
    else:
        # Total coverage only
        return _calculate_total_coverage(image, class_masks, class_names, results)


def _calculate_grid_coverage(image, grid_shape, class_masks, class_names, results):
    """Calculate coverage per grid cell."""
    img_height, img_width = image.shape[:2]
    rows, cols = grid_shape
    row_height = img_height // rows
    col_width = img_width // cols
    
    grid_areas = {}
    grid_detections = {}
    
    for i in range(rows):
        for j in range(cols):
            grid_areas[(i, j)] = {}
            grid_detections[(i, j)] = []
    
    # Calculate coverage per grid cell per class
    for class_id in range(len(class_names)):
        class_name = class_names[class_id]
        mask = class_masks[class_id]
        
        for i in range(rows):
            for j in range(cols):
                y_start = i * row_height
                y_end = (i + 1) * row_height
                x_start = j * col_width
                x_end = (j + 1) * col_width
                
                grid_mask = mask[y_start:y_end, x_start:x_end]
                segmented_pixels = np.sum(grid_mask > 127)  # More robust threshold
                grid_cell_area = (y_end - y_start) * (x_end - x_start)
                percentage_coverage = (segmented_pixels / grid_cell_area) * 100
                
                if percentage_coverage > 0:
                    grid_areas[(i, j)][class_name] = percentage_coverage
    
    # Track detections per grid cell
    if results[0].boxes is not None:
        boxes = results[0].boxes
        for box_idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            grid_row = center_y // row_height
            grid_col = center_x // col_width
            
            if grid_row < rows and grid_col < cols:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                class_name = class_names[class_id]
                
                grid_detections[(grid_row, grid_col)].append({
                    'class': class_name,
                    'confidence': confidence,
                    'box': (int(x1), int(y1), int(x2), int(y2))
                })
    
    return grid_areas, grid_detections, class_masks


def _calculate_total_coverage(image, class_masks, class_names, results):
    """Calculate total coverage without grid."""
    img_height, img_width = image.shape[:2]
    total_area = img_height * img_width
    
    total_areas = {}
    total_detections = []
    
    # Calculate coverage per class
    for class_id in range(len(class_names)):
        class_name = class_names[class_id]
        mask = class_masks[class_id]
        
        segmented_pixels = np.sum(mask > 127)  # More robust threshold
        percentage_coverage = (segmented_pixels / total_area) * 100
        
        if percentage_coverage > 0:
            total_areas[class_name] = percentage_coverage
    
    # Track all detections
    if results[0].boxes is not None:
        boxes = results[0].boxes
        for box_idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = class_names[class_id]
            
            total_detections.append({
                'class': class_name,
                'confidence': confidence,
                'box': (int(x1), int(y1), int(x2), int(y2))
            })
    
    return total_areas, total_detections, class_masks


def visualize_results(image, grid_areas, grid_detections, class_masks, class_colors, 
                      use_grid=True, grid_shape=(4, 4), frame_coords=None):
    """
    Visualize segmentation with optional grid overlay.
    
    Args:
        image: Input image
        grid_areas: Coverage data (dict for grid or total)
        grid_detections: Detection data
        class_masks: Per-class masks
        class_colors: Color mapping for classes
        use_grid: Boolean for grid visualization
        grid_shape: Grid dimensions
        frame_coords: Optional frame coordinates to draw frame boundary
    
    Returns:
        viz_img: Visualization image
    """
    viz_img = image.copy()
    overlay = viz_img.copy()
    
    # Create overlay with each class getting its unique color
    for class_id, mask in class_masks.items():
        color = class_colors.get(class_id, (0, 255, 0))
        overlay[mask > 127] = color  # Use threshold
    
    # Blend overlay with original image (increased alpha for better visibility)
    alpha = 0.6  # Increased from 0.4
    viz_img = cv.addWeighted(viz_img, 1 - alpha, overlay, alpha, 0)
    
    # Draw frame boundary if provided
    if frame_coords is not None:
        # This would be on the original image, not cropped
        pass
    
    if use_grid:
        # Draw grid lines
        viz_img = grid_lines(viz_img, grid_shape, color=(255, 255, 255), thickness=2)
        
        # Add percentage text to each grid cell
        img_height, img_width = image.shape[:2]
        rows, cols = grid_shape
        row_height = img_height // rows
        col_width = img_width // cols
        
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2  # Slightly smaller for more text to fit
        thickness = 3
        text_color = (255, 255, 0)
        bg_color = (0, 0, 0)
        line_height = 40
        
        for i in range(rows):
            for j in range(cols):
                y_start = i * row_height
                x_start = j * col_width
                
                # Get all classes and their coverages for this cell
                classes_in_cell = grid_areas[(i, j)]
                
                if classes_in_cell:
                    text_lines = [f"{cls}: {cov:.1f}%" for cls, cov in classes_in_cell.items()]
                else:
                    text_lines = ["Bare"]
                
                # Calculate available height in grid cell
                available_height = row_height - 10
                max_lines = available_height // line_height
                text_lines = text_lines[:max_lines]
                
                # Start position
                y_offset = y_start + line_height
                
                for text in text_lines:
                    text_size = cv.getTextSize(text, font, font_scale, thickness)[0]
                    text_width = text_size[0]
                    
                    text_x = max(x_start + 3, min(x_start + col_width - text_width - 3, x_start + 3))
                    text_y = y_offset
                    
                    # Draw background rectangle
                    cv.rectangle(viz_img,
                                (text_x - 3, text_y - text_size[1] - 2),
                                (text_x + text_width + 3, text_y + 2),
                                bg_color, -1)
                    
                    # Draw text
                    cv.putText(viz_img, text, (text_x, text_y),
                              font, font_scale, text_color, thickness)
                    
                    y_offset += line_height
    else:
        # Show total coverage as text overlay
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        text_color = (255, 255, 0)
        bg_color = (0, 0, 0)
        y_offset = 40
        
        for class_name, coverage in grid_areas.items():
            text = f"{class_name}: {coverage:.2f}%"
            text_size = cv.getTextSize(text, font, font_scale, thickness)[0]
            
            # Draw background
            cv.rectangle(viz_img,
                        (10, y_offset - text_size[1] - 5),
                        (10 + text_size[0] + 10, y_offset + 5),
                        bg_color, -1)
            
            # Draw text
            cv.putText(viz_img, text, (15, y_offset),
                      font, font_scale, text_color, thickness)
            
            y_offset += 50
    
    return viz_img


def create_excel_report(grid_areas, grid_detections, class_names, scale_factor, 
                       use_grid=True, grid_shape=(4, 4), output_path=OUTPUT_EXCEL):
    """
    Create Excel report with analysis data.
    
    Args:
        grid_areas: Coverage data
        grid_detections: Detection data
        class_names: List of class names
        scale_factor: Pixels per cm
        use_grid: Boolean for grid-based report
        grid_shape: Grid dimensions
        output_path: Path to save Excel file
    
    Returns:
        df: Pandas DataFrame with the report
    """
    data = []
    
    if use_grid:
        rows, cols = grid_shape
        grid_cell_area_cm2 = (FRAME_SIZE_CM / rows) * (FRAME_SIZE_CM / cols)
        
        for i in range(rows):
            for j in range(cols):
                coverage_dict = grid_areas[(i, j)]
                detections = grid_detections[(i, j)]
                
                row_data = {
                    'Grid Row': i,
                    'Grid Column': j,
                    'Grid Cell Area (cm²)': round(grid_cell_area_cm2, 2),
                    'Total Coverage (%)': round(sum(coverage_dict.values()), 2),
                }
                
                # Add per-class coverage
                for class_name in class_names:
                    coverage_pct = coverage_dict.get(class_name, 0)
                    row_data[f'{class_name} Coverage (%)'] = round(coverage_pct, 2)
                    # Calculate actual area in cm²
                    actual_area_cm2 = (coverage_pct / 100) * grid_cell_area_cm2
                    row_data[f'{class_name} Area (cm²)'] = round(actual_area_cm2, 4)
                
                row_data['Total Detections'] = len(detections)
                
                # Add detection details
                for det_idx, det in enumerate(detections):
                    row_data[f'Det_{det_idx+1}_Class'] = det['class']
                    row_data[f'Det_{det_idx+1}_Confidence'] = round(det['confidence'], 4)
                
                data.append(row_data)
    else:
        # Total coverage report
        total_area_cm2 = FRAME_SIZE_CM * FRAME_SIZE_CM
        
        row_data = {
            'Total Area (cm²)': total_area_cm2,
            'Total Coverage (%)': round(sum(grid_areas.values()), 2),
        }
        
        # Add per-class coverage
        for class_name in class_names:
            coverage_pct = grid_areas.get(class_name, 0)
            row_data[f'{class_name} Coverage (%)'] = round(coverage_pct, 2)
            actual_area_cm2 = (coverage_pct / 100) * total_area_cm2
            row_data[f'{class_name} Area (cm²)'] = round(actual_area_cm2, 4)
        
        row_data['Total Detections'] = len(grid_detections)
        
        # Add detection details
        for det_idx, det in enumerate(grid_detections):
            row_data[f'Det_{det_idx+1}_Class'] = det['class']
            row_data[f'Det_{det_idx+1}_Confidence'] = round(det['confidence'], 4)
        
        data.append(row_data)
    
    df = pd.DataFrame(data)
    df.to_excel(output_path, sheet_name='Coverage Analysis', index=False)
    print(f"Excel report saved to: {output_path}")
    return df


# ========== MAIN WORKFLOW ==========
def main():
    print("="*80)
    print("PHOTOQUADRAT SEGMENTATION ANALYSIS WORKFLOW")
    print("="*80)
    
    # Load image
    print("\n[1/6] Loading image...")
    img = cv.imread(IMAGE_PATH)
    if img is None:
        print(f"Error: Could not load image from {IMAGE_PATH}")
        exit(1)
    print(f"Image loaded: {img.shape}")
    
    # Load models
    print("\n[2/6] Loading models...")
    frame_model = YOLO(FRAME_MODEL_PATH)
    seg_model = YOLO(SEGMENTATION_MODEL_PATH)
    
    print(f"\nFrame detection model loaded:")
    print(f"  Task: {frame_model.task}")
    print(f"  Classes: {len(frame_model.names)}")
    
    print(f"\nSegmentation model loaded:")
    print(f"  Task: {seg_model.task}")  # SHOULD SAY 'segment'
    print(f"  Classes: {len(seg_model.names)}")
    
    for idx, name in seg_model.names.items():
        print(f"    Class {idx}: {name}")
    
    # Detect frame
    print(f"\n[3/6] Detecting photoquadrat frame...")
    frame_coords, frame_polygon, frame_conf = detect_frame(img, frame_model, FRAME_CONFIDENCE)
    
    # Crop to frame
    print(f"\n[4/6] Cropping to frame region...")
    cropped_img, scale_factor, crop_offset = crop_to_frame(img, frame_coords, frame_polygon)
    
    # Run segmentation
    print(f"\n[5/6] Running segmentation (confidence: {CONFIDENCE_THRESHOLD})...")
    seg_results = seg_model(cropped_img, conf=CONFIDENCE_THRESHOLD)
    
    # ========== DIAGNOSTIC OUTPUT ==========
    print("\n" + "="*80)
    print("SEGMENTATION DETECTION SUMMARY")
    print("="*80)
    
    if seg_results[0].boxes is not None and len(seg_results[0].boxes) > 0:
        detected_classes = seg_results[0].boxes.cls.cpu().numpy()
        detected_confidences = seg_results[0].boxes.conf.cpu().numpy()
        
        print(f"\nTotal detections: {len(detected_classes)}")
        print(f"\nPer-class breakdown:")
        
        for class_id in range(len(seg_model.names)):
            class_name = seg_model.names[class_id]
            count = np.sum(detected_classes == class_id)
            if count > 0:
                class_confs = detected_confidences[detected_classes == class_id]
                print(f"  Class {class_id} ({class_name}):")
                print(f"    Detections: {count}")
                print(f"    Confidence - Min: {np.min(class_confs):.3f}, Max: {np.max(class_confs):.3f}, Avg: {np.mean(class_confs):.3f}")
            else:
                print(f"  Class {class_id} ({class_name}): 0 detections")
        
        print(f"\nAll detections:")
        for idx, (cls, conf) in enumerate(zip(detected_classes, detected_confidences)):
            print(f"  [{idx}] Class {int(cls)} ({seg_model.names[int(cls)]}) - Confidence: {conf:.3f}")
    else:
        print("\nWARNING: NO DETECTIONS FOUND!")
        print("\nPossible reasons:")
        print("  1. Confidence threshold too high (try lowering CONFIDENCE_THRESHOLD)")
        print("  2. Model not suitable for this image")
        print("  3. Image quality/lighting issues")
        print("  4. Objects too small or obscured")
        
    if seg_results[0].masks is not None:
        print(f"\n✓ Total masks generated: {len(seg_results[0].masks.data)}")
    else:
        print("\nWARNING: NO MASKS FOUND!")
        print("  Check if the model was trained for instance segmentation (task='segment')")
        print(f"  Current model task: {seg_model.task}")
    
    print("="*80 + "\n")
    
    # Get class info
    class_names = seg_results[0].names
    num_classes = len(class_names)
    class_colors = get_class_colors(num_classes)
    
    # Calculate coverage
    print(f"Calculating coverage (Grid: {USE_GRID})...")
    if USE_GRID:
        grid_areas, grid_detections, class_masks = calculate_segmentation_coverage(
            cropped_img, seg_results[0].masks, seg_results, 
            use_grid=True, grid_shape=GRID_SHAPE
        )
    else:
        total_areas, total_detections, class_masks = calculate_segmentation_coverage(
            cropped_img, seg_results[0].masks, seg_results, 
            use_grid=False
        )
        grid_areas = total_areas
        grid_detections = total_detections
    
    # Print results
    print(f"\n{'='*80}")
    print("COVERAGE RESULTS")
    print(f"{'='*80}")
    
    if USE_GRID:
        total_coverage_all = 0
        cells_with_coverage = 0
        
        for (row, col) in sorted(grid_areas.keys()):
            print(f"\nGrid ({row}, {col}):")
            if grid_areas[(row, col)]:
                for class_name, coverage in grid_areas[(row, col)].items():
                    print(f"  {class_name}: {coverage:.2f}%")
                cell_total = sum(grid_areas[(row, col)].values())
                total_coverage_all += cell_total
                cells_with_coverage += 1
            else:
                print(f"  Bare (no coverage)")
            
            detection_count = len(grid_detections[(row, col)])
            print(f"  Detections in cell: {detection_count}")
        
        print(f"\n{'='*80}")
        print(f"Summary:")
        print(f"  Total cells: {GRID_SHAPE[0] * GRID_SHAPE[1]}")
        print(f"  Cells with coverage: {cells_with_coverage}")
        print(f"  Cells bare: {GRID_SHAPE[0] * GRID_SHAPE[1] - cells_with_coverage}")
        print(f"  Average coverage per cell: {total_coverage_all / (GRID_SHAPE[0] * GRID_SHAPE[1]):.2f}%")
    else:
        print("\nTotal Coverage:")
        if grid_areas:
            for class_name, coverage in grid_areas.items():
                print(f"  {class_name}: {coverage:.2f}%")
            print(f"\n  Overall total: {sum(grid_areas.values()):.2f}%")
        else:
            print("  No coverage detected")
        print(f"  Total Detections: {len(grid_detections)}")
    
    print(f"{'='*80}\n")
    
    # Visualize
    print(f"[6/6] Generating visualization...")
    viz_img = visualize_results(
        cropped_img, grid_areas, grid_detections, class_masks, class_colors,
        use_grid=USE_GRID, grid_shape=GRID_SHAPE
    )
    
    # Save visualization
    cv.imwrite(OUTPUT_VIZ, viz_img)
    print(f"✓ Visualization saved to: {OUTPUT_VIZ}")
    
    # Display
    plt.figure(figsize=(14, 10))
    plt.imshow(cv.cvtColor(viz_img, cv.COLOR_BGR2RGB))
    plt.title(f"Segmentation Analysis | Grid: {USE_GRID} | Conf: {CONFIDENCE_THRESHOLD}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    
    # Create Excel report
    print("\nGenerating Excel report...")
    df = create_excel_report(
        grid_areas, grid_detections, list(class_names.values()), scale_factor,
        use_grid=USE_GRID, grid_shape=GRID_SHAPE, output_path=OUTPUT_EXCEL
    )
    
    print(f"\n{'='*80}")
    print("WORKFLOW COMPLETE!")
    print(f"{'='*80}")
    print(f"Frame confidence: {frame_conf:.3f}")
    print(f"Scale factor: {scale_factor:.2f} pixels/cm")
    print(f"Total quadrat area: {FRAME_SIZE_CM}x{FRAME_SIZE_CM} cm = {FRAME_SIZE_CM**2} cm²")
    print(f"Visualization: {OUTPUT_VIZ}")
    print(f"Excel report: {OUTPUT_EXCEL}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()