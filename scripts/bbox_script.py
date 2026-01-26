import cv2
import numpy as np

#Load the image
image_path = "input/file/path"
image = cv2.imread(image_path)

# Image dimensions
img_height, img_width, _ = image.shape

# Define grid segmentation (e.g., 4x4 grid)
grid_rows, grid_cols = 4, 4
segment_height = img_height // grid_rows
segment_width = img_width // grid_cols

# Bounding boxes (replace with your detected bounding boxes)
# Format: [x_min, y_min, x_max, y_max, confidence]
bounding_boxes = [
    [50, 50, 150, 150, 0.8],
    [300, 300, 400, 400, 0.6],
    # Add more bounding boxes as needed
]

# Initialize a grid to store total coverage per segment
segment_coverage = np.zeros((grid_rows, grid_cols))

# Calculate coverage for each bounding box
for box in bounding_boxes:
    x_min, y_min, x_max, y_max, confidence = box
    
    # Filter by confidence threshold
    if confidence < 0.5:
        continue

    # Bounding box area
    box_area = (x_max - x_min) * (y_max - y_min)

    # Determine which segments the bounding box overlaps
    for row in range(grid_rows):
        for col in range(grid_cols):
            # Segment boundaries
            seg_x_min = col * segment_width
            seg_y_min = row * segment_height
            seg_x_max = seg_x_min + segment_width
            seg_y_max = seg_y_min + segment_height

            # Calculate overlap between bounding box and segment
            overlap_x_min = max(x_min, seg_x_min)
            overlap_y_min = max(y_min, seg_y_min)
            overlap_x_max = min(x_max, seg_x_max)
            overlap_y_max = min(y_max, seg_y_max)

            if overlap_x_min < overlap_x_max and overlap_y_min < overlap_y_max:
                overlap_area = (overlap_x_max - overlap_x_min) * (overlap_y_max - overlap_y_min)
                segment_coverage[row, col] += overlap_area

# Calculate percentage cover for each segment
segment_percent_cover = (segment_coverage / (segment_width * segment_height)) * 100

# Display results
for row in range(grid_rows):
    for col in range(grid_cols):
        print(f"Segment ({row}, {col}) cover: {segment_percent_cover[row, col]:.2f}%")
