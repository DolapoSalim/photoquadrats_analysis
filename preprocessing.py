import cv2
import numpy as np
from pathlib import Path

# Settings
INPUT_IMAGE = "C:\\Users\\dolap\\OneDrive\\Documents\\DOLAPO\\data-analysis\\photoquadrats_analysis\\img\\sample_img.jpg"
OUTPUT_IMAGE = "quadrat_calibrated.jpg"
QUADRAT_SIZE_CM = 20

# Load image
img = cv2.imread(INPUT_IMAGE)
if img is None:
    print(f"Error: Could not load image {INPUT_IMAGE}")
    exit(1)

H, W = img.shape[:2]
print(f"Image size: {W}x{H}")

# Convert to HSV for green detection
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define green color range (adjust if needed)
lower_green = np.array([35, 40, 40])
upper_green = np.array([90, 255, 255])

# Create mask for green pixels
mask = cv2.inRange(hsv, lower_green, upper_green)

# Morphological operations to clean up
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) < 4:
    print(f"Error: Found only {len(contours)} contours, expected 4 corners")
    exit(1)

# Get the 4 largest contours (the corners)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]

# Get center of each contour (corner position)
corners = []
for contour in contours:
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        corners.append([cx, cy])

corners = np.array(corners, dtype=np.float32)

# Sort corners: top-left, top-right, bottom-right, bottom-left
center = corners.mean(axis=0)
angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
corners = corners[np.argsort(angles)]

print(f"Detected corners (pixels):")
for i, c in enumerate(corners):
    print(f"  Corner {i+1}: ({c[0]:.1f}, {c[1]:.1f})")

# Calculate pixel-to-cm conversion from corner distances
# Distance between corners should be 20cm
dist_px = np.linalg.norm(corners[1] - corners[0])  # top edge
pixels_per_cm = dist_px / QUADRAT_SIZE_CM
cm_per_pixel = 1 / pixels_per_cm

print(f"\nCalibration:")
print(f"  Distance between corners: {dist_px:.1f} pixels")
print(f"  Pixels per cm: {pixels_per_cm:.2f}")
print(f"  cm per pixel: {cm_per_pixel:.6f}")

# Define destination points (perfect square)
output_size = int(QUADRAT_SIZE_CM * pixels_per_cm)
dst_corners = np.array([
    [0, 0],
    [output_size, 0],
    [output_size, output_size],
    [0, output_size]
], dtype=np.float32)

# Get perspective transform matrix
matrix = cv2.getPerspectiveTransform(corners, dst_corners)

# Apply perspective transform
calibrated_img = cv2.warpPerspective(img, matrix, (output_size, output_size))

# Save calibrated image
cv2.imwrite(OUTPUT_IMAGE, calibrated_img)
print(f"\nCalibrated image saved to {OUTPUT_IMAGE}")
print(f"Output size: {output_size}x{output_size} pixels")

# Save calibration data
calib_data = {
    'pixels_per_cm': pixels_per_cm,
    'cm_per_pixel': cm_per_pixel,
    'output_size': output_size,
    'quadrat_size_cm': QUADRAT_SIZE_CM
}

calib_file = OUTPUT_IMAGE.replace('.jpg', '_calib.txt')
with open(calib_file, 'w') as f:
    for key, val in calib_data.items():
        f.write(f"{key}: {val}\n")

print(f"Calibration data saved to {calib_file}")

# Optional: Display result
print("\nPreprocessing complete! Ready for species analysis.")