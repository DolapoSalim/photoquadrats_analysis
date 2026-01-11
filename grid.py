# script for drawing grid on image

import cv2 as cv
import numpy as np

def grid_lines(image, grid_shape, color=(0, 255, 0), thickness=1):
    """
    Draws a grid on the given image.

    Parameters:
    - image: The input image on which to draw the grid.
    - grid_shape: A tuple (rows, cols) specifying the number of grid cells.
    - color: The color of the grid lines (default is green).
    - thickness: The thickness of the grid lines (default is 1).

    Returns:
    - The image with the grid drawn on it.
    """
    img_height, img_width = image.shape[:2]
    rows, cols = grid_shape

    # Calculate the spacing between lines
    row_height = img_height // rows
    col_width = img_width // cols

    # Draw horizontal lines
    for i in range(1, rows):
        y = i * row_height
        cv.line(image, (0, y), (img_width, y), color, thickness)

    # Draw vertical lines
    for j in range(1, cols):
        x = j * col_width
        cv.line(image, (x, 0), (x, img_height), color, thickness)

    return image

image = r"C:\Users\dolap\OneDrive\Documents\DOLAPO\data-analysis\photoquadrats_analysis\img\sample_img.JPG"
img = cv.imread(image)
grid_img = grid_lines(img, (4, 4), color=(255, 0, 0), thickness=2)
cv.imshow("Grid Image", grid_img)