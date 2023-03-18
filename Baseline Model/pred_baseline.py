import cv2
import numpy as np
import os

def predict(filename, output_dir):
    # Load the image
    img = cv2.imread(filename)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blur, 100, 200)

    # Dilate the edges to close gaps
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open file to write predictions
    output_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(filename))[0] + ".txt").replace("image", "label")
    output_file = open(output_filename, "w")

    # Loop through contours and draw bounding boxes
    for contour in contours:
        # Calculate the contour area
        area = cv2.contourArea(contour)

        if area > 50:  # Only draw contours with area greater than 100 pixels
            # Get the bounding box coordinates
            x, y, w, h = cv2.boundingRect(contour)
            x_center = x + w / 2
            y_center = y + h / 2
            # Write label to output file
            output_file.write(f"0 {(x_center / img.shape[1]):.8f} {(y_center / img.shape[0]):.8f} {(w / img.shape[1]):.8f} {(h / img.shape[0]):.8f}\n")
    
    # Close output file
    output_file.close()

# Directory containing images to predict
input_dir = r"C:\Users\16474\Desktop\potholes-baseline\final_dataset\test\images"

# Directory to save predictions and images with bounding boxes
output_dir = r"C:\Users\16474\Desktop\potholes-baseline\final_dataset\test\predictions"

# Loop through images in input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"):
        # Call predict function on image
        input_path = os.path.join(input_dir, filename)
        predict(input_path, output_dir)