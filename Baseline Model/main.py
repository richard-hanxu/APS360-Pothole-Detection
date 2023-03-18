import cv2
import numpy as np

# Load the image
img = cv2.imread('1.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite("1gray.jpg", gray)

# Apply Gaussian blur to reduce noise
blur = cv2.GaussianBlur(gray, (21, 21), 0)

cv2.imwrite("2blur.jpg", blur)

# Apply Canny edge detection
edges = cv2.Canny(blur, 100, 200)

cv2.imwrite("3edges.jpg", edges)

# Dilate the edges to close gaps
kernel = np.ones((5, 5), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=1)

cv2.imwrite("4dilated.jpg", dilated)

# Find contours
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw bounding boxes around each contour
for contour in contours:
    # Calculate the contour area
    area = cv2.contourArea(contour)
    if area > 100:  # Only draw contours with area greater than 100 pixels
        # Get the bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)
        # Draw the bounding box
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

# Show the result
# cv2.imshow('Result', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite("5contours.jpg", img)