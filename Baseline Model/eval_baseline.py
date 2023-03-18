import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# IOU calculation function
def calculate_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def calculate_f1(gt_boxes, pred_boxes, iou_threshold):
    tp = 0
    fp = 0
    fn = 0
    
    for pred_box in pred_boxes:
        iou_scores = []
        for gt_box in gt_boxes:
            iou_score = calculate_iou(pred_box, gt_box)
            iou_scores.append(iou_score)
        max_iou = max(iou_scores) if iou_scores else 0
        if max_iou >= iou_threshold:
            tp += 1
        else:
            fp += 1
    
    fn = len(gt_boxes) - tp
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    return f1

# get list of image files in /images directory
img_dir = r"C:\Users\16474\Desktop\potholes-baseline\final_dataset\test\images"
img_files = os.listdir(img_dir)
img_files = [f for f in img_files if f.endswith(".jpg")]

scores = []
f1_scores = []

# loop through image files and corresponding label files in /labels directory
for img_file in img_files:
    label_file = os.path.join(r"C:\Users\16474\Desktop\potholes-baseline\final_dataset\test\labels", img_file.replace(".jpg", ".txt").replace("image", "label"))
    pred_label_file = os.path.join(r"C:\Users\16474\Desktop\potholes-baseline\final_dataset\test\predictions", img_file.replace(".jpg", ".txt").replace("image", "label"))

    if os.stat(label_file).st_size == 0 or os.stat(pred_label_file).st_size == 0:
        scores.append(0)
        continue

    # load image and label files
    img = Image.open(os.path.join(img_dir, img_file))
    label = np.loadtxt(label_file)
    pred_label = np.loadtxt(pred_label_file)

    if len(label.shape) == 1:
        label = np.array([label])
    
    if len(pred_label.shape) == 1:
        pred_label = np.array([pred_label])

    # loop through predicted labels and calculate IOU scores with ground-truth labels
    for pred_box in pred_label:
        iou_scores = []
        for label_box in label:
            iou_score = calculate_iou(pred_box, label_box)
            iou_scores.append(iou_score)
        max_iou = max(iou_scores)
        # print(f"IOU score for {img_file}: {max_iou}")
        scores.append(max_iou)

# loop through image files and corresponding label files in /labels directory
for img_file in img_files:
    label_file = os.path.join(r"C:\Users\16474\Desktop\potholes-baseline\final_dataset\test\labels", img_file.replace(".jpg", ".txt").replace("image", "label"))
    pred_label_file = os.path.join(r"C:\Users\16474\Desktop\potholes-baseline\final_dataset\test\predictions", img_file.replace(".jpg", ".txt").replace("image", "label"))

    if os.stat(label_file).st_size == 0 and os.stat(pred_label_file).st_size != 0:
        pred_label = np.loadtxt(pred_label_file)
        
        if len(pred_label.shape) == 1:
            pred_label = np.array([pred_label])
        
        tp = 0
        fp = len(pred_label)
        fn = 0

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        f1_scores.append(f1)

        continue
    if os.stat(label_file).st_size != 0 and os.stat(pred_label_file).st_size == 0:
        label = np.loadtxt(label_file)
        
        if len(label.shape) == 1:
            label = np.array([label])
        
        tp = 0
        fp = 0
        fn = len(label)

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        f1_scores.append(f1)

        continue
    if os.stat(label_file).st_size == 0 and os.stat(pred_label_file).st_size == 0:
        continue

    # load image and label files
    img = Image.open(os.path.join(img_dir, img_file))
    label = np.loadtxt(label_file)
    pred_label = np.loadtxt(pred_label_file)

    if len(label.shape) == 1:
        label = np.array([label])
    
    if len(pred_label.shape) == 1:
        pred_label = np.array([pred_label])

    f1_score = calculate_f1(label, pred_label, 0.5)
    f1_scores.append(f1_score)

print("average f1 score")
print(sum(f1_scores) / len(f1_scores))
print("average iou score")
print(sum(scores) / len(scores))

# Set up the plot
fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

# Plot the histograms side by side
axs[0].hist(f1_scores, bins=10, color='blue', alpha=0.5)
axs[1].hist(scores, bins=10, color='red', alpha=0.5)

# Set the y-axis limits for each subplot based on the maximum value in the data
axs[0].set_ylim([0, max(axs[0].get_ylim()[1], max(axs[0].get_yticks()))])
axs[1].set_ylim([0, max(axs[1].get_ylim()[1], max(axs[1].get_yticks()))])

# Set the labels for the plot
axs[0].set_title('F1 scores')
axs[1].set_title('IOU scores')
axs[0].set_xlabel('F1')
axs[1].set_xlabel('IOU')
axs[0].set_ylabel('Frequency')

# Show the plot
plt.tight_layout()
plt.show()