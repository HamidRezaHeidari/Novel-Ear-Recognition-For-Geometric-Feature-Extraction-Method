# Machine Vision Project / Hamid Reza Heidari  / Milad Mohammadi

# PHASE 7 --> Enrollment Enhancement

# Import library
import cv2
from ultralytics import YOLO

def Enhancement(image, clipLimit=2.0, tileGridSize=(8, 8)):

    trained_model = YOLO("best.onnx", task="detect")
    objects = trained_model.predict(source=image, max_det=1)

    # calculate bounding box
    for r in objects:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]

    # Define the crop region (x1, y1, x2, y2)
    x1, y1, x2, y2 = b
    x1 = int(x1.item())
    x2 = int(x2.item())
    y1 = int(y1.item())
    y2 = int(y2.item())

    # Crop the image
    cropped_image = image[y1:y2, x1:x2]

    # Convert to grayscale
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # Create CLAHE object
    # Clip limit: Threshold for contrast limiting
    # Tile grid size: Size of grid for histogram equalization
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    # Apply CLAHE
    enhanced_image = clahe.apply(gray)

    return enhanced_image

# image = cv2.imread("001_1.bmp")
# r = Enhancement(image)
# cv2.imshow('Enhanced Image', r)
# cv2.waitKey(0)
# cv2.destroyAllWindows()