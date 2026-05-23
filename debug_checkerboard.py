import cv2
import glob
import numpy as np

CHESSBOARD_SIZE = (9, 6)

images = glob.glob('calibration_images/*.jpg')
print(f"Testing {len(images)} images...\n")

detected_count = 0
not_detected_count = 0

for idx, fname in enumerate(images[:5]):  # Test first 5 images
    img = cv2.imread(fname)
    if img is None:
        print(f"Image {idx}: {fname} - Failed to read")
        continue
    
    print(f"\nImage {idx}: {fname}")
    print(f"  Shape: {img.shape}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Try standard detection
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
    print(f"  Standard detection: {ret}")
    
    if not ret:
        # Try with adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        ret2, corners2 = cv2.findChessboardCorners(thresh, CHESSBOARD_SIZE, None)
        print(f"  Adaptive threshold detection: {ret2}")
        
        # Try histogram equalization
        equalized = cv2.equalizeHist(gray)
        ret3, corners3 = cv2.findChessboardCorners(equalized, CHESSBOARD_SIZE, None)
        print(f"  Histogram equalized detection: {ret3}")
    
    if ret:
        detected_count += 1
    else:
        not_detected_count += 1

print(f"\n\nSummary:")
print(f"Detected: {detected_count}/5")
print(f"Not detected: {not_detected_count}/5")
