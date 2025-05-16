import os
from matplotlib import pyplot as plt
from ultralytics import YOLO

from image_processing.config import CORNER_MODEL, MASK_MODEL
from .corners import get_corner_number, get_corner_points, euclidean_distance

import numpy as np
import cv2

corner_model = YOLO(CORNER_MODEL)
mask_model = YOLO(MASK_MODEL)

def perspective_transform(image, card_corners):    
    (tl, tr, br, bl) = card_corners

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype=np.float32), dst)
    warped = cv2.warpPerspective(image.copy(), M, (maxWidth, maxHeight))

    return warped

def find_card_corners(contour):
    min_rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(min_rect)
    box = np.int0(box)
    
    card_corners = []
    for corner in box:
        distances = np.linalg.norm(contour - corner, axis=2)
        min_dist_index = np.argmin(distances)
        card_corners.append(contour[min_dist_index][0])
    
    return np.array(card_corners)

def calculate_coords_to_transform(img, corner_model, mask_model):
    # Student IDs are simpler rectangles, so we can use basic contour detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("Không thể nhận diện được thẻ sinh viên. Hãy thử lại với hình ảnh khác")
        
    # Find the largest contour - should be the ID card
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.int32)
    
    # Sort points to get proper order: top-left, top-right, bottom-right, bottom-left
    ordered_coords = order_points(box)
    
    return ordered_coords
    
    
    if corner_number == 4:
        return process_four_corners(ordered_coords)
    elif corner_number == 3:
        return process_three_corners(img, ordered_coords, mask_model)
    
def process_four_corners(ordered_coords):
    return np.array(ordered_coords)

def process_three_corners(img, ordered_coords, mask_model):
    masks = mask_model.predict(img)
    contour = masks[0][0].masks.xy[0]
    contour = contour.astype(np.int32).reshape(-1, 1, 2)
    
    unordered_coords = find_card_corners(contour)
    
    coords_to_transform = assign_closest_coords(ordered_coords, unordered_coords)
    return coords_to_transform

def assign_closest_coords(ordered_coords, unordered_coords):
    coords_to_transform = [[], [], [], []]
    none_idx = None
    
    for i, coord_1 in enumerate(ordered_coords):
        if coord_1 is None:
            none_idx = i
            continue
        
        distances = [euclidean_distance(coord_1, coord_2) for coord_2 in unordered_coords]
        coord_idx = np.argmin(distances)
        
        coords_to_transform[i] = unordered_coords[coord_idx]
        unordered_coords = np.delete(unordered_coords, coord_idx, axis=0)
    
    if none_idx is not None:
        coords_to_transform[none_idx] = unordered_coords[0]
    
    return coords_to_transform

def order_points(pts):
    # Initialize ordered coordinates array
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # Top-left will have smallest sum
    # Bottom-right will have largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right will have smallest difference
    # Bottom-left will have largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def transform_card_image(image_path):
    img = cv2.imread(image_path)
    
    coords_to_transform = calculate_coords_to_transform(img, corner_model, mask_model)
        
    transformed = perspective_transform(img, coords_to_transform)
    return transformed