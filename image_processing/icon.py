from ultralytics import YOLO

from image_processing.config import ICON_MODEL
from image_processing.utils import get_bounding_box

icon_model = YOLO(ICON_MODEL)

def filter_bboxes(img, bboxes):
    # For student ID cards, we'll filter based on the image dimensions
    # We want to exclude the header area with the school name and logo
    height, width = img.shape[:2]
    header_height = height * 0.25  # Assume top 25% contains header
    
    filtered_bboxes = [bbox for bbox in [get_bounding_box(box) for box in bboxes] 
                      if bbox[1] > header_height]  # Only keep boxes below header

    return filtered_bboxes
