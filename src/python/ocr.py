# -*- coding: utf-8 -*-
import cv2
import numpy as np
import re
import logging
import os
import torch
import easyocr
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from typing import Optional, Tuple
from utils import detect_card_region, improved_preprocessing, extract_student_info

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

app = FastAPI(
    title="Student ID Card OCR API",
    description="API for OCR processing of Vietnamese student ID cards",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js development server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize EasyOCR reader with Vietnamese optimization
try:
    # Use default configuration since we want to use the built-in model
    reader = easyocr.Reader(
        ['vi'],  # Vietnamese only for better accuracy
        gpu=True if torch.cuda.is_available() else False,
        # Using default Vietnamese model
        download_enabled=True,
        verbose=False
    )
except Exception as e:
    logging.error(f"OCR initialization failed: {str(e)}")
    raise RuntimeError("Could not initialize OCR system")

class ExtractedText(BaseModel):
    text: str
    confidence: float
    line_number: int

class StudentInfo(BaseModel):
    student_id: Optional[str] = None
    full_name: Optional[str] = None
    birth_date: Optional[str] = None
    class_name: Optional[str] = None
    major: Optional[str] = None
    home_town: Optional[str] = None
    enrollment_type: Optional[str] = None
    academic_year: Optional[str] = None

class OCRResult(BaseModel):
    overall_confidence: float
    rotation_angle: int
    detected_text: list[ExtractedText]
    extracted_info: StudentInfo
    debug_images: dict[str, str]  # Image type -> base64 encoded image

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

def process_image_for_ocr(img: np.ndarray) -> tuple[str, float, int]:
    """Process image and run OCR with multiple attempts and debug logging."""
    attempts = [
        lambda x: x,  # Original preprocessed image
        lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),  # Rotated 90
        lambda x: cv2.rotate(x, cv2.ROTATE_180),  # Rotated 180
        lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE),  # Rotated 270
    ]
    
    best_text = []
    max_avg_confidence = 0
    best_rotation = 0
    debug_dir = os.path.join(os.path.dirname(__file__), 'debug_ocr')
    os.makedirs(debug_dir, exist_ok=True)
    
    # Initial preprocessing with enhanced contrast
    preprocessed = improved_preprocessing(img)
    cv2.imwrite(os.path.join(debug_dir, 'ocr_input.jpg'), preprocessed)
    
    # Parameters for text detection
    MIN_CONFIDENCE = 0.4
    MIN_TEXT_HEIGHT = 20
    
    for i, attempt in enumerate(attempts):
        try:
            rotated = attempt(preprocessed)
            cv2.imwrite(os.path.join(debug_dir, f'rotation_{i}.jpg'), rotated)            # Run EasyOCR with improved settings for Vietnamese
            results = reader.readtext(
                rotated,
                batch_size=8,  # Increased batch size for better context
                detail=1,
                paragraph=False,  # Disable paragraph mode for ID cards
                # Fine-tuned parameters for student ID cards
                height_ths=0.3,  # Lower height threshold for better text grouping
                width_ths=0.3,   # Lower width threshold to detect shorter text
                y_ths=0.5,       # Increased y threshold to better separate lines
                x_ths=0.1,       # Lower x threshold to group text within lines
                slope_ths=0.2    # Allow more rotation within text lines
            )
            
            if results:
                # Process and group results
                filtered_results = []
                for (box, text, conf) in results:
                    if conf > MIN_CONFIDENCE and text.strip():
                        # Calculate box dimensions
                        box_points = np.array(box)
                        x_coords = box_points[:, 0]
                        y_coords = box_points[:, 1]
                        width = max(x_coords) - min(x_coords)
                        height = max(y_coords) - min(y_coords)
                        center_y = (max(y_coords) + min(y_coords)) / 2
                          # Clean up Vietnamese text
                        clean_text = text.strip()
                        # Fix common OCR mistakes
                        clean_text = re.sub(r'PT\s*(?:IT)?', 'PTIT', clean_text)  # Fix PTIT
                        clean_text = re.sub(r'VIL\s*$', 'VIỆN', clean_text)      # Fix VIỆN
                        clean_text = re.sub(r'Hp\s*(?:va|và)\s*ten', 'Họ và tên', clean_text)  # Fix "Họ và tên"
                        clean_text = re.sub(r'Slnh\s*ngày', 'Sinh ngày', clean_text)  # Fix "Sinh ngày"
                        clean_text = re.sub(r'(\d{2})h(\d{2,4})', r'\1/\2', clean_text)  # Fix date format
                        clean_text = re.sub(r'["\',]', ' ', clean_text)
                        clean_text = re.sub(r'\s+', ' ', clean_text)
                        
                        filtered_results.append({
                            'text': clean_text,
                            'confidence': conf,
                            'center_y': center_y,
                            'width': width,
                            'height': height,
                            'x_start': min(x_coords)
                        })
                
                if filtered_results:                    # First, identify the main regions (header, body) based on text size
                    avg_height = sum(r['height'] for r in filtered_results) / len(filtered_results)
                    header_results = []
                    body_results = []
                    
                    for result in filtered_results:
                        # Header text is usually larger
                        if result['height'] > avg_height * 1.2:
                            header_results.append(result)
                        else:
                            body_results.append(result)
                    
                    # Sort headers by y position first
                    header_results.sort(key=lambda x: x['center_y'])
                    
                    # Sort body text by y position, then x position
                    body_results.sort(key=lambda x: (x['center_y'], x['x_start']))
                    
                    # Process headers and body separately
                    text_lines = []
                    current_line = []
                    last_y = None
                    total_conf = 0
                    
                    for result in filtered_results:
                        if last_y is None:
                            current_line = [result]
                        elif abs(result['center_y'] - last_y) > avg_height * 1.5:
                            text_lines.append(current_line)
                            current_line = [result]
                        else:
                            current_line.append(result)
                        
                        last_y = result['center_y']
                        total_conf += result['confidence']
                    
                    if current_line:
                        text_lines.append(current_line)
                      # Format text with structure
                    text_parts = []
                    
                    # Add header lines first
                    for result in header_results:
                        text_parts.append(result['text'])
                    
                    # Process body text in left-to-right, top-to-bottom order
                    current_y = None
                    current_line = []
                    
                    for result in body_results:
                        if current_y is None:
                            current_y = result['center_y']
                            current_line = [result['text']]
                        elif abs(result['center_y'] - current_y) <= avg_height * 0.5:
                            # Same line
                            current_line.append(result['text'])
                        else:
                            # New line
                            text_parts.append(' '.join(current_line))
                            current_y = result['center_y']
                            current_line = [result['text']]
                    
                    # Add last line if any
                    if current_line:
                        text_parts.append(' '.join(current_line))
                    
                    avg_confidence = total_conf / len(filtered_results)
                    
                    # Update best result if confidence is higher
                    if avg_confidence > max_avg_confidence:
                        max_avg_confidence = avg_confidence
                        best_text = text_parts
                        best_rotation = i
                        
                    # Log with better structure
                    logging.info(f"\nKết quả OCR (Xoay {i * 90}°):")
                    logging.info(f"Tìm thấy {len(filtered_results)} vùng văn bản")
                    logging.info(f"Độ tin cậy trung bình: {avg_confidence:.2f}")
                    for line in text_parts:
                        logging.info(line)
            
        except Exception as e:
            logging.error(f"Lỗi trong lần thử {i}: {str(e)}")
    
    if not best_text:
        raise HTTPException(status_code=400, detail="Không thể đọc được text từ ảnh")
      # Join with newlines for better structure
    full_text = '\n'.join(best_text)
    logging.info(f"\nKết quả tốt nhất (Xoay {best_rotation * 90}°):\n{full_text}")
    
    return full_text, max_avg_confidence, best_rotation

@app.post("/api/ocr", 
         description="Perform OCR on a Vietnamese student ID card image",
         response_model=OCRResult,
         responses={
             400: {"description": "Invalid image or OCR failed"},
             415: {"description": "Unsupported media type"}
         })
async def perform_ocr(file: UploadFile = File(..., description="Image file to process")):
    """
    Upload an image of a Vietnamese student ID card for OCR processing.
    
    The image should be:
    - A clear photo of the student ID card
    - In JPG, PNG, or BMP format
    - Card should be well-lit and oriented properly
    
    Returns extracted student information including ID, name, birth date, etc.
    """
    try:
        logging.info(f"Received file: {file.filename}, content_type: {file.content_type}")
          # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Log image properties
        logging.info(f"Image shape: {img.shape}, dtype: {img.dtype}")
        
        # Detect and crop card region
        card_img = detect_card_region(img)
          # Perform OCR with rotation info
        text, confidence, best_rotation = process_image_for_ocr(card_img)
        
        # Extract student information
        student_info = extract_student_info(text)
          # Calculate overall confidence combining OCR and info extraction
        info_confidence = len([v for v in student_info.values() if v]) / len(student_info)
        final_confidence = (confidence + info_confidence) / 2

        # Convert debug images to base64
        debug_images = {}
        debug_dir = os.path.join(os.path.dirname(__file__), 'debug_ocr')
        for img_file in ['ocr_input.jpg', 'warped_card.jpg']:
            try:
                img_path = os.path.join(debug_dir, img_file)
                if os.path.exists(img_path):
                    with open(img_path, "rb") as f:
                        img_data = f.read()
                        img_b64 = base64.b64encode(img_data).decode()
                        debug_images[img_file] = f"data:image/jpeg;base64,{img_b64}"
            except Exception as e:
                logging.error(f"Error encoding debug image {img_file}: {str(e)}")

        # Create structured text lines
        detected_lines = []
        for idx, line in enumerate(text.split('\n'), 1):
            if line.strip():
                detected_lines.append(ExtractedText(
                    text=line.strip(),
                    confidence=confidence,  # Using overall confidence per line
                    line_number=idx
                ))
        
        return OCRResult(
            overall_confidence=final_confidence,
            rotation_angle=best_rotation * 90,
            detected_text=detected_lines,
            extracted_info=StudentInfo(**student_info),
            debug_images=debug_images
        )
        
    except Exception as e:
        logging.error(f"OCR failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)