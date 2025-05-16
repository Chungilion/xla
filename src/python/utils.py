# -*- coding: utf-8 -*-
import cv2
import numpy as np
import re
import logging
from PIL import Image
from typing import Optional
import os

def detect_card_region(img: np.ndarray) -> np.ndarray:
    """Detect and crop the card region with improved edge detection."""
    # Create working copy
    working_img = img.copy()
    height, width = working_img.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(working_img, cv2.COLOR_BGR2GRAY)
    
    # Bilateral filter to reduce noise while preserving edges
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Edge detection with auto-calculated thresholds
    median = np.median(blurred)
    lower = int(max(0, (1.0 - 0.33) * median))
    upper = int(min(255, (1.0 + 0.33) * median))
    edges = cv2.Canny(blurred, lower, upper)
    
    # Dilate edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Debug image for contours
    debug_dir = os.path.join(os.path.dirname(__file__), 'debug_ocr')
    os.makedirs(debug_dir, exist_ok=True)
    contour_debug = img.copy()
    cv2.drawContours(contour_debug, contours, -1, (0,255,0), 2)
    cv2.imwrite(os.path.join(debug_dir, 'contours_debug.jpg'), contour_debug)
    
    # Find the card contour
    max_area = 0
    card_contour = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > width * height * 0.2:  # Must be at least 20% of image
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:  # Must be rectangular
                if area > max_area:
                    max_area = area
                    card_contour = approx
    
    if card_contour is not None:
        # Sort points
        pts = np.float32(card_contour.reshape(4, 2))
        rect = np.zeros((4, 2), dtype=np.float32)
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left
        
        # Get width and height
        width = int(max(np.linalg.norm(rect[1] - rect[0]), np.linalg.norm(rect[2] - rect[3])))
        height = int(max(np.linalg.norm(rect[3] - rect[0]), np.linalg.norm(rect[2] - rect[1])))
        
        # Transform perspective
        dst_pts = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(rect, dst_pts)
        warped = cv2.warpPerspective(img, matrix, (width, height))
        
        # Save debug image
        cv2.imwrite(os.path.join(debug_dir, 'warped_card.jpg'), warped)
        
        return warped
    
    return img  # Return original if no card found

def enhance_text_regions(img: np.ndarray) -> np.ndarray:
    """Enhance text regions for better OCR accuracy."""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Increase contrast
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    
    # Apply unsharp masking
    gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
    unsharp = cv2.addWeighted(gray, 2.0, gaussian, -1.0, 0)
    
    return unsharp

def detect_text_regions(img: np.ndarray) -> list:
    """Detect potential text regions using MSER."""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Create MSER detector
    mser = cv2.MSER_create(
        delta=5,
        min_area=100,
        max_area=2000,
        max_variation=0.5
    )
    
    # Detect regions
    regions, _ = mser.detectRegions(gray)
    
    # Filter and merge text regions
    text_regions = []
    for region in regions:
        x, y, w, h = cv2.boundingRect(region)
        if w/h > 0.1 and w/h < 20:  # Filter out non-text-like regions
            text_regions.append((x, y, w, h))
    
    return text_regions

def improved_preprocessing(img: np.ndarray) -> np.ndarray:
    """Enhanced preprocessing pipeline optimized for printed text and Vietnamese characters."""
    # Create a larger image with higher resolution
    height, width = img.shape[:2]
    target_width = 1600  # Increased target width for better detail
    scale = target_width / width
    target_height = int(height * scale)
    img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

    # Convert to HSV for better handling of lighting variations
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Enhance value channel with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16, 16))
    v_equalized = clahe.apply(v)

    # Merge channels back and convert to BGR
    hsv_merged = cv2.merge([h, s, v_equalized])
    enhanced = cv2.cvtColor(hsv_merged, cv2.COLOR_HSV2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to reduce noise while preserving edges
    bilateral = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Sharpen image with a 5x5 kernel optimized for text
    kernel = np.array([[-1,-1,-1,-1,-1],
                      [-1, 2, 2, 2,-1],
                      [-1, 2, 8, 2,-1],
                      [-1, 2, 2, 2,-1],
                      [-1,-1,-1,-1,-1]]) / 8.0
    sharpened = cv2.filter2D(bilateral, -1, kernel)

    # Save debug images
    debug_dir = os.path.join(os.path.dirname(__file__), 'debug_ocr')
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, 'enhanced.jpg'), enhanced)
    cv2.imwrite(os.path.join(debug_dir, 'bilateral.jpg'), bilateral)
    cv2.imwrite(os.path.join(debug_dir, 'sharpened.jpg'), sharpened)

    # Apply local thresholding for better text separation
    block_size = 35
    thresh = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        10
    )

    # Clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Final debug image
    cv2.imwrite(os.path.join(debug_dir, 'final_preprocessed.jpg'), cleaned)

    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

def normalize_vietnamese_text(text: str) -> str:
    """Normalize Vietnamese text by handling common OCR issues."""
    # Common OCR mistakes in Vietnamese characters
    replacements = {
        # Basic Vietnamese characters
        'ă': ['a~', 'ã', 'ă', 'ằ', 'ẵ', 'a'],
        'â': ['â', 'ã', 'ẫ', 'ấ', 'a'],
        'đ': ['d-', 'd~', 'ð', 'đ', 'd', 'D'],
        'ê': ['e^', 'ê', 'ế', 'e'],
        'ô': ['o^', 'ô', 'ố', 'o'],
        'ơ': ['o*', 'ơ', 'ớ', 'o'],
        'ư': ['u+', 'ư', 'ứ', 'u'],
        
        # Tone marks and common mistakes
        'à': ['a`', 'à', 'a'],
        'á': ['a´', 'á', 'a'],
        'ạ': ['a.', 'ạ', 'a'],
        'ả': ['a?', 'ả', 'a'],
        'ã': ['a~', 'ã', 'a'],
        'ế': ['e^\'', 'ế', 'é'],
        'ề': ['e`', 'ề', 'è'],
        'ể': ['e?', 'ể', 'ẻ'],
        'ễ': ['e~', 'ễ', 'ẽ'],
        'ệ': ['e.', 'ệ', 'ẹ'],
        
        # Common word corrections
        'VIỆN': ['VIẸ', 'VIEN', 'VIẸN', 'VIL', 'VIE', 'Vie'],
        'PTIT': ['PIT', 'PTIT', 'PT', 'P7IT'],
        'SINH': ['SINN', 'SINH', 'SlNH', 'S1NH'],
        'VIÊN': ['VIEN', 'VIÊN', 'V1EN', 'VIÉN'],
        'KHOA': ['KHÓA', 'KHOA', 'KH0A'],
        'CÔNG': ['CONG', 'CÔNG', 'C0NG'],
        'NGHỆ': ['NGHE', 'NGHẸ', 'NGHÊ', 'NGHỆ', 'NGHL'],
        'THÔNG': ['THONG', 'THÔNG', 'TH0NG'],
        'TIN': ['TIN', 'TÍN', 'T1N'],
        'TRƯỜNG': ['TRUONG', 'TRƯONG', 'TRƯỜNG', 'TRU0NG'],
        'ĐẠI': ['DAI', 'ĐAI', 'ĐẠI', 'DA1'],
        'HỌC': ['HOC', 'HỌC', 'H0C'],
        'VÀ': ['VA', 'VÀ', 'Va', 'và'],
        'TÊN': ['TEN', 'TÊN', 'Ten', 'tên'],
        'MÃ': ['MA', 'MÃ', 'Ma', 'mã'],
        'SỐ': ['SO', 'SỐ', 'S0', 'số'],
        'HỆ': ['HE', 'HỆ', 'He', 'hệ'],
        'NGÀY': ['NGAY', 'NGÀY', 'ngay', 'ngày'],
        'SINH': ['SINH', 'SlNH', 'sinh'],
        'LỚP': ['LOP', 'LỚP', 'lớp'],
        'NGÀNH': ['NGANH', 'NGÀNH', 'ngành'],
        'CHÍNH': ['CHINH', 'CHÍNH', 'chính'],
        'QUY': ['QUY', 'quy'],
    }

    # First normalize diacritical marks and special characters
    normalized = text.upper()  # Convert to uppercase for consistency

    # Remove unwanted characters
    normalized = re.sub(r'["\',]', ' ', normalized)
    
    # Handle number substitutions first
    number_fixes = {
        '0': 'O',
        '1': 'I',
        '7': 'T',
        '8': 'B',
    }
    for num, letter in number_fixes.items():
        normalized = re.sub(rf'\b{num}\b', letter, normalized)
    
    # Apply word-level corrections first (longer strings first to avoid partial matches)
    word_replacements = {k: v for k, v in replacements.items() if len(k) > 1}
    char_replacements = {k: v for k, v in replacements.items() if len(k) == 1}
    
    # Sort word replacements by length (longest first)
    for correct in sorted(word_replacements.keys(), key=len, reverse=True):
        variants = word_replacements[correct]
        for variant in variants:
            # Case-insensitive replacement using regex
            normalized = re.sub(rf'\b{re.escape(variant)}\b', correct, normalized, flags=re.IGNORECASE)
    
    # Then apply character-level corrections
    for correct, variants in char_replacements.items():
        for variant in variants:
            normalized = normalized.replace(variant, correct)
    
    # Clean up multiple spaces
    normalized = re.sub(r'\s+', ' ', normalized)
    
    return normalized.strip()

def extract_student_info(text: str) -> dict:
    """Extract student information from OCR text with improved Vietnamese support."""
    # Clean and normalize text
    text = re.sub(r'["\',]', ' ', text)  # Remove quotes and commas that might interfere
    text = normalize_vietnamese_text(text)
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    logging.info(f"Normalized text: {text}")
    
    # Initialize result dictionary
    info = {
        'student_id': None,
        'full_name': None,
        'birth_date': None,
        'class_name': None,
        'major': None,
        'home_town': None,
        'enrollment_type': None,
        'academic_year': None
    }
    
    # More flexible patterns with case insensitive matching
    patterns = {
        'student_id': [
            # Support for both MSV and full "Mã số sinh viên" with variations
            r'(?:(?:mã|ma)\s*(?:số|so)\s*(?:sinh\s*viên|sv)|msv|mssv)[:\s]*([B][0-9][A-Z0-9]{6,8})',
            r'(?:Student\s*ID|ID)[:\s]*([B][0-9][A-Z0-9]{6,8})',
            r'(?:[B][0-9][A-Z0-9]{6,8})',  # Direct ID pattern
            r'(?:^|\s)([B][0-9][A-Z0-9]{6,8})(?:\s|$)',  # ID anywhere in text
        ],
        'full_name': [
            # Support for both "họ và tên" and "họ tên" with common OCR mistakes
            r'(?:họ\s*(?:và|va)\s*tên|họ\s*tên|ho\s*(?:va|và)\s*ten)[:\s]*([A-ZĐÀÁẢÃẠÈÉẺẼẸÌÍỈĨỊÒÓỎÕỌÙÚỦŨỤỲÝỶỸỴÂĂÔƠƯ\s]+?)(?=\s*(?:sinh\s*ngày|ngày\s*sinh|sinh\s*ngay|ngay\s*sinh|$))',
            r'(?:Full\s*name|Name)[:\s]*([A-ZĐÀÁẢÃẠÈÉẺẼẸÌÍỈĨỊÒÓỎÕỌÙÚỦŨỤỲÝỶỸỴÂĂÔƠƯ\s]+?)(?=\s*(?:Date|DOB|$))',
        ],
        'birth_date': [
            # Support various date formats and separators
            r'(?:ngày\s*sinh|sinh\s*ngày|ngay\s*sinh|sinh\s*ngay)[:\s]*(\d{1,2}[-./]\d{1,2}[-./]\d{2,4})',
            r'(?:Date\s*of\s*birth|DOB)[:\s]*(\d{1,2}[-./]\d{1,2}[-./]\d{2,4})',
        ],
        'class_name': [
            # Support for both Vietnamese and English class identifiers
            r'(?:lớp|lop)[:\s]*([A-Z0-9-]+)',
            r'(?:Class|Group)[:\s]*([A-Z0-9-]+)',
        ],
        'major': [
            # Support various ways to write "ngành" with diacritics
            r'(?:ngành|nganh|chuyên\s*ngành|chuyen\s*nganh)[:\s]*([A-ZĐÀÁẢÃẠÈÉẺẼẸÌÍỈĨỊÒÓỎÕỌÙÚỦŨỤỲÝỶỸỴÂĂÔƠƯ\s-]+?)(?=\s*(?:khóa|khoa|$))',
            r'(?:Major|Programme)[:\s]*([A-ZĐÀÁẢÃẠÈÉẺẼẸÌÍỈĨỊÒÓỎÕỌÙÚỦŨỤỲÝỶỸỴÂĂÔƠƯ\s-]+?)(?=\s*(?:Year|$))',
        ],
        'home_town': [
            # Support for both quê quán and nơi sinh
            r'(?:quê\s*quán|que\s*quan|nơi\s*sinh|noi\s*sinh)[:\s]*([A-ZĐÀÁẢÃẠÈÉẺẼẸÌÍỈĨỊÒÓỎÕỌÙÚỦŨỤỲÝỶỸỴÂĂÔƠƯ\s,]+)',
            r'(?:Place\s*of\s*birth|Home\s*town)[:\s]*([A-ZĐÀÁẢÃẠÈÉẺẼẸÌÍỈĨỊÒÓỎÕỌÙÚỦŨỤỲÝỶỸỴÂĂÔƠƯ\s,]+)',
        ],
        'enrollment_type': [
            # Support for various program types with common variations
            r'(?:hệ|he)[:\s]*([^,\n]*?(?:chính\s*quy|chinh\s*quy|CHÍNH\s*QUY|liên\s*thông|lien\s*thong|văn\s*bằng\s*2)[^,\n]*)',
            r'(?:Program\s*type|Type)[:\s]*([^,\n]*?(?:Regular|Transfer|Second\s*Degree)[^,\n]*)',
        ],
        'academic_year': [
            # Support for khóa/khoa with year ranges
            r'(?:khóa|khoa|khoá)[:\s]*(\d{4}(?:\s*[-]\s*\d{4}|\s*\d{4})?)',
            r'(?:Academic\s*year|Year)[:\s]*(\d{4}(?:\s*[-]\s*\d{4}|\s*\d{4})?)',
        ],
    }
    
    # Try to extract each field with improved logging
    for field, field_patterns in patterns.items():
        for pattern in field_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match.group(1).strip()
                if value and (not info[field] or len(value) > len(info[field])):
                    info[field] = value
                    logging.info(f"Found {field}: {value} using pattern: {pattern}")

    # Post-process the extracted information
    if info['student_id']:
        info['student_id'] = info['student_id'].upper()  # Ensure student ID is uppercase
    
    if info['full_name']:
        info['full_name'] = ' '.join(word.capitalize() for word in info['full_name'].split())  # Proper name capitalization
    
    if info['birth_date']:
        # Standardize date format
        date_parts = re.split(r'[-./]', info['birth_date'])
        if len(date_parts) == 3:
            day, month, year = date_parts
            if len(year) == 2:
                year = '20' + year if int(year) < 25 else '19' + year
            info['birth_date'] = f"{day}/{month}/{year}"

    return info
