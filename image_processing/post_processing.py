import re

from image_processing.config import (
    FIELD_MAPPINGS, 
    SKIP_ROI_TEXT, 
    DEFAULT_FIELDS,
    CLASS_CODE_PATTERNS,
    ACADEMIC_TERM_PATTERNS
)

def clean_string(s):
    """Clean a string by removing special characters but preserving Vietnamese diacritics and important symbols"""
    if not s:
        return ""
    # More carefully preserve Vietnamese characters AND special characters used in class codes
    s = re.sub(r'[^\w\s\-áàảãạâấầẩẫậăắằẳẵặđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵÁÀẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ:/.,-]', '', s)
    return re.sub(r'\s+', ' ', s).strip()

def normalize_text(text):
    """Normalize text by converting to lowercase and removing diacritics and extra whitespace"""
    if not text:
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Replace đ/Đ with d/D
    text = text.replace('đ', 'd').replace('Đ', 'D')
    
    # Remove diacritics - both uppercase and lowercase variants
    text = re.sub(r'[áàảãạâấầẩẫậăắằẳẵặÁÀẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶ]', 'a', text)
    text = re.sub(r'[éèẻẽẹêếềểễệÉÈẺẼẸÊẾỀỂỄỆ]', 'e', text)
    text = re.sub(r'[íìỉĩịÍÌỈĨỊ]', 'i', text)
    text = re.sub(r'[óòỏõọôốồổỗộơớờởỡợÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢ]', 'o', text)
    text = re.sub(r'[úùủũụưứừửữựÚÙỦŨỤƯỨỪỬỮỰ]', 'u', text)
    text = re.sub(r'[ýỳỷỹỵÝỲỶỸỴ]', 'y', text)
    
    # Remove extra whitespace and trim
    return ' '.join(text.split())

def check_student_id(text):
    """Check if text contains a valid student ID pattern"""
    pattern = r'[BĐ]\d{2}[A-Z]+[0O\d]{3,4}'
    return bool(re.search(pattern, text))

def handle_student_id(text):
    """Extract and format student ID"""
    pattern = r'[BĐ]\d{2}[A-Z]+[0O\d]{3,4}'
    match = re.search(pattern, text)
    if match:
        student_id = match.group(0).upper()
        # Handle O/0 confusion only in the number part
        main_part = student_id[:-3]
        number_part = student_id[-3:].replace('O', '0')
        return main_part + number_part
    return None

def extract_after_label(text, label):
    """Extract text that appears after a label"""
    if label in text:
        return text.split(label, 1)[1].strip()
    return None

def process_text(raw_text):
    """Process OCR text from student ID cards and format for frontend display"""
    # Initialize result structure
    result = {
        "student_id": None,
        "fields": {
            "full_name": {"value": ""},
            "date_of_birth": {"value": ""},
            "place_of_origin": {"value": ""},
            "class": {"value": "", "extra": ""},
            "major_and_term": {"value": "", "term_value": ""}
        }
    }
    
    # Format raw text lines and print for debugging
    raw_text = [line.strip() for line in raw_text if line.strip()]
    print("\nProcessing lines:")
    for line in raw_text:
        print(f"Line: {line}")
    print("\n")
    
    # Skip title line if present
    raw_text = [line for line in raw_text if "THẺ SINH VIÊN" not in line.upper()]
    
    current_field = None
    for line in raw_text:
        # Skip empty lines
        if not line.strip():
            continue
            
        # Handle student ID - prioritize format with no prefix
        if re.match(r'^[BĐ]\d{2}[A-Z]+[0O\d]{3,4}$', line):
            if not result["student_id"]:  # Only set if not already set
                id_text = line.strip()
                main = id_text[:-3]
                nums = id_text[-3:].replace('O', '0')
                result["student_id"] = main + nums
            continue
        
        # Skip lines that only contain "Mã SV"
        if line.strip().upper() == "MÃ SV":
            continue
            
        # Handle full name
        if "Họ và tên:" in line:
            current_field = "full_name"
            name_value = extract_after_label(line, "Họ và tên:")
            if name_value and not result["fields"]["full_name"]["value"]:
                result["fields"]["full_name"]["value"] = name_value
                current_field = None
        elif current_field == "full_name" and not result["fields"]["full_name"]["value"]:
            result["fields"]["full_name"]["value"] = line.strip()
            current_field = None
            
        # Handle birth date
        elif "Sinh ngày:" in line:
            current_field = "birth_date"
            date_value = extract_after_label(line, "Sinh ngày:")
            if date_value and not result["fields"]["date_of_birth"]["value"]:
                result["fields"]["date_of_birth"]["value"] = date_value
                current_field = None
        elif current_field == "birth_date" and not result["fields"]["date_of_birth"]["value"]:
            if re.match(r'\d{1,2}/\d{1,2}/\d{4}', line):
                result["fields"]["date_of_birth"]["value"] = line.strip()
            current_field = None
            
        # Handle place of origin
        elif "Hộ khẩu TT:" in line:
            current_field = "origin"
            origin_value = extract_after_label(line, "Hộ khẩu TT:")
            if origin_value and not result["fields"]["place_of_origin"]["value"]:
                result["fields"]["place_of_origin"]["value"] = origin_value
                current_field = None
        elif current_field == "origin" and not result["fields"]["place_of_origin"]["value"]:
            result["fields"]["place_of_origin"]["value"] = line.strip()
            current_field = None
            
        # Handle class and education type
        elif "Lớp:" in line:
            current_field = "class"
            # Extract class value
            if "Hệ:" in line:
                # Handle combined format "Lớp: XXX Hệ: YYY"
                parts = line.split("Hệ:")
                class_part = extract_after_label(parts[0], "Lớp:")
                if class_part:
                    result["fields"]["class"]["value"] = class_part.strip()
                if len(parts) > 1:
                    result["fields"]["class"]["extra"] = parts[1].strip()
                current_field = None
            else:
                # Handle split format
                class_value = extract_after_label(line, "Lớp:")
                if class_value:
                    result["fields"]["class"]["value"] = class_value.strip()
        elif "Hệ:" in line and (not result["fields"]["class"]["extra"] or result["fields"]["class"]["extra"].strip() == ""):
            education_type = extract_after_label(line, "Hệ:")
            if education_type:
                result["fields"]["class"]["extra"] = education_type.strip()
            current_field = None
        elif current_field == "class":
            # Only set education type if not already set and line looks like an education type
            if not result["fields"]["class"]["extra"] and "đại học" in line.lower():
                result["fields"]["class"]["extra"] = line.strip()
            current_field = None
            
        # Handle major and term
        elif "Ngành:" in line:
            current_field = "major"
            if "Khóa:" in line:
                # Handle combined format "Ngành: XXX Khóa: YYY"
                parts = line.split("Khóa:")
                major_part = extract_after_label(parts[0], "Ngành:")
                if major_part:
                    result["fields"]["major_and_term"]["value"] = major_part.strip()
                if len(parts) > 1:
                    result["fields"]["major_and_term"]["term_value"] = parts[1].strip()
                current_field = None
            else:
                # Handle split format
                major_value = extract_after_label(line, "Ngành:")
                if major_value:
                    result["fields"]["major_and_term"]["value"] = major_value.strip()
            
        elif "Khóa:" in line and not result["fields"]["major_and_term"]["term_value"]:
            term_value = extract_after_label(line, "Khóa:")
            if term_value:
                result["fields"]["major_and_term"]["term_value"] = term_value.strip()
            current_field = None
            
        elif current_field == "major":
            # This is a continuation of the major field
            if not result["fields"]["major_and_term"]["value"]:
                result["fields"]["major_and_term"]["value"] = line.strip()
            current_field = None

    return result

def label_text(content_text):
    """For backward compatibility - directly return processed text if it's a dict"""
    if isinstance(content_text, dict):
        return content_text
        
    return process_text(content_text)