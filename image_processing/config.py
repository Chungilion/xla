CORNER_MODEL = "./models/corners.pt"
ICON_MODEL = "./models/icon.pt"
MASK_MODEL = "./models/mask.pt"
CRAFT_MODEL = "./models/craft_mlt_25k.pth"

VIETOCR_MODEL = "vgg_transformer"

CLASS_NAMES = {
    0: "bottom_left",
    1: "bottom_right",
    2: "top_left",
    3: "top_right"
}

# Text to skip when processing OCR text
SKIP_ROI_TEXT = [
    "THẺ", "BƯU", "VIỄN", "CHÍNH", "HỌC VIỆN",
    "HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG",
    "POSTS AND TELECOMMUNICATIONS INSTITUTE OF TECHNOLOGY"
]

# Common typos and variations in class codes
CLASS_CODE_PATTERNS = [
    r'[DE]\d{2}CQ[A-Z]+\d*-?[A-Z]?',  # Standard format: D21CQCNO8-B
    r'[DE]\d{2}[A-Z]+\d*-?[A-Z]?',     # Without CQ: D21CNO8-B
]

# Academic term patterns
ACADEMIC_TERM_PATTERNS = [
    r'HK\s*[1234]\s*(\d{4})\s*-\s*(\d{4})',         # HK1 2021-2022
    r'HKII?\s*(\d{4})\s*-\s*(\d{4})',               # HK1/HKI 2021-2022
    r'HKIII?\s*(\d{4})\s*-\s*(\d{4})',              # HKII/HK2 2021-2022
    r'Học kỳ\s*[1234]\s*(\d{4})\s*-\s*(\d{4})',     # Học kỳ 1 2021-2022
    r'(\d{4})\s*-\s*(\d{4})'                         # Plain year range 2021-2022
]

# Field mapping configuration
FIELD_MAPPINGS = {
    'student_id': {
        'label_variations': ['Mã sinh viên:', 'MSV:', 'MaSV:', 'MSSV:'],
        'extraction_type': 'after_colon',
        'output_field': 'student_id'
    },
    'full_name': {
        'label_variations': ['Họ và tên:', 'Họ tên:', 'HỌ VÀ TÊN:'],
        'extraction_type': 'after_colon',
        'output_field': 'full_name'
    },
    'class': {
        'label_variations': ['Lớp:', 'LỚP:'],
        'extraction_type': 'after_colon',
        'output_field': 'class'
    },
    'major': {
        'label_variations': ['Ngành:', 'NGÀNH:', 'Chuyên ngành:', 'CHUYÊN NGÀNH:'],
        'extraction_type': 'after_colon',
        'output_field': 'major'
    },
    'academic_term': {
        'label_variations': ['Học kỳ:', 'HK:', 'Kỳ:'],
        'extraction_type': 'after_colon',
        'output_field': 'academic_term',
        'value_type': 'year_range'
    },
    'place_of_origin': {
        'label_variations': ['Quê quán:', 'Quequan:', 'QUÊ QUÁN:'],
        'extraction_type': 'after_colon',
        'output_field': 'place_of_origin'
    },
    'date_of_birth': {
        'label_variations': ['Ngày sinh:', 'NGÀY SINH:', 'Sinh ngày:'],
        'extraction_type': 'after_colon',
        'output_field': 'date_of_birth',
        'value_type': 'date'
    }
}

# Default fields map
DEFAULT_FIELDS = {
    'student_id': None,
    'full_name': None,
    'class': None,
    'major': None,
    'academic_term': None,
    'place_of_origin': None,
    'date_of_birth': None
}