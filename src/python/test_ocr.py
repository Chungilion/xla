import requests

def test_ocr_endpoint():
    url = 'http://localhost:8000/api/ocr'
    # Replace 'test_image.jpg' with the path to your test image
    files = {'file': ('test_image.jpg', open('test_image.jpg', 'rb'), 'image/jpeg')}
    
    try:
        response = requests.post(url, files=files)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        files['file'][1].close()

if __name__ == "__main__":
    test_ocr_endpoint()
