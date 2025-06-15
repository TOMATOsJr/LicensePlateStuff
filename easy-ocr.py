import easyocr
import cv2

# Define the characters you want EasyOCR to recognize
custom_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# Initialize EasyOCR with the recognition network optimized for alphanumeric characters
reader = easyocr.Reader(['en'], gpu=False, recog_network='english_g2')  # 'english_g2' is suitable for alphanumeric text

for i in range(2, 10):
    image_path = f'license_plate_crops/lic{i}.png'
    image = cv2.imread(image_path)
    # Run OCR
    results = reader.readtext(image)
    
    print(f"lic{i}.png")
    # Print results
    for (bbox, text, confidence) in results:
        print(f"Detected: {text} | Confidence: {confidence:.2f}")