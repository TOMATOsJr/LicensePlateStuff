from paddleocr import PaddleOCR
# from paddleocr import TextRecognition

model = PaddleOCR(ocr_version="PP-OCRv4",use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False, lang="en")
# model = TextRecognition()
output = model.predict(input="./license_plate_crops/lic9.png")
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
