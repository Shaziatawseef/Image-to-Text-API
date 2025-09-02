import os
import requests
from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
from PIL import Image
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)

# Initialize PaddleOCR once (English only for speed)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

@app.route('/')
def home():
    return jsonify({
        "status": True,
        "message": "Image-to-Text API is running. Use /ocr?url=IMAGE_URL"
    })

@app.route('/ocr', methods=['GET'])
def ocr_from_url():
    image_url = request.args.get("url")

    if not image_url:
        return jsonify({
            "status": False,
            "error": "Query parameter 'url' is required. Example: /ocr?url=https://example.com/captcha.png"
        }), 400

    try:
        # Fetch image from URL
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()

        # Open as image
        img = Image.open(BytesIO(response.content)).convert("RGB")

        # Run OCR
        result = ocr.ocr(img, cls=False)

        # Extract text
        extracted_texts = []
        for line in result:
            for box, (text, confidence) in line:
                extracted_texts.append(text)

        return jsonify({
            "status": True,
            "url": image_url,
            "extracted_text": " ".join(extracted_texts).strip(),
            "texts": extracted_texts
        })

    except Exception as e:
        return jsonify({
            "status": False,
            "error": str(e)
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
