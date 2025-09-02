import os
import requests
from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
from PIL import Image
from io import BytesIO

# Initialize PaddleOCR (English only for speed)
ocr_model = PaddleOCR(use_angle_cls=False, lang='en')

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "message": "Image-to-Text API is running. Use /ocr?url=IMAGE_URL",
        "status": True
    })

@app.route("/ocr", methods=["GET"])
def ocr():
    try:
        img_url = request.args.get("url")
        if not img_url:
            return jsonify({
                "status": False,
                "error": "Query parameter 'url' is required. Example: /ocr?url=https://example.com/captcha.png"
            })

        # Download image from URL
        response = requests.get(img_url, timeout=10)
        if response.status_code != 200:
            return jsonify({
                "status": False,
                "error": f"Failed to fetch image. HTTP {response.status_code}"
            })

        # Load image
        img = Image.open(BytesIO(response.content))

        # Run OCR
        result = ocr_model.ocr(img, cls=False)  # No unexpected keyword error

        # Extract text
        texts = []
        for line in result[0]:
            texts.append(line[1][0])

        return jsonify({
            "status": True,
            "url": img_url,
            "text": " ".join(texts) if texts else "",
            "details": texts
        })

    except Exception as e:
        return jsonify({
            "status": False,
            "error": str(e)
        })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
