import os
import requests
from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
import cv2
import numpy as np

# Initialize PaddleOCR with English and angle classifier
ocr = PaddleOCR(use_angle_cls=True, lang='en')

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "message": "Captcha OCR API is running. Use /ocr?url=IMAGE_URL",
        "status": True
    })

def url_to_image(url: str):
    """Download image from URL and convert to OpenCV numpy array"""
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        return None
    img_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

@app.route("/ocr", methods=["GET"])
def ocr_api():
    image_url = request.args.get("url")
    if not image_url:
        return jsonify({
            "status": False,
            "error": "Query parameter 'url' is required. Example: /ocr?url=https://example.com/captcha.png"
        }), 400

    img = url_to_image(image_url)
    if img is None:
        return jsonify({"status": False, "error": "Failed to download image"}), 400

    try:
        # Run OCR directly on numpy image
        results = ocr.ocr(img, cls=True)

        text_output = []
        for line in results[0]:
            text_output.append({
                "text": line[1][0],
                "confidence": float(line[1][1])
            })

        return jsonify({
            "status": True,
            "url": image_url,
            "results": text_output,
            "text": " ".join([x["text"] for x in text_output])
        })

    except Exception as e:
        return jsonify({"status": False, "error": str(e)}), 500


if __name__ == "__main__":
    # Render needs host=0.0.0.0
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
