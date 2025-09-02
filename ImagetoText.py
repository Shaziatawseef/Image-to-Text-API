import os
import requests
from flask import Flask, request, jsonify
from paddleocr import PaddleOCR

# Initialize OCR with cls enabled (no cls inside predict!)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "message": "Captcha OCR API is running. Use /ocr?url=IMAGE_URL",
        "status": True
    })

@app.route("/ocr", methods=["GET"])
def ocr_api():
    image_url = request.args.get("url")
    if not image_url:
        return jsonify({
            "status": False,
            "error": "Query parameter 'url' is required. Example: /ocr?url=https://example.com/captcha.png"
        })

    try:
        # Download image
        response = requests.get(image_url, timeout=10)
        if response.status_code != 200:
            return jsonify({"status": False, "error": "Failed to fetch image"})
        
        # Save temporary file
        image_path = "captcha.png"
        with open(image_path, "wb") as f:
            f.write(response.content)

        # Run OCR
        results = ocr.ocr(image_path, cls=True)

        text_output = []
        for line in results:
            for box, (text, confidence) in line:
                text_output.append({"text": text, "confidence": float(confidence)})

        return jsonify({
            "status": True,
            "url": image_url,
            "results": text_output
        })

    except Exception as e:
        return jsonify({
            "status": False,
            "error": str(e)
        })

if __name__ == "__main__":
    # Render needs host=0.0.0.0
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
