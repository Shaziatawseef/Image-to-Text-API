import io
import requests
from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
from PIL import Image

# Initialize Flask
app = Flask(__name__)

# Initialize PaddleOCR (English only for speed)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

@app.route("/")
def home():
    return jsonify({"status": True, "message": "Image to Text API Running"})

@app.route("/ocr", methods=["POST"])
def ocr_api():
    try:
        if "url" in request.json:  # If image URL is provided
            image_url = request.json["url"]
            response = requests.get(image_url, timeout=10)
            img_bytes = io.BytesIO(response.content)
        elif "file" in request.files:  # If image file is uploaded
            img_bytes = request.files["file"]
        else:
            return jsonify({"status": False, "error": "No image provided"}), 400

        # Convert to PIL Image
        image = Image.open(img_bytes).convert("RGB")

        # Run OCR
        result = ocr.ocr(image, cls=True)

        # Extract text
        extracted_text = " ".join([line[1][0] for res in result for line in res])

        return jsonify({
            "status": True,
            "text": extracted_text
        })

    except Exception as e:
        return jsonify({"status": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
