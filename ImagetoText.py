from flask import Flask, request, jsonify
import easyocr
import requests
from io import BytesIO
from PIL import Image
import torch

# Initialize Flask app
app = Flask(__name__)

# Initialize EasyOCR reader (English only for speed, you can add more langs like ['en', 'hi'])
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

@app.route('/')
def home():
    return jsonify({
        "status": True,
        "message": "Image-to-Text API is running. Use /ocr?url=IMAGE_URL"
    })

@app.route('/ocr', methods=['GET'])
def ocr_from_url():
    # Get image URL from query parameter
    image_url = request.args.get("url")
    if not image_url:
        return jsonify({
            "status": False,
            "error": "Query parameter 'url' is required. Example: /ocr?url=https://example.com/image.jpg"
        })

    try:
        # Fetch image from URL
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()

        # Load image into memory
        image = Image.open(BytesIO(response.content)).convert("RGB")

        # Run OCR
        result = reader.readtext(response.content)

        # Extract only text parts
        texts = [text for _, text, _ in result]

        return jsonify({
            "status": True,
            "url": image_url,
            "extracted_text": texts
        })

    except Exception as e:
        return jsonify({
            "status": False,
            "error": str(e)
        })

if __name__ == "__main__":
    # Render expects PORT from environment
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)        }), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
