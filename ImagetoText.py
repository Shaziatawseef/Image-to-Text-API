from flask import Flask, request, jsonify
import requests
from PIL import Image
import pytesseract
from io import BytesIO

app = Flask(__name__)

@app.route('/ocr', methods=['GET'])
def ocr_from_url():
    # Get image URL from query parameter
    image_url = request.args.get("url")
    if not image_url:
        return jsonify({
            "status": False,
            "error": "Query parameter 'url' is required. Example: /ocr?url=https://example.com/image.jpg"
        }), 400

    try:
        # Download image
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))

        # Extract text
        extracted_text = pytesseract.image_to_string(img)

        return jsonify({
            "status": True,
            "url": image_url,
            "text": extracted_text.strip()
        })
    except Exception as e:
        return jsonify({
            "status": False,
            "error": str(e)
        }), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
