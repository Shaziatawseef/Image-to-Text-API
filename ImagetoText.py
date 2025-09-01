import uvicorn
import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
import io
from PIL import Image

# Initialize FastAPI app
app = FastAPI(title="Image to Text OCR API", version="1.0")

# Initialize PaddleOCR (English + Numbers)
ocr = PaddleOCR(use_angle_cls=True, lang="en")  # Load once, reuse = faster

@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...)):
    """
    OCR endpoint: Upload an image and get extracted text
    """
    try:
        # Read uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Save to temp file for OCR
        image.save("temp.png")

        # Perform OCR
        result = ocr.ocr("temp.png", cls=True)

        # Extract text
        text_results = []
        for line in result:
            for word in line:
                text_results.append(word[1][0])

        return JSONResponse({
            "status": True,
            "text": " ".join(text_results),
            "details": text_results
        })

    except Exception as e:
        return JSONResponse({"status": False, "error": str(e)})


@app.post("/ocr_url")
async def ocr_from_url(url: str = Form(...)):
    """
    OCR endpoint: Send an image URL instead of uploading
    """
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return JSONResponse({"status": False, "error": "Failed to fetch image"})

        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        image.save("temp.png")

        result = ocr.ocr("temp.png", cls=True)

        text_results = []
        for line in result:
            for word in line:
                text_results.append(word[1][0])

        return JSONResponse({
            "status": True,
            "text": " ".join(text_results),
            "details": text_results
        })

    except Exception as e:
        return JSONResponse({"status": False, "error": str(e)})


# Run locally
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
