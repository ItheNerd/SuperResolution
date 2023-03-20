import cv2
import numpy as np
import io
import pickle
from fastapi import FastAPI, File, UploadFile
from starlette.responses import StreamingResponse

# Load the pickled model from a file
with open("realesrgan_x4plus_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()


@app.post("/enhance_image")
async def enhance_image(file: UploadFile = File(...)):
    # Read the image file and convert it to a NumPy array
    file_contents = await file.read()
    nparr = np.fromstring(file_contents, np.uint8)
    input_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Enhance the input image using the RealESRGAN model
    output_image, _ = model.enhance(input_image, outscale=4)

    # Encode the output image as PNG binary data
    success, encoded_image = cv2.imencode(".png", output_image)
    if not success:
        return {"message": "Failed to encode image"}

    # Return the enhanced image as a file download
    return StreamingResponse(
        io.BytesIO(encoded_image.tobytes()), media_type="image/png"
    )
