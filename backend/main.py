import cv2
import numpy as np
import io
import pickle
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

# Load the pickled model from a file
with open("realesrgan_x4plus_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/enhance_image")
async def enhance_image(image: UploadFile = File(...)):
    # Read the image file and convert it to a NumPy array
    file_contents = await image.read()
    nparr = np.fromstring(file_contents, np.uint8)
    input_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print("Input Recieved")
    # Enhance the input image using the RealESRGAN model
    output_image, _ = model.enhance(input_image, outscale=4)
    cv2.imwrite("output.png", output_image)

    print("Output generated")
    # Encode the output image as PNG binary data
    success, encoded_image = cv2.imencode(".png ", output_image)
    if not success:
        return {"message": "Failed to encode image"}

    # Return the enhanced image as a file download
    print("Output sent")
    response = StreamingResponse(
        io.BytesIO(encoded_image.tobytes()), media_type="image/png"
    )
    return response
