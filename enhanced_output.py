import pickle
import cv2

# Load the pickled model from a file
with open("realesrgan_x4plus_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the input image
input_image = cv2.imread("input.jpg", cv2.IMREAD_UNCHANGED)

# Enhance the input image using the RealESRGAN model
output_image, _ = model.enhance(input_image, outscale=4)

# Save the output image
cv2.imwrite("output.png", output_image)
