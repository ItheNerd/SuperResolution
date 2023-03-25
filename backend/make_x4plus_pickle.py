import pickle
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# Load the pre-trained RealESRGAN model
model = RealESRGANer(
    scale=3,
    model_path="weights/RealESRGAN_x4plus.pth",
    dni_weight=None,
    model=RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
    ),
)

# Pickle the model to a file
with open("realesrgan_x4plus_model.pkl", "wb") as f:
    pickle.dump(model, f)
