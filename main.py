import os
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from fastapi import FastAPI, Response, HTTPException
from pydantic_settings import BaseSettings, SettingsConfigDict
import requests  # New import for fetching the image from URL

class Settings(BaseSettings):
    model_name: str = "briaai/RMBG-2.0"
    # Removed input_image_path since we'll use a fixed URL
    model_config = SettingsConfigDict(
        protected_namespaces=('settings_',)
    )
    
settings = Settings()

app = FastAPI()
device = torch.device('cpu')  # Force using CPU
model = None

@app.on_event("startup")
def load_model():
    global model
    print("Loading model...")
    try:
        model = AutoModelForImageSegmentation.from_pretrained(settings.model_name, trust_remote_code=True)
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

def round_up_to_multiple(val, divisor=32):
    return (val + divisor - 1) // divisor * divisor

@app.get("/remove_bg")
def remove_background():
    image_url = "https://deletmetest.awesomeheap.com/bg_test.jpeg"
    try:
        # Fetch the image from the URL
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an error for bad status codes
        original_image = Image.open(BytesIO(response.content)).convert("RGBA")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error fetching image: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

    orig_w, orig_h = original_image.size

    # Calculate padded dimensions
    W_pad = round_up_to_multiple(orig_w, 32)
    H_pad = round_up_to_multiple(orig_h, 32)

    # Create a new padded image
    padded_image = Image.new("RGB", (W_pad, H_pad), (0, 0, 0))
    left = (W_pad - orig_w) // 2
    top = (H_pad - orig_h) // 2
    padded_image.paste(original_image.convert("RGB"), (left, top))

    # Define image transformations
    transform_image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    input_tensor = transform_image(padded_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        preds = outputs[-1].sigmoid().cpu()

    pred_mask = preds[0].squeeze(0)
    mask_pil = transforms.ToPILImage()(pred_mask)
    mask_cropped = mask_pil.crop((left, top, left + orig_w, top + orig_h))

    result_image = original_image.copy()
    result_image.putalpha(mask_cropped)

    # Save the result to a buffer
    buffer = BytesIO()
    result_image.save(buffer, format="PNG")
    buffer.seek(0)

    return Response(content=buffer.getvalue(), media_type="image/png")
