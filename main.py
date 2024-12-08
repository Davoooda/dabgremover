import os
import time  # New import for tracking execution time
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
    print("Startup event triggered: Loading model...")
    try:
        model = AutoModelForImageSegmentation.from_pretrained(settings.model_name, trust_remote_code=True)
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

def round_up_to_multiple(val, divisor=32):
    rounded = (val + divisor - 1) // divisor * divisor
    print(f"Rounding up {val} to the nearest multiple of {divisor}: {rounded}")
    return rounded

@app.get("/")
def read_root():
    print("Root endpoint '/' called.")
    return {"message": "Welcome to the Background Removal API. Use the /remove_bg endpoint to remove image backgrounds."}

@app.get("/remove_bg")
def remove_background():
    print("Endpoint '/remove_bg' called.")
    
    # Start tracking execution time
    start_time = time.perf_counter()
    
    image_url = "https://deletmetest.awesomeheap.com/tst_img.png"
    print(f"Fetching image from URL: {image_url}")
    
    try:
        # Fetch the image from the URL
        response = requests.get(image_url)
        print(f"Received response with status code: {response.status_code}")
        response.raise_for_status()  # Raise an error for bad status codes
        original_image = Image.open(BytesIO(response.content)).convert("RGBA")
        print("Image fetched and opened successfully.")
    except requests.exceptions.RequestException as e:
        print(f"RequestException while fetching image: {e}")
        raise HTTPException(status_code=400, detail=f"Error fetching image: {e}")
    except Exception as e:
        print(f"Unexpected error while fetching image: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

    orig_w, orig_h = original_image.size
    print(f"Original image size: width={orig_w}, height={orig_h}")

    # Calculate padded dimensions
    W_pad = round_up_to_multiple(orig_w, 32)
    H_pad = round_up_to_multiple(orig_h, 32)
    print(f"Padded image size: width={W_pad}, height={H_pad}")

    # Create a new padded image
    padded_image = Image.new("RGB", (W_pad, H_pad), (0, 0, 0))
    left = (W_pad - orig_w) // 2
    top = (H_pad - orig_h) // 2
    print(f"Padded image will have the original image pasted at: left={left}, top={top}")
    padded_image.paste(original_image.convert("RGB"), (left, top))
    print("Original image pasted onto the padded image.")

    # Define image transformations
    transform_image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    print("Image transformations defined.")

    input_tensor = transform_image(padded_image).unsqueeze(0).to(device)
    print(f"Input tensor shape: {input_tensor.shape}")

    try:
        with torch.no_grad():
            print("Performing model inference...")
            outputs = model(input_tensor)
            print("Model inference completed.")
            preds = outputs[-1].sigmoid().cpu()
            print("Predictions processed with sigmoid activation.")
    except Exception as e:
        print(f"Error during model inference: {e}")
        raise HTTPException(status_code=500, detail=f"Model inference error: {e}")

    pred_mask = preds[0].squeeze(0)
    print(f"Prediction mask shape: {pred_mask.shape}")
    mask_pil = transforms.ToPILImage()(pred_mask)
    print("Converted prediction mask to PIL image.")
    mask_cropped = mask_pil.crop((left, top, left + orig_w, top + orig_h))
    print("Cropped the mask to the original image dimensions.")

    result_image = original_image.copy()
    result_image.putalpha(mask_cropped)
    print("Applied the mask to the original image to remove the background.")

    # Save the result to a buffer
    buffer = BytesIO()
    result_image.save(buffer, format="PNG")
    buffer.seek(0)
    print("Result image saved to buffer.")

    # Calculate and log execution time
    elapsed_time = time.perf_counter() - start_time
    print(f"Script execution time: {elapsed_time:.2f} seconds")

    print("Returning the processed image.")
    return Response(content=buffer.getvalue(), media_type="image/png")
