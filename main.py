import os
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from fastapi import FastAPI, Response
from pydantic_settings import BaseSettings
# Пример настройки приложения (опционально)
class Settings(BaseSettings):
    model_name: str = "briaai/RMBG-2.0"
    input_image_path: str = "tst_img.png"

settings = Settings()

app = FastAPI()
device = torch.device('cpu')  # Принудительно на CPU
model = None

@app.on_event("startup")
def load_model():
    global model
    print("Загрузка модели...")
    model = AutoModelForImageSegmentation.from_pretrained(settings.model_name, trust_remote_code=True)
    model.to(device)
    model.eval()
    print("Модель загружена.")

def round_up_to_multiple(val, divisor=32):
    return (val + divisor - 1) // divisor * divisor

@app.get("/remove_bg")
def remove_background():
    # Загружаем исходное изображение
    original_image = Image.open(settings.input_image_path).convert("RGBA")
    orig_w, orig_h = original_image.size

    # Вычисляем размеры с паддингом
    W_pad = round_up_to_multiple(orig_w, 32)
    H_pad = round_up_to_multiple(orig_h, 32)

    padded_image = Image.new("RGB", (W_pad, H_pad), (0, 0, 0))
    left = (W_pad - orig_w) // 2
    top = (H_pad - orig_h) // 2
    padded_image.paste(original_image.convert("RGB"), (left, top))

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

    # Сохраняем результат в буфер
    buffer = BytesIO()
    result_image.save(buffer, format="PNG")
    buffer.seek(0)

    return Response(content=buffer.getvalue(), media_type="image/png")
