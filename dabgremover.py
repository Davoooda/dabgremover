import os
# Не будем использовать MPS, чтобы избежать несоответствий в операциях
# и упрощать отладку. При необходимости можно вернуть на MPS или CUDA.
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from tqdm import tqdm

# Принудительно на CPU для предсказуемости
device = torch.device('cpu')
print("Используется CPU.")

# Покажем прогресс выполнения шагов:
with tqdm(total=6, desc="Общий прогресс") as pbar:
    # Загрузка модели
    model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
    model.to(device)
    model.eval()
    pbar.update(1)  # Обновим прогресс после загрузки модели

    # Путь к исходному изображению
    input_image_path = "tst_img.png"
    original_image = Image.open(input_image_path).convert("RGBA")

    # Оригинальные размеры
    orig_w, orig_h = original_image.size

    # Некоторые модели сегментации могут требовать, чтобы обе стороны были кратны 32.
    # Чтобы не ломать пропорции, мы не будем искажать изображение.
    # Вместо этого мы добавим отступы (паддинги) до ближайшего размера, кратного 32.
    def round_up_to_multiple(val, divisor=32):
        return (val + divisor - 1) // divisor * divisor

    W_pad = round_up_to_multiple(orig_w, 32)
    H_pad = round_up_to_multiple(orig_h, 32)

    pbar.update(1)  # Обновим прогресс после вычисления нужных размеров

    # Создаем новое изображение большего размера и помещаем исходное по центру
    padded_image = Image.new("RGB", (W_pad, H_pad), (0, 0, 0))  # заполним чёрным, но так как у нас маска будет альфа, это не критично
    # Рассчитаем координаты для центрирования
    left = (W_pad - orig_w) // 2
    top = (H_pad - orig_h) // 2
    padded_image.paste(original_image.convert("RGB"), (left, top))

    # Трансформации без изменения пропорций: просто нормализация, т.к. мы уже подготовили размеры
    transform_image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    input_tensor = transform_image(padded_image).unsqueeze(0).to(device)

    pbar.update(1)  # Обновим прогресс после подготовки входных данных

    with torch.no_grad():
        outputs = model(input_tensor)
        # Предполагается, что нужная маска в последнем элементе outputs
        preds = outputs[-1].sigmoid().cpu()

    pbar.update(1)  # Обновим прогресс после получения предсказания

    pred_mask = preds[0].squeeze(0)  # размер H_pad x W_pad

    # Превращаем предсказание в PIL
    mask_pil = transforms.ToPILImage()(pred_mask)

    # Вырежем из маски ту область, где находится исходное изображение
    mask_cropped = mask_pil.crop((left, top, left + orig_w, top + orig_h))

    # Применим маску как альфа-канал к исходному изображению в исходных пропорциях
    # Маска уже содержит мягкие градации от 0 до 255, поэтому будет плавная прозрачность.
    result_image = original_image.copy()
    result_image.putalpha(mask_cropped)

    pbar.update(1)  # Обновим прогресс после обработки маски и картинки

    output_path = "no_bg_image.png"
    result_image.save(output_path)
    print(f"Изображение сохранено в {output_path}. Проверяйте прозрачность в редакторе с альфа-каналом.")

    pbar.update(1)  # Обновим прогресс после сохранения файла
