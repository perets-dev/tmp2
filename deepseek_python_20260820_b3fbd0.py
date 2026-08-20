from PIL import Image
import os
import json

def crop_tables_from_image(image_path, layout_results, output_dir="cropped_tables", padding=10):
    """
    Обрезает все таблицы из изображения по координатам из layout detection
    
    Args:
        image_path (str): Путь к исходному изображению
        layout_results (list): Результат из res.json() от LayoutDetection
        output_dir (str): Директория для сохранения обрезанных таблиц
        padding (int): Отступ вокруг таблицы (в пикселях) для захвата границ
    
    Returns:
        list: Список путей к сохраненным изображениям таблиц
    """
    # Создаем директорию для сохранения
    os.makedirs(output_dir, exist_ok=True)
    
    # Загружаем изображение
    image = Image.open(image_path).convert("RGB")
    img_width, img_height = image.size
    
    # Фильтруем только таблицы
    tables = [item for item in layout_results if item.get("label") == "table"]
    
    if not tables:
        print("Таблицы не найдены на изображении")
        return []
    
    # Сохраняем каждую таблицу
    saved_paths = []
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    for i, table in enumerate(tables):
        # Извлекаем координаты
        coords = table.get("coordinate", [])
        if not coords:
            continue
            
        x1, y1, x2, y2 = coords
        
        # Добавляем отступы (чтобы не обрезать края таблицы)
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_width, x2 + padding)
        y2 = min(img_height, y2 + padding)
        
        # Обрезаем изображение
        cropped = image.crop((x1, y1, x2, y2))
        
        # Сохраняем
        output_path = os.path.join(output_dir, f"{base_name}_table_{i+1}.png")
        cropped.save(output_path)
        saved_paths.append(output_path)
        
        print(f"Таблица {i+1} сохранена: {output_path}")
        print(f"  Координаты: [{x1}, {y1}, {x2}, {y2}]")
        print(f"  Размер: {cropped.size[0]}x{cropped.size[1]} пикселей")
    
    return saved_paths

# Пример использования
if __name__ == "__main__":
    from paddleocr import LayoutDetection
    
    # 1. Получаем layout detection
    model = LayoutDetection(model_name="PP-DocLayout_plus-L")
    output = model.predict("your_document.png", batch_size=1, layout_nms=True)
    
    # 2. Извлекаем JSON результаты
    layout_json = []
    for res in output:
        layout_json.extend(res.json())
    
    # 3. Обрезаем все таблицы
    cropped_tables = crop_tables_from_image(
        image_path="your_document.png",
        layout_results=layout_json,
        output_dir="cropped_tables",
        padding=10
    )
    
    # 4. Теперь можно обработать каждую таблицу отдельно
    for table_path in cropped_tables:
        print(f"Обработка: {table_path}")
        # Здесь можно добавить запуск SLANeXt для распознавания структуры
        # или передать в Qwen-VL для OCR