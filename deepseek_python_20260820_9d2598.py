from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os

def draw_layout_boxes(image_path, layout_results, output_path=None, show_labels=True, save=True):
    """
    Функция для визуализации прямоугольников layout detection
    
    Args:
        image_path (str): Путь к исходному изображению
        layout_results (list): Результат из res.json() от LayoutDetection
        output_path (str, optional): Путь для сохранения результата. Если None, 
                                     сохранит с префиксом "layout_"
        show_labels (bool): Показывать ли подписи классов
        save (bool): Сохранять ли результат в файл
    
    Returns:
        PIL.Image.Image: Изображение с нарисованными прямоугольниками
    """
    # Загружаем изображение
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    # Попробуем загрузить шрифт для подписей
    try:
        # Попытка использовать системный шрифт
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 16)
    except:
        try:
            # Для Windows
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            # Если шрифт не найден - используем дефолтный
            font = ImageFont.load_default()
    
    # Цвета для разных классов (можно расширить)
    colors = {
        "title": "#FF0000",       # Красный
        "text": "#00AA00",        # Зеленый
        "table": "#0066FF",       # Синий
        "figure": "#FF8800",      # Оранжевый
        "header": "#9900CC",      # Фиолетовый
        "footer": "#00CCCC",      # Бирюзовый
        "caption": "#CC0066",     # Розовый
        "formula": "#666600",     # Оливковый
        "reference": "#006666",   # Темно-бирюзовый
    }
    default_color = "#FF00FF"     # Маджента для неизвестных классов
    
    # Проходим по всем обнаруженным объектам
    for item in layout_results:
        # Извлекаем координаты
        coords = item.get("coordinate", [])
        if not coords:
            continue
            
        x1, y1, x2, y2 = coords
        label = item.get("label", "unknown")
        score = item.get("score", 0)
        
        # Выбираем цвет для класса
        color = colors.get(label, default_color)
        
        # Рисуем прямоугольник
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Добавляем подпись с названием класса и уверенностью
        if show_labels:
            text = f"{label}: {score:.2f}"
            
            # Рисуем фон для текста (чтобы было читаемо)
            text_bbox = draw.textbbox((x1, y1), text, font=font)
            draw.rectangle(text_bbox, fill="white", outline="white")
            
            # Рисуем текст
            draw.text((x1, y1), text, fill=color, font=font)
    
    # Сохраняем результат
    if save:
        if output_path is None:
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            output_path = f"layout_{name}.png"
        
        image.save(output_path)
        print(f"Визуализация сохранена в: {output_path}")
    
    return image

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
    
    # 3. Визуализируем
    img_with_boxes = draw_layout_boxes(
        image_path="your_document.png",
        layout_results=layout_json,
        output_path="visualized_document.png",
        show_labels=True,
        save=True
    )
    
    # 4. Показываем через matplotlib
    # plt.figure(figsize=(15, 10))
    # plt.imshow(img_with_boxes)
    # plt.axis('off')
    # plt.show()