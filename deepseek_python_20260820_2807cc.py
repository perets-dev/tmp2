import numpy as np
import cv2

def crop_tables_from_array(image_array, layout_results, padding=10, min_size=(50, 50)):
    """
    Обрезает все таблицы из изображения (numpy array) по координатам из layout detection
    
    Args:
        image_array (numpy.ndarray): Исходное изображение в формате (H, W, C) 
                                     (BGR или RGB, поддерживается любое)
        layout_results (list): Результат из res.json() от LayoutDetection
        padding (int): Отступ вокруг таблицы (в пикселях) для захвата границ
        min_size (tuple): Минимальный размер таблицы (ширина, высота) для фильтрации
    
    Returns:
        list: Список словарей с обрезанными таблицами
              [{"array": numpy.ndarray, "coords": [x1, y1, x2, y2], "score": float}, ...]
    """
    # Проверяем входные данные
    if not isinstance(image_array, np.ndarray):
        raise TypeError("image_array должен быть numpy.ndarray")
    
    if len(image_array.shape) == 2:
        # Если изображение черно-белое, преобразуем в 3 канала
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
    
    img_height, img_width = image_array.shape[:2]
    
    # Фильтруем только таблицы с достаточным размером
    tables = []
    for item in layout_results:
        if item.get("label") != "table":
            continue
        
        coords = item.get("coordinate", [])
        if not coords:
            continue
            
        x1, y1, x2, y2 = coords
        width = x2 - x1
        height = y2 - y1
        
        # Фильтруем слишком маленькие таблицы
        if width < min_size[0] or height < min_size[1]:
            continue
            
        tables.append({
            "coords": coords,
            "score": item.get("score", 0),
            "width": width,
            "height": height
        })
    
    if not tables:
        print("Таблицы не найдены или все слишком маленькие")
        return []
    
    # Обрезаем каждую таблицу
    cropped_tables = []
    
    for i, table in enumerate(tables):
        x1, y1, x2, y2 = table["coords"]
        
        # Добавляем отступы
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_width, x2 + padding)
        y2 = min(img_height, y2 + padding)
        
        # Обрезаем массив
        cropped = image_array[y1:y2, x1:x2].copy()
        
        cropped_tables.append({
            "array": cropped,
            "coords": [x1, y1, x2, y2],
            "original_coords": table["coords"],
            "score": table["score"],
            "width": cropped.shape[1],
            "height": cropped.shape[0],
            "index": i
        })
    
    print(f"✅ Обрезано таблиц: {len(cropped_tables)}")
    for i, tbl in enumerate(cropped_tables):
        print(f"   Таблица {i+1}: размер {tbl['width']}x{tbl['height']}px, уверенность {tbl['score']:.2f}")
    
    return cropped_tables

# Расширенная версия с дополнительными возможностями
def crop_tables_advanced(image_array, layout_results, padding=10, 
                         min_size=(50, 50), return_coords_only=False, 
                         preserve_aspect_ratio=False):
    """
    Расширенная версия обрезки таблиц из numpy array
    
    Args:
        image_array (numpy.ndarray): Исходное изображение
        layout_results (list): Результат из res.json() от LayoutDetection
        padding (int): Отступ вокруг таблицы
        min_size (tuple): Минимальный размер таблицы (ширина, высота)
        return_coords_only (bool): Если True, возвращает только координаты без обрезки
        preserve_aspect_ratio (bool): Сохранять ли соотношение сторон при обрезке
    
    Returns:
        dict: {
            "tables": [numpy.ndarray, ...] или [{"array": ..., "coords": ...}, ...],
            "metadata": [...]  # информация о каждой таблице
        }
    """
    if not isinstance(image_array, np.ndarray):
        raise TypeError("image_array должен быть numpy.ndarray")
    
    if len(image_array.shape) == 2:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
    
    img_height, img_width = image_array.shape[:2]
    
    # Извлекаем таблицы
    tables = []
    for item in layout_results:
        if item.get("label") != "table":
            continue
        
        coords = item.get("coordinate", [])
        if not coords:
            continue
        
        x1, y1, x2, y2 = coords
        width = x2 - x1
        height = y2 - y1
        
        if width >= min_size[0] and height >= min_size[1]:
            tables.append({
                "coords": coords,
                "score": item.get("score", 0)
            })
    
    if not tables:
        return {"tables": [], "metadata": []}
    
    cropped_tables = []
    metadata = []
    
    for i, table in enumerate(tables):
        x1, y1, x2, y2 = table["coords"]
        
        # Добавляем отступы
        pad = padding
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(img_width, x2 + pad)
        y2 = min(img_height, y2 + pad)
        
        # Если нужно сохранить соотношение сторон
        if preserve_aspect_ratio:
            # Расширяем до квадрата или другой пропорции
            current_width = x2 - x1
            current_height = y2 - y1
            
            if current_width > current_height:
                diff = current_width - current_height
                y1 = max(0, y1 - diff // 2)
                y2 = min(img_height, y2 + diff // 2)
            else:
                diff = current_height - current_width
                x1 = max(0, x1 - diff // 2)
                x2 = min(img_width, x2 + diff // 2)
        
        # Обрезаем массив
        cropped = image_array[y1:y2, x1:x2].copy()
        
        # Сохраняем метаданные
        meta = {
            "index": i,
            "coords": [x1, y1, x2, y2],
            "original_coords": table["coords"],
            "score": table["score"],
            "size": [cropped.shape[1], cropped.shape[0]],
            "dtype": str(cropped.dtype),
            "shape": cropped.shape
        }
        metadata.append(meta)
        
        cropped_tables.append({
            "array": cropped,
            "metadata": meta
        })
    
    return {
        "tables": cropped_tables,  # список словарей с array и metadata
        "metadata": metadata        # просто список метаданных
    }

# Простая версия - только массивы
def crop_tables_simple(image_array, layout_results, padding=10):
    """
    Простейшая версия - возвращает только список обрезанных массивов
    
    Args:
        image_array (numpy.ndarray): Исходное изображение
        layout_results (list): Результат из res.json() от LayoutDetection
        padding (int): Отступ вокруг таблицы
    
    Returns:
        list: Список обрезанных массивов (numpy.ndarray)
    """
    cropped_arrays = []
    img_height, img_width = image_array.shape[:2]
    
    for item in layout_results:
        if item.get("label") != "table":
            continue
        
        coords = item.get("coordinate", [])
        if not coords:
            continue
        
        x1, y1, x2, y2 = coords
        
        # Добавляем отступы
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_width, x2 + padding)
        y2 = min(img_height, y2 + padding)
        
        cropped = image_array[y1:y2, x1:x2].copy()
        cropped_arrays.append(cropped)
    
    return cropped_arrays

# Пример использования
if __name__ == "__main__":
    import cv2
    from paddleocr import LayoutDetection
    
    # 1. Загружаем изображение как numpy array
    image_path = "your_document.png"
    image = cv2.imread(image_path)  # BGR формат
    
    # 2. Получаем layout detection
    model = LayoutDetection(model_name="PP-DocLayout_plus-L")
    output = model.predict(image_path, batch_size=1, layout_nms=True)
    
    # 3. Извлекаем JSON результаты
    layout_json = []
    for res in output:
        layout_json.extend(res.json())
    
    # 4. Обрезаем таблицы (разные варианты)
    
    # Вариант 1: Полная информация
    result = crop_tables_from_array(image, layout_json, padding=10)
    for table in result:
        print(f"Таблица: {table['array'].shape}, координаты: {table['coords']}")
        # table["array"] - обрезанное изображение
    
    # Вариант 2: Только массивы
    table_arrays = crop_tables_simple(image, layout_json, padding=10)
    for i, arr in enumerate(table_arrays):
        print(f"Таблица {i+1}: {arr.shape}")
        cv2.imwrite(f"table_{i+1}.png", arr)
    
    # Вариант 3: Расширенная версия с метаданными
    result_advanced = crop_tables_advanced(
        image, 
        layout_json, 
        padding=15,
        min_size=(100, 50),
        preserve_aspect_ratio=False
    )
    
    for table in result_advanced["tables"]:
        arr = table["array"]
        meta = table["metadata"]
        print(f"Таблица {meta['index']+1}: размер {meta['size']}, уверенность {meta['score']:.2f}")
        # arr - обрезанное изображение