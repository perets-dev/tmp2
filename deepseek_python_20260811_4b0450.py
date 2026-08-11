import cv2
import numpy as np
from scipy import ndimage
from collections import Counter
from scipy.signal import find_peaks

def preprocess_image(image):
    """
    Предобработка изображения
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Улучшение контраста
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Бинаризация
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Морфологическая очистка
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary

def find_connected_components(binary_img, min_area=30):
    """
    Находит связные компоненты (буквы)
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_img, connectivity=8
    )
    
    components = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_area:
            components.append({
                'x': stats[i, cv2.CC_STAT_LEFT],
                'y': stats[i, cv2.CC_STAT_TOP],
                'width': stats[i, cv2.CC_STAT_WIDTH],
                'height': stats[i, cv2.CC_STAT_HEIGHT],
                'area': area,
                'aspect_ratio': stats[i, cv2.CC_STAT_WIDTH] / stats[i, cv2.CC_STAT_HEIGHT] 
                               if stats[i, cv2.CC_STAT_HEIGHT] > 0 else 0
            })
    
    return components

def analyze_letter_shape(roi):
    """
    Анализирует форму буквы для определения наличия выступов
    """
    if roi.size == 0:
        return None
    
    height, width = roi.shape
    if height < 5 or width < 5:
        return None
    
    # Горизонтальная и вертикальная проекции
    row_sum = np.sum(roi, axis=1)
    col_sum = np.sum(roi, axis=0)
    
    has_ascender = False
    has_descender = False
    
    # Анализ выступов сверху и снизу
    if len(row_sum) >= 5:
        top_part = row_sum[:max(1, height//4)]
        bottom_part = row_sum[max(1, 3*height//4):]
        middle_part = row_sum[max(1, height//4):max(1, 3*height//4)]
        
        if len(middle_part) > 0:
            # Выступ сверху (буквы р, т, Т)
            if len(top_part) > 0 and np.mean(top_part) > np.mean(middle_part) * 1.5:
                has_ascender = True
            
            # Выступ снизу (буквы у, д, з)
            if len(bottom_part) > 0 and np.mean(bottom_part) > np.mean(middle_part) * 1.5:
                has_descender = True
    
    # Дополнительная проверка для буквы Т
    if not has_ascender and not has_descender:
        if width > height * 0.7 and len(col_sum) > 5:
            top_half = col_sum[:len(col_sum)//2]
            bottom_half = col_sum[len(col_sum)//2:]
            if np.mean(top_half) > np.mean(bottom_half) * 1.5:
                has_ascender = True
    
    return {
        'has_ascender': has_ascender,
        'has_descender': has_descender
    }

def extract_text_features(binary_img, min_area=30):
    """
    Извлекает признаки из всех букв на изображении
    """
    components = find_connected_components(binary_img, min_area)
    
    features = []
    for comp in components:
        roi = binary_img[comp['y']:comp['y']+comp['height'], 
                        comp['x']:comp['x']+comp['width']]
        
        shape_features = analyze_letter_shape(roi)
        if shape_features:
            features.append({
                **comp,
                **shape_features
            })
    
    return features

def evaluate_orientation(text_features, img_shape, angle):
    """
    Оценивает качество ориентации текста при заданном угле
    """
    height, width = img_shape[:2] if len(img_shape) == 2 else img_shape[:2]
    score = 0
    
    for feature in text_features:
        y = feature['y']
        x = feature['x']
        comp_height = feature['height']
        comp_width = feature['width']
        
        if angle == 0:
            # Нормальная ориентация: выступы сверху в верхней части, снизу - в нижней
            if feature['has_ascender'] and y < height * 0.5:
                score += 1
            elif feature['has_descender'] and y + comp_height > height * 0.5:
                score += 1
                
        elif angle == 180:
            # Перевернутая: все наоборот
            if feature['has_ascender'] and y + comp_height > height * 0.5:
                score += 1
            elif feature['has_descender'] and y < height * 0.5:
                score += 1
                
        elif angle == 90:
            # Поворот на 90°: проверяем по горизонтали
            if feature['has_ascender'] and x < width * 0.5:
                score += 0.5
            elif feature['has_descender'] and x + comp_width > width * 0.5:
                score += 0.5
                
        elif angle == 270:
            if feature['has_ascender'] and x + comp_width > width * 0.5:
                score += 0.5
            elif feature['has_descender'] and x < width * 0.5:
                score += 0.5
    
    return score

def detect_orientation_russian(image, debug=False):
    """
    Определяет угол поворота изображения с русским текстом
    
    Args:
        image: изображение (numpy array)
        debug: если True, выводит отладочную информацию
    
    Returns:
        tuple: (best_angle, confidence)
    """
    # Предобработка
    binary = preprocess_image(image)
    
    # Извлечение признаков букв
    text_features = extract_text_features(binary)
    
    if len(text_features) < 3:
        if debug:
            print("Недостаточно букв для анализа")
        return 0, 0.0
    
    # Проверяем все углы
    angles = [0, 90, 180, 270]
    scores = []
    
    for angle in angles:
        score = evaluate_orientation(text_features, binary.shape, angle)
        scores.append(score)
        if debug:
            print(f"Угол {angle}°: оценка {score}")
    
    # Выбираем лучший угол
    best_idx = np.argmax(scores)
    best_angle = angles[best_idx]
    
    # Вычисляем уверенность
    total_score = sum(scores)
    confidence = scores[best_idx] / total_score if total_score > 0 else 0
    
    # Проверяем альтернативный вариант (180° vs 0°)
    # Если разница маленькая, возможно текст не содержит характерных букв
    if abs(scores[0] - scores[2]) < 2 and len(text_features) < 10:
        if debug:
            print("Низкая уверенность: мало букв с выступами")
        confidence = min(confidence, 0.5)
    
    return best_angle, confidence

def rotate_image(image, angle):
    """
    Поворачивает изображение на заданный угол
    """
    if angle == 0:
        return image
    
    if len(image.shape) == 3:
        height, width = image.shape[:2]
    else:
        height, width = image.shape
    
    center = (width // 2, height // 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    
    return rotated

def correct_orientation(image, min_confidence=0.3, debug=False):
    """
    Основная функция: определяет и исправляет ориентацию
    
    Args:
        image: изображение или путь к файлу
        min_confidence: минимальная уверенность для применения поворота
        debug: если True, выводит отладочную информацию
    
    Returns:
        tuple: (исправленное_изображение, угол_поворота, уверенность)
    """
    # Загрузка изображения
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image}")
    else:
        img = image.copy()
    
    # Определение ориентации
    angle, confidence = detect_orientation_russian(img, debug)
    
    if debug:
        print(f"Обнаруженный угол: {angle}°")
        print(f"Уверенность: {confidence:.2f}")
    
    # Применяем поворот только если уверенность достаточна
    if angle != 0 and confidence >= min_confidence:
        corrected = rotate_image(img, angle)
        return corrected, angle, confidence
    else:
        return img, angle, confidence

def batch_correct_orientation(image_paths, output_dir=None, min_confidence=0.3):
    """
    Обработка нескольких изображений
    
    Args:
        image_paths: список путей к изображениям
        output_dir: директория для сохранения результатов (опционально)
        min_confidence: минимальная уверенность для поворота
    
    Returns:
        list: список результатов
    """
    results = []
    
    for img_path in image_paths:
        try:
            corrected, angle, confidence = correct_orientation(img_path, min_confidence)
            
            result = {
                'path': img_path,
                'angle': angle,
                'confidence': confidence,
                'success': True,
                'corrected': corrected
            }
            
            # Сохранение результата
            if output_dir and corrected is not None:
                import os
                filename = os.path.basename(img_path)
                output_path = os.path.join(output_dir, f"corrected_{filename}")
                cv2.imwrite(output_path, corrected)
                result['saved_path'] = output_path
            
            results.append(result)
            
        except Exception as e:
            results.append({
                'path': img_path,
                'success': False,
                'error': str(e)
            })
    
    return results

# Вспомогательная функция для визуализации
def visualize_correction(image_path):
    """
    Визуализирует процесс коррекции ориентации
    """
    import matplotlib.pyplot as plt
    
    # Загрузка изображения
    original = cv2.imread(image_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Коррекция
    corrected, angle, confidence = correct_orientation(original, debug=True)
    corrected_rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
    
    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original_rgb)
    axes[0].set_title(f"Оригинал\n(неизвестная ориентация)")
    axes[0].axis('off')
    
    axes[1].imshow(corrected_rgb)
    axes[1].set_title(f"Исправлено\nУгол: {angle}°, Уверенность: {confidence:.2f}")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return corrected, angle, confidence

# Пример использования
if __name__ == "__main__":
    # Простой пример
    img = cv2.imread("document.jpg")
    if img is not None:
        # Определяем и исправляем ориентацию
        corrected_img, angle, confidence = correct_orientation(
            img, 
            min_confidence=0.3, 
            debug=True
        )
        
        # Сохраняем результат
        if angle != 0:
            cv2.imwrite("document_corrected.jpg", corrected_img)
            print(f"Изображение исправлено (угол: {angle}°)")
        else:
            print("Изображение уже в правильной ориентации")
    
    # Пакетная обработка
    # results = batch_correct_orientation(
    #     ["doc1.jpg", "doc2.jpg", "doc3.jpg"],
    #     output_dir="./corrected",
    #     min_confidence=0.3
    # )
    # for res in results:
    #     print(f"{res['path']}: угол {res.get('angle', 'error')}°")