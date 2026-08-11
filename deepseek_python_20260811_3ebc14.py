import cv2
import numpy as np
from scipy import ndimage
from skimage.transform import radon
from collections import Counter
import matplotlib.pyplot as plt

def preprocess_robust(image):
    """
    Улучшенная предобработка для плохих документов
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 1. Улучшение контраста (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 2. Шумоподавление (сохраняет края)
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # 3. Адаптивная бинаризация (лучше для плохого освещения)
    binary = cv2.adaptiveThreshold(
        denoised, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 2
    )
    
    # 4. Морфологическая очистка
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary

def detect_orientation_radon_fft(image):
    """
    Определение ориентации через преобразование Радона + FFT
    Возвращает: 'horizontal' или 'vertical'
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 1. Подготовка
    gray = cv2.resize(gray, (min(gray.shape[1], 800), min(gray.shape[0], 800)))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. Преобразование Радона
    theta = np.linspace(0., 180., 180, endpoint=False)
    sinogram = radon(binary, theta=theta)
    
    # 3. Анализ вариаций
    variance = np.var(sinogram, axis=0)
    
    # 4. Находим пики с учетом шума
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(variance, height=np.mean(variance) + np.std(variance))
    
    if len(peaks) > 0:
        # Берем самый высокий пик
        best_peak = peaks[np.argmax(variance[peaks])]
        best_angle = theta[best_peak]
    else:
        # Если пиков нет, берем максимум
        best_angle = theta[np.argmax(variance)]
    
    # Определяем группу: для текста 90° в Радоне = горизонтальный текст
    # 0° в Радоне = вертикальный текст
    if 85 <= best_angle <= 95:
        return 'horizontal', 0.8
    elif best_angle < 5 or best_angle > 175:
        return 'vertical', 0.7
    else:
        # Если угол между 5° и 85°, используем дополнительный анализ
        # Сравниваем вариации при 90° и 0°
        var_90 = variance[90] if 90 < len(variance) else 0
        var_0 = variance[0] if 0 < len(variance) else 0
        
        if var_90 > var_0:
            return 'horizontal', var_90 / (var_90 + var_0 + 0.01)
        else:
            return 'vertical', var_0 / (var_90 + var_0 + 0.01)

def detect_orientation_by_morphology(image):
    """
    Определение ориентации через анализ текстуры (морфология)
    Возвращает: 'horizontal' или 'vertical'
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 1. Усиление текстуры
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 2. Градиенты (подсвечивают направление текста)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 3. Вычисляем направление
    angle = np.arctan2(sobely, sobelx) * 180 / np.pi
    
    # 4. Гистограмма направлений
    hist, bins = np.histogram(angle, bins=36, range=(-90, 90))
    
    # 5. Считаем количество горизонтальных и вертикальных краев
    # Горизонтальные: от -15° до 15° (индексы 15-21)
    horizontal_edges = np.sum(hist[15:21])
    # Вертикальные: около 90° (индексы 6-12) и -90° (индексы 24-30)
    vertical_edges = np.sum(hist[6:12]) + np.sum(hist[24:30])
    
    # 6. Определяем группу
    total = horizontal_edges + vertical_edges + 1
    h_score = horizontal_edges / total
    v_score = vertical_edges / total
    
    if h_score > v_score:
        return 'horizontal', h_score
    else:
        return 'vertical', v_score

def detect_orientation_by_layout(image):
    """
    Определение ориентации по структуре документа
    Возвращает: 'horizontal' или 'vertical'
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 1. Усиление границ
    edges = cv2.Canny(gray, 30, 100)
    
    # 2. Морфологическое закрытие для объединения линий
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    horizontal = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    vertical = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # 3. Сравниваем количество горизонтальных и вертикальных линий
    h_lines = np.sum(horizontal) / 255
    v_lines = np.sum(vertical) / 255
    
    # 4. Определяем группу
    total = h_lines + v_lines + 1
    h_score = h_lines / total
    v_score = v_lines / total
    
    if h_score > v_score:
        return 'horizontal', h_score
    else:
        return 'vertical', v_score

def detect_orientation_group(image, debug=False):
    """
    Определяет группу ориентации: 'horizontal' (0/180) или 'vertical' (90/270)
    """
    results = []
    weights = []
    
    # Метод 1: Радон
    try:
        group1, conf1 = detect_orientation_radon_fft(image)
        results.append(group1)
        weights.append(0.4 * conf1)
        if debug:
            print(f"Radon: {group1} (conf={conf1:.2f})")
    except Exception as e:
        if debug:
            print(f"Radon: ошибка - {e}")
    
    # Метод 2: Морфология
    try:
        group2, conf2 = detect_orientation_by_morphology(image)
        results.append(group2)
        weights.append(0.3 * conf2)
        if debug:
            print(f"Morphology: {group2} (conf={conf2:.2f})")
    except Exception as e:
        if debug:
            print(f"Morphology: ошибка - {e}")
    
    # Метод 3: Структура документа
    try:
        group3, conf3 = detect_orientation_by_layout(image)
        results.append(group3)
        weights.append(0.3 * conf3)
        if debug:
            print(f"Layout: {group3} (conf={conf3:.2f})")
    except Exception as e:
        if debug:
            print(f"Layout: ошибка - {e}")
    
    # Голосование с весами
    if not results:
        if debug:
            print("Все методы не сработали, возвращаем 'horizontal' по умолчанию")
        return 'horizontal', 0.0
    
    # Взвешенное голосование
    votes = {'horizontal': 0.0, 'vertical': 0.0}
    for group, weight in zip(results, weights):
        votes[group] = votes.get(group, 0.0) + weight
    
    total_weight = sum(weights)
    if total_weight == 0:
        return 'horizontal', 0.0
    
    # Нормализуем и выбираем победителя
    for group in votes:
        votes[group] = votes[group] / total_weight
    
    best_group = max(votes, key=votes.get)
    confidence = votes[best_group]
    
    if debug:
        print(f"\nГолоса: horizontal={votes['horizontal']:.2f}, vertical={votes['vertical']:.2f}")
        print(f"Итог: {best_group} (уверенность: {confidence:.2f})")
    
    return best_group, confidence

def detect_orientation_ensemble(image, debug=False):
    """
    Обертка для обратной совместимости - возвращает угол
    Теперь определяет только группу: 0 (для горизонтальных) или 90 (для вертикальных)
    """
    group, confidence = detect_orientation_group(image, debug)
    
    if group == 'horizontal':
        return 0, confidence
    else:
        return 90, confidence

def correct_orientation_group(image, min_confidence=0.3, debug=False):
    """
    Коррекция ориентации с определением группы
    """
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image}")
    else:
        img = image.copy()
    
    # Определяем группу
    group, confidence = detect_orientation_group(img, debug)
    
    # Если уверенность низкая, пробуем перебор с оценкой качества
    if confidence < min_confidence:
        if debug:
            print(f"Низкая уверенность ({confidence:.2f}), пробуем перебор...")
        
        # Пробуем горизонтальную и вертикальную ориентацию
        orientations = [
            ('horizontal', 0),
            ('vertical', 90)
        ]
        
        scores = []
        for name, angle in orientations:
            rotated = rotate_image(img, angle)
            score = evaluate_image_quality(rotated)
            scores.append((name, angle, score))
            
            if debug:
                print(f"{name} (угол {angle}°): качество {score:.2f}")
        
        # Выбираем лучшую
        best = max(scores, key=lambda x: x[2])
        group = best[0]
        confidence = best[2] / (sum(s[2] for s in scores) + 0.01)
        
        if debug:
            print(f"Выбрано: {group} (качество: {best[2]:.2f})")
    
    # Определяем угол для поворота
    if group == 'horizontal':
        angle = 0
    else:
        angle = 90
    
    if debug:
        print(f"Финальный результат: группа={group}, угол={angle}°, уверенность={confidence:.2f}")
    
    # Применяем поворот только если это вертикальная группа (90°)
    # Так как для горизонтальной группы поворот не нужен
    if angle != 0 and confidence >= min_confidence:
        corrected = rotate_image(img, angle)
        return corrected, angle, confidence
    else:
        return img, angle, confidence

def evaluate_image_quality(image):
    """
    Оценка качества изображения для выбора лучшей ориентации
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 1. Резкость (градиент)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = np.var(laplacian)
    
    # 2. Контраст
    contrast = np.std(gray) / 255.0
    
    # 3. Количество краев (текст создает края)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges) / (gray.shape[0] * gray.shape[1])
    
    # 4. Энтропия (информативность)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / (hist.sum() + 1e-7)
    entropy = -np.sum(hist * np.log(hist + 1e-7))
    
    # Комбинированная оценка
    score = (
        sharpness / 1000 +
        contrast * 10 +
        edge_density * 100 +
        entropy / 10
    )
    
    return score

def rotate_image(image, angle):
    """
    Поворот изображения
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

def batch_correct_orientation_group(image_paths, output_dir=None, min_confidence=0.3):
    """
    Пакетная обработка с определением группы
    """
    results = []
    
    for img_path in image_paths:
        try:
            corrected, angle, confidence = correct_orientation_group(
                img_path, 
                min_confidence=min_confidence,
                debug=False
            )
            
            group = 'horizontal' if angle == 0 else 'vertical'
            
            result = {
                'path': img_path,
                'group': group,
                'angle': angle,
                'confidence': confidence,
                'success': True,
                'corrected': corrected
            }
            
            if output_dir and corrected is not None:
                import os
                filename = os.path.basename(img_path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}_corrected{ext}")
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

def visualize_group_detection(image_path):
    """
    Визуализация определения группы
    """
    img = cv2.imread(image_path)
    if img is None:
        print("Не удалось загрузить изображение")
        return
    
    # Определяем группу
    group, confidence = detect_orientation_group(img, debug=True)
    
    # Показываем результат
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Оригинал\nГруппа: {group}, уверенность: {confidence:.2f}")
    plt.axis('off')
    
    # Показываем повернутое изображение (если нужно)
    if group == 'vertical':
        corrected = rotate_image(img, 90)
        plt.subplot(2, 2, 2)
        plt.imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
        plt.title(f"Исправлено (поворот на 90°)")
        plt.axis('off')
    else:
        plt.subplot(2, 2, 2)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Горизонтальная группа (поворот не нужен)")
        plt.axis('off')
    
    # Показываем методы
    methods_info = []
    try:
        group_r, conf_r = detect_orientation_radon_fft(img)
        methods_info.append(f"Радон: {group_r} ({conf_r:.2f})")
    except:
        methods_info.append("Радон: ошибка")
    
    try:
        group_m, conf_m = detect_orientation_by_morphology(img)
        methods_info.append(f"Морфология: {group_m} ({conf_m:.2f})")
    except:
        methods_info.append("Морфология: ошибка")
    
    try:
        group_l, conf_l = detect_orientation_by_layout(img)
        methods_info.append(f"Структура: {group_l} ({conf_l:.2f})")
    except:
        methods_info.append("Структура: ошибка")
    
    plt.subplot(2, 2, 3)
    plt.axis('off')
    plt.text(0.1, 0.5, "\n".join(methods_info), fontsize=12, verticalalignment='center')
    plt.title("Результаты методов")
    
    plt.subplot(2, 2, 4)
    plt.axis('off')
    plt.text(0.1, 0.5, 
           f"ИТОГОВЫЙ РЕЗУЛЬТАТ:\n"
           f"Группа: {group}\n"
           f"Уверенность: {confidence:.2f}\n"
           f"Угол для поворота: {0 if group == 'horizontal' else 90}°",
           fontsize=14, verticalalignment='center',
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen"))
    
    plt.tight_layout()
    plt.show()

# Пример использования
if __name__ == "__main__":
    # Обработка одного изображения
    img = cv2.imread("bad_document.jpg")
    if img is not None:
        corrected, angle, confidence = correct_orientation_group(
            img, 
            min_confidence=0.3,
            debug=True
        )
        
        group = 'horizontal' if angle == 0 else 'vertical'
        print(f"\nГруппа: {group}, угол: {angle}°, уверенность: {confidence:.2f}")
        
        if angle != 0:
            cv2.imwrite("corrected_document.jpg", corrected)
            print("Изображение сохранено как 'corrected_document.jpg'")
    
    # Визуализация
    # visualize_group_detection("bad_document.jpg")
    
    # Пакетная обработка
    # results = batch_correct_orientation_group(
    #     ["doc1.jpg", "doc2.jpg", "doc3.jpg"],
    #     output_dir="./corrected",
    #     min_confidence=0.3
    # )
    # for res in results:
    #     print(f"{res['path']}: группа {res.get('group', 'error')}")