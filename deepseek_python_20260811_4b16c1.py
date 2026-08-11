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
    Работает даже на очень плохих изображениях
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
    
    # 5. Для текста: угол в Радоне сдвинут на 90°
    rotation_angle = (best_angle - 90) % 180
    
    # 6. Округляем до ближайшего кратного 90
    return round(rotation_angle / 90) * 90

def detect_orientation_by_morphology(image):
    """
    Определение ориентации через анализ текстуры (морфология)
    Хорошо работает на документах с плохим текстом
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
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    angle = np.arctan2(sobely, sobelx) * 180 / np.pi
    
    # 4. Гистограмма направлений
    hist, bins = np.histogram(angle, bins=36, range=(-90, 90))
    
    # 5. Находим главное направление
    main_direction = bins[np.argmax(hist)]
    
    # 6. Округляем до 0, 90, 180, 270
    if abs(main_direction) < 45:
        return 0
    elif main_direction > 45:
        return 90
    else:
        return 270

def detect_orientation_by_layout(image):
    """
    Определение ориентации по структуре документа
    Работает даже когда текст почти не виден
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
    
    # 4. Определяем ориентацию документа
    if h_lines > v_lines * 1.5:
        # Больше горизонтальных линий
        return 0
    elif v_lines > h_lines * 1.5:
        # Больше вертикальных линий
        return 90
    else:
        # Неопределенно
        return None

def detect_orientation_ensemble(image, debug=False):
    """
    Ансамбль методов для максимальной стабильности
    """
    results = []
    weights = []
    
    # Метод 1: Радон (надежный)
    try:
        angle1 = detect_orientation_radon_fft(image)
        results.append(angle1)
        weights.append(0.4)
        if debug:
            print(f"Radon: {angle1}°")
    except:
        pass
    
    # Метод 2: Морфология
    try:
        angle2 = detect_orientation_by_morphology(image)
        results.append(angle2)
        weights.append(0.3)
        if debug:
            print(f"Morphology: {angle2}°")
    except:
        pass
    
    # Метод 3: Анализ выступов (если текст читаемый)
    try:
        from orientation_advanced import detect_orientation_russian
        angle3, conf = detect_orientation_russian(image)
        if conf > 0.3:
            results.append(angle3)
            weights.append(0.3 * conf)
            if debug:
                print(f"Letters: {angle3}° (conf={conf:.2f})")
    except:
        pass
    
    # Метод 4: Структура документа
    try:
        angle4 = detect_orientation_by_layout(image)
        if angle4 is not None:
            results.append(angle4)
            weights.append(0.2)
            if debug:
                print(f"Layout: {angle4}°")
    except:
        pass
    
    # Голосование с весами
    if not results:
        return 0, 0.0
    
    # Взвешенное голосование
    angle_votes = {}
    for angle, weight in zip(results, weights):
        angle_votes[angle] = angle_votes.get(angle, 0) + weight
    
    best_angle = max(angle_votes, key=angle_votes.get)
    confidence = angle_votes[best_angle] / sum(weights)
    
    if debug:
        print(f"\nИтог: {best_angle}° (уверенность: {confidence:.2f})")
        print(f"Голоса: {angle_votes}")
    
    return best_angle, confidence

def correct_orientation_robust(image, min_confidence=0.3, debug=False):
    """
    Основная функция коррекции для плохих документов
    """
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image}")
    else:
        img = image.copy()
    
    # Определяем ориентацию ансамблем методов
    angle, confidence = detect_orientation_ensemble(img, debug)
    
    # Если уверенность низкая, пробуем перебор с оценкой
    if confidence < min_confidence:
        if debug:
            print("Низкая уверенность, пробуем перебор...")
        
        # Пробуем все углы и оцениваем качество
        scores = []
        for test_angle in [0, 90, 180, 270]:
            rotated = rotate_image(img, test_angle)
            # Оцениваем качество: резкость, контраст, наличие текста
            score = evaluate_image_quality(rotated)
            scores.append(score)
            
            if debug:
                print(f"Угол {test_angle}°: качество {score:.2f}")
        
        best_idx = np.argmax(scores)
        angle = [0, 90, 180, 270][best_idx]
        confidence = scores[best_idx] / max(scores) if max(scores) > 0 else 0
    
    # Применяем поворот
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
        sharpness / 1000 +  # Нормализуем
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

def batch_correct_orientation_robust(image_paths, output_dir=None, min_confidence=0.25):
    """
    Пакетная обработка с улучшенным алгоритмом
    """
    results = []
    
    for img_path in image_paths:
        try:
            corrected, angle, confidence = correct_orientation_robust(
                img_path, 
                min_confidence=min_confidence,
                debug=False
            )
            
            result = {
                'path': img_path,
                'angle': angle,
                'confidence': confidence,
                'success': True,
                'corrected': corrected
            }
            
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

# Пример использования
if __name__ == "__main__":
    # Обработка одного изображения
    img = cv2.imread("bad_document.jpg")
    if img is not None:
        corrected, angle, confidence = correct_orientation_robust(
            img, 
            min_confidence=0.25,
            debug=True
        )
        
        print(f"Угол: {angle}° (уверенность: {confidence:.2f})")
        cv2.imwrite("corrected_document.jpg", corrected)
    
    # Пакетная обработка
    # results = batch_correct_orientation_robust(
    #     ["doc1.jpg", "doc2.jpg", "doc3.jpg"],
    #     output_dir="./corrected",
    #     min_confidence=0.25
    # )