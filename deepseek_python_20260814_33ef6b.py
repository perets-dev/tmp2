import re

def fix_personal_numbers(text: str) -> str:
    """
    Находит секцию между @ СПИСОК ВСЕХ ЛИЧНЫХ НОМЕРОВ ГРАЖДАН РБ: и @
    и заменяет все 8 на B внутри этой секции
    """
    # Шаблон: находим всю секцию между маркерами
    # (.*?) - захватываем все содержимое между маркерами
    pattern = r'(@\s*СПИСОК\s+ВСЕХ\s+ЛИЧНЫХ\s+НОМЕРОВ\s+ГРАЖДАН\s+РБ\s*:\s*)(.*?)(@)'
    
    def replace_in_section(match):
        prefix = match.group(1)  # Начальный маркер
        content = match.group(2)  # Содержимое с номерами
        suffix = match.group(3)   # Конечный маркер @
        
        # Заменяем все 8 на B внутри секции
        fixed_content = content.replace('8', 'B')
        
        # Возвращаем обновленную секцию
        return prefix + fixed_content + suffix
    
    # Применяем замену
    result = re.sub(pattern, replace_in_section, text, flags=re.IGNORECASE | re.DOTALL)
    
    return result

# Пример использования
if __name__ == "__main__":
    sample_text = """
    Текст документа...
    
    @ СПИСОК ВСЕХ ЛИЧНЫХ НОМЕРОВ ГРАЖДАН РБ:
    1234567A123ABC8, 8765432B456DEF1, 1111111C789GHI2, 2222222D012JKL8
    @
    
    Продолжение документа
    """
    
    print("Исходный текст:")
    print(sample_text)
    print("\n" + "="*50 + "\n")
    
    fixed_text = fix_personal_numbers(sample_text)
    
    print("Исправленный текст:")
    print(fixed_text)