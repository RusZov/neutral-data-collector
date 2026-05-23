"""
Модуль для балансировки датасета.

Поддерживает:
1. Одноклассовую задачу (просто накопление до заданного объёма)
2. Многоклассовую задачу с балансировкой по классам
3. Дообучение на небольшом вручную размеченном якоре
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetBalancer:
    """Балансировщик датасета."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.balancing_config = config.get('balancing', {})
        self.enabled = self.balancing_config.get('enabled', False)
        self.anchor_file = self.balancing_config.get('anchor_file', '')
        
        # Целевое распределение классов
        self.target_distribution = self.balancing_config.get(
            'class_distribution',
            {}
        )
    
    def load_anchor_data(self) -> Optional[pd.DataFrame]:
        """Загрузить данные ручной разметки (якорь)."""
        if not self.anchor_file or not Path(self.anchor_file).exists():
            return None
        
        print(f"Загрузка якорных данных из {self.anchor_file}")
        
        # Поддержка CSV и JSONL
        if self.anchor_file.endswith('.csv'):
            df = pd.read_csv(self.anchor_file)
        elif self.anchor_file.endswith('.jsonl'):
            df = pd.read_json(self.anchor_file, lines=True)
        elif self.anchor_file.endswith('.json'):
            df = pd.read_json(self.anchor_file)
        else:
            print(f"Неподдерживаемый формат якорного файла: {self.anchor_file}")
            return None
        
        print(f"Загружено {len(df)} якорных примеров")
        return df
    
    def classify_by_keywords(
        self,
        text: str,
        keywords: Dict[str, List[str]]
    ) -> Optional[str]:
        """
        Классифицировать текст по ключевым словам.
        
        Args:
            text: Текст для классификации
            keywords: Словарь {класс: [ключевые_слова]}
        
        Returns:
            str или None: Название класса или None если не определено
        """
        text_lower = text.lower()
        scores = {}
        
        for class_name, words in keywords.items():
            score = sum(1 for word in words if word.lower() in text_lower)
            if score > 0:
                scores[class_name] = score
        
        if not scores:
            return None
        
        # Возвращаем класс с максимальнымscore
        return max(scores, key=scores.get)
    
    def auto_label(
        self,
        data: List[Dict],
        task_description: str
    ) -> List[Dict]:
        """
        Автоматически проставить метки на основе описания задачи.
        
        Для юридической задачи: "за", "против", "нейтрально"
        Для медицинской: "диагностика", "лечение", "профилактика"
        и т.д.
        """
        # Простая эвристика на основе ключевых слов
        keyword_sets = {
            'legal': {
                'за': ['право', 'разрешено', 'можно', 'следует', 'обязан'],
                'против': ['запрещено', 'нельзя', 'недопустимо', 'нарушение'],
                'нейтрально': ['статья', 'закон', 'кодекс', 'пункт', 'часть']
            },
            'medical': {
                'диагностика': ['симптом', 'диагноз', 'обследование', 'анализ'],
                'лечение': ['лечение', 'терапия', 'препарат', 'лекарство'],
                'профилактика': ['профилактика', 'предотвращение', 'вакцина']
            },
            'technical': {
                'инструкция': ['как', 'шаг', 'способ', 'метод', 'алгоритм'],
                'описание': ['устройство', 'структура', 'компонент', 'модуль'],
                'примеры': ['пример', 'код', 'snippet', 'демо']
            }
        }
        
        # Определяем тип задачи
        task_type = 'legal'  # по умолчанию
        if 'медицин' in task_description.lower() or 'medical' in task_description.lower():
            task_type = 'medical'
        elif 'техническ' in task_description.lower() or 'technical' in task_description.lower():
            task_type = 'technical'
        
        keywords = keyword_sets.get(task_type, keyword_sets['legal'])
        
        # Проставляем метки
        labeled_data = []
        for item in data:
            text = item.get('text', '')
            label = self.classify_by_keywords(text, keywords)
            
            if label:
                item['label'] = label
                labeled_data.append(item)
            else:
                # Неразмеченные помечаем как 'other'
                item['label'] = 'other'
                labeled_data.append(item)
        
        return labeled_data
    
    def balance_classes(
        self,
        data: List[Dict],
        target_size: int
    ) -> List[Dict]:
        """
        Сбалансировать классы в датасете.
        
        Args:
            data: Данные с метками
            target_size: Целевой размер датасета
        
        Returns:
            List[Dict]: Сбалансированные данные
        """
        # Группировка по классам
        by_class = {}
        for item in data:
            label = item.get('label', 'unknown')
            if label not in by_class:
                by_class[label] = []
            by_class[label].append(item)
        
        print("Распределение по классам до балансировки:")
        for label, items in by_class.items():
            print(f"  {label}: {len(items)}")
        
        # Если целевое распределение не задано, делаем равномерное
        if not self.target_distribution:
            n_classes = len(by_class)
            target_per_class = target_size // n_classes
        else:
            # Используем заданное распределение
            total_weight = sum(self.target_distribution.values())
            target_per_class = {
                cls: int(target_size * weight / total_weight)
                for cls, weight in self.target_distribution.items()
            }
        
        # Сэмплирование
        balanced_data = []
        stats = {'original': {}, 'sampled': {}}
        
        for label, items in by_class.items():
            stats['original'][label] = len(items)
            
            if isinstance(target_per_class, dict):
                target_count = target_per_class.get(label, len(items))
            else:
                target_count = target_per_class
            
            if len(items) > target_count:
                # Downsampling
                sampled = random.sample(items, target_count)
                stats['sampled'][label] = target_count
            else:
                # Upsampling (с повторениями если нужно)
                sampled = items.copy()
                while len(sampled) < target_count:
                    sampled.append(random.choice(items))
                stats['sampled'][label] = len(sampled)
            
            balanced_data.extend(sampled)
        
        # Перемешивание
        random.shuffle(balanced_data)
        
        print("\nРаспределение после балансировки:")
        for label, count in stats['sampled'].items():
            print(f"  {label}: {count} (было: {stats['original'].get(label, 0)})")
        
        return balanced_data
    
    def augment_with_anchor(
        self,
        collected_data: List[Dict],
        anchor_data: pd.DataFrame,
        target_size: int
    ) -> List[Dict]:
        """
        Дополнить собранные данные якорными примерами.
        
        Args:
            collected_data: Собранные данные
            anchor_data: Якорные данные (ручная разметка)
            target_size: Целевой размер
        
        Returns:
            List[Dict]: Объединённые данные
        """
        # Конвертация anchor_data в список словарей
        anchor_dicts = anchor_data.to_dict('records')
        
        # Проверка на дубликаты
        collected_texts = {item.get('text', '') for item in collected_data}
        unique_anchors = [
            a for a in anchor_dicts
            if a.get('text', '') not in collected_texts
        ]
        
        print(f"Уникальных якорных примеров: {len(unique_anchors)}")
        
        # Объединение
        combined = collected_data + unique_anchors
        
        # Если нужно больше данных - дополняем якорями с повторениями
        if len(combined) < target_size and unique_anchors:
            needed = target_size - len(combined)
            additional = random.choices(unique_anchors, k=needed)
            combined.extend(additional)
            print(f"Дополнено {needed} якорных примеров")
        
        return combined[:target_size]
    
    def process(
        self,
        data: List[Dict],
        target_size: int,
        task_description: str = ""
    ) -> List[Dict]:
        """
        Основной метод обработки данных.
        
        Args:
            data: Исходные данные
            target_size: Целевой размер датасета
            task_description: Описание задачи для авто-лейблинга
        
        Returns:
            List[Dict]: Обработанные данные
        """
        if not self.enabled:
            # Просто обрезаем до target_size
            print("Балансировка отключена. Одноклассовая задача.")
            return data[:target_size]
        
        print("Включена многоклассовая балансировка")
        
        # Загрузка якорных данных если есть
        anchor_data = self.load_anchor_data()
        
        if anchor_data is not None:
            # Объединение с якорными данными
            data = self.augment_with_anchor(data, anchor_data, target_size)
        
        # Авто-лейблинг если нет меток
        if data and 'label' not in data[0]:
            print("Автоматическая разметка данных...")
            data = self.auto_label(data, task_description)
        
        # Балансировка классов
        if self.target_distribution or len(set(item.get('label', '') for item in data)) > 1:
            data = self.balance_classes(data, target_size)
        
        return data[:target_size]


def create_stratified_split(
    data: List[Dict],
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Создать стратифицированное разбиение на train/val/test.
    
    Args:
        data: Данные с метками
        test_size: Доля тестовой выборки
        val_size: Доля валидационной выборки
        random_state: Seed для воспроизводимости
    
    Returns:
        Tuple[train_data, val_data, test_data]
    """
    df = pd.DataFrame(data)
    
    if 'label' not in df.columns:
        # Если нет меток, просто случайное разбиение
        indices = np.random.permutation(len(df))
        test_end = int(len(df) * test_size)
        val_end = test_end + int(len(df) * val_size)
        
        test_data = df.iloc[indices[:test_end]].to_dict('records')
        val_data = df.iloc[indices[test_end:val_end]].to_dict('records')
        train_data = df.iloc[indices[val_end:]].to_dict('records')
    else:
        # Стратифицированное разбиение
        train_df, temp_df = train_test_split(
            df,
            test_size=test_size + val_size,
            stratify=df['label'],
            random_state=random_state
        )
        
        if val_size > 0:
            val_ratio = val_size / (test_size + val_size)
            val_df, test_df = train_test_split(
                temp_df,
                test_size=1 - val_ratio,
                stratify=temp_df['label'],
                random_state=random_state
            )
        else:
            val_df = pd.DataFrame()
            test_df = temp_df
        
        train_data = train_df.to_dict('records')
        val_data = val_df.to_dict('records')
        test_data = test_df.to_dict('records')
    
    print(f"Разбиение данных:")
    print(f"  Train: {len(train_data)}")
    print(f"  Val: {len(val_data)}")
    print(f"  Test: {len(test_data)}")
    
    return train_data, val_data, test_data
