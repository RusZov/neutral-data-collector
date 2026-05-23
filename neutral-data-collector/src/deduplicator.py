"""
Модуль для дедупликации текстов с использованием SimHash.
"""

import hashlib
from typing import Dict, List, Set, Tuple
from collections import defaultdict

import numpy as np
from simhash import Simhash, SimhashIndex


class TextDeduplicator:
    """Дедупликатор текстов на основе SimHash."""
    
    def __init__(self, threshold: int = 3):
        """
        Инициализировать дедупликатор.
        
        Args:
            threshold: Порог различия хэшей (0-64). Меньше = строже дедупликация.
        """
        self.threshold = threshold
        self.index = SimhashIndex([], k=self.threshold)
        self.processed_ids: Set[str] = set()
    
    def _extract_features(self, text: str) -> List[int]:
        """
        Извлечь признаки из текста для хеширования.
        
        Использует n-граммы символов для лучшей работы с русским текстом.
        """
        # Нормализация
        text = text.lower().strip()
        
        # Генерация n-грамм (символьных)
        n = 3
        features = []
        for i in range(len(text) - n + 1):
            ngram = text[i:i + n]
            # Хэш n-граммы
            h = hashlib.md5(ngram.encode('utf-8')).hexdigest()
            features.append(int(h, 16))
        
        return features
    
    def compute_simhash(self, text: str) -> Simhash:
        """Вычислить SimHash для текста."""
        features = self._extract_features(text)
        return Simhash(features)
    
    def is_duplicate(self, text: str, existing_id: str = None) -> bool:
        """
        Проверить, является ли текст дубликатом.
        
        Args:
            text: Текст для проверки
            existing_id: ID существующего элемента (для обновления индекса)
        
        Returns:
            bool: True если дубликат
        """
        simhash = self.compute_simhash(text)
        
        # Поиск похожих в индексе
        duplicates = self.index.get_near_dups(simhash)
        
        if duplicates:
            # Если есть дубликаты и мы не обновляем существующий
            if not existing_id or existing_id not in duplicates:
                return True
        
        return False
    
    def add_to_index(self, text: str, item_id: str):
        """Добавить текст в индекс."""
        simhash = self.compute_simhash(text)
        self.index.add(item_id, simhash)
        self.processed_ids.add(item_id)
    
    def deduplicate(
        self,
        data: List[Dict],
        use_url: bool = True
    ) -> List[Dict]:
        """
        Удалить дубликаты из набора данных.
        
        Args:
            data: Список словарей с данными
            use_url: Учитывать URL для быстрой дедупликации
        
        Returns:
            List[Dict]: Данные без дубликатов
        """
        unique_data = []
        seen_urls: Set[str] = set()
        stats = {
            'total': len(data),
            'url_duplicates': 0,
            'content_duplicates': 0,
            'unique': 0
        }
        
        print(f"Дедупликация {len(data)} элементов...")
        
        for item in data:
            item_id = item.get('id', '')
            url = item.get('source_url', '')
            text = item.get('text', '')
            
            # Быстрая проверка по URL
            if use_url and url in seen_urls:
                stats['url_duplicates'] += 1
                continue
            
            # Проверка по содержимому
            if self.is_duplicate(text, item_id):
                stats['content_duplicates'] += 1
                continue
            
            # Добавление в уникальные
            unique_data.append(item)
            seen_urls.add(url)
            self.add_to_index(text, item_id)
            stats['unique'] += 1
        
        print(f"Дедупликация завершена:")
        print(f"  - Дубликатов по URL: {stats['url_duplicates']}")
        print(f"  - Дубликатов по содержимому: {stats['content_duplicates']}")
        print(f"  - Уникальных: {stats['unique']}")
        
        return unique_data
    
    def find_duplicate_groups(
        self,
        data: List[Dict]
    ) -> List[List[Dict]]:
        """
        Найти группы дубликатов.
        
        Args:
            data: Список словарей с данными
        
        Returns:
            List[List[Dict]]: Список групп дубликатов
        """
        # Временный индекс для группировки
        temp_index = SimhashIndex([], k=self.threshold)
        groups = defaultdict(list)
        
        for item in data:
            text = item.get('text', '')
            simhash = self.compute_simhash(text)
            item_id = item.get('id', '')
            
            # Поиск ближайших соседей
            near_dups = temp_index.get_near_dups(simhash)
            
            if near_dups:
                # Добавляем к существующей группе
                group_id = near_dups[0]
                groups[group_id].append(item)
            else:
                # Создаём новую группу
                groups[item_id].append(item)
            
            # Добавляем в индекс
            temp_index.add(item_id, simhash)
        
        # Возвращаем только группы с дубликатами (>1 элемента)
        duplicate_groups = [g for g in groups.values() if len(g) > 1]
        
        print(f"Найдено {len(duplicate_groups)} групп дубликатов")
        
        return duplicate_groups


class MinHashDeduplicator:
    """Альтернативный дедупликатор на основе MinHash (для больших датасетов)."""
    
    def __init__(self, num_perm: int = 128, threshold: float = 0.8):
        """
        Инициализировать MinHash дедупликатор.
        
        Args:
            num_perm: Количество перестановок для MinHash
            threshold: Порог схожести Jaccard (0-1)
        """
        self.num_perm = num_perm
        self.threshold = threshold
        self.hash_functions = self._generate_hash_functions()
    
    def _generate_hash_functions(self) -> List[Tuple[int, int]]:
        """Сгенерировать хэш-функции для MinHash."""
        np.random.seed(42)
        a = np.random.randint(1, 2**32 - 1, size=self.num_perm, dtype=np.uint64)
        b = np.random.randint(0, 2**32 - 1, size=self.num_perm, dtype=np.uint64)
        return list(zip(a.tolist(), b.tolist()))
    
    def _compute_minhash(self, text: str) -> np.ndarray:
        """Вычислить MinHash подпись для текста."""
        # Токенизация
        tokens = set(text.lower().split())
        
        if not tokens:
            return np.zeros(self.num_perm, dtype=np.uint64)
        
        minhash = np.full(self.num_perm, 2**64 - 1, dtype=np.uint64)
        
        for token in tokens:
            token_hash = int(hashlib.md5(token.encode()).hexdigest(), 16)
            
            for i, (a, b) in enumerate(self.hash_functions):
                h = (a * token_hash + b) % (2**64 - 1)
                minhash[i] = min(minhash[i], h)
        
        return minhash
    
    def jaccard_similarity(self, mh1: np.ndarray, mh2: np.ndarray) -> float:
        """Вычислить схожесть Jaccard между двумя MinHash подписями."""
        matches = np.sum(mh1 == mh2)
        return matches / self.num_perm
    
    def deduplicate(self, data: List[Dict]) -> List[Dict]:
        """
        Удалить дубликаты используя MinHash.
        
        Примечание: O(n²) сложность, подходит для датасетов до ~10K элементов.
        Для больших датасетов используйте LSH (Locality Sensitive Hashing).
        """
        if not data:
            return []
        
        unique_data = []
        signatures = []
        
        print(f"MinHash дедупликация {len(data)} элементов...")
        
        for item in data:
            text = item.get('text', '')
            sig = self._compute_minhash(text)
            
            is_dup = False
            for existing_sig in signatures:
                similarity = self.jaccard_similarity(sig, existing_sig)
                if similarity >= self.threshold:
                    is_dup = True
                    break
            
            if not is_dup:
                unique_data.append(item)
                signatures.append(sig)
        
        print(f"Уникальных после MinHash: {len(unique_data)}")
        
        return unique_data
