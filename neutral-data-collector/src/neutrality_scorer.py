"""
Модуль для оценки нейтральности текста.

Поддерживает два режима:
1. Rule-based (на основе правил) - быстрый, не требует GPU
2. Transformer-based (на основе RuBERT) - более точный, но медленнее
"""

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from langdetect import detect, DetectorFactory, LangDetectException

# Фиксируем seed для воспроизводимости результатов langdetect
DetectorFactory.seed = 0


class EmotionalLexicon:
    """Словарь эмоционально окрашенных слов для русского языка."""
    
    # Позитивные слова
    POSITIVE_WORDS = {
        'отличный', 'прекрасный', 'замечательный', 'хороший', 'лучший',
        'восхищение', 'радость', 'счастье', 'удовольствие', 'восторг',
        'люблю', 'обожаю', 'нравится', 'приятно', 'комфортно',
        'успех', 'достижение', 'победа', 'прогресс', 'развитие',
        'рекомендую', 'советую', 'одобряю', 'поддерживаю', 'благодарю',
        'эффективный', 'качественный', 'надёжный', 'безопасный', 'полезный'
    }
    
    # Негативные слова
    NEGATIVE_WORDS = {
        'плохой', 'ужасный', 'страшный', 'отвратительный', 'мерзкий',
        'боль', 'страдание', 'горе', 'печаль', 'разочарование',
        'ненавижу', 'презираю', 'не нравится', 'неприятно', 'дискомфорт',
        'провал', 'поражение', 'регресс', 'упадок', 'деградация',
        'не рекомендую', 'предупреждаю', 'осуждаю', 'критикую', 'жалоба',
        'неэффективный', 'некачественный', 'ненадёжный', 'опасный', 'вредный',
        'катастрофа', 'кошмар', 'бедствие', 'кризис', 'проблема'
    }
    
    # Слова усиления (модификаторы интенсивности)
    INTENSIFIERS = {
        'очень': 1.5,
        'крайне': 1.8,
        'чрезвычайно': 2.0,
        'особенно': 1.4,
        'действительно': 1.3,
        'абсолютно': 1.7,
        'полностью': 1.6,
        'совершенно': 1.7,
        'невероятно': 1.9,
        'ужасно': 1.8,
        'страшно': 1.6
    }
    
    # Слова ослабления
    DOWNGRADERS = {
        'немного': 0.5,
        'слегка': 0.4,
        'чуть': 0.4,
        'довольно': 0.7,
        'относительно': 0.6,
        'более-менее': 0.5,
        'почти': 0.3,
        'практически': 0.4
    }
    
    @classmethod
    def normalize_word(cls, word: str) -> str:
        """Нормализовать слово (привести к нижней регистру, удалить окончания)."""
        word = word.lower()
        # Простая лемматизация - удаление типичных окончаний
        endings = ['ый', 'ий', 'ый', 'ое', 'ее', 'ая', 'яя', 'ым', 'им', 'ого', 'его']
        for ending in endings:
            if word.endswith(ending) and len(word) > 4:
                word = word[:-len(ending)]
                break
        return word
    
    @classmethod
    def get_emotional_score(cls, text: str) -> Tuple[float, float, float]:
        """
        Вычислить эмоциональную оценку текста.
        
        Returns:
            Tuple[positive_score, negative_score, total_emotional]
        """
        words = re.findall(r'\b[а-яa-z]+\b', text.lower(), re.IGNORECASE | re.UNICODE)
        
        positive_count = 0
        negative_count = 0
        modifier = 1.0
        
        for i, word in enumerate(words):
            normalized = cls.normalize_word(word)
            
            # Проверка на модификаторы
            if word in cls.INTENSIFIERS:
                modifier = max(modifier, cls.INTENSIFIERS[word])
            elif word in cls.DOWNGRADERS:
                modifier = min(modifier, cls.DOWNGRADERS[word])
            # Проверка на позитивные слова
            elif any(normalized in cls.normalize_word(pw) for pw in cls.POSITIVE_WORDS) or \
                 any(normalized in pw for pw in cls.POSITIVE_WORDS):
                positive_count += modifier
                modifier = 1.0  # сброс модификатора
            # Проверка на негативные слова
            elif any(normalized in cls.normalize_word(nw) for nw in cls.NEGATIVE_WORDS) or \
                 any(normalized in nw for nw in cls.NEGATIVE_WORDS):
                negative_count += modifier
                modifier = 1.0  # сброс модификатора
        
        total_words = len(words) if words else 1
        emotional_total = positive_count + negative_count
        
        return positive_count / total_words, negative_count / total_words, emotional_total


class NeutralityScorer:
    """Оценщик нейтральности текста."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_config = config.get('neutrality_model', {})
        self.model_type = self.model_config.get('type', 'rule_based')
        self.batch_size = self.model_config.get('batch_size', 16)
        
        # Загрузка трансформера если нужно
        self.transformer_model = None
        self.transformer_tokenizer = None
        
        if self.model_type == 'transformer':
            self._load_transformer()
    
    def _load_transformer(self):
        """Загрузить предобученную модель для анализа тональности."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            model_name = self.model_config.get(
                'transformer_name',
                'blanchefort/rubert-base-cased-sentiment-rusentiment'
            )
            
            print(f"Загрузка модели {model_name}...")
            self.transformer_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.transformer_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype='auto',
                device_map='cpu'
            )
            self.transformer_model.eval()
            print("Модель загружена успешно")
            
        except Exception as e:
            print(f"Не удалось загрузить трансформер: {e}")
            print("Переключение на rule-based режим")
            self.model_type = 'rule_based'
    
    def detect_language(self, text: str) -> Optional[str]:
        """Определить язык текста."""
        try:
            if len(text) < 10:
                return None
            return detect(text)
        except LangDetectException:
            return None
    
    def is_valid_language(self, text: str) -> bool:
        """Проверить, соответствует ли язык требованиям."""
        lang = self.detect_language(text)
        if lang is None:
            return False
        
        primary_lang = self.config.get('language', {}).get('primary', 'ru')
        allow_secondary = self.config.get('language', {}).get('allow_secondary', False)
        secondary_lang = self.config.get('language', {}).get('secondary', 'en')
        
        if lang == primary_lang:
            return True
        
        if allow_secondary and lang == secondary_lang:
            return True
        
        return False
    
    def score_rule_based(self, text: str) -> float:
        """
        Оценить нейтральность текста на основе правил.
        
        Метрика нейтральности:
        - Чем меньше эмоциональных слов относительно общего числа слов, тем выше нейтральность
        - Баланс между позитивными и негативными словами увеличивает нейтральность
        - Наличие маркеров субъективности снижает нейтральность
        
        Returns:
            float: Оценка нейтральности от 0 до 1
        """
        pos_score, neg_score, emotional_total = EmotionalLexicon.get_emotional_score(text)
        
        # Общее число слов
        words = re.findall(r'\b[а-яa-z]+\b', text.lower(), re.IGNORECASE | re.UNICODE)
        total_words = len(words) if words else 1
        
        # Доля эмоциональных слов
        emotional_ratio = emotional_total / total_words
        
        # Баланс между позитивом и негативом (чем ближе к 0.5, тем лучше)
        if pos_score + neg_score > 0:
            balance = min(pos_score, neg_score) / (pos_score + neg_score)
        else:
            balance = 0.5  # Если нет эмоций, считаем идеально сбалансированным
        
        # Маркеры субъективности
        subjective_markers = [
            'я считаю', 'я думаю', 'по моему мнению', 'мне кажется',
            'i think', 'i believe', 'in my opinion', 'i feel'
        ]
        text_lower = text.lower()
        has_subjectivity = any(marker in text_lower for marker in subjective_markers)
        
        # Маркеры категоричности
        categorical_markers = [
            'всегда', 'никогда', 'однозначно', 'безусловно', 'несомненно',
            'always', 'never', 'definitely', 'undoubtedly'
        ]
        has_categoricity = any(marker in text_lower for marker in categorical_markers)
        
        # Расчёт итоговой оценки
        # Базовая оценка: обратная пропорция от эмоциональности
        base_score = 1.0 - min(emotional_ratio * 10, 1.0)  # Штраф за эмоции
        
        # Бонус за баланс
        balance_bonus = balance * 0.2
        
        # Штраф за субъективность
        subjectivity_penalty = 0.15 if has_subjectivity else 0
        
        # Штраф за категоричность
        categoricity_penalty = 0.1 if has_categoricity else 0
        
        # Итоговая оценка
        neutrality_score = base_score + balance_bonus - subjectivity_penalty - categoricity_penalty
        
        # Нормализация к диапазону [0, 1]
        neutrality_score = max(0.0, min(1.0, neutrality_score))
        
        return neutrality_score
    
    def score_transformer(self, texts: List[str]) -> List[float]:
        """
        Оценить нейтральность текстов с помощью трансформера.
        
        Returns:
            List[float]: Список оценок нейтральности
        """
        import torch
        
        scores = []
        
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                
                inputs = self.transformer_tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                outputs = self.transformer_model(**inputs)
                logits = outputs.logits
                
                # Для sentiment моделей: чем ближе к нейтральному классу, тем выше оценка
                # Предполагаем, что модель имеет классы: negative, neutral, positive
                if logits.shape[1] >= 3:
                    # Берём вероятность нейтрального класса
                    neutral_probs = torch.softmax(logits, dim=1)[:, 1]  # индекс 1 = neutral
                    scores.extend(neutral_probs.cpu().numpy().tolist())
                else:
                    # Бинарная классификация: используем отклонение от 0.5
                    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                    if isinstance(probs, np.ndarray):
                        neutrality_scores = 1.0 - np.abs(probs - 0.5) * 2
                    else:
                        neutrality_scores = 1.0 - abs(probs - 0.5) * 2
                    scores.extend(neutrality_scores.tolist() if hasattr(neutrality_scores, 'tolist') else [neutrality_scores])
        
        return scores
    
    def score(self, text: str) -> float:
        """
        Оценить нейтральность одного текста.
        
        Args:
            text: Текст для оценки
        
        Returns:
            float: Оценка нейтральности от 0 до 1
        """
        if self.model_type == 'transformer' and self.transformer_model:
            scores = self.score_transformer([text])
            return scores[0] if scores else 0.5
        else:
            return self.score_rule_based(text)
    
    def score_batch(self, texts: List[str]) -> List[float]:
        """
        Оценить нейтральность нескольких текстов.
        
        Args:
            texts: Список текстов для оценки
        
        Returns:
            List[float]: Список оценок нейтральности
        """
        if self.model_type == 'transformer' and self.transformer_model:
            return self.score_transformer(texts)
        else:
            return [self.score_rule_based(text) for text in texts]
    
    def filter_by_neutrality(
        self,
        data: List[Dict],
        min_score: float = 0.7
    ) -> List[Dict]:
        """
        Отфильтровать данные по порогу нейтральности.
        
        Args:
            data: Список словарей с данными (должен содержать 'text')
            min_score: Минимальный порог нейтральности
        
        Returns:
            List[Dict]: Отфильтрованные данные с добавленным полем 'neutrality_score'
        """
        filtered = []
        texts = [item['text'] for item in data]
        
        print(f"Оценка нейтральности {len(texts)} текстов...")
        
        # Оценка пачками
        scores = self.score_batch(texts)
        
        for item, score in zip(data, scores):
            item['neutrality_score'] = round(score, 4)
            
            # Проверка языка
            if not self.is_valid_language(item['text']):
                continue
            
            if score >= min_score:
                filtered.append(item)
        
        print(f"Прошли фильтр нейтральности: {len(filtered)} из {len(data)}")
        return filtered
