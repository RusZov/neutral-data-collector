"""
Модуль для скрапинга и извлечения текста из веб-страниц.
"""

import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from readability import Document as ReadabilityDocument
from trafilatura import extract as trafilatura_extract
from tqdm import tqdm


class CacheManager:
    """Управление кэшированием запросов."""
    
    def __init__(self, cache_dir: str = "cache", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600
        self.search_cache_file = self.cache_dir / "search_cache.json"
        self.content_cache_dir = self.cache_dir / "content"
        self.content_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Загрузка кэша поиска
        self.search_cache = {}
        if self.search_cache_file.exists():
            with open(self.search_cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Проверка TTL
                current_time = time.time()
                self.search_cache = {
                    k: v for k, v in data.items()
                    if current_time - v.get('timestamp', 0) < self.ttl_seconds
                }
    
    def _get_content_cache_path(self, url: str) -> Path:
        """Получить путь к файлу кэша для URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.content_cache_dir / f"{url_hash}.html"
    
    def get_search_results(self, query: str) -> Optional[List[Dict]]:
        """Получить результаты поиска из кэша."""
        return self.search_cache.get(query)
    
    def save_search_results(self, query: str, results: List[Dict]):
        """Сохранить результаты поиска в кэш."""
        self.search_cache[query] = {
            'results': results,
            'timestamp': time.time()
        }
        with open(self.search_cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.search_cache, f, ensure_ascii=False, indent=2)
    
    def get_content(self, url: str) -> Optional[str]:
        """Получить HTML содержимое из кэша."""
        cache_path = self._get_content_cache_path(url)
        if cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                return f.read()
        return None
    
    def save_content(self, url: str, html: str):
        """Сохранить HTML содержимое в кэш."""
        cache_path = self._get_content_cache_path(url)
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(html)


class WebScraper:
    """Скрапер для сбора данных из веба."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config.get('scraping', {}).get(
                'user_agent',
                'Mozilla/5.0 (compatible; NeutralDataCollector/1.0)'
            )
        })
        
        # Кэширование
        cache_config = config.get('cache', {})
        self.cache = CacheManager(
            ttl_hours=cache_config.get('ttl_hours', 24)
        ) if cache_config.get('enabled', True) else None
        
        # Параметры скрапинга
        self.delay = config.get('scraping', {}).get('delay_between_requests', 1.0)
        self.max_retries = config.get('scraping', {}).get('max_retries', 3)
        
        # Фильтры доменов
        self.domain_blacklist = set(config.get('domains', {}).get('blacklist', []))
        self.domain_whitelist = set(config.get('domains', {}).get('whitelist', []))
        
        # Стоп-слова
        self.stop_words = [w.lower() for w in config.get('stop_words', [])]
        
        # Языковые настройки
        self.primary_lang = config.get('language', {}).get('primary', 'ru')
        self.allow_secondary = config.get('language', {}).get('allow_secondary', False)
        self.secondary_lang = config.get('language', {}).get('secondary', 'en')
    
    def is_domain_allowed(self, url: str) -> bool:
        """Проверить, разрешён ли домен."""
        domain = urlparse(url).netloc.lower()
        
        # Удалить www. префикс
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Проверка blacklist
        for blocked in self.domain_blacklist:
            if blocked in domain:
                return False
        
        # Проверка whitelist (если задан)
        if self.domain_whitelist:
            return any(allowed in domain for allowed in self.domain_whitelist)
        
        return True
    
    def search(self, query: str, max_results: int = 100) -> List[Dict]:
        """Поиск через DuckDuckGo."""
        if self.cache:
            cached = self.cache.get_search_results(query)
            if cached:
                return cached[:max_results]
        
        results = []
        try:
            with DDGS() as ddgs:
                # DuckDuckGo search с разбивкой на страницы
                for result in ddgs.text(query, max_results=max_results):
                    if self.is_domain_allowed(result.get('href', '')):
                        results.append({
                            'title': result.get('title', ''),
                            'url': result.get('href', ''),
                            'snippet': result.get('body', '')
                        })
        except Exception as e:
            print(f"Ошибка при поиске '{query}': {e}")
        
        # Сохранение в кэш
        if self.cache and results:
            self.cache.save_search_results(query, results)
        
        # Уважение к серверам
        time.sleep(self.delay)
        
        return results
    
    def fetch_content(self, url: str) -> Optional[str]:
        """Загрузить HTML содержимое страницы."""
        if self.cache:
            cached = self.cache.get_content(url)
            if cached:
                return cached
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                html = response.text
                
                if self.cache:
                    self.cache.save_content(url, html)
                
                time.sleep(self.delay)
                return html
            
            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    print(f"Не удалось загрузить {url}: {e}")
                else:
                    time.sleep(self.delay * (attempt + 1))
        
        return None
    
    def extract_text(self, html: str, url: str) -> Optional[str]:
        """Извлечь основной текст из HTML."""
        if not html:
            return None
        
        text = None
        
        # Попытка 1: trafilatura (лучшее качество)
        try:
            text = trafilatura_extract(
                html,
                url=url,
                include_comments=False,
                include_tables=False,
                no_fallback=False
            )
        except Exception:
            pass
        
        # Попытка 2: readability-lxml
        if not text:
            try:
                doc = ReadabilityDocument(html)
                text = doc.summary()
                # Очистка от HTML тегов
                soup = BeautifulSoup(text, 'lxml')
                text = soup.get_text(separator=' ', strip=True)
            except Exception:
                pass
        
        # Попытка 3: базовая очистка BeautifulSoup
        if not text:
            try:
                soup = BeautifulSoup(html, 'lxml')
                # Удаление скриптов и стилей
                for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                    tag.decompose()
                text = soup.get_text(separator=' ', strip=True)
            except Exception:
                pass
        
        return text
    
    def clean_text(self, text: str) -> str:
        """Очистить текст от мусора."""
        if not text:
            return ""
        
        # Нормализация пробелов
        text = re.sub(r'\s+', ' ', text)
        
        # Удаление шаблонных фраз
        patterns = [
            r'cookie\s*политика',
            r'согласен\s*на\s*обработку',
            r'реклама',
            r'подпишитесь?\s*(на|в)?',
            r'©\s*\d{4}',
            r'все\s*права\s*защищены',
            r'использование\s*сайта',
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Проверка на стоп-слова
        text_lower = text.lower()
        if any(sw in text_lower for sw in self.stop_words):
            # Если много стоп-слов, вернуть пустую строку
            stop_word_count = sum(1 for sw in self.stop_words if sw in text_lower)
            if stop_word_count > 2:
                return ""
        
        return text.strip()
    
    def validate_text(self, text: str) -> bool:
        """Проверить текст на соответствие требованиям."""
        if not text:
            return False
        
        min_len = self.config.get('text_processing', {}).get('min_length', 100)
        max_len = self.config.get('text_processing', {}).get('max_length', 10000)
        
        if len(text) < min_len or len(text) > max_len:
            return False
        
        return True
    
    def collect(self, queries: List[str], target_size: int) -> List[Dict]:
        """
        Собрать данные по списку запросов.
        
        Args:
            queries: Список поисковых запросов
            target_size: Целевое количество текстов
        
        Returns:
            Список словарей с данными
        """
        collected_data = []
        all_urls = set()  # Для дедупликации по URL
        
        print(f"Начинаем сбор данных. Цель: {target_size} текстов")
        print(f"Поисковые запросы: {queries}")
        
        with tqdm(total=target_size, desc="Сбор данных") as pbar:
            for query in queries:
                if len(collected_data) >= target_size:
                    break
                
                # Поиск
                search_results = self.search(query, max_results=100)
                
                for result in search_results:
                    if len(collected_data) >= target_size:
                        break
                    
                    url = result['url']
                    
                    # Пропускаем уже обработанные URL
                    if url in all_urls:
                        continue
                    all_urls.add(url)
                    
                    # Загрузка контента
                    html = self.fetch_content(url)
                    if not html:
                        continue
                    
                    # Извлечение текста
                    text = self.extract_text(html, url)
                    if not text:
                        continue
                    
                    # Очистка
                    text = self.clean_text(text)
                    
                    # Валидация
                    if not self.validate_text(text):
                        continue
                    
                    # Добавление в коллекцию
                    collected_data.append({
                        'id': hashlib.md5(f"{url}_{datetime.now()}".encode()).hexdigest(),
                        'text': text,
                        'source_url': url,
                        'title': result.get('title', ''),
                        'snippet': result.get('snippet', ''),
                        'timestamp': datetime.now().isoformat(),
                        'query': query
                    })
                    
                    pbar.update(1)
        
        print(f"Собрано {len(collected_data)} текстов")
        return collected_data
