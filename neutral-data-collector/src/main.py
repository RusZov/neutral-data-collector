#!/usr/bin/env python3
"""
Главный модуль сборщика нейтральных данных.

Использование:
    python main.py --config config.yaml
    
Или программно:
    from src.main import DataCollector
    collector = DataCollector(config)
    collector.run()
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


class DataCollector:
    """Основной класс для сбора и обработки нейтральных данных."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Инициализировать сборщик данных.
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        self.config = self._load_config(config_path)
        self.stats = {
            'search_queries': 0,
            'pages_fetched': 0,
            'texts_extracted': 0,
            'after_neutrality_filter': 0,
            'after_deduplication': 0,
            'final_size': 0,
            'sources': {},
            'neutrality_scores': []
        }
        
        # Импорт компонентов
        from .scraper import WebScraper
        from .neutrality_scorer import NeutralityScorer
        from .deduplicator import TextDeduplicator
        from .balancer import DatasetBalancer
        
        # Инициализация компонентов
        self.scraper = WebScraper(self.config)
        self.scorer = NeutralityScorer(self.config)
        self.deduplicator = TextDeduplicator(threshold=3)
        self.balancer = DatasetBalancer(self.config)
    
    def _load_config(self, config_path: str) -> Dict:
        """Загрузить конфигурацию из YAML файла."""
        config_file = Path(config_path)
        
        if not config_file.exists():
            # Поиск в родительских директориях
            for parent in [Path.cwd()] + list(Path.cwd().parents):
                test_path = parent / config_path
                if test_path.exists():
                    config_file = test_path
                    break
        
        if not config_file.exists():
            print(f"Файл конфигурации не найден: {config_path}")
            print("Использую конфигурацию по умолчанию")
            return self._default_config()
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"Конфигурация загружена из {config_file}")
        return config
    
    def _default_config(self) -> Dict:
        """Конфигурация по умолчанию."""
        return {
            'task': {
                'description': 'нейтральные тексты',
                'keywords': ['информация', 'факты']
            },
            'collection': {
                'target_size': 1000,
                'min_neutrality_score': 0.7,
                'max_results_per_query': 50
            },
            'language': {
                'primary': 'ru',
                'allow_secondary': False
            },
            'domains': {
                'blacklist': ['vk.com', 'facebook.com'],
                'whitelist': []
            },
            'stop_words': ['реклама', 'подпишитесь'],
            'text_processing': {
                'min_length': 100,
                'max_length': 10000
            },
            'cache': {
                'enabled': True,
                'ttl_hours': 24
            },
            'scraping': {
                'delay_between_requests': 1.0,
                'max_retries': 3
            },
            'neutrality_model': {
                'type': 'rule_based'
            },
            'output': {
                'csv_path': 'data/processed/dataset.csv',
                'jsonl_path': 'data/processed/dataset.jsonl',
                'stats_path': 'data/processed/statistics.json'
            }
        }
    
    def generate_queries(self) -> List[str]:
        """Сгенерировать поисковые запросы на основе задачи."""
        task_config = self.config.get('task', {})
        description = task_config.get('description', '')
        keywords = task_config.get('keywords', [])
        
        queries = []
        
        # Добавляем ключевые слова как запросы
        queries.extend(keywords)
        
        # Генерируем комбинированные запросы
        if description:
            queries.append(description)
            
            # Вариации с вопросами
            question_words = ['что такое', 'как', 'почему', 'когда', 'где']
            for qw in question_words[:3]:  # Ограничиваем количество
                queries.append(f"{qw} {description}")
        
        # Удаляем дубликаты
        queries = list(dict.fromkeys(queries))
        
        print(f"Сгенерировано {len(queries)} поисковых запросов")
        return queries
    
    def collect_data(self) -> List[Dict]:
        """Собрать сырые данные."""
        queries = self.generate_queries()
        target_size = self.config.get('collection', {}).get('target_size', 1000)
        
        # Увеличиваем целевой размер для компенсации фильтрации
        # (ожидаем что ~50% пройдут фильтры)
        extended_target = int(target_size * 2)
        
        raw_data = self.scraper.collect(queries, extended_target)
        
        self.stats['texts_extracted'] = len(raw_data)
        self.stats['search_queries'] = len(queries)
        
        # Статистика по источникам
        for item in raw_data:
            source = item.get('source_url', '')
            domain = source.split('/')[2] if '//' in source else 'unknown'
            self.stats['sources'][domain] = self.stats['sources'].get(domain, 0) + 1
        
        return raw_data
    
    def filter_by_neutrality(self, data: List[Dict]) -> List[Dict]:
        """Отфильтровать данные по нейтральности."""
        min_score = self.config.get('collection', {}).get('min_neutrality_score', 0.7)
        
        filtered = self.scorer.filter_by_neutrality(data, min_score)
        
        self.stats['after_neutrality_filter'] = len(filtered)
        self.stats['neutrality_scores'] = [
            item.get('neutrality_score', 0) for item in filtered
        ]
        
        return filtered
    
    def deduplicate(self, data: List[Dict]) -> List[Dict]:
        """Удалить дубликаты."""
        unique_data = self.deduplicator.deduplicate(data)
        self.stats['after_deduplication'] = len(unique_data)
        return unique_data
    
    def balance(self, data: List[Dict]) -> List[Dict]:
        """Сбалансировать датасет."""
        target_size = self.config.get('collection', {}).get('target_size', 1000)
        task_description = self.config.get('task', {}).get('description', '')
        
        balanced_data = self.balancer.process(data, target_size, task_description)
        self.stats['final_size'] = len(balanced_data)
        
        return balanced_data
    
    def save_dataset(
        self,
        data: List[Dict],
        output_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Сохранить датасет в различных форматах.
        
        Returns:
            Dict[str, str]: Пути к сохранённым файлам
        """
        output_config = self.config.get('output', {})
        
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = Path(output_config.get('csv_path', 'data/processed')).parent
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # CSV
        csv_path = output_path / 'dataset.csv'
        df = pd.DataFrame(data)
        
        # Выбираем нужные колонки
        columns = ['id', 'text', 'source_url', 'neutrality_score', 'length', 'timestamp']
        available_columns = [c for c in columns if c in df.columns]
        
        # Добавляем length если нет
        if 'length' not in df.columns:
            df['length'] = df['text'].apply(len)
        
        df[available_columns].to_csv(csv_path, index=False, encoding='utf-8')
        saved_files['csv'] = str(csv_path)
        print(f"Сохранено в CSV: {csv_path}")
        
        # JSONL (для Hugging Face Datasets)
        jsonl_path = output_path / 'dataset.jsonl'
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for item in data:
                # Подготовка для HF Datasets
                hf_item = {
                    'text': item.get('text', ''),
                    'source': item.get('source_url', ''),
                    'neutrality_score': item.get('neutrality_score', 0.0),
                    'length': item.get('length', len(item.get('text', ''))),
                    'timestamp': item.get('timestamp', '')
                }
                # Добавляем label если есть
                if 'label' in item:
                    hf_item['label'] = item['label']
                
                f.write(json.dumps(hf_item, ensure_ascii=False) + '\n')
        
        saved_files['jsonl'] = str(jsonl_path)
        print(f"Сохранено в JSONL: {jsonl_path}")
        
        # Statistics JSON
        stats_path = output_path / 'statistics.json'
        self._save_statistics(stats_path)
        saved_files['stats'] = str(stats_path)
        print(f"Статистика сохранена: {stats_path}")
        
        return saved_files
    
    def _save_statistics(self, path: Path):
        """Сохранить статистику."""
        stats = self.stats.copy()
        
        # Добавляем агрегированные метрики
        if stats['neutrality_scores']:
            scores = stats['neutrality_scores']
            stats['neutrality_metrics'] = {
                'mean': float(np.mean(scores)),
                'median': float(np.median(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'histogram': self._compute_histogram(scores)
            }
        
        # Сериализуемые типы
        stats['timestamp'] = datetime.now().isoformat()
        stats['config'] = self.config
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
    
    def _compute_histogram(
        self,
        values: List[float],
        bins: int = 10
    ) -> Dict[str, int]:
        """Вычислить гистограмму значений."""
        hist, bin_edges = np.histogram(values, bins=bins, range=(0, 1))
        
        histogram_dict = {}
        for i, count in enumerate(hist):
            bin_label = f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}"
            histogram_dict[bin_label] = int(count)
        
        return histogram_dict
    
    def print_statistics(self):
        """Вывести статистику в консоль."""
        print("\n" + "="*60)
        print("СТАТИСТИКА СБОРА ДАННЫХ")
        print("="*60)
        
        print(f"\n📊 Общие показатели:")
        print(f"  Поисковых запросов: {self.stats['search_queries']}")
        print(f"  Текстов извлечено: {self.stats['texts_extracted']}")
        print(f"  После фильтра нейтральности: {self.stats['after_neutrality_filter']}")
        print(f"  После дедупликации: {self.stats['after_deduplication']}")
        print(f"  Финальный размер: {self.stats['final_size']}")
        
        print(f"\n🌐 Распределение по источникам (топ-10):")
        sorted_sources = sorted(
            self.stats['sources'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        for domain, count in sorted_sources:
            print(f"  {domain}: {count}")
        
        if self.stats['neutrality_scores']:
            scores = self.stats['neutrality_scores']
            print(f"\n📈 Метрики нейтральности:")
            print(f"  Среднее: {np.mean(scores):.4f}")
            print(f"  Медиана: {np.median(scores):.4f}")
            print(f"  Стд. отклонение: {np.std(scores):.4f}")
            print(f"  Мин: {np.min(scores):.4f}")
            print(f"  Макс: {np.max(scores):.4f}")
        
        print("\n" + "="*60)
    
    def run(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Запустить полный пайплайн сбора данных.
        
        Args:
            output_dir: Директория для вывода (опционально)
        
        Returns:
            Dict[str, str]: Пути к сохранённым файлам
        """
        print("\n" + "="*60)
        print("ЗАПУСК СБОРЩИКА НЕЙТРАЛЬНЫХ ДАННЫХ")
        print("="*60)
        
        task_desc = self.config.get('task', {}).get('description', 'не задана')
        target_size = self.config.get('collection', {}).get('target_size', 1000)
        print(f"\n🎯 Задача: {task_desc}")
        print(f"📦 Целевой размер: {target_size} текстов")
        
        # Шаг 1: Сбор данных
        print("\n[1/5] Сбор данных...")
        raw_data = self.collect_data()
        
        if not raw_data:
            print("❌ Не удалось собрать данные. Проверьте подключение к интернету.")
            return {}
        
        # Шаг 2: Фильтрация по нейтральности
        print("\n[2/5] Фильтрация по нейтральности...")
        neutral_data = self.filter_by_neutrality(raw_data)
        
        if not neutral_data:
            print("❌ Ни один текст не прошёл фильтр нейтральности.")
            print("   Попробуйте снизить порог min_neutrality_score в конфиге.")
            return {}
        
        # Шаг 3: Дедупликация
        print("\n[3/5] Дедупликация...")
        unique_data = self.deduplicate(neutral_data)
        
        # Шаг 4: Балансировка
        print("\n[4/5] Балансировка...")
        balanced_data = self.balance(unique_data)
        
        # Шаг 5: Сохранение
        print("\n[5/5] Сохранение результатов...")
        saved_files = self.save_dataset(balanced_data, output_dir)
        
        # Вывод статистики
        self.print_statistics()
        
        print("\n✅ Сбор данных завершён успешно!")
        print(f"\nФайлы сохранены:")
        for format_name, path in saved_files.items():
            print(f"  {format_name.upper()}: {path}")
        
        return saved_files


def main():
    """Точка входа для CLI."""
    parser = argparse.ArgumentParser(
        description='Сборщик нейтральных данных для обучения языковых моделей',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main.py                          # Использовать config.yaml по умолчанию
  python main.py --config my_config.yaml  # Использовать свой конфиг
  python main.py --output ./my_data       # Сохранить в указанную директорию
  python main.py --target-size 5000       # Переопределить целевой размер
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Путь к файлу конфигурации (по умолчанию: config.yaml)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Директория для сохранения результатов'
    )
    
    parser.add_argument(
        '--target-size',
        type=int,
        default=None,
        help='Целевой размер датасета (переопределяет config.yaml)'
    )
    
    parser.add_argument(
        '--min-neutrality',
        type=float,
        default=None,
        help='Минимальный порог нейтральности (переопределяет config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Создание сборщика
    collector = DataCollector(args.config)
    
    # Переопределение параметров если указаны
    if args.target_size is not None:
        collector.config['collection']['target_size'] = args.target_size
    
    if args.min_neutrality is not None:
        collector.config['collection']['min_neutrality_score'] = args.min_neutrality
    
    # Запуск
    try:
        collector.run(output_dir=args.output)
    except KeyboardInterrupt:
        print("\n\n⚠️  Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
