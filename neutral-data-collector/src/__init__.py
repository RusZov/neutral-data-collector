"""
Модуль инициализации пакета.
"""

from .scraper import WebScraper, CacheManager
from .neutrality_scorer import NeutralityScorer, EmotionalLexicon
from .deduplicator import TextDeduplicator, MinHashDeduplicator
from .balancer import DatasetBalancer, create_stratified_split

__version__ = "1.0.0"
__author__ = "Neutral Data Collector"

__all__ = [
    "WebScraper",
    "CacheManager",
    "NeutralityScorer",
    "EmotionalLexicon",
    "TextDeduplicator",
    "MinHashDeduplicator",
    "DatasetBalancer",
    "create_stratified_split",
]
