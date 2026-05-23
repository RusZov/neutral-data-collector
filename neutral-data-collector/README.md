# Neutral Data Collector

Пайплайн для сбора нейтральных текстовых данных из открытых источников для обучения локальных языковых моделей.

## 🎯 Возможности

- **Поиск данных**: Автоматический поиск через DuckDuckGo по ключевым словам
- **Извлечение текста**: Умное извлечение основного контента из HTML (trafilatura, readability-lxml)
- **Детекция языка**: Фильтрация по языку (русский + опционально английский)
- **Оценка нейтральности**: 
  - Rule-based подход (быстрый, CPU-only)
  - Transformer-based подход (RuBERT, более точный)
- **Дедупликация**: SimHash для удаления дубликатов
- **Балансировка**: Поддержка многоклассовых задач с балансировкой по классам
- **Кэширование**: Сохранение результатов запросов для повторного использования
- **Вежливый скрапинг**: Задержки между запросами, уважение к серверам

## 📁 Структура проекта

```
neutral-data-collector/
├── README.md              # Этот файл
├── LICENSE                # MIT License
├── requirements.txt       # Зависимости Python
├── config.yaml           # Конфигурация
├── src/
│   ├── __init__.py       # Инициализация пакета
│   ├── scraper.py        # Скрапинг и извлечение текста
│   ├── neutrality_scorer.py  # Оценка нейтральности
│   ├── deduplicator.py   # Дедупликация (SimHash)
│   ├── balancer.py       # Балансировка датасета
│   └── main.py           # Основной пайплайн
├── cache/                # Кэш запросов
├── data/
│   ├── raw/             # Сырые данные
│   └── processed/       # Обработанные датасеты
└── notebooks/
    └── explore_dataset.ipynb  # Исследование датасета
```

## ⚙️ Установка

```bash
# Клонируйте репозиторий
cd neutral-data-collector

# Создайте виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

# Установите зависимости
pip install -r requirements.txt
```

### Для transformer-based оценки нейтральности:

Убедитесь, что у вас достаточно памяти (минимум 4GB RAM для CPU):

```bash
# Дополнительно можно установить оптимизации
pip install optimum[onnxruntime]
```

## 🚀 Быстрый старт

### 1. Настройте задачу в `config.yaml`

```yaml
task:
  description: "юридические консультации"
  keywords:
    - "юридическая помощь"
    - "правовые консультации"
    - "законодательство"

collection:
  target_size: 5000          # Целевой размер датасета
  min_neutrality_score: 0.7  # Минимальная нейтральность (0..1)
```

### 2. Запустите сборщик

```bash
# Из директории проекта
python -m src.main

# Или напрямую
python src/main.py --config config.yaml

# С переопределением параметров
python src/main.py --target-size 10000 --min-neutrality 0.8
```

### 3. Проверьте результаты

Файлы будут сохранены в `data/processed/`:
- `dataset.csv` - таблица с колонками: id, text, source_url, neutrality_score, length, timestamp
- `dataset.jsonl` - формат для импорта в Hugging Face Datasets
- `statistics.json` - подробная статистика сбора

## 📖 Конфигурация

### Основные параметры

#### `task` - Целевая задача
```yaml
task:
  description: "медицинские FAQ"  # Описание задачи
  keywords:                        # Ключевые слова для поиска
    - "симптомы заболеваний"
    - "диагностика"
    - "лечение"
```

#### `collection` - Параметры сбора
```yaml
collection:
  target_size: 50000            # Целевой размер датасета
  min_neutrality_score: 0.7     # Порог нейтральности (0..1)
  max_results_per_query: 100    # Макс. результатов на запрос
```

#### `neutrality_model` - Модель оценки нейтральности

**Rule-based (быстро, CPU):**
```yaml
neutrality_model:
  type: "rule_based"
```

**Transformer-based (точнее, требует больше ресурсов):**
```yaml
neutrality_model:
  type: "transformer"
  transformer_name: "blanchefort/rubert-base-cased-sentiment-rusentiment"
  batch_size: 16
```

#### `domains` - Фильтры доменов
```yaml
domains:
  blacklist:
    - "vk.com"
    - "facebook.com"
    - "youtube.com"
  whitelist: []  # Если не пустой, используются только эти домены
```

#### `balancing` - Балансировка классов
```yaml
balancing:
  enabled: true                    # Включить балансировку
  anchor_file: "anchor_data.csv"   # Файл с ручной разметкой
  class_distribution:              # Желаемое распределение
    positive: 0.33
    negative: 0.33
    neutral: 0.34
```

## 📊 Интерпретация метрик нейтральности

### Rule-based оценка

Метрика вычисляется по формуле:

```
neutrality_score = base_score + balance_bonus - subjectivity_penalty - categoricity_penalty
```

Где:
- **base_score** (0..1): Обратная пропорция от доли эмоциональных слов
- **balance_bonus** (0..0.2): Бонус за баланс позитива/негатива
- **subjectivity_penalty** (0.15): Штраф за маркеры субъективности ("я думаю", "по моему мнению")
- **categoricity_penalty** (0.1): Штраф за категоричность ("всегда", "никогда")

### Диапазоны значений:

| Score | Интерпретация |
|-------|--------------|
| 0.9-1.0 | Высоконейтральный (научный, официальный стиль) |
| 0.7-0.9 | Нейтральный (новости, документация) |
| 0.5-0.7 | Слабоэмоциональный (блоги, статьи) |
| < 0.5 | Эмоциональный (отзывы, мнения) |

### Transformer-based оценка

Использует предобученную модель sentiment analysis. Оценка — вероятность класса "neutral".

## 🔧 Дообучение модели на собранном датасете

### Пример для DistilBERT (Hugging Face)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import torch

# Загрузка датасета
dataset = load_dataset('json', data_files='data/processed/dataset.jsonl')

# Токенизация
model_name = "distilrubert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Подготовка меток (если есть)
if 'label' in tokenized_datasets['train'].column_names:
    num_labels = len(set(tokenized_datasets['train']['label']))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
else:
    # Для продолжения предобучения (language modeling)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("sberbank-ai/ruGPT-3-small")

# Параметры обучения
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    fp16=False,  # True если есть GPU
    save_steps=500,
    logging_steps=100,
)

# Трейнер
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)

# Обучение
trainer.train()

# Сохранение
trainer.save_model("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")
```

### Пример для ruGPT-3 (генерация текста)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Загрузка базовой модели
model_name = "sberbank-ai/ruGPT-3-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Подготовка данных
dataset = load_dataset('json', data_files='data/processed/dataset.jsonl')

def prepare_examples(examples):
    texts = [f"{text}" for text in examples["text"]]
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings

tokenized_dataset = dataset.map(prepare_examples, batched=True)

# Fine-tuning как выше...
```

## 📈 Анализ датасета

Используйте Jupyter notebook для исследования:

```bash
jupyter notebook notebooks/explore_dataset.ipynb
```

Пример анализа:

```python
import pandas as pd
import matplotlib.pyplot as plt
import json

# Загрузка статистики
with open('data/processed/statistics.json', 'r') as f:
    stats = json.load(f)

# Гистограмма нейтральности
plt.figure(figsize=(10, 6))
hist = stats['neutrality_metrics']['histogram']
plt.bar(hist.keys(), hist.values())
plt.xlabel('Диапазон нейтральности')
plt.ylabel('Количество текстов')
plt.title('Распределение по нейтральности')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Топ источников
df = pd.read_csv('data/processed/dataset.csv')
source_counts = df['source_url'].apply(lambda x: x.split('/')[2]).value_counts().head(10)
print(source_counts)
```

## 🛠️ Расширение функциональности

### Добавление нового источника поиска

```python
# В src/scraper.py добавьте новый метод в WebScraper
def search_google(self, query: str) -> List[Dict]:
    """Поиск через Google Custom Search API."""
    from googleapiclient.discovery import build
    
    api_key = self.config.get('google_api_key', '')
    cse_id = self.config.get('google_cse_id', '')
    
    service = build("customsearch", "v1", developerKey=api_key)
    results = service.cse().list(q=query, cx=cse_id).execute()
    
    return [
        {'title': r['title'], 'url': r['link'], 'snippet': r['snippet']}
        for r in results.get('items', [])
    ]
```

### Добавление собственной метрики нейтральности

```python
# В src/neutrality_scorer.py
class CustomNeutralityScorer(NeutralityScorer):
    def score_custom(self, text: str) -> float:
        # Ваша логика оценки
        pass
```

## ⚠️ Важные замечания

1. **Юридические аспекты**: Убедитесь, что сбор данных соответствует условиям использования сайтов и законодательству.

2. **Rate limiting**: По умолчанию установлена задержка 1 секунда между запросами. Не уменьшайте её без необходимости.

3. **Авторское право**: Собранные данные могут быть защищены авторским правом. Используйте ответственно.

4. **Качество данных**: Всегда проверяйте качество собранных данных перед обучением моделей.

## 📝 Лицензия

MIT License - см. файл LICENSE

## 🤝 Вклад

Pull requests приветствуются! Пожалуйста, создавайте issue для обсуждения значительных изменений.

## 📧 Контакты

Для вопросов и предложений создавайте issue в репозитории.
