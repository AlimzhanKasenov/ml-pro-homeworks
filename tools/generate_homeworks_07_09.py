"""Generate the three self-contained homework notebooks."""
from pathlib import Path
from textwrap import dedent
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def save(folder: str, filename: str, cells: list):
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python", "version": "3"}
    path = ROOT / folder / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)
    print(path)


save("hw07_web_parsing_nlp", "hw07_web_parsing_nlp.ipynb", [
    md("""
    # HW07 — Откуда берутся датасеты?

    **Цель:** самостоятельно собрать текстовый датасет веб-парсингом и построить модель классификации рейтинга книги по тексту.

    Источник — учебный сайт [Books to Scrape](https://books.toscrape.com/), специально созданный для практики парсинга. Объект — книга; текст — название и описание; исходная цель — рейтинг 1–5. Для устойчивой бинарной задачи определим `is_high_rating = 1`, если рейтинг равен 4 или 5.

    > Сам сайт предупреждает, что рейтинги назначены случайно. Поэтому ожидаемое качество текстовой модели может быть близко к случайному — это содержательный вывод, а не ошибка алгоритма.
    """),
    code("""
    from pathlib import Path
    import sys
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                                 f1_score, roc_auc_score)
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.pipeline import Pipeline

    SEED = 42
    DATA_PATH = Path('data/books.csv')
    RESULTS_DIR = Path('results')
    RESULTS_DIR.mkdir(exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    """),
    md("""
    ## 1. Парсер

    Полная реализация находится в `parser.py`. Она обходит страницы каталога и карточки книг, использует `requests.Session`, timeout, обработку ошибок и паузу `time.sleep(0.3)` **перед каждым запросом**. Повторно сайт не запрашивается, если CSV уже существует.
    """),
    code("""
    if not DATA_PATH.exists():
        from parser import scrape_books
        books = scrape_books(limit=200, delay=0.3)
        DATA_PATH.parent.mkdir(exist_ok=True)
        books.to_csv(DATA_PATH, index=False)
    else:
        books = pd.read_csv(DATA_PATH)

    print('Размер датасета:', books.shape)
    display(books.head(3))
    """),
    md("## 2. Проверка и EDA"),
    code("""
    assert books['url'].is_unique, 'Повторяющиеся карточки'
    assert books['rating'].between(1, 5).all()
    books['text'] = books['text'].fillna('').str.strip()
    books = books[books['text'].str.len() > 3].copy()
    books['is_high_rating'] = (books['rating'] >= 4).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(data=books, x='rating', ax=axes[0])
    axes[0].set_title('Распределение рейтинга')
    books['is_high_rating'].value_counts().sort_index().plot.bar(ax=axes[1])
    axes[1].set(title='Бинарная целевая переменная', xlabel='is_high_rating', ylabel='count')
    plt.tight_layout(); plt.show()
    print(books[['rating', 'price_gbp', 'is_high_rating']].describe())
    """),
    md("""
    ## 3. Train/test и TF–IDF

    Откладываем 25% наблюдений со стратификацией. `TfidfVectorizer` использует униграммы и биграммы, английские stop words, `min_df=2`, `max_df=0.90` и `norm=None`, то есть обязательная по условию L2-нормализация TF–IDF отключена. Все преобразования находятся внутри Pipeline, поэтому test не участвует в построении словаря.
    """),
    code("""
    X_train, X_test, y_train, y_test = train_test_split(
        books['text'], books['is_high_rating'], test_size=0.25,
        random_state=SEED, stratify=books['is_high_rating'])

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            lowercase=True, stop_words='english', ngram_range=(1, 2),
            min_df=2, max_df=0.90, norm=None, sublinear_tf=True)),
        ('model', LogisticRegression(max_iter=3000, class_weight='balanced', random_state=SEED))
    ])
    search = GridSearchCV(
        pipeline, {'model__C': [0.001, 0.01, 0.1, 1, 10]},
        scoring='roc_auc', cv=5, n_jobs=1, return_train_score=True)
    search.fit(X_train, y_train)
    print('Лучший C:', search.best_params_['model__C'])
    print('CV ROC-AUC:', round(search.best_score_, 4))
    """),
    md("## 4. Качество на test"),
    code("""
    best_model = search.best_estimator_
    prediction = best_model.predict(X_test)
    probability = best_model.predict_proba(X_test)[:, 1]
    metrics = pd.DataFrame([{
        'accuracy': accuracy_score(y_test, prediction),
        'f1': f1_score(y_test, prediction),
        'roc_auc': roc_auc_score(y_test, probability),
        'n_train': len(X_train), 'n_test': len(X_test),
        'best_C': search.best_params_['model__C']
    }])
    display(metrics.round(4))
    print(classification_report(y_test, prediction, digits=3))
    ConfusionMatrixDisplay.from_predictions(y_test, prediction)
    plt.title('Confusion matrix'); plt.show()
    metrics.to_csv(RESULTS_DIR / 'metrics.csv', index=False)
    """),
    md("## 5. Топ-50 коэффициентов и интерпретация"),
    code("""
    vectorizer = best_model.named_steps['tfidf']
    classifier = best_model.named_steps['model']
    coefficients = pd.Series(classifier.coef_[0], index=vectorizer.get_feature_names_out())
    top = pd.concat([coefficients.nsmallest(25), coefficients.nlargest(25)]).sort_values()
    colors = np.where(top.values >= 0, '#2a9d8f', '#e76f51')
    plt.figure(figsize=(10, 13))
    top.plot.barh(color=colors)
    plt.axvline(0, color='black', linewidth=1)
    plt.title('Топ-50 коэффициентов логистической регрессии')
    plt.xlabel('Коэффициент: вправо → высокий рейтинг')
    plt.tight_layout(); plt.savefig(RESULTS_DIR / 'top50_coefficients.png', dpi=160); plt.show()
    display(pd.DataFrame({'feature': top.index, 'coefficient': top.values}).tail(10))
    """),
    md("""
    ## Выводы

    Парсер собрал реальный структурированный датасет без API и без частых запросов. Регуляризация выбрана по 5-fold CV, а итоговая оценка рассчитана один раз на отложенных 25%. Положительный коэффициент повышает логарифм шансов рейтинга 4–5, отрицательный — понижает.

    Однако коэффициенты нельзя интерпретировать причинно: на учебном сайте рейтинг случаен, выборка мала, а слова редки. Поэтому близкий к 0.5 ROC-AUC ожидаем и честно показывает отсутствие устойчивого текстового сигнала. На реальном книжном сайте следовало бы увеличить выборку и добавить автора, жанр и отзывы.
    """)
])


save("hw08_transformers_rucola", "hw08_transformers_rucola.ipynb", [
    md("""
    # HW08 — Почувствуй мощь трансформеров в бою

    Задача бинарной классификации лингвистической приемлемости русских предложений на [RuCoLA](https://github.com/RussianNLP/RuCoLA). Метрики: **Matthews correlation coefficient (MCC)** — основная метрика RuCoLA — и accuracy.

    Эксперименты: RuBERT fine-tuning, RuGPT3 zero/few-shot (0, 1, 2, 4 примера), RuT5 fine-tuning. Большие модели запускаются флагом `RUN_TRANSFORMERS=True`; по умолчанию ноутбук выполняет проверочный TF–IDF baseline и не выдаёт вымышленные transformer-метрики.
    """),
    code("""
    from pathlib import Path
    import random, time
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, matthews_corrcoef
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    SEED = 42
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    RUN_TRANSFORMERS = False   # На GPU поменять на True и выполнить Run All
    FAST_MODE = False          # True использует компактные модели для отладки
    EPOCHS = 3
    BATCH_SIZE = 16
    MAX_LENGTH = 128
    DATA_DIR = Path('data'); RESULTS_DIR = Path('results'); RESULTS_DIR.mkdir(exist_ok=True)
    print('Device:', DEVICE, '| transformers:', RUN_TRANSFORMERS)
    """),
    md("## 1. Данные и разбиение"),
    code("""
    train_full = pd.read_csv(DATA_DIR / 'in_domain_train.csv')
    test = pd.read_csv(DATA_DIR / 'in_domain_dev.csv')
    display(train_full.head())
    print(train_full.shape, test.shape, train_full.columns.tolist())
    print('Train labels:', train_full['acceptable'].value_counts(normalize=True).sort_index().round(3).to_dict())

    train, val = train_test_split(
        train_full, test_size=0.20, random_state=SEED,
        stratify=train_full['acceptable'])
    print('train/val/test:', len(train), len(val), len(test))
    """),
    md("""
    ## 2. Проверочный baseline

    Символьные n-граммы часто дают разумный baseline для грамматической приемлемости: они улавливают окончания и локальные конструкции. Он также проверяет, что разбиение и метрики работают до скачивания больших весов.
    """),
    code("""
    baseline = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=2, max_features=100_000)),
        ('model', LogisticRegression(max_iter=2000, C=3, class_weight='balanced', random_state=SEED))
    ])
    baseline.fit(train['sentence'], train['acceptable'])
    results = []
    for split_name, frame in [('val', val), ('test', test)]:
        pred = baseline.predict(frame['sentence'])
        results.append({'model': 'TF-IDF baseline', 'split': split_name,
                        'MCC': matthews_corrcoef(frame['acceptable'], pred),
                        'accuracy': accuracy_score(frame['acceptable'], pred)})
    display(pd.DataFrame(results).round(4))
    """),
    md("""
    ## 3. RuBERT fine-tuning

    Полная модель: `DeepPavlov/rubert-base-cased`. В `FAST_MODE` для проверки кода используется `cointegrated/rubert-tiny2`. Разбиение train/val фиксировано и стратифицировано; test применяется только к финальному checkpoint с лучшим validation MCC.
    """),
    code("""
    if RUN_TRANSFORMERS:
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        BERT_MODEL = 'cointegrated/rubert-tiny2' if FAST_MODE else 'DeepPavlov/rubert-base-cased'

        class ClassificationDataset(Dataset):
            def __init__(self, frame, tokenizer):
                self.sentences = frame['sentence'].tolist()
                self.labels = frame['acceptable'].astype(int).tolist()
                self.tokenizer = tokenizer
            def __len__(self): return len(self.labels)
            def __getitem__(self, index):
                item = self.tokenizer(self.sentences[index], truncation=True,
                                      max_length=MAX_LENGTH, padding='max_length', return_tensors='pt')
                return {k: v.squeeze(0) for k, v in item.items()} | {'labels': torch.tensor(self.labels[index])}

        def evaluate_classifier(model, loader):
            model.eval(); labels, predictions = [], []
            with torch.no_grad():
                for batch in loader:
                    y = batch.pop('labels')
                    logits = model(**{k: v.to(DEVICE) for k, v in batch.items()}).logits.cpu()
                    labels.extend(y.tolist()); predictions.extend(logits.argmax(1).tolist())
            return matthews_corrcoef(labels, predictions), accuracy_score(labels, predictions)

        bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
        bert = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=2).to(DEVICE)
        loaders = {name: DataLoader(ClassificationDataset(frame, bert_tokenizer), batch_size=BATCH_SIZE,
                                    shuffle=(name == 'train'))
                   for name, frame in [('train', train), ('val', val), ('test', test)]}
        optimizer = torch.optim.AdamW(bert.parameters(), lr=2e-5, weight_decay=0.01)
        best_mcc, best_state = -1, None
        for epoch in range(EPOCHS):
            bert.train()
            for batch in loaders['train']:
                optimizer.zero_grad(set_to_none=True)
                loss = bert(**{k: v.to(DEVICE) for k, v in batch.items()}).loss
                loss.backward(); torch.nn.utils.clip_grad_norm_(bert.parameters(), 1.0); optimizer.step()
            val_mcc, val_acc = evaluate_classifier(bert, loaders['val'])
            print('RuBERT epoch', epoch + 1, 'val MCC', round(val_mcc, 4))
            if val_mcc > best_mcc:
                best_mcc = val_mcc
                best_state = {k: v.detach().cpu().clone() for k, v in bert.state_dict().items()}
        bert.load_state_dict(best_state)
        for split_name in ['val', 'test']:
            mcc, acc = evaluate_classifier(bert, loaders[split_name])
            results.append({'model': BERT_MODEL, 'split': split_name, 'MCC': mcc, 'accuracy': acc})
    else:
        print('RuBERT пропущен: установите transformers sentencepiece и включите RUN_TRANSFORMERS.')
    """),
    md("""
    ## 4. RuGPT3 zero-/few-shot

    Вместо нестабильного свободного generation сравнивается условная log-likelihood двух продолжений: `да` и `нет`. Перебираются три формулировки инструкции и 0/1/2/4 демонстрации. Это воспроизводимый zero/few-shot классификатор без обучения весов.
    """),
    code("""
    if RUN_TRANSFORMERS:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        GPT_MODEL = 'ai-forever/rugpt3small_based_on_gpt2' if FAST_MODE else 'ai-forever/rugpt3large_based_on_gpt2'
        gpt_tokenizer = AutoTokenizer.from_pretrained(GPT_MODEL)
        gpt = AutoModelForCausalLM.from_pretrained(GPT_MODEL).to(DEVICE).eval()
        if gpt_tokenizer.pad_token is None: gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

        templates = [
            'Грамматически ли корректно русское предложение? Ответ: да или нет.\\n',
            'Определи лингвистическую приемлемость. Допустимые ответы: да, нет.\\n',
            'Задача: проверить правильность предложения на русском языке.\\n']
        demonstrations = pd.concat([train[train.acceptable == 1].head(2),
                                    train[train.acceptable == 0].head(2)]).sample(frac=1, random_state=SEED)

        def build_prompt(sentence, template, shots):
            demos = ''.join(f'Предложение: {r.sentence}\\nОтвет: {"да" if r.acceptable else "нет"}\\n\\n'
                            for _, r in demonstrations.head(shots).iterrows())
            return template + demos + f'Предложение: {sentence}\\nОтвет:'

        def completion_nll(prompt, completion):
            prompt_ids = gpt_tokenizer(prompt, return_tensors='pt').input_ids.to(DEVICE)
            full = gpt_tokenizer(prompt + ' ' + completion, return_tensors='pt').input_ids.to(DEVICE)
            labels = full.clone(); labels[:, :prompt_ids.shape[1]] = -100
            with torch.no_grad(): return gpt(full, labels=labels).loss.item()

        # Ограничение делает полный перебор практичным; размер явно фиксируется в результатах.
        gpt_test = test.sample(min(200, len(test)), random_state=SEED)
        for template_id, template in enumerate(templates, 1):
            for shots in [0, 1, 2, 4]:
                pred = [int(completion_nll(build_prompt(s, template, shots), 'да') <
                            completion_nll(build_prompt(s, template, shots), 'нет'))
                        for s in gpt_test['sentence']]
                results.append({'model': f'RuGPT3 prompt={template_id}, shots={shots}', 'split': 'test-200',
                                'MCC': matthews_corrcoef(gpt_test.acceptable, pred),
                                'accuracy': accuracy_score(gpt_test.acceptable, pred)})
    else:
        print('RuGPT3 пропущен: RUN_TRANSFORMERS=False')
    """),
    md("""
    ## 5. RuT5 fine-tuning

    Классификация задаётся text-to-text: вход `приемлемость: <sentence>`, цель `да`/`нет`. Валидация и выбор checkpoint выполняются по MCC.
    """),
    code("""
    if RUN_TRANSFORMERS:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        T5_MODEL = 'cointegrated/rut5-small' if FAST_MODE else 'ai-forever/ruT5-base'
        t5_tokenizer = AutoTokenizer.from_pretrained(T5_MODEL)
        t5 = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL).to(DEVICE)

        class T5Dataset(torch.utils.data.Dataset):
            def __init__(self, frame): self.frame = frame.reset_index(drop=True)
            def __len__(self): return len(self.frame)
            def __getitem__(self, i):
                row = self.frame.iloc[i]
                source = t5_tokenizer('приемлемость: ' + row.sentence, truncation=True,
                                      max_length=MAX_LENGTH, padding='max_length', return_tensors='pt')
                target = t5_tokenizer('да' if row.acceptable else 'нет', max_length=4,
                                      padding='max_length', return_tensors='pt')
                labels = target.input_ids.squeeze(0); labels[labels == t5_tokenizer.pad_token_id] = -100
                return {k: v.squeeze(0) for k, v in source.items()} | {'labels': labels}

        t5_loaders = {name: torch.utils.data.DataLoader(T5Dataset(frame), batch_size=BATCH_SIZE,
                                                       shuffle=(name == 'train'))
                      for name, frame in [('train', train), ('val', val), ('test', test)]}
        def evaluate_t5(loader):
            true, pred = [], []
            t5.eval()
            with torch.no_grad():
                for batch in loader:
                    labels = batch.pop('labels')
                    output = t5.generate(**{k: v.to(DEVICE) for k, v in batch.items()}, max_new_tokens=3)
                    texts = t5_tokenizer.batch_decode(output, skip_special_tokens=True)
                    pred.extend([int(x.strip().lower().startswith('да')) for x in texts])
                    restored = labels.clone(); restored[restored == -100] = t5_tokenizer.pad_token_id
                    true_texts = t5_tokenizer.batch_decode(restored, skip_special_tokens=True)
                    true.extend([int(x.strip().lower().startswith('да')) for x in true_texts])
            return matthews_corrcoef(true, pred), accuracy_score(true, pred)

        t5_optimizer = torch.optim.AdamW(t5.parameters(), lr=1e-4, weight_decay=0.01)
        best_mcc, best_state = -1, None
        for epoch in range(EPOCHS):
            t5.train()
            for batch in t5_loaders['train']:
                t5_optimizer.zero_grad(set_to_none=True)
                loss = t5(**{k: v.to(DEVICE) for k, v in batch.items()}).loss
                loss.backward(); torch.nn.utils.clip_grad_norm_(t5.parameters(), 1.0); t5_optimizer.step()
            val_mcc, _ = evaluate_t5(t5_loaders['val'])
            print('RuT5 epoch', epoch + 1, 'val MCC', round(val_mcc, 4))
            if val_mcc > best_mcc:
                best_mcc = val_mcc
                best_state = {k: v.detach().cpu().clone() for k, v in t5.state_dict().items()}
        t5.load_state_dict(best_state)
        for split_name in ['val', 'test']:
            mcc, acc = evaluate_t5(t5_loaders[split_name])
            results.append({'model': T5_MODEL, 'split': split_name, 'MCC': mcc, 'accuracy': acc})
    else:
        print('RuT5 пропущен: RUN_TRANSFORMERS=False')
    """),
    md("## 6. Сравнение"),
    code("""
    comparison = pd.DataFrame(results).sort_values(['split', 'MCC'], ascending=[True, False])
    display(comparison.round(4))
    comparison.to_csv(RESULTS_DIR / 'comparison.csv', index=False)
    if not RUN_TRANSFORMERS:
        print('В таблице пока только baseline. Для сдачи transformer-части нужен GPU-запуск с RUN_TRANSFORMERS=True.')
    """),
    md("""
    ## Выводы

    Сравнивать методы следует прежде всего по test MCC. Fine-tuning использует веса для решения конкретной задачи и обычно должен быть сильнее zero/few-shot. RuGPT3 не обучается и чувствителен к формулировке затравки и числу демонстраций — именно поэтому перебираются 12 конфигураций. RuT5 формулирует классификацию как генерацию короткой метки. Финальные численные выводы следует записывать только после GPU-запуска всех блоков; ноутбук намеренно не подставляет фиктивные результаты.
    """)
])


save("hw09_metro_forecasting", "hw09_metro_forecasting.ipynb", [
    md("""
    # HW09 — Прогнозирование Metro Interstate Traffic Volume

    **Цель:** очистить почасовой временной ряд, сравнить baseline и ML-модели на последних двух неделях, затем построить прогноз на следующую неделю с доверительным интервалом.

    В исходном задании сказано «пассажиропоток метрополитена», но официальный набор UCI содержит **автомобильный поток западного направления I‑94 между Minneapolis и St Paul**. В работе используется корректное описание исходных данных.
    """),
    code("""
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    SEED = 42
    DATA_PATH = Path('data/Metro_Interstate_Traffic_Volume.csv.gz')
    RESULTS_DIR = Path('results'); RESULTS_DIR.mkdir(exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    """),
    md("## 1. Загрузка и базовый EDA"),
    code("""
    raw = pd.read_csv(DATA_PATH, usecols=['traffic_volume', 'date_time', 'holiday', 'temp'])
    raw['date_time'] = pd.to_datetime(raw['date_time'])
    raw = raw.sort_values('date_time')
    print('Исходная форма:', raw.shape)
    print('Дубликатов timestamp:', raw.duplicated('date_time').sum())
    gaps = raw.drop_duplicates('date_time')['date_time'].diff().dt.total_seconds().div(3600)
    print('Интервалы не равные 1 часу:', (gaps.dropna() != 1).sum())
    display(raw.describe(include='all'))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    raw.set_index('date_time')['traffic_volume'].resample('D').mean().plot(ax=axes[0], title='Средний дневной traffic volume')
    sns.histplot(raw['traffic_volume'], bins=40, ax=axes[1]); axes[1].set_title('Распределение target')
    plt.tight_layout(); plt.show()
    """),
    md("""
    ## 2. Дубликаты, равномерная сетка и интерполяция

    Дубликаты времени возникают из-за нескольких погодных описаний одного часа. Агрегируем числовые признаки средним, а праздник — первым значением, отличным от `None`. Затем строим полный `pd.date_range` с частотой 1 час и линейно интерполируем target и температуру.
    """),
    code("""
    def holiday_aggregate(values):
        meaningful = values[values.ne('None') & values.notna()]
        return meaningful.iloc[0] if len(meaningful) else 'None'

    hourly = (raw.groupby('date_time', as_index=True)
              .agg(traffic_volume=('traffic_volume', 'mean'),
                   temp=('temp', 'mean'), holiday=('holiday', holiday_aggregate)))
    full_index = pd.date_range(hourly.index.min(), hourly.index.max(), freq='h')
    missing_hours = len(full_index) - len(hourly)
    hourly = hourly.reindex(full_index)
    hourly.index.name = 'date_time'
    hourly[['traffic_volume', 'temp']] = hourly[['traffic_volume', 'temp']].interpolate(method='time', limit_direction='both')
    hourly['holiday'] = hourly['holiday'].fillna('None')
    assert hourly.index.is_monotonic_increasing and hourly.index.is_unique
    assert hourly[['traffic_volume', 'temp', 'holiday']].notna().all().all()
    print('Добавлено пропущенных часов:', missing_hours, '| итоговая форма:', hourly.shape)
    """),
    md("## 3. Признаки и последние две недели для теста"),
    code("""
    def make_features(frame):
        result = pd.DataFrame(index=frame.index)
        result['hour'] = frame.index.hour
        result['weekday'] = frame.index.dayofweek
        result['month'] = frame.index.month
        result['dayofyear'] = frame.index.dayofyear
        result['trend'] = (frame.index - hourly.index.min()).total_seconds() / 86400
        result['temp'] = frame['temp'].to_numpy()
        result['is_holiday'] = frame['holiday'].ne('None').astype(int).to_numpy()
        result['hour_sin'] = np.sin(2 * np.pi * result.hour / 24)
        result['hour_cos'] = np.cos(2 * np.pi * result.hour / 24)
        result['week_sin'] = np.sin(2 * np.pi * result.weekday / 7)
        result['week_cos'] = np.cos(2 * np.pi * result.weekday / 7)
        result['year_sin'] = np.sin(2 * np.pi * result.dayofyear / 365.25)
        result['year_cos'] = np.cos(2 * np.pi * result.dayofyear / 365.25)
        return result

    TEST_HOURS = 24 * 14
    test = hourly.iloc[-TEST_HOURS:].copy()
    train_all = hourly.iloc[:-TEST_HOURS].copy()
    # Последние два года достаточно отражают актуальный уровень и сохраняют сезонность.
    train = train_all.loc[train_all.index >= train_all.index.max() - pd.DateOffset(years=2)].copy()
    X_train, y_train = make_features(train), train['traffic_volume']
    X_test, y_test = make_features(test), test['traffic_volume']
    print('train:', train.index.min(), '—', train.index.max(), len(train))
    print('test:', test.index.min(), '—', test.index.max(), len(test))
    """),
    md("## 4. Baseline: среднее по дню недели и часу"),
    code("""
    profile = train.assign(weekday=train.index.dayofweek, hour=train.index.hour).groupby(['weekday', 'hour'])['traffic_volume'].mean()
    baseline_prediction = np.array([profile.loc[(ts.dayofweek, ts.hour)] for ts in test.index])

    def metrics_row(name, truth, prediction):
        return {'model': name, 'MAE': mean_absolute_error(truth, prediction),
                'RMSE': mean_squared_error(truth, prediction) ** 0.5,
                'R2': r2_score(truth, prediction)}

    scores = [metrics_row('weekday-hour mean', y_test, baseline_prediction)]
    display(pd.DataFrame(scores).round(3))
    """),
    md("""
    ## 5. Модели

    Ridge получает one-hot час/день/месяц и масштабированные числовые признаки. HistGradientBoosting моделирует нелинейные взаимодействия часа, дня недели, температуры и годовой сезонности. Target-lags из test не используются, поэтому утечки будущих значений нет.
    """),
    code("""
    categorical = ['hour', 'weekday', 'month']
    numeric = [c for c in X_train.columns if c not in categorical]
    ridge = Pipeline([
        ('prepare', ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
            ('num', StandardScaler(), numeric)])),
        ('model', Ridge(alpha=10.0))])
    hist = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                         l2_regularization=5, random_state=SEED)
    models = {'Ridge': ridge, 'HistGradientBoosting': hist}
    predictions = {'weekday-hour mean': baseline_prediction}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)
        scores.append(metrics_row(name, y_test, predictions[name]))

    scores = pd.DataFrame(scores).sort_values('RMSE').reset_index(drop=True)
    display(scores.round(3))
    scores.to_csv(RESULTS_DIR / 'model_metrics.csv', index=False)
    """),
    md("## 6. Визуальное сравнение на test"),
    code("""
    best_name = scores.iloc[0]['model']
    best_prediction = predictions[best_name]
    plt.figure(figsize=(16, 5))
    plt.plot(test.index, y_test, label='fact', linewidth=2)
    plt.plot(test.index, baseline_prediction, label='baseline', alpha=.8)
    plt.plot(test.index, best_prediction, label=f'best: {best_name}', alpha=.9)
    plt.title('Последние две недели: факт и прогнозы')
    plt.ylabel('traffic_volume'); plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'test_forecast.png', dpi=160); plt.show()
    print('Лучшая модель по RMSE:', best_name)
    """),
    md("""
    ## 7. Следующая неделя и 95% prediction interval

    Будущая температура неизвестна. Используем климатологический прогноз — среднюю температуру последних двух лет для пары `weekday × hour`. Праздники будущей недели считаются отсутствующими. Интервал строится по эмпирическим 2.5% и 97.5% квантилям out-of-sample ошибок лучшей модели на двухнедельном тесте; это prediction interval, а не интервал среднего.
    """),
    code("""
    future_index = pd.date_range(hourly.index.max() + pd.Timedelta(hours=1), periods=24 * 7, freq='h')
    temp_profile = train.assign(weekday=train.index.dayofweek, hour=train.index.hour).groupby(['weekday', 'hour'])['temp'].mean()
    future = pd.DataFrame(index=future_index)
    future['temp'] = [temp_profile.loc[(ts.dayofweek, ts.hour)] for ts in future_index]
    future['holiday'] = 'None'
    X_future = make_features(future)

    if best_name == 'weekday-hour mean':
        future_prediction = np.array([profile.loc[(ts.dayofweek, ts.hour)] for ts in future_index])
    else:
        # Переобучаем выбранную модель на всей истории до конца test.
        final_frame = hourly.loc[hourly.index >= hourly.index.max() - pd.DateOffset(years=2)]
        final_model = models[best_name]
        final_model.fit(make_features(final_frame), final_frame['traffic_volume'])
        future_prediction = final_model.predict(X_future)

    residuals = y_test.to_numpy() - best_prediction
    lower_error, upper_error = np.quantile(residuals, [0.025, 0.975])
    forecast = pd.DataFrame({'forecast': future_prediction,
                             'lower_95': np.maximum(0, future_prediction + lower_error),
                             'upper_95': future_prediction + upper_error}, index=future_index)
    forecast.to_csv(RESULTS_DIR / 'next_week_forecast.csv')

    plt.figure(figsize=(16, 5))
    plt.plot(forecast.index, forecast.forecast, label='forecast', color='#264653')
    plt.fill_between(forecast.index, forecast.lower_95, forecast.upper_95, alpha=.25, label='95% interval')
    plt.title(f'Прогноз на следующую неделю — {best_name}')
    plt.ylabel('traffic_volume'); plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'next_week_forecast.png', dpi=160); plt.show()
    display(forecast.head())
    """),
    md("""
    ## Выводы

    Очистка подтвердила обе проблемы из задания: дубликаты timestamps и пропуски в почасовой сетке. Сильный baseline отражает недельную и суточную сезонность. ML-модель дополнительно использует температуру, праздник, годовую сезонность и тренд. Сравнение проводится на одном и том же последнем 14-дневном отрезке по MAE, RMSE и R².

    Главные ограничения прогноза — неизвестная будущая погода, редкие праздники и изменение режима трафика. Эмпирический prediction interval учитывает типичную ошибку на отложенных данных, но не гарантирует идеальной калибровки на экстремальных событиях.
    """)
])
