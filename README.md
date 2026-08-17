# ML Pro Homeworks

Репозиторий домашних заданий по курсу Machine Learning.

## Домашние задания

### HW01 — Gradient Boosting

Сравнение алгоритмов градиентного бустинга на задаче предсказания оттока клиентов Telco Customer Churn.

В работе выполнены:

- Exploratory Data Analysis (EDA);
- обработка пропущенных значений;
- Feature Engineering;
- кодирование категориальных признаков при помощи LabelEncoder;
- разделение данных на train/test;
- обучение GradientBoostingClassifier;
- обучение XGBoost;
- обучение CatBoost;
- обучение LightGBM;
- сравнение моделей с параметрами по умолчанию;
- подбор гиперпараметров с RandomizedSearchCV;
- 5-fold Stratified Cross-Validation;
- финальное сравнение настроенных моделей.

## Результаты моделей с параметрами по умолчанию

| Model | ROC-AUC |
|---|---:|
| GradientBoosting | 0.8387 |
| LightGBM | 0.8271 |
| CatBoost | 0.8260 |
| XGBoost | 0.8067 |

Лучший результат с параметрами по умолчанию показал GradientBoostingClassifier.

## Результаты после настройки гиперпараметров

| Model | ROC-AUC |
|---|---:|
| XGBoost | 0.8383 |
| CatBoost | 0.8381 |
| GradientBoosting | 0.8365 |
| LightGBM | 0.8350 |

Среди настроенных моделей лучший результат на test-выборке показал XGBoost.

Наибольший прирост после настройки гиперпараметров также получил XGBoost: ROC-AUC увеличился примерно с 0.8067 до 0.8383.

При этом GradientBoostingClassifier с параметрами по умолчанию показал ROC-AUC 0.8387, что оказалось немного выше результата настроенного XGBoost.

## Структура проекта

```text
ml-pro-homeworks/
├── hw01_boosting/
│   ├── data/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── notebooks/
│       └── hw01_boosting.ipynb
├── .gitignore
├── requirements.txt
└── README.md