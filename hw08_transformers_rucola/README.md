# HW08 — Трансформеры на RuCoLA

Ноутбук содержит четыре эксперимента:

1. выполненный TF–IDF baseline;
2. fine-tuning RuBERT;
3. RuGPT3 zero/few-shot с тремя prompts и 0/1/2/4 примерами;
4. fine-tuning RuT5 в text-to-text постановке.

Официальные `in_domain_train.csv` и `in_domain_dev.csv` находятся в `data/`.
Baseline выполнен полностью: test MCC = 0.239, accuracy = 0.691.

Большие transformer-модели требуют GPU и по умолчанию отключены. Для полного
эксперимента установить зависимости, изменить `RUN_TRANSFORMERS = True` в первой
ячейке и выполнить `Restart Kernel and Run All`. `FAST_MODE=True` использует
компактные модели для технической проверки, но для финальной сдачи нужен
`FAST_MODE=False`.

```powershell
python -m pip install transformers accelerate sentencepiece
```

Не следует выдавать baseline-таблицу за полное сравнение трансформеров.

