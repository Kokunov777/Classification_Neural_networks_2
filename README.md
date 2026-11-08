# Лабораторная работа №2 — Titanic Survival

Бинарная классификация: предсказание выживаемости пассажиров «Титаника» (`Survived`). Лучший результат — **LDA**: Accuracy 0.849, F1 0.797, AUC-ROC 0.862.

## Структура проекта

```
laba_2/
├── LR2_classification_nn.ipynb   — основной выполненный ноутбук: EDA, классификаторы,
│                                    подбор гиперпараметров, нейросеть, TensorBoard, выводы
├── titanic.csv                   — датасет (891 пассажир, 12 столбцов)
├── figures_titanic/              — 14 рисунков из ноутбука (EDA, ROC, матрицы ошибок,
│                                    кривые обучения, дерево решений)
├── results_titanic/              — числовые результаты в CSV
│   ├── baseline_metrics.csv      — метрики базовых моделей
│   ├── tuned_metrics.csv         — метрики после подбора гиперпараметров
│   ├── nn_experiments.csv        — результаты экспериментов с нейросетью
│   └── comparison.csv            — итоговое сравнение всех подходов
├── logs_tb_titanic/              — «сырые» event-логи TensorBoard (13 запусков)
├── logs/                         — готовые графики TensorBoard для отчёта
│   ├── tensorboard_baseline_curves.png  — train/val кривые loss, accuracy, AUC
│   ├── tensorboard_learning_rate.png    — сравнение learning rate
│   ├── tensorboard_runs_comparison.png  — сравнение конфигураций сети
│   └── tensorboard_scalars.csv          — все скаляры в табличном виде
├── export_tensorboard.py         — пересборка logs/ из logs_tb_titanic/
├── backup/                       — старая версия ноутбука (вариант Bank Marketing)
├── requirements.txt              — зависимости
└── .venv/                        — готовое Python-окружение
```

## Что где смотреть

- **Ход работы и выводы** — `LR2_classification_nn.ipynb` (графики TensorBoard — в разделе 6.5).
- **Итоговые метрики** — `results_titanic/comparison.csv`.
- **Картинки для отчёта** — `figures_titanic/` и `logs/`.
- **Интерактивный TensorBoard** — `tensorboard --logdir logs_tb_titanic`, затем http://localhost:6006.

## Запуск

```bash
.venv\Scripts\python.exe -m jupyter notebook LR2_classification_nn.ipynb
```

Пересобрать графики в `logs/`:

```bash
.venv\Scripts\python.exe export_tensorboard.py
```
