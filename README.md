# RuBERT Multi-task NER & Classification

[English](#english) | Русский

Портфолио-версия учебного проекта, выполненного в рамках обучения в Яндекс Практикуме. Репозиторий содержит реализацию и агрегированные результаты; тексты задания, материалы курса, датасет и комментарии ревьюера не включены.

В этом проекте я исследовала, как одна модель может одновременно находить именованные сущности и определять характеристики текста. Использовала RuBERT с двумя выходными головами и attention для классификации. Также сравнила исходную модель с INT8-версией, чтобы оценить, как квантизация влияет на скорость, размер и качество.

## Что реализовано

Общий `DeepPavlov/rubert-base-cased` с двумя выходами:

- NER: линейная голова, BIO-метки по первому subword, `ignore_index=-100`.
- Multi-label CLS: `[CLS]` + masked mean + обучаемый attention для каждого класса.

Совместный loss использует uncertainty weighting. Обучение включает AdamW, раздельные learning rates энкодера и голов, warmup, gradient clipping, early stopping и сохранение checkpoint. Порог классификации и эпоха выбираются по validation. Для сравнения используется частотный CLS baseline. Отдельный эксперимент оценивает динамическую INT8-квантизацию линейных слоёв на CPU.

## Результаты исходного запуска

| Вариант | NER macro-F1 | CLS micro-F1 | CPU, мс/документ | Веса, МиБ |
|---|---:|---:|---:|---:|
| FP32 | 0.7675 | 0.8009 | 155.3199 | 681.1885 |
| Dynamic INT8 | 0.7325 | 0.7937 | 86.9188 | 434.7126 |

Частотный baseline: CLS micro-F1 **0.7226**. Выбрана эпоха **21**, единый порог **0.475**. Основная модель — FP32.

NER считается по BIO-меткам без `O`, на первых subword исходных слов; это не entity-level F1. Метрики качества получены на 75 test-документах. Скорость измерена на восьми документах на одном CPU: после прогрева, медиана трёх повторов, без токенизации. Ускорение INT8 около 1.79 раза относится только к этому измерению. Модель обучалась в Google Colab с A100.

## Данные и энкодер

Внешние ресурсы скачиваются при запуске:

- [NEREL на Hugging Face](https://huggingface.co/datasets/danasone/nerel)
- [RuBERT](https://huggingface.co/DeepPavlov/rubert-base-cased)

Веса и данные в архив не включены. Их лицензии и условия использования следует смотреть у исходных поставщиков; этот репозиторий не предоставляет прав на внешние ресурсы. Полное описание учебного задания не публикуется.

## Запуск

1. Открыть `nerel_multitask.ipynb` в Jupyter или Google Colab.
2. Выбрать GPU и выполнить ячейки сверху вниз. Первая ячейка кода устанавливает основные зависимости; для отдельной среды также есть `requirements.txt`.
3. Модели и данные загружаются из Hugging Face. Нужны интернет и достаточно GPU/системной памяти. Исходный batch size — 8; точные результаты при изменении среды или параметров могут отличаться.
4. `RUN_QUANTIZATION=True` запускает дополнительный CPU-эксперимент после обучения. Для первого знакомства его можно отключить; основная оценка уже выполнена до него.

Результаты сохраняются локально в `nerel_attention_outputs/`, каталог исключён через `.gitignore`. Чтобы использовать модель без повторного обучения, нужно отдельно сохранить checkpoint, токенизатор и конфигурацию после своего запуска.

## Публичная редакция

Агрегированные графики и метрики сохранены из исходного выполненного ноутбука. Выводы текстов, отдельных записей датасета и инструкционные блоки удалены. Диагностика ошибок заменена агрегатами, демонстрационный текст — новым вымышленным предложением. У изменённых диагностических ячеек outputs очищены: новые прогнозы не выдаются за результат исходного запуска. Эта редакция не переобучалась при подготовке к публикации.

## Ограничения

Небольшая выборка, дисбаланс классов и усечение длинных документов; наблюдается переобучение. Split: 596 train / 75 validation / 75 test, seed 42. Test просматривался в нескольких итерациях разработки, поэтому оценка исследовательская; она не является независимым подтверждением качества. Сопоставимое исследование multi-task против single-task не проводилось. Рост качества нельзя приписать исключительно attention без контролируемого сравнения.

## Основные результаты

Модель использует общий энкодер RuBERT для распознавания именованных сущностей и многометочной классификации документов. Классификационная голова объединяет `[CLS]`, усреднённые эмбеддинги и attention для каждого класса.

В выполненном эксперименте получены:

- NER macro-F1 — **0,7675**;
- CLS micro-F1 — **0,8009**;
- CLS micro-F1 частотного baseline — **0,7226**.

Динамическая INT8-квантизация сократила время CPU-инференса примерно в **1,79 раза**, а размер файла весов — примерно на **36%**. При этом качество снизилось, поэтому основной версией осталась FP32.

Результаты носят исследовательский характер: тестовая выборка просматривалась в нескольких итерациях разработки. Измеренное ускорение относится к использованному CPU и выбранным восьми документам.

## Состав репозитория

- `nerel_multitask.ipynb` — код эксперимента, графики и агрегированные результаты.
- `requirements.txt` — зависимости.
- `.gitignore` — исключения для локальных данных, весов и временных файлов.

Датасет и веса модели в репозиторий не включены. Внешние ресурсы скачиваются при запуске ноутбука.

<a id="english"></a>

# RuBERT Multi-task NER & Classification

A portfolio version of an educational project completed as part of the Yandex Practicum program. This repository contains the implementation and aggregate results. Assignment instructions, course materials, the dataset, and reviewer comments are not included.

In this project, I explored how a single model can identify named entities and predict document characteristics at the same time. I used RuBERT with two output heads and attention for classification. I also compared the original model with an INT8 version to assess how quantization affects inference speed, model size, and predictive performance.

## Implementation

A shared `DeepPavlov/rubert-base-cased` encoder with two outputs:

- **NER:** a linear head with BIO labels assigned to the first subword of each original word; remaining positions are ignored using `ignore_index=-100`.
- **Multi-label classification:** `[CLS]` + masked mean pooling + learned attention for each class.

The joint loss uses uncertainty weighting. Training includes AdamW, separate learning rates for the encoder and output heads, warmup, gradient clipping, early stopping, and checkpoint saving. The classification threshold and checkpoint epoch are selected using the validation set.

A label-frequency baseline is included for comparison. A separate experiment evaluates dynamic INT8 quantization of linear layers on CPU.

## Original Run Results

| Model | NER macro-F1 | CLS micro-F1 | CPU latency, ms/document | Serialized weights, MiB |
|---|---:|---:|---:|---:|
| FP32 | 0.7675 | 0.8009 | 155.3199 | 681.1885 |
| Dynamic INT8 | 0.7325 | 0.7937 | 86.9188 | 434.7126 |

The label-frequency baseline achieved a CLS micro-F1 of **0.7226**. The selected checkpoint was from epoch **21**, with a single classification threshold of **0.475**. FP32 remains the primary model.

NER macro-F1 is calculated over BIO labels excluding `O`, at the first subword of each original word. It is not entity-level F1.

Predictive performance was evaluated on 75 test documents. CPU latency was measured on eight documents after warmup, using the median of three repeated runs and excluding tokenization. The approximately **1.79× INT8 speedup** applies only to this benchmark. Training was performed in Google Colab with an A100 GPU.

## Dataset and Encoder

External resources are downloaded at runtime:

- [NEREL on Hugging Face](https://huggingface.co/datasets/danasone/nerel)
- [RuBERT](https://huggingface.co/DeepPavlov/rubert-base-cased)

Model weights and data are not included in the repository. Refer to the original providers for their licenses and terms of use; this repository does not grant rights to external resources. The full educational assignment is not published.

## Running the Notebook

1. Open `nerel_multitask.ipynb` in Jupyter or Google Colab.
2. Select a GPU runtime and execute the cells from top to bottom. The first code cell installs the main dependencies; `requirements.txt` is also provided for a separate environment.
3. The model and dataset are downloaded from Hugging Face. Internet access and sufficient GPU and system memory are required. The original batch size is 8. Exact results may differ across environments or when parameters are changed.
4. `RUN_QUANTIZATION=True` enables the additional CPU experiment after training. You can disable it for an initial run; the main evaluation takes place before this step.

Outputs are saved locally to `nerel_attention_outputs/`, which is excluded by `.gitignore`. To use the model without retraining, save the checkpoint, tokenizer, and configuration after your own run.

## Public Version

Aggregate plots and metrics have been retained from the original executed notebook. Outputs containing document text or individual dataset records, as well as assignment instructions, have been removed.

Error diagnostics now report aggregate statistics, and the demonstration text has been replaced with a new fictional sentence. Outputs of modified diagnostic cells have been cleared so that new predictions are not presented as results of the original run. The model was not retrained when preparing this public version.

## Limitations

The dataset is small, class frequencies are imbalanced, long documents are truncated, and the training logs show signs of overfitting.

The split contains **596 training, 75 validation, and 75 test documents**, with seed **42**. The test set was inspected across multiple development iterations, so these results are exploratory rather than an independent confirmation of generalization.

A controlled comparison of multi-task and single-task training was not conducted. Performance improvements cannot be attributed solely to attention without a controlled experiment.

## Key Findings

The model uses a shared RuBERT encoder for named entity recognition and multi-label document classification. The classification head combines `[CLS]`, mean-pooled embeddings, and class-specific attention.

The executed experiment achieved:

- **NER macro-F1:** 0.7675;
- **CLS micro-F1:** 0.8009;
- **Label-frequency baseline CLS micro-F1:** 0.7226.

Dynamic INT8 quantization provided an approximately **1.79× CPU inference speedup** and reduced the serialized weight size by approximately **36%**. Predictive performance decreased, so FP32 was retained as the primary model.

These findings are exploratory because the test set was inspected during multiple development iterations. The measured speedup applies to the CPU and eight documents used in this benchmark.

## Repository Contents

- `nerel_multitask.ipynb` — experiment code, plots, and aggregate results.
- `requirements.txt` — dependencies.
- `.gitignore` — exclusions for local data, model weights, and temporary files.

The dataset and model weights are not included in the repository. External resources are downloaded when the notebook is run.
