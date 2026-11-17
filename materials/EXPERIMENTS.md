# Журнал решений Avito ML Cup 2026

Для каждого запуска записываем модель, признаки, объём train, seed, локальные метрики, ссылки на артефакты и подтверждённый результат Public. Локальное качество сравниваем при одинаковой выборке и правилах расчёта. Public хранится отдельно. Private определяет итоговое место.

Из пяти попыток в день на команду выделяем сабмит для кандидатов с заметным приростом относительно лучшего уже отправленного решения. Рабочий ориентир — около `+0.003 NDCG` или больше на позднем val при сравнении на тех же лентах. Смотрим, сохраняется ли выигрыш на раннем val и не зависит ли он от одной группы лент. Ориентир помогает выбирать, когда отправлять, и может меняться по мере накопления результатов.

**Лучший известный результат команды на Public — `nn-calibration-joint-024`, NDCG `0.3652139518`.** Полный val `0.375812`, поздний val `0.372957`. Сравниваем следующие варианты с этим ансамблем. Ориентир `+0.003 NDCG` соответствует примерно `0.37596` на тех же поздних лентах val. [Сабмит](../experiments/nn-calibration-joint-024/submission-reserve.parquet), [проверка](../experiments/nn-calibration-joint-024/submission_audit.json) и [метрики](../experiments/nn-calibration-joint-024/metrics.json).

Вариант `lgb-ensemble-priors-024` дал `+0.00138` на позднем val и остался в резерве. Его компоненты вошли в новый ансамбль. [Сабмит](../experiments/lgb-ensemble-priors-024/submission.parquet) и [проверка](../experiments/lgb-ensemble-priors-024/submission_audit.json).

| Решение | Деревьев | NDCG ранний val | NDCG поздний val | NDCG полный val | NDCG Public |
| --- | ---: | ---: | ---: | ---: | ---: |
| [TEAM-004, лучший из четырёх известных участнику запусков](../experiments/team-004/submission.parquet) | — | — | — | — | **0.3540182787523695** |
| [LightGBM LambdaMART, 9 базовых признаков](../experiments/lgb-basic-006/metrics.json) | 120 | 0.348375 | 0.339201 | 0.342868 | — |
| [LambdaMART, признаки внутри ленты](../experiments/lgb-context-006/metrics.json) | 80 | 0.347612 | 0.338538 | 0.342166 | — |
| [LambdaMART, признаки внутри ленты и история](../experiments/lgb-history-006/metrics.json) | 40 | 0.361861 | 0.353437 | 0.356804 | — |
| [LambdaMART с поправкой на позиционное смещение, история](../experiments/lgb-debiased-006/metrics.json) | 80 | 0.362635 | 0.355946 | 0.358620 | — |
| [XE-NDCG, история](../experiments/lgb-xendcg-006/metrics.json) | 40 | 0.365549 | 0.357696 | 0.360835 | — |
| [Бинарный LightGBM, история](../experiments/lgb-binary-006/metrics.json) | 80 | 0.367484 | 0.360453 | 0.363264 | — |
| [Ансамбль XE-NDCG + LambdaMART с поправкой](../experiments/lgb-ensemble-006/metrics.json) | 40 + 80 | 0.368044 | 0.360634 | 0.363597 | — |
| [Ансамбль XE-NDCG + LambdaMART с поправкой + бинарная модель](../experiments/lgb-ensemble3-006/metrics.json) | 40 + 80 + 80 | **0.369184** | **0.361991** | **0.364867** | — |

TEAM-004 — предыдущий лучший результат команды на Public. Результат сообщён участником. Параметры модели, предсказания на val и результаты трёх предыдущих попыток пока неизвестны.

В первом цикле победил `lgb-ensemble3-006` по полному NDCG на поздней части val. Признаки истории дали +0.01424 на поздней части относительно базового LambdaMART. Ансамбль прибавил +0.00154 к лучшей отдельной модели. Его NDCG@10 / NDCG@30 на полном val составили `0.236150 / 0.304976`. Поздняя часть использована для сравнения вариантов.

[Проверенный сабмит тройного ансамбля](../experiments/lgb-ensemble3-006/submission.parquet) содержит 2 904 806 пар. Первый кандидат совпадает с TEAM-004 в 12 303 из 28 595 лент (43,03%), при равенстве scores для диагностики берётся меньший `item_id`. Для оценки ансамбля с TEAM-004 на локальном val нужны его предсказания на val. [Проверка сабмита и SHA-256](../experiments/lgb-ensemble3-006/submission_audit.json).

Проверка TEAM-004 подтвердила 2 904 806 уникальных пар шаблона, 28 595 лент, типы `int32, int32, float32` и конечные scores. В 217 лентах есть равные scores. SHA-256 оригинала и архивной копии совпадает: `486455487fd2111e942222c127db80f6ed4dad694c9f3a791e1a5db53df9b8ab`.

## Первый цикл локальных экспериментов

- [x] Проверить ключи и scores TEAM-004, зафиксировать контрольную сумму.
- [x] На одних train-лентах сравнить LambdaMART с базовыми признаками, с признаками внутри ленты и с историей пользователя. Сохранить предсказания val для повторного сравнения и ансамблей.
- [x] Использовать раннюю часть val для выбора числа деревьев. Сравнить полный NDCG, NDCG@10 и NDCG@30 на поздней части. Зафиксировать отдельный результат на полном val.
- [x] Подготовить и проверить тестовый Parquet для лучшего локального решения.

Выборка train для первого цикла — шесть целых Parquet-файлов из 512, выбраны без возвращения с seed `2026`. В них 36 928 лент и 2 866 780 кандидатов. В обучении воспроизводимо перемешан порядок кандидатов в каждой ленте. Исходная позиция `pos` и идентификаторы кандидатов и пользователей отсутствуют среди входных признаков модели. ID объявлений используются только для проверки присутствия кандидата в его собственной истории.

`basic` повторяет девять входов официального CatBoost. `context` добавляет ранги значений и нормализацию внутри ленты, размеры групп и нормы векторов. `history` добавляет частоты категорий и типов событий, свежесть и сходство с векторами истории. LambdaMART обучается с `lambdarank_truncation_level=40`. Поправка берёт `pos` из train как служебную разметку групп позиций. Модель получает те же 51 входных признака при обучении и предсказании. В ансамблях scores каждой модели стандартизируются внутри ленты, затем берётся среднее с равными весами.

Ранний val содержит 5 766 лент. Поздний начинается с `timestamp=1790091904` и содержит 8 657 лент. Модели, результаты в трёх метриках и предсказания val лежат в папках `experiments/lgb-*-006`. Значения параметров и список train-файлов зафиксированы в `metrics.json` каждого запуска.

На полном val случайная сортировка с seed `2026` дала NDCG `0.288681`, косинус между кандидатом и пользователем — `0.341688`, максимум косинуса кандидата к событиям истории — `0.342283`. Значения служат контрольными точками. [Три метрики по всем простым признакам](../experiments/cache-006/val_reference.json).

[Страница соревнования](https://ods.ai/competitions/avitotechrecsys2026) и [FAQ](https://ods.ai/competitions/avitotechrecsys2026/faq). Пять сабмитов в день делятся на всю команду. Опубликованный [лидерборд](https://ods.ai/competitions/avitotechrecsys2026/leaderboard) на момент просмотра показывал лидера с NDCG 0.3982555809.

## Воспроизведение выбранного решения

Команды PowerShell из корня проекта. Они используют отдельные выходные папки и выбранные в первом цикле размеры моделей. Для повторения поиска числа деревьев использовать `rounds` из соответствующих `metrics.json`.

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$rankingPackages = @('--no-project', '--python', '3.11', '--with', 'numpy==1.26.4', '--with', 'polars==1.14.0', '--with', 'pyarrow==19.0.1', '--with', 'lightgbm==4.6.0')
uv run @rankingPackages python experiments/ranking.py prepare train experiments/repro-cache --shards 6 --stage-embeddings
uv run @rankingPackages python experiments/ranking.py prepare val experiments/repro-cache
uv run @rankingPackages python experiments/ranking.py prepare test experiments/repro-cache
uv run @rankingPackages python experiments/ranking.py train experiments/repro-cache experiments/repro-xendcg history --objective rank_xendcg --rounds 40 --threads 6
uv run @rankingPackages python experiments/ranking.py train experiments/repro-cache experiments/repro-debiased history --position-debias --rounds 80 --threads 6
uv run @rankingPackages python experiments/ranking.py train experiments/repro-cache experiments/repro-binary history --objective binary --rounds 80 --threads 6
uv run @rankingPackages python experiments/ranking.py blend experiments/repro-cache experiments/repro-ensemble experiments/repro-xendcg experiments/repro-debiased experiments/repro-binary
uv run @rankingPackages python experiments/ranking.py blend experiments/repro-cache experiments/repro-ensemble experiments/repro-xendcg experiments/repro-debiased experiments/repro-binary --test
```

## Второй цикл: 24 обучающие части

Сохраняем все шесть исходных файлов и добавляем 18 с seed `2026`. Проверяем тот же набор признаков, параметры всех трёх моделей, равные веса ансамбля и ту же хронологическую границу val. Сравниваем с `lgb-ensemble3-006` по полному NDCG на позднем val.

- [x] Подготовить train и проверить, что в нём присутствуют все шесть исходных файлов и что позиции соответствуют ключам кандидатов.
- [x] Обучить XE-NDCG, бинарный LightGBM и LambdaMART с поправкой на позиционное смещение.
- [x] Сравнить ансамбли на 6 и 24 частях. Сохранить лучший вариант и проверить его тестовый Parquet.

В кэше `experiments/cache-024/train.npz` 147 684 ленты и 11 566 844 кандидата. Для пяти лент из каждого файла сверены тройки `(item_id, pos, target)` с исходным Parquet. Список шардов и seed сохранены в самом кэше и в отчётах моделей.

| Решение | Деревьев | NDCG ранний val | NDCG поздний val | NDCG полный val | NDCG Public |
| --- | ---: | ---: | ---: | ---: | ---: |
| [XE-NDCG, 24 части](../experiments/lgb-xendcg-024/metrics.json) | 120 | 0.368826 | 0.362049 | 0.364758 | — |
| [LambdaMART с поправкой, 24 части](../experiments/lgb-debiased-024/metrics.json) | 120 | 0.369539 | 0.361510 | 0.364719 | — |
| [Бинарный LightGBM, 24 части](../experiments/lgb-binary-024/metrics.json) | 120 | 0.368076 | 0.362645 | 0.364816 | — |
| [Ансамбль трёх моделей, 24 части](../experiments/lgb-ensemble3-024/metrics.json) | 120 + 120 + 120 | 0.371151 | **0.364808** | 0.367344 | **0.3572675727** |
| [Ансамбль шести моделей, 6 и 24 части](../experiments/lgb-ensemble6-006-024/metrics.json) | шесть моделей | 0.371871 | 0.364616 | **0.367517** | — |

Ансамбль на 24 частях прибавил `+0.002817` на позднем val и `+0.002477` на полном val относительно первого ансамбля. Комбинация шести моделей лидировала на полном val, но уступила на позднем `0.000192`. Для дальнейшей работы выбран ансамбль на 24 частях по заранее заданному критерию позднего val. Разница между двумя новыми ансамблями мала.

Пользователь загрузил сабмит ансамбля на 24 частях и сообщил Public NDCG `0.3572675727`. Прирост относительно TEAM-004 составил `+0.0032492939476305`.

При сравнении с первым ансамблем прирост позднего NDCG по числу кандидатов составил `+0.00158` для 5–30, `+0.00217` для 31–100 и `+0.00433` для 101–300. Парный бутстрэп по лентам дал ориентировочный 95% интервал `[+0.00060, +0.00510]` для прироста по сравнению с первым ансамблем. [Результаты сравнения](../experiments/lgb-ensemble3-024/scale_comparison.json).

Для воспроизведения второго цикла использовать `$rankingPackages` из предыдущего раздела. Кэш val/test сохранён с первого цикла.

```powershell
uv run @rankingPackages python experiments/ranking.py prepare train experiments/repro-cache-024 --shards 24 --stage-embeddings --include-train-shards experiments/cache-006/train_shards.json
uv run @rankingPackages python experiments/ranking.py train experiments/cache-006 experiments/repro-xendcg-024 history --train-file experiments/repro-cache-024/train.npz --objective rank_xendcg --rounds 160 --threads 6
uv run @rankingPackages python experiments/ranking.py train experiments/cache-006 experiments/repro-debiased-024 history --train-file experiments/repro-cache-024/train.npz --position-debias --rounds 200 --threads 6
uv run @rankingPackages python experiments/ranking.py train experiments/cache-006 experiments/repro-binary-024 history --train-file experiments/repro-cache-024/train.npz --objective binary --rounds 120 --threads 6
uv run @rankingPackages python experiments/ranking.py blend experiments/cache-006 experiments/repro-ensemble-024 experiments/repro-xendcg-024 experiments/repro-debiased-024 experiments/repro-binary-024
uv run @rankingPackages python experiments/ranking.py blend experiments/cache-006 experiments/repro-ensemble-024 experiments/repro-xendcg-024 experiments/repro-debiased-024 experiments/repro-binary-024 --test
```

## Третий цикл: статистика объявлений

На тех же 24 частях и том же val проверяем четыре признака: логарифм числа показов, логарифм числа целевых действий, долю положительных со сглаживанием и логарифм показов за последние пять доступных дней train. Счётчики построены по полному train. Для каждой обучающей ленты они используют только дни до даты самой ленты по Москве. Val и test используют статистику до конца train.

Исторические показы доступны для 68,33% кандидатов на обучении, 65,43% на val и 53,82% на test. Такое различие учитываем при выборе ансамбля. [Профиль признаков](../experiments/cache-priors-024/profile.json).

| Решение | Деревьев | NDCG ранний val | NDCG поздний val | NDCG полный val | NDCG Public |
| --- | ---: | ---: | ---: | ---: | ---: |
| [XE-NDCG с новыми признаками](../experiments/lgb-xendcg-priors-024/metrics.json) | 160 | 0.369310 | 0.362805 | 0.365406 | — |
| [Бинарный LightGBM с новыми признаками](../experiments/lgb-binary-priors-024/metrics.json) | 120 | 0.371277 | 0.365537 | 0.367832 | — |
| [Смешанный ансамбль из пяти моделей](../experiments/lgb-ensemble-mixed-priors-024/metrics.json) | пять моделей | 0.372845 | 0.365686 | 0.368548 | — |
| [Ансамбль из обновлённых XE-NDCG и бинарной модели и исходного LambdaMART](../experiments/lgb-ensemble-priors-024/metrics.json) | 160 + 120 + 120 | **0.373049** | **0.366185** | **0.368929** | — |

Лучший вариант прибавил `+0.001377` на позднем val относительно решения с подтверждённым баллом Public. Ориентировочный 95% интервал парного бутстрэпа по лентам составляет `[-0.00055, +0.00335]`. Покрытие на test ниже, поэтому перенос этого прироста на лидерборд пока не подтверждён. [Сравнение](../experiments/lgb-ensemble-priors-024/comparison.json).

Для воспроизведения использовать `$rankingPackages` из первого цикла и сохранённый LambdaMART второго цикла.

```powershell
uv run @rankingPackages python experiments/item_priors.py experiments/cache-024/train.npz experiments/cache-006 experiments/repro-cache-priors-024
uv run @rankingPackages python experiments/ranking.py train experiments/repro-cache-priors-024 experiments/repro-xendcg-priors-024 history-priors --objective rank_xendcg --rounds 160 --threads 6
uv run @rankingPackages python experiments/ranking.py train experiments/repro-cache-priors-024 experiments/repro-binary-priors-024 history-priors --objective binary --rounds 120 --threads 6
uv run @rankingPackages python experiments/ranking.py blend experiments/repro-cache-priors-024 experiments/repro-ensemble-priors-024 experiments/repro-xendcg-priors-024 experiments/lgb-debiased-024 experiments/repro-binary-priors-024
uv run @rankingPackages python experiments/ranking.py blend experiments/repro-cache-priors-024 experiments/repro-ensemble-priors-024 experiments/repro-xendcg-priors-024 experiments/lgb-debiased-024 experiments/repro-binary-priors-024 --test
```

## Четвёртый цикл: недавний интерес и позиционное смещение

К 55 существующим признакам добавили 24. Среди них максимальное сходство по типу события и по окнам в 4, 16 и 64 события, спад веса с давностью, позиция, тип и источник наиболее похожего события, сходство с недавней историей в той же категории и с центром векторов кандидатов ленты. Сохранили 24 исходные части train и ключи кандидатов. Для val независимо сверены ключи, метки и точное совпадение прежних 55 признаков.

Отдельно проверяем бинарный LightGBM на 55 признаках с исходной позицией при обучении. При оценке каждый кандидат получает пять одинаковых для всех кандидатов опорных позиций в диапазоне `0..300`. Предсказанные вероятности усредняются с весами, пропорциональными длине интервалов позиций. Это приближение равномерного распределения для эксперимента, фактические позиции val/test неизвестны.

| Решение | Деревьев | NDCG ранний val | NDCG поздний val | NDCG полный val | NDCG Public |
| --- | ---: | ---: | ---: | ---: | ---: |
| [XE-NDCG с признаками недавнего интереса](../experiments/lgb-xendcg-intent-024/metrics.json) | 80 | 0.371021 | 0.364542 | 0.367132 | — |
| [Бинарный LightGBM с признаками недавнего интереса](../experiments/lgb-binary-intent-024/metrics.json) | 160 | 0.371735 | 0.367480 | 0.369181 | — |
| [Бинарный LightGBM с усреднением по опорным позициям](../experiments/lgb-binary-refpos-024/metrics.json) | 40 | 0.358514 | 0.353585 | 0.355556 | — |
| [Две модели с недавним интересом и исходный LambdaMART](../experiments/lgb-ensemble-intent-024/metrics.json) | три модели | 0.374108 | 0.368165 | 0.370541 | — |
| [Ансамбль из пяти моделей со статистикой и недавним интересом](../experiments/lgb-ensemble-mixed-intent-024/metrics.json) | пять моделей | 0.373914 | **0.368208** | 0.370489 | **0.3594949565** |

После загрузки пользователь сообщил Public NDCG `0.3594949565`. Прирост к предыдущему лучшему сабмиту `lgb-ensemble3-024` составил `+0.0022273838`. Общий прирост к TEAM-004 составил `+0.0054766777`.

Ансамбль из пяти моделей выбран по позднему val. Отправленный ансамбль он опередил на `+0.003400` на позднем val, `+0.002763` на раннем и `+0.003145` на полном. Прирост позднего NDCG составил `+0.00549`, `+0.00485` и `+0.00051` для лент с 5–30, 31–100 и 101–300 кандидатами соответственно. Ориентировочный 95% интервал парного бутстрэпа по лентам `[+0.00129, +0.00554]`. Тройной ансамбль практически совпал с ним на позднем val, разница `0.000043`. Модель с усреднением по опорным позициям уступила исходной бинарной модели со статистикой показов на `0.01195` позднего NDCG и осталась отдельным экспериментом. [Парное сравнение](../experiments/lgb-ensemble-mixed-intent-024/intent_comparison.json).

Пять моделей объединены после стандартизации scores внутри ленты с равными весами. Для воспроизведения использовать `$rankingPackages` первого цикла и компоненты предыдущих циклов.

```powershell
uv run @rankingPackages python experiments/intent_features.py train experiments/cache-priors-024 experiments/repro-cache-intent-024 --workers 6
uv run @rankingPackages python experiments/intent_features.py val experiments/cache-priors-024 experiments/repro-cache-intent-024 --workers 2
uv run @rankingPackages python experiments/intent_features.py test experiments/cache-priors-024 experiments/repro-cache-intent-024 --workers 5
uv run @rankingPackages python experiments/ranking.py train experiments/repro-cache-intent-024 experiments/repro-xendcg-intent-024 history-intent --objective rank_xendcg --rounds 160 --threads 6
uv run @rankingPackages python experiments/ranking.py train experiments/repro-cache-intent-024 experiments/repro-binary-intent-024 history-intent --objective binary --rounds 160 --threads 6
uv run @rankingPackages python experiments/ranking.py train experiments/cache-priors-024 experiments/repro-binary-refpos-024 history-priors --objective binary --rounds 120 --threads 6 --position-standardize
uv run @rankingPackages python experiments/ranking.py blend experiments/repro-cache-intent-024 experiments/repro-mixed-intent-024 experiments/repro-xendcg-intent-024 experiments/lgb-xendcg-priors-024 experiments/lgb-debiased-024 experiments/repro-binary-intent-024 experiments/lgb-binary-priors-024 --test
```

## Пятый цикл: больше обучающих лент

Точка отсчёта — отправленный ансамбль `lgb-ensemble-mixed-intent-024` с поздним NDCG `0.368208` и Public `0.3594949565`. Для проверки масштаба добавляем 24 целых train-файла к прежним 24, с seed `2026`. Прежние ленты, порядок кандидатов и готовые 79 признаков сохраняются. Для новых лент считаем тот же набор признаков и статистику показов по дням до даты ленты. Val и test остаются прежними.

Обучаем бинарный LightGBM и XE-NDCG на 48 частях. На 24 частях отдельно проверяем LambdaMART со встроенной поправкой на позиционное смещение и полным набором признаков недавнего интереса. Результаты сравниваем на 8 657 поздних лентах val и проверяем прирост на раннем val.

В выборке 48 частей 295 376 лент и 23 111 632 кандидата. Признаки исходных 24 частей сверены побайтово, а ключи кандидатов, позиции и метки для 120 новых лент сверены с исходными файлами. [Проверка кэша](../experiments/cache-intent-048/profile.json).

| Решение | Деревьев | NDCG ранний val | NDCG поздний val | NDCG полный val |
| --- | ---: | ---: | ---: | ---: |
| [LambdaMART с поправкой и 79 признаками, 24 части](../experiments/lgb-debiased-intent-024/metrics.json) | 160 | 0.370115 | 0.363443 | 0.366110 |
| [Отправленный ансамбль с заменой только LambdaMART](../experiments/lgb-ensemble-intent-rank024/metrics.json) | пять моделей | 0.373673 | 0.368315 | 0.370457 |
| [Бинарный LightGBM с 79 признаками, 48 частей](../experiments/lgb-binary-intent-048/metrics.json) | 160 | 0.372035 | 0.368395 | 0.369850 |
| [XE-NDCG с 79 признаками, 48 частей](../experiments/lgb-xendcg-intent-048/metrics.json) | 160 | 0.373270 | 0.364272 | 0.367869 |
| [Замена двух моделей на варианты с 48 частями](../experiments/lgb-ensemble-replace-intent-048/metrics.json) | пять моделей | 0.375359 | 0.368294 | 0.371119 |
| [Модели с 24 и 48 частями, равные веса](../experiments/lgb-ensemble-equal-intent-048/metrics.json) | семь моделей | 0.374898 | 0.368705 | 0.371181 |
| [Модели с 24 и 48 частями, веса по раннему val](../experiments/lgb-ensemble-weighted-intent-048/metrics.json) | семь моделей | 0.375040 | **0.368731** | **0.371253** |

На раннем val проверили суммарные веса двух новых моделей `25%, 40%, 55%, 70%, 85%, 100%`. Выбрали 40%, оставшиеся 60% распределены поровну между пятью моделями отправленного ансамбля. Поздний прирост `+0.000523`, ориентировочный 95% интервал парного бутстрэпа `[-0.00074, +0.00171]`. В лентах с 5–30 кандидатами NDCG снизился на `0.00082`. Улучшение недостаточно для сабмита. [Сравнение вариантов и веса](../experiments/cache-intent-048/scale_comparison.json).

```powershell
uv run @rankingPackages python experiments/expand_train.py experiments/cache-intent-024/train.npz experiments/repro-cache-intent-048/train.npz --workers 8
uv run @rankingPackages python experiments/ranking.py train experiments/cache-intent-024 experiments/repro-binary-intent-048 history-intent --train-file experiments/repro-cache-intent-048/train.npz --objective binary --rounds 160 --threads 8
uv run @rankingPackages python experiments/ranking.py train experiments/cache-intent-024 experiments/repro-xendcg-intent-048 history-intent --train-file experiments/repro-cache-intent-048/train.npz --objective rank_xendcg --rounds 160 --threads 8
uv run @rankingPackages python experiments/ranking.py blend experiments/cache-intent-024 experiments/repro-ensemble-weighted-intent-048 experiments/lgb-xendcg-intent-024 experiments/lgb-xendcg-priors-024 experiments/lgb-debiased-024 experiments/lgb-binary-intent-024 experiments/lgb-binary-priors-024 experiments/repro-xendcg-intent-048 experiments/repro-binary-intent-048 --weights .12 .12 .12 .12 .12 .2 .2
```

## Шестой цикл: вес ленты при обучении

У бинарного LightGBM в исходной настройке кандидаты имеют одинаковый вес. Длинные ленты поэтому вносят больше в функцию потерь, хотя финальная NDCG усредняется по лентам. На прежних 24 частях отдельно проверяем веса кандидата `1/√M` и `1/M`, где `M` — число кандидатов ленты. Средний вес кандидата нормирован к единице, остальные параметры совпадают с `lgb-binary-intent-024`.

| Решение | Деревьев | NDCG ранний val | NDCG поздний val | NDCG полный val |
| --- | ---: | ---: | ---: | ---: |
| [Бинарная модель с обычными весами](../experiments/lgb-binary-intent-024/metrics.json) | 160 | 0.371735 | 0.367480 | 0.369181 |
| [Бинарная модель, вес 1/√M](../experiments/lgb-binary-feed-sqrt-024/metrics.json) | 160 | 0.369914 | 0.364447 | 0.366633 |
| [Бинарная модель, вес 1/M](../experiments/lgb-binary-feed-equal-024/metrics.json) | 80 | 0.369335 | 0.363984 | 0.366124 |
| [Отправленный ансамбль с заменой бинарной модели на 1/√M](../experiments/lgb-ensemble-feed-sqrt-024/metrics.json) | пять моделей | 0.373599 | 0.367580 | 0.369986 |
| [Отправленный ансамбль с заменой бинарной модели на 1/M](../experiments/lgb-ensemble-feed-equal-024/metrics.json) | пять моделей | 0.373872 | 0.367634 | 0.370128 |
| [Взвешенный ансамбль 48 частей с добавлением модели 1/√M](../experiments/lgb-ensemble-scale-feed-048/metrics.json) | восемь моделей | 0.375394 | 0.368259 | 0.371112 |

Ни один из вариантов не улучшил ансамбль с моделями на 48 частях на позднем val. У обычной бинарной модели NDCG на коротких лентах `0.51837`, у `1/√M` — `0.51444`, у `1/M` — `0.51814`. Поэтому гипотеза об улучшении коротких лент тоже не подтвердилась. Лучший локальный результат этих двух циклов остаётся `0.368731` на позднем val. Для Public сохраняем подтверждённый ансамбль `lgb-ensemble-mixed-intent-024` с `0.3594949565`. [Сравнение во всех группах лент](../experiments/cache-intent-048/feed_weight_comparison.json).

```powershell
uv run @rankingPackages python experiments/ranking.py train experiments/cache-intent-024 experiments/repro-binary-feed-sqrt-024 history-intent --objective binary --rounds 160 --threads 6 --feed-weight-power .5
uv run @rankingPackages python experiments/ranking.py train experiments/cache-intent-024 experiments/repro-binary-feed-equal-024 history-intent --objective binary --rounds 160 --threads 6 --feed-weight-power 1
```

## Седьмой цикл: нейросетевое ранжирование внутри ленты

На тех же 24 частях train обучаем нейросеть на CPU. Она кодирует кандидатов, объединяет их представления внутри ленты по среднему и максимуму и использует этот общий контекст при выдаче оценок. Функция потерь распределяет вероятность внутри ленты между положительными кандидатами. Небольшой бинарный член учитывает положительные и отрицательные отдельно.

Сравниваем три варианта: 79 прежних признаков, те же признаки с исходными векторами пользователя и кандидатов и векторным средним 48 недавних событий, а также вариант с дополнительным попарным штрафом на восемь самых высоко оценённых отрицательных кандидатов. Векторы истории усредняются с убывающим по позиции весом. Схема исходных признаков и временное разбиение val прежние. Эпоху, веса ансамбля и способ нормализации выбираем на раннем val. Поздний val используем для проверки результата.

Для всех трёх запусков использован PyTorch `2.10.0+cpu`, 32 000 лент за эпоху и seed `2026`. Пропуски в векторах обрабатываются нулевыми представлениями, а в числовых признаках — отдельной маской. У исходных векторов сверены все ключи val и проверено содержимое в 40 лентах по исходным Parquet и `data.memmap`.

| Решение | Эпоха | NDCG ранний val | NDCG поздний val | NDCG полный val | NDCG Public |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Нейросеть на 79 признаках](../experiments/nn-context-024/metrics.json) | 10 | 0.372897 | 0.366298 | 0.368936 | — |
| [Смесь отправленного ансамбля и сети на 79 признаках](../experiments/nn-context-early-blend-024/comparison.json) | 10 | 0.376165 | 0.368540 | 0.371588 | — |
| [Нейросеть с исходными векторами](../experiments/nn-vectors-024/metrics.json) | 9 | 0.374278 | 0.368819 | 0.371001 | — |
| [Нейросеть с векторами и попарным штрафом](../experiments/nn-hard-pairs-024/metrics.json) | 4 | 0.370109 | 0.364026 | 0.366458 | — |
| [Ансамбль моделей LightGBM на 24 и 48 частях и сети с векторами](../experiments/nn-final-blend-024/comparison.json) | 9 у сети | 0.377552 | 0.371212 | 0.373746 | — |
| [Тот же ансамбль с разрешением ничьих](../experiments/nn-final-blend-024/resolved_comparison.json) | 9 у сети | **0.377574** | **0.371229** | **0.373766** | **0.3625182218** |

После загрузки пользователь сообщил Public NDCG `0.3625182218`. Прирост к `lgb-ensemble-mixed-intent-024` составил `+0.0030232653` на Public и `+0.0030211229` на позднем val. Общий прирост к TEAM-004 составил `+0.0084999430`.

В итоговой смеси вес LightGBM `55%`, сети с векторами `45%`. Смешиваются процентильные ранги двух прогнозов внутри ленты. Выбор всей конфигурации по раннему val дал `+0.003638` на раннем и `+0.003003` на позднем относительно Public-решения. Прирост позднего NDCG составляет `+0.00355` для 5–30, `+0.00499` для 31–100 и `+0.00075` для 101–300 кандидатов. Ориентировочный 95% интервал парного бутстрэпа `[+0.00117, +0.00483]`. Попарный штраф снизил качество и в выбранный ансамбль не вошёл.

В 12 997 тестовых лентах исходная смесь дала одинаковые scores хотя бы двум кандидатам. Разрешили ничьи по смеси стандартизованных исходных оценок двух моделей и записали уникальные ранги как scores. Порядок кандидатов с различными оценками сохранился. После этого поздний NDCG составил `0.371229`, а выигрыш к отправленному решению `+0.003021`. При проверке 2 904 806 пар test равных scores внутри лент не осталось.

Воспроизведение из корня проекта с `$rankingPackages` из первого цикла. Для шага `predict` нужен тестовый прогноз LightGBM из пятого цикла. Его создаёт `ranking.py blend` с теми же компонентами и весами, `--test` и сохранённым каталогом ансамбля.

```powershell
$neuralPackages = $rankingPackages + @('--with', 'torch==2.10.0')
foreach ($split in 'train','val','test') {
  uv run @neuralPackages python experiments/embedding_cache.py $split experiments/cache-intent-024 experiments/repro-embedding-cache-024
}
uv run @neuralPackages python experiments/set_ranker.py train experiments/cache-intent-024 experiments/repro-nn-context-024 --epochs 11 --feeds-per-epoch 32000
uv run @neuralPackages python experiments/set_ranker.py train experiments/cache-intent-024 experiments/repro-nn-vectors-024 --embeddings experiments/repro-embedding-cache-024 --epochs 11 --feeds-per-epoch 32000
uv run @neuralPackages python experiments/set_ranker.py train experiments/cache-intent-024 experiments/repro-nn-hard-pairs-024 --embeddings experiments/repro-embedding-cache-024 --epochs 10 --feeds-per-epoch 32000 --pairwise-weight .5
uv run @neuralPackages python experiments/neural_blend.py select experiments/cache-intent-024 experiments/repro-nn-final-024 --baselines experiments/lgb-ensemble-mixed-intent-024 experiments/lgb-ensemble-weighted-intent-048 --neural experiments/repro-nn-context-024 experiments/repro-nn-vectors-024 experiments/repro-nn-hard-pairs-024
uv run @neuralPackages python experiments/set_ranker.py predict experiments/cache-intent-024 experiments/repro-nn-vectors-024/model.pt experiments/repro-nn-vectors-024/test_predictions.parquet --embeddings experiments/repro-embedding-cache-024
uv run @neuralPackages python experiments/neural_blend.py predict experiments/cache-intent-024 experiments/repro-nn-final-024/comparison.json experiments/repro-nn-final-024/submission-resolved.parquet experiments/lgb-ensemble-weighted-intent-048/submission.parquet experiments/repro-nn-vectors-024/test_predictions.parquet --resolve-ties
```

## Восьмой цикл: больше данных для сети и отдельные события истории

Сравниваем с отправленным `nn-final-blend-024`, Public `0.3625182218` и поздний val `0.371229`. Для сравнения используем файл `val_predictions_resolved.parquet`, соответствующий разрешению ничьих в сабмите.

Первый запуск использует ту же архитектуру с исходными векторами, 48 частей train (295 376 лент, включая исходные 24) и 44 000 лент за эпоху. Размер выборки и число обновлений выросли вместе. Val/test и архитектура совпадают с `nn-vectors-024`. На большом train исходная матрица признаков читается через отображение файла в память.

Второй запуск на 24 частях добавляет до 16 индивидуальных событий к вектору средней истории. Берём восемь самых свежих событий с вектором, затем по одному доступному событию начиная от заранее заданных старших позиций. Веса внимания зависят от кандидата, сходства векторов и обучаемых представлений типа, источника, категории и давности события. В среднем на train доступно 15,10 события на ленту, на val 15,07. Ключи val сверены полностью, содержимое 40 лент проверено по исходным событиям и `data.memmap`.

Третий запуск на тех же 24 частях добавляет каждому кандидату вектор ближайшего события из всей доступной истории. Позицию события берём из сохранённого `intent_match_event_rank`. Нужный вектор доступен для 98,76% кандидатов train и 98,56% кандидатов val. Ключи val сверены полностью. В 40 лентах ближайшие векторы проверены по исходному `data.memmap`.

| Решение | Эпоха | NDCG ранний val | NDCG поздний val | NDCG полный val |
| --- | ---: | ---: | ---: | ---: |
| [Нейросеть на 48 частях](../experiments/nn-vectors-048/metrics.json) | 8 | 0.377259 | 0.368014 | 0.371710 |
| [Нейросеть с вниманием к 16 событиям](../experiments/nn-attention-024/metrics.json) | 10 | 0.375915 | 0.367191 | 0.370679 |
| [Нейросеть с вектором ближайшего события](../experiments/nn-nearest-024/metrics.json) | 9 | 0.374390 | 0.369464 | 0.371433 |
| [Ансамбль, выбранный по раннему val](../experiments/nn-history-final-024/comparison.json) | три компонента | 0.379475 | 0.369641 | 0.373573 |
| [Тот же ансамбль с весами по длине ленты](../experiments/nn-history-size-blend-024/comparison.json) | три компонента | 0.379620 | 0.369125 | 0.373321 |

Сеть с ближайшим событием улучшила отдельную сеть `nn-vectors-024` на `+0.000645` позднего NDCG. Для ансамбля пользы не хватило. Из четырёх семейств смесей с весами по раннему val максимальный поздний NDCG `0.371044` дал zscore-ансамбль с тремя новыми сетями и LightGBM. Он уступил отправленному `0.371229`. Победитель по раннему val уступил отправленному ансамблю на `0.001588` на позднем val. Вариант с весами по длине ленты уступил на `0.002104`.

Разбивка по датам для смеси двух новых сетей и LightGBM показала `+0.00065` на 22 сентября и `−0.00112` на 24 сентября к отправленному ансамблю. Ранний val целиком лежит в пределах 22 сентября, поэтому прироста только на нём недостаточно. Новый сабмит для этого цикла не нужен. [Разбивка по датам](../experiments/nn-history-scale-blend-024/temporal_breakdown.json).

```powershell
uv run @neuralPackages python experiments/embedding_cache.py train experiments/cache-intent-048 experiments/repro-embedding-cache-048
uv run @neuralPackages python experiments/set_ranker.py train experiments/cache-intent-024 experiments/repro-nn-vectors-048 --embeddings experiments/embedding-cache-024 --train-file experiments/cache-intent-048/train.npz --train-embeddings experiments/repro-embedding-cache-048/train.npz --mmap-train --epochs 11 --feeds-per-epoch 44000
foreach ($split in 'train','val') {
  uv run @neuralPackages python experiments/history_event_cache.py $split experiments/cache-intent-024 experiments/repro-history-event-cache-024
  uv run @neuralPackages python experiments/history_event_cache.py $split experiments/cache-intent-024 experiments/repro-history-nearest-cache-024 --nearest
}
uv run @neuralPackages python experiments/set_ranker.py train experiments/cache-intent-024 experiments/repro-nn-attention-024 --embeddings experiments/embedding-cache-024 --events experiments/repro-history-event-cache-024 --epochs 11 --feeds-per-epoch 32000
uv run @neuralPackages python experiments/set_ranker.py train experiments/cache-intent-024 experiments/repro-nn-nearest-024 --embeddings experiments/embedding-cache-024 --nearest experiments/repro-history-nearest-cache-024 --epochs 11 --feeds-per-epoch 32000
uv run @neuralPackages python experiments/neural_blend.py select experiments/cache-intent-024 experiments/repro-nn-history-024 --baselines experiments/nn-final-blend-024 experiments/lgb-ensemble-weighted-intent-048 --neural experiments/repro-nn-attention-024 experiments/repro-nn-vectors-048 experiments/repro-nn-nearest-024 --reference-scores experiments/nn-final-blend-024/val_predictions_resolved.parquet --reference-test-scores experiments/nn-final-blend-024/submission-resolved.parquet
uv run @neuralPackages python experiments/neural_blend.py adapt experiments/cache-intent-024 experiments/repro-nn-history-024/comparison.json experiments/repro-nn-history-size-024
```

## Девятый цикл: поправка оценок на случайно перемешанных лентах

Заморозили все основные модели и подбирали небольшую линейную поправку к их scores только на раннем val. Из 5 766 ранних лент 3 749 использованы для подгонки коэффициентов и следующие 2 017 для выбора состава признаков и регуляризации. Затем коэффициенты переоценены на полном раннем val. Поздние 8 657 лент используют готовые коэффициенты. Для сравнения взяты оценки отправленного `nn-final-blend-024` с разрешёнными ничьими.

Линейная поправка использует оценку отправленного ансамбля, отдельно scores деревьев и нейросети, а также признаки объявления и истории. Числовые признаки стандартизируются внутри ленты; у самых сильных сигналов проверены нелинейные поправки в хвостах. Основная функция потерь оценивает распределение вероятности для всей ленты. В попарном варианте веса сравнения положительного и отрицательного кандидатов зависят от потери NDCG при их перестановке в исходном порядке.

| Решение | NDCG ранний val | NDCG поздний val | NDCG полный val | Прирост на позднем |
| --- | ---: | ---: | ---: | ---: |
| [Линейная поправка](../experiments/nn-calibration-024/metrics.json) | 0.380130 | 0.372044 | 0.375277 | +0.000815 |
| [Линейная поправка с весом по длине ленты](../experiments/nn-calibration-size-024/metrics.json) | 0.380143 | 0.371811 | 0.375142 | +0.000581 |
| [Попарная поправка только на верхних отрицательных](../experiments/nn-calibration-pairs-024/metrics.json) | 0.271114 | 0.269549 | 0.270175 | −0.101680 |
| [Попарная поправка с выборкой по всем рангам](../experiments/nn-calibration-stratified-024/metrics.json) | 0.376864 | 0.371775 | 0.373809 | +0.000545 |
| [Линейная поправка с оценками двух дополнительных нейросетей](../experiments/nn-calibration-extra-024/metrics.json) | 0.379938 | 0.372428 | 0.375430 | +0.001198 |
| [Смесь поправок по длине ленты](../experiments/nn-calibration-joint-024/metrics.json) | 0.380098 | **0.372957** | **0.375812** | **+0.001728** |

Исключение всех отрицательных ниже первых 32 привело к смещённому набору пар. На раннем val средняя оценка отправленного ансамбля у положительных `0.696`, у всех отрицательных `0.502`. После отбора верхних отрицательных и взвешивания пар средний отступ положительного от отрицательного по логарифмическому рангу стал отрицательным (`−0.985`), а коэффициент деревьев сменил знак. Исправили выборку. В лентах с более чем 32 отрицательными сохраняем верхние восемь, а остальные представляем 24 интервалами с весами по числу кандидатов. Исправленная схема улучшила короткие ленты на `+0.00182`.

У лучшей смеси веса исходного прогноза, линейной и исправленной попарной поправок выбраны на внутренних 2 017 лентах. Для 5–30 кандидатов веса `(⅓, ⅓, ⅓)`, для 31–100 `(0, 1, 0)`, для 101–300 `(0, ½, ½)`. Приросты на позднем val по этим группам составили `+0.00267`, `+0.00233` и `+0.00047`. Парный бутстрэп по поздним лентам дал ориентировочный 95% интервал `[+0.00069, +0.00287]`. Проверка гладкой аппроксимации NDCG не прошла отбор на внутреннем раннем блоке. [Метрики аппроксимации](../experiments/nn-calibration-smooth-024/metrics.json) и [описание ApproxNDCG](https://doi.org/10.1007/s10791-009-9124-x).

Резервный кандидат прошёл проверку всех 2 904 806 тестовых пар. Ничьих внутри лент нет. Поздний прирост меньше принятого ориентира `+0.003`, поэтому пока бережём попытку на Public. Подтверждённый результат команды `0.3625182218`. [Резервный файл](../experiments/nn-calibration-joint-024/submission-reserve.parquet) и [проверка](../experiments/nn-calibration-joint-024/submission_audit.json).

Воспроизведение из корня проекта с `$rankingPackages` из первого цикла. Сохранённый конфиг смеси содержит выбранные веса. Тестовые predictions двух дополнительных нейросетей сохранены в каталогах их запусков.

```powershell
$calValScores = @('experiments/nn-final-blend-024/val_predictions_resolved.parquet','experiments/lgb-ensemble-weighted-intent-048/val_predictions.parquet','experiments/nn-vectors-024/val_predictions.parquet')
$calTestScores = @('experiments/nn-final-blend-024/submission-resolved.parquet','experiments/lgb-ensemble-weighted-intent-048/submission.parquet','experiments/nn-vectors-024/test_predictions.parquet')
$calExtraVal = $calValScores + @('experiments/nn-nearest-024/val_predictions.parquet','experiments/nn-vectors-048/val_predictions.parquet')
$calExtraTest = $calTestScores + @('experiments/nn-nearest-024/test_predictions.parquet','experiments/nn-vectors-048/test_predictions.parquet')
uv run @rankingPackages python experiments/score_calibration.py select experiments/cache-intent-024 experiments/repro-calibration-extra-024 @calExtraVal
uv run @rankingPackages python experiments/score_calibration.py select experiments/cache-intent-024 experiments/repro-calibration-pairs-024 @calValScores --pairwise
uv run @rankingPackages python experiments/score_calibration.py predict experiments/cache-intent-024 experiments/repro-calibration-extra-024/metrics.json experiments/repro-calibration-extra-024/test_predictions.parquet @calExtraTest
uv run @rankingPackages python experiments/score_calibration.py predict experiments/cache-intent-024 experiments/repro-calibration-pairs-024/metrics.json experiments/repro-calibration-pairs-024/test_predictions.parquet @calTestScores
uv run @rankingPackages python experiments/score_calibration.py combine experiments/cache-intent-024 experiments/nn-calibration-joint-024/metrics.json experiments/nn-calibration-joint-024/submission-repro.parquet experiments/nn-final-blend-024/submission-resolved.parquet experiments/repro-calibration-extra-024/test_predictions.parquet experiments/repro-calibration-pairs-024/test_predictions.parquet
```

## Быстрая проверка перед одной попыткой Public

При двух оставшихся попытках на 15 ноября проверили, поможет ли сильнее учитывать совпадения в верхней части ленты. Для каждого прогноза сравнили линейные ранги и преобразование `1/(k + ранг)` для `k = 8, 20, 60`. Веса по длине ленты подобраны на двух последовательных ранних блоках. Для каждого блока поправки обучались только на предшествующих лентах. Поздние ленты не использованы в выборе.

| Вариант | NDCG поздний val | Разница с отправленным |
| --- | ---: | ---: |
| Отправленный ансамбль | 0.371229 | — |
| Выбранные на раннем val линейные ранги | 0.372652 | +0.001423 |
| Обратный ранг, `k=8` | 0.372408 | +0.001178 |
| Обратный ранг, `k=20` | 0.372410 | +0.001180 |
| Обратный ранг, `k=60` | 0.372395 | +0.001166 |
| Резервная смесь поправок | **0.372957** | **+0.001728** |

Для одной попытки выбрали уже подготовленный резерв. Его прирост к предыдущему отправленному ансамблю составил `+0.00188` и `+0.00157` в первой и второй временных половинах позднего val. Ещё раз проверены все 2 904 806 тестовых пар и отсутствие ничьих. После загрузки пользователь сообщил NDCG **0.3652139518 на Public**. Прирост к предыдущему сабмиту составил `+0.0026957300`, общий прирост к TEAM-004 составил `+0.0111956730`. [Сабмит](../experiments/nn-calibration-joint-024/submission-reserve.parquet), [сравнение смесей](../experiments/nn-calibration-top-fusion-024/metrics.json) и [проверка по половинам](../experiments/nn-calibration-top-fusion-024/reserve_verification.json).

```powershell
uv run @rankingPackages python experiments/top_rank_fusion.py select experiments/cache-intent-024 experiments/repro-top-rank-024 experiments/nn-final-blend-024/val_predictions_resolved.parquet experiments/nn-calibration-joint-024/val_predictions.parquet experiments/nn-calibration-extra-024 experiments/nn-calibration-stratified-024
```

## Сопоставление позднего val и Public

Загрузки, для которых есть оценки обоих решений на одних и тех же поздних лентах.

| Изменение | Прирост на позднем val | Прирост на Public |
| --- | ---: | ---: |
| Признаки недавнего интереса и смесь с более простыми моделями | +0.003400 | +0.002227 |
| Нейросеть с исходными векторами в ансамбле с LightGBM | +0.003021 | +0.003023 |
| Поправка оценок по данным случайно перемешанных лент | +0.001728 | +0.002696 |
| Поправка по истории показов с учётом позиции, длины и категории | +0.000328 | −0.000294 |
| 10% GPU Transformer после 140 408 лент | −0.000062 | −0.000344 |

- В первых трёх переходах поздний val верно показал направление изменения. Поправка по истории показов выиграла в обеих поздних половинах, а Public снизился. Для небольших приростов учитываем парный интервал и сравниваем покрытие сигнала в val и test. У этой поправки интервал включал ноль, а покрытие кандидатов было 62.77% на позднем val и 53.82% на test.
- Сильные шаги добавляли новый сигнал или исправляли использование старого. История, обучение на исходных векторах и небольшая поправка по лентам с политикой показа как в test помогли. Удвоение train, более сложная обработка истории и усиление верхних рангов в смеси иногда улучшали ранний val, но не выигрывали на позднем. Держим точные прогнозы отправленного решения как контроль. Подбираем веса на ранних лентах и сравниваем замороженный вариант на поздних.
- У нового ансамбля на длинные ленты с 101–300 кандидатами приходится 37,5% позднего val. Здесь NDCG вырос лишь на `+0.00047`, против `+0.00267` и `+0.00233` у коротких и средних. В длинных лентах хотя бы один положительный есть в топ-10 у 24,7% лент и в топ-30 у 50,6%. У 25,9% таких лент первый положительный находится на местах 11–30. Это измеренная область для следующей проверки.
- Простое объединение топ-10 текущего ансамбля, деревьев и трёх сетей в длинных лентах расширило список примерно до 16,4 кандидата. Положительный появился у 34,3% лент. Верхние кандидаты текущего ансамбля при том же числе позиций дали 34,6%, поэтому для первого опыта хватает текущего списка. Добавление объединения к топ-30 дало 51,16% против 51,09% при равном числе кандидатов. [Разбор ошибок и покрытие при равном бюджете](../experiments/nn-calibration-joint-024/late_error_analysis.json).

Следующая гипотеза с приоритетом — переставлять первые 30 кандидатов только в длинных лентах. Начать с небольшой поправки, сравнивающей исходные оценки LightGBM и нейросетей, свежесть истории и согласие моделей. Для настройки брать оценки, полученные без обучения на рассматриваемой ленте. На позднем val смотреть полный NDCG, NDCG@10 в длинных лентах и прирост в двух временных половинах. Позиции ниже 30 в этом контролируемом опыте остаются в исходном порядке. Если задача требует приблизиться к `+0.003` только за счёт длинных лент, в этой группе нужен выигрыш порядка `+0.008`.

## Десятый цикл: перестановка топ-30 в длинных лентах

16 ноября проверили гипотезу на лентах со 101–300 кандидатами. Переранжирование ограничено первыми 30 объявлениями текущего ансамбля `nn-calibration-joint-024`. Остальные scores фиксированы. Линейная модель учитывает исходный ранг, оценки шести компонентов, согласие верхних позиций и до 12 признаков кандидата и истории.

В раннем val восстановили прогнозы калибровочных компонентов на четырёх последовательных блоках. Коэффициенты каждого компонента для блока получены по предшествующим лентам. Из этих примеров 801 длинная лента использована для подгонки модели топ-30, 749 следующих — для выбора размера поправки и регуляризации. После выбора модель переоценена на всех 1 550 ранних длинных лентах, у 791 из них был положительный в исходном топ-30. На позднем val не подбирались параметры. Оценки ранних лент в итоговых `val_predictions.parquet` получены после переоценки, поэтому их нельзя использовать как независимые прогнозы для следующего ансамбля.

| Вариант | NDCG поздний val | Прирост на позднем val | Прирост в длинных лентах |
| --- | ---: | ---: | ---: |
| Текущий отправленный ансамбль | 0.372957 | — | — |
| [Свободная перестановка топ-30](../experiments/long-head-rerank-verified-024/metrics.json) | 0.372957 | 0 | 0 |
| [Первые три места фиксированы, сдвиг не более пяти позиций](../experiments/long-head-rerank-conservative-024/metrics.json) | 0.372963 | +0.000006 | +0.000016 |

В свободном варианте ни одна ненулевая поправка не обошла контроль на внутреннем блоке; выбрано сохранение исходного порядка. В осторожном варианте NDCG@10 длинных лент вырос с `0.097326` до `0.097476`. Прирост полного NDCG в временных половинах позднего val имел разные знаки (`−0.0000066`, `+0.0000184`). В 131 из 3 245 поздних длинных лент изменился состав топ-10, в трёх лентах там появился положительный, которого раньше не было. Проверка всех 14 423 лент подтвердила неизменность scores вне затронутых топ-30, сохранение первых трёх и предел в пять позиций. [Проверка ограничений](../experiments/long-head-rerank-conservative-024/scope_audit.json).

Прирост слишком мал для отправки. Лучший Public остаётся `0.3652139518`. Разбор предыдущих ошибок подтвердил наличие положительных ниже топ-10, но простая поправка из доступных scores и признаков почти не помогла отличить их от соседей. Начальный запуск с ошибкой в дисконтировании исключён; в таблице только пересчитанные результаты.

```powershell
uv run @rankingPackages python experiments/long_head_rerank.py select experiments experiments/repro-long-head-free-024
uv run @rankingPackages python experiments/long_head_rerank.py select experiments experiments/repro-long-head-conservative-024 --conservative
```

## Одиннадцатый цикл. Поправка на показ, перевзвешивание train и переходы между товарами

В каждом опыте параметры выбраны на первых 5 766 лентах val, итоговая проверка выполнена на следующих 8 657 лентах. Контроль — точные val-оценки сабмита с Public `0.3652139518`. В таблице показана полная NDCG после разрешения ничьих в смесях. Временные половины позднего блока содержат 4 317 и 4 340 лент.

| Вариант | Поздний val | Прирост к сабмиту | Прирост по временным половинам |
| --- | ---: | ---: | ---: |
| Отправленный ансамбль | **0.372957** | — | — |
| [Смесь с нейросетью, учитывающей позицию показа](../experiments/exposure-blend-024/branch_audit.json) | 0.371664 | −0.001293 | −0.001357 / −0.001228 |
| [Смесь с перевзвешенным LightGBM, выбранная на раннем val](../experiments/domain-blend-024/branch_audit.json) | 0.372957 | 0 | 0 / 0 |
| [Диагностика смеси с 5% перевзвешенного LightGBM](../experiments/domain-blend-024/branch_audit.json) | 0.373116 | +0.000159 | −0.000009 / +0.000325 |
| [Совместный поиск весов двух новых моделей](../experiments/exposure-domain-blend-024/comparison.json) | 0.371664 | −0.001293 | −0.001357 / −0.001228 |
| [Поправка по переходам между товарами](../experiments/transition-probe-024/comparison.json) | 0.372942 | −0.000015 | +0.000058 / −0.000088 |

- **Позиция показа.** Предсказание отклика в train — произведение сигмоиды релевантности и обучаемой монотонной функции позиции. Функция применяется в loss, на val используется оценка релевантности. На одинаковых 24 файлах train нейросеть с этой функцией дала `0.365315` против `0.368819` у предыдущей нейросети с исходными векторами. Одновременно изменился баланс потерь, поэтому сравнение оценивает вариант целиком. В смеси выбран вес `0.30` для новой ветки. Ранний прирост после разрешения ничьих `+0.000949` сменился ухудшением в обеих половинах позднего блока. Функция позиции описывает наблюдаемый train, где исходный порядок связан с релевантностью. [Метрики обучения](../experiments/nn-exposure-024/metrics.json).
- **Перевзвешивание по сходству с val.** Доменный классификатор учился на признаках кандидатов 24 train-файлов и 3 749 ранних val-лент. Для проверки AUC использовали отдельные train-ленты и 2 017 оставшихся ранних val-лент. В классификации домена метки отклика val не участвовали. AUC `0.82545`; оценки отношения плотностей ограничены диапазоном `[0.25, 4]` и сглажены квадратным корнем. Эффективный размер выборки — `91.2%`. При тех же 160 деревьях поздняя NDCG бинарного LightGBM снизилась с `0.367480` до `0.365835`, хотя ранняя выросла. В сетке весов ранний блок выбрал ноль. Дополнительная проверка веса `0.05` дала малый выигрыш, сосредоточенный во второй половине. При совместном поиске на раннем блоке снова выбрана только новая нейросеть с весом `0.30`. [Диагностика весов](../experiments/domain-adapt-024/metrics.json) и [метрики LightGBM](../experiments/lgb-binary-domain-024/metrics.json).
- **Переходы между товарами.** По всем 512 train-файлам посчитали переходы от разных товаров среди 12 недавних событий к товару с положительной меткой в ленте. Граф содержит 138 740 направленных пар, ограниченных товарами, встречающимися среди истории и кандидатов val. При проверке по поздним лентам граф затронул `6.79%` лент, `0.085%` кандидатов и `0.258%` положительных кандидатов. Точные совпадения слишком редки для заметного улучшения. Для следующей проверки этого подхода стоит обобщать переходы между группами товаров по эмбеддингам и сначала измерять покрытие на случайно перемешанных лентах. [Профиль покрытия](../experiments/transition-probe-024/comparison.json).

Лучший результат Public в журнале — `0.3652139518`. Ни одна проверенная смесь в этом цикле не дала оснований тратить попытку. Для следующей ветки имеет смысл проверять знак прироста в двух последовательных частях раннего блока до итоговой оценки. В этом цикле все три отдельные модели или поправки показали, что выигрыш на ранних лентах может исчезнуть на следующих.

```powershell
uv run @rankingPackages --with torch==2.10.0 python experiments/set_ranker.py train experiments/cache-intent-024 experiments/repro-nn-exposure-024 --embeddings experiments/embedding-cache-024 --exposure-model --epochs 11 --feeds-per-epoch 32000 --threads 6
uv run @rankingPackages python experiments/domain_weights.py experiments/cache-intent-024 experiments/repro-domain-adapt-024 --threads 6
uv run @rankingPackages python experiments/ranking.py train experiments/cache-intent-024 experiments/repro-lgb-domain-024 history-intent --objective binary --rounds 160 --threads 6 --sample-weights experiments/repro-domain-adapt-024/train_weights.npz
uv run @rankingPackages python experiments/transition_probe.py experiments/cache-intent-024 experiments/nn-calibration-joint-024/val_predictions.parquet experiments/repro-transition-probe-024
uv run @rankingPackages python experiments/neural_blend.py select experiments/cache-intent-024 experiments/repro-exposure-domain-blend-024 --baselines experiments/nn-calibration-joint-024 --neural experiments/repro-nn-exposure-024 experiments/repro-lgb-domain-024 --reference-scores experiments/nn-calibration-joint-024/val_predictions.parquet --reference-test-scores experiments/nn-calibration-joint-024/submission-reserve.parquet
```

## Двенадцатый цикл. Переходы между группами похожих товаров

По 150 000 эмбеддингов train оценили главные компоненты и пороги по медианам. Первые 8 и 10 компонент задают до 256 и 1 024 групп. Категории служат контролем. Для каждой группы посчитали направленные переходы от 12 последних событий истории. Один сигнал сравнивает число положительных откликов с общим распределением положительных по группам. Второй учитывает частоту попадания группы в ленты. Для её оценки взяли до 16 кандидатов из каждой train-ленты с весом, компенсирующим долю отбора. Это поправка по наблюдаемому составу ленты; позицию показа она не моделирует. [Граф и статистика обучения](../experiments/semantic-transition-fit/fit.json).

Обработаны все 512 файлов, 3 151 208 лент, 5 446 872 положительных отклика и 46 287 708 отобранных показов. Проверили 84 сочетания сигнала, веса и области применения. Вес выбирали по меньшему из приростов на двух последовательных ранних блоках из 3 749 и 2 017 лент. Затем оценили вариант на прежних 8 657 поздних лентах. Контроль — прогнозы сабмита с Public `0.3652139518`.

| Вариант | Покрытие положительных на позднем val | NDCG | NDCG@10 | Положительных в топ-10 |
| --- | ---: | ---: | ---: | ---: |
| Отправленный ансамбль | — | 0.372957 | 0.245700 | 4 864 |
| Граф точных ID из предыдущего цикла | 0.258% | 0.372942 | 0.245684 | — |
| Выбранные переходы между 1 024 группами с учётом показов | 98.846% | **0.373050** | **0.245809** | 4 866 |
| Такая же поправка только в лентах со 101+ кандидатом | 98.846% по всему блоку до ограничения | 0.373013 | 0.245783 | 4 866 |

У выбранного варианта вес `0.05`. Прирост полной NDCG `+0.000093`, парный 95%-й бутстрэп-интервал по лентам `[−0.000178, +0.000360]`. По временным половинам `−0.000135` и `+0.000320`. В семи лентах в топ-10 стало больше положительных, в пяти меньше. Граф с 256 группами без поправки на состав ленты дал диагностические `4 886` положительных в топ-10, но проиграл на втором раннем блоке и первой половине позднего. Поддерживаемые графом переходы доступны почти всем положительным кандидатам. Сама поправка в проверенном виде даёт слишком мало устойчивых изменений порядка, чтобы тратить попытку Public. [Подробное сравнение](../experiments/semantic-transition-eval-024/diagnostics.json) и [сетка раннего поиска](../experiments/semantic-transition-eval-024/early_search.json).

В `val_graph_scores.npz` сохранены шесть признаков с проверенными ключами лент и кандидатов. Они пригодны для будущего сравнения обучаемого ранжирования. Следующий эксперимент может проверить, дают ли переходы дополнительный сигнал при обучении на train с признаками графа, посчитанными только по предшествующим дням.

```powershell
uv run @rankingPackages python experiments/semantic_transitions.py experiments/repro-semantic-transition-fit
uv run @rankingPackages python experiments/semantic_transition_rerank.py select experiments/cache-intent-024 experiments/repro-semantic-transition-fit/graph.npz experiments/nn-calibration-joint-024/val_predictions.parquet experiments/repro-semantic-transition-eval-024
```

## Тринадцатый цикл. Графовые признаки внутри LightGBM

Добавили к 79 базовым признакам 12 графовых. Для категорий и двух разбиений эмбеддингов сохраняются средние по истории оценки перехода к группе кандидата и те же оценки относительно конкурентов в ленте. Использовали прежнее разбиение по эмбеддингам. Частоты переходов и ожидаемое число откликов для каждой train-ленты получены только по предыдущим завершённым дням. Для val использованы все дни train. В 1 105 615 строках первого дня кэша графовые признаки отсутствуют, как и должно быть при пустом префиксе. По всем 11 566 844 строкам train и 1 389 971 строке val проверены даты, ключи и числа кандидатов. [Аудит графа и выравнивания](../experiments/daily-transition-024/alignment_audit.json).

Обе модели и их контрольные варианты обучены на одинаковых 24 файлах с одинаковыми гиперпараметрами и бюджетом в 160 деревьев. Число деревьев отдельно выбрано по раннему val. Для бинарной ветки это 120 с графом и 160 в контроле, для `rank_xendcg` — 160 и 80.

| Ветка | Поздний val без графа | Поздний val с графом | Δ NDCG | Δ NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| [Бинарная](../experiments/lgb-binary-graph-024/metrics.json) | 0.367480 | 0.368020 | +0.000540 | −0.000614 |
| [`rank_xendcg`](../experiments/lgb-xendcg-graph-024/metrics.json) | 0.364542 | 0.364866 | +0.000325 | +0.000276 |

Бинарная модель улучшила полную NDCG в обеих поздних половинах (`+0.000849`, `+0.000232`). У `rank_xendcg` поздний эффект поменял знак (`+0.002069`, `−0.001410`). Внутри лент на местах 11–30 отдельные графовые признаки слабо различают положительные и отрицательные объявления. У сигнала по положительным переходам между 1 024 группами попарный AUC на этих местах составил `0.531` и `0.512` в двух половинах. [Попарная диагностика](../experiments/daily-transition-024/conditional_head_audit.json).

Проверили 36 сочетаний веса и сигнала поверх отправленного ансамбля. Сравнили добавление полных стандартизованных оценок новой модели и добавление только разницы с её контролем. Лучший по двум ранним блокам вариант — `5%` полной оценки бинарной ветки. На позднем val он дал **0.373001** против `0.372957` (`+0.000044`). В топ-10 оказалось 4 869 положительных вместо 4 864. По половинам эффект `−0.000075` и `+0.000163`. Парный 95%-й интервал `[−0.000130, +0.000214]`. Попытку Public с этим вариантом тратить не стоит. Лучший подтверждённый Public — `0.3652139518`. [Сравнение моделей и смесей](../experiments/daily-graph-comparison-024/assessment.json).

Следующая независимая проверка — сравнить тот же ранжировщик на свежем train и на случайной выборке такого же размера. Это проверит переносимость по времени напрямую. Подсчёт переходов по большему числу файлов уже дал широкий охват, но эта ветка добавила слишком мало к итоговому ансамблю.

```powershell
uv run @rankingPackages python experiments/daily_transition_features.py prepare experiments/semantic-transition-fit/graph.npz experiments/cache-intent-024 experiments/repro-daily-transition-024
uv run @rankingPackages python experiments/ranking.py train experiments/cache-intent-024 experiments/repro-lgb-binary-graph-024 history-intent --objective binary --rounds 160 --threads 6 --extra-features experiments/repro-daily-transition-024
uv run @rankingPackages python experiments/ranking.py train experiments/cache-intent-024 experiments/repro-lgb-xendcg-graph-024 history-intent --objective rank_xendcg --rounds 160 --threads 6 --extra-features experiments/repro-daily-transition-024
uv run @rankingPackages python experiments/compare_graph_rankers.py experiments/cache-intent-024 experiments/nn-calibration-joint-024/val_predictions.parquet experiments/repro-daily-graph-comparison-024 --pair experiments/repro-lgb-binary-graph-024 experiments/lgb-binary-intent-024 --pair experiments/repro-lgb-xendcg-graph-024 experiments/lgb-xendcg-intent-024
```

## Четырнадцатый цикл. Свежесть обучающих лент

Из одного кэша 48 файлов собрали два набора по 45 844 целых ленты и 3 575 775 кандидатов. Свежий набор — 18–19 сентября, два последних дня с полным объёмом данных. Контроль выбран случайно по всему train с точным совпадением числа лент каждой длины. В контроле 7 111 лент из свежего набора, поскольку случайный отбор допускает все даты. Доля положительных отличается вместе с составом данных: 81 208 в свежем и 78 955 в случайном. Для третьей ветки использовали те же случайные ленты с уменьшением веса вдвое за каждые четыре дня. Средний вес кандидата равен единице, эффективный размер выборки по весам — 62.8%. [Состав выборок](../experiments/time-train-048/cohorts.json) и [аудит весов и парных разниц](../experiments/time-train-048/paired_audit.json).

Все три ветки используют бинарный LightGBM, 79 прежних признаков, одинаковые гиперпараметры и бюджет 160 деревьев. Количество деревьев выбрано по раннему val. Итог проверен на прежних 8 657 поздних лентах.

| Train | Выбранные деревья | Ранний NDCG | Поздний NDCG | Поздний NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| [Случайный, выровненный по длинам](../experiments/lgb-random-matched-048/metrics.json) | 160 | 0.368074 | **0.364715** | 0.235500 |
| [Только 18–19 сентября](../experiments/lgb-recent-048/metrics.json) | 120 | **0.371535** | 0.364513 | **0.235839** |
| [Случайный с перевзвешиванием по дате](../experiments/lgb-recency-weighted-048/metrics.json) | 80 | 0.369229 | 0.362613 | 0.232564 |

Свежий train прибавил `+0.003461` на раннем val и дал `−0.000202` на позднем против случайного контроля. Поздний интервал по парному бутстрэпу `[−0.002918, +0.002516]`, временные половины `−0.000650` и `+0.000244`. Перевзвешивание дало `−0.002102` на позднем блоке и проиграло в обеих его половинах. Один доменный классификатор на признаках кандидатов чуть хуже отделяет свежий train от раннего val, чем случайный train (`0.841` против `0.851` AUC). Сближение с ранним val по этим признакам не гарантирует прироста на следующих датах. [Диагностика домена](../experiments/time-train-048/covariate_audit.json).

У 36 вариантов смеси с текущим ансамблем веса выбирались по двум ранним блокам. Ни одна ненулевая поправка не улучшила оба блока, выбран нулевой вес. Точное сравнение воспроизводит `0.372957` NDCG и 4 864 положительных в топ-10 на позднем val. Лучший подтверждённый Public — `0.3652139518`. [Сравнение смесей](../experiments/time-train-comparison-048/assessment.json).

Следующая проверка при фиксированном бюджете кандидатов — увеличить число разных обучающих лент, оставляя все положительные и отбирая ограниченное число отрицательных. Подсравнение со случайным отбором отрицательных покажет вклад трудных примеров. Полный бинарный контроль на 48 файлах и 23.1 млн строк ранее давал поздний NDCG `0.368395`; его преимущество над выборкой 3.58 млн строк может зависеть и от числа лент, и от числа кандидатов на ленту.

```powershell
uv run @rankingPackages python experiments/time_train.py prepare experiments/cache-intent-048/train.npz experiments/repro-time-train-048
uv run @rankingPackages python experiments/time_train.py weights experiments/repro-time-train-048/random.npz experiments/repro-time-train-048/random-recency-weights.npz --half-life 4
uv run @rankingPackages python experiments/ranking.py train experiments/cache-intent-024 experiments/repro-lgb-recent-048 history-intent --objective binary --rounds 160 --threads 5 --train-file experiments/repro-time-train-048/recent.npz
uv run @rankingPackages python experiments/ranking.py train experiments/cache-intent-024 experiments/repro-lgb-random-matched-048 history-intent --objective binary --rounds 160 --threads 5 --train-file experiments/repro-time-train-048/random.npz
uv run @rankingPackages python experiments/ranking.py train experiments/cache-intent-024 experiments/repro-lgb-recency-weighted-048 history-intent --objective binary --rounds 160 --threads 5 --train-file experiments/repro-time-train-048/random.npz --sample-weights experiments/repro-time-train-048/random-recency-weights.npz
uv run @rankingPackages python experiments/compare_graph_rankers.py experiments/cache-intent-024 experiments/nn-calibration-joint-024/val_predictions.parquet experiments/repro-time-train-comparison-048 --pair experiments/repro-lgb-recent-048 experiments/repro-lgb-random-matched-048 --pair experiments/repro-lgb-recency-weighted-048 experiments/repro-lgb-random-matched-048 --variant-label temporal
```

## Пятнадцатый цикл. Больше лент при фиксированном бюджете строк

**Протокол до просмотра позднего val.** Используем все 295 376 лент из train-кэша на 48 файлах и сохраняем все положительные. Каждой ленте с отрицательными выделяем один отрицательный пример. Остаток распределяем пропорционально свободным местам методом наибольших остатков до общего бюджета 3 575 775 строк. Равные остатки разрешаем по случайному приоритету. Seed `2026`.

Для обеих веток используем одни ленты, квоты и случайные приоритеты отрицательных. В случайной ветке берём кандидатов в порядке этих приоритетов. В смешанном отборе `floor(квота / 2)` мест занимают отрицательные с наибольшим средним `user_cos_rank` и `intent_rank_recent_max`. Пропуск ранга заменяем на `0.5` только при отборе. При равном среднем используем случайный приоритет. Остальные места заполняем в порядке случайных приоритетов среди ещё свободных отрицательных. Обе модели получают исходные 79 признаков полной ленты и обучаются тем же бинарным LightGBM, что и контроль из целых лент, с бюджетом 160 деревьев, пятью потоками и выбором числа деревьев по раннему val.

На двух ранних блоках подбираем вес каждой новой модели в смеси с отправленным ансамблем. Стандартизируем scores внутри ленты, проверяем веса `0`, `0.025`, `0.05`, `0.10`, `0.20`, `0.35`, `0.50`. Максимизируем меньший из двух ранних приростов полной NDCG, при равенстве выбираем по общей ранней NDCG. Ненулевой вес допускается при положительном приросте в каждой части. При равных оценках смеси порядок уточняет отправленный ансамбль, затем `item_id`. Ненулевой вес, выбранный на раннем val, оправдывает попытку при позднем приросте около `+0.003` и положительном эффекте в обеих половинах. Для меньшего прироста ориентир от `+0.001` с положительной нижней границей парного 95%-го бутстрэп-интервала по лентам и ростом NDCG@10 и числа положительных в топ-10 в обеих половинах.

В каждой новой выборке 295 376 лент, 510 341 положительный и 3 065 434 отрицательных. В контроле из целых лент 45 844 ленты и 78 955 положительных при тех же 3 575 775 строках. Контраст по охвату лент включает сопутствующее изменение доли положительных с `2.21%` до `14.27%`. В сравнении способов отбора положительные, ленты и квоты совпадают. В смешанном отборе по трудности заполнено 1 464 700 мест (`47.8%` отрицательных). Средний ранг отрицательных вырос с `0.4975` до `0.6779`. В обе выборки вошли 1 814 318 общих отрицательных (`59.2%`). Для всех строк проверены `feed`, `counts`, `item`, `target`, `timestamp`, `position`, сохранение положительных и побитовое совпадение всех 79 признаков с исходным кэшем. [Аудит выборок](../experiments/feed-width-048/cohorts.json).

Все бинарные модели в таблице сохранили 160 деревьев по раннему val. Поздний блок содержит 8 657 лент, две его временные половины содержат 4 317 и 4 340 лент.

| Модель | Ранний NDCG | Поздний NDCG | Поздний NDCG@10 | Поздний NDCG@30 | Положительных в позднем топ-10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Контроль из целых лент](../experiments/lgb-random-matched-048/metrics.json) | 0.368074 | 0.364715 | 0.235500 | 0.304280 | 4 700 |
| [Случайное прореживание](../experiments/lgb-width-random-048/metrics.json) | 0.373807 | **0.368203** | 0.239348 | 0.308950 | 4 752 |
| [Трудные и случайные отрицательные](../experiments/lgb-width-hard-048/metrics.json) | 0.333352 | 0.326783 | 0.186369 | 0.255776 | 3 901 |
| [Все 23.1 млн кандидатов, справочно](../experiments/lgb-binary-intent-048/metrics.json) | 0.372035 | 0.368395 | 0.240126 | 0.309488 | 4 782 |
| Отправленный ансамбль с Public `0.3652139518` | 0.380098 | **0.372957** | 0.245700 | 0.315384 | 4 864 |

- При одинаковом бюджете широкий набор со случайным отбором дал `+0.003488` полной NDCG к контролю из целых лент. По поздним половинам `+0.004538` и `+0.002444`. Парный 95%-й бутстрэп-интервал по лентам `[+0.001133, +0.005951]`. В топ-10 добавилось 52 положительных. Разница с бинарной моделью на полном кэше составила `−0.000192` NDCG при использовании `15.47%` её строк.
- Замена части случайных отрицательных трудными на тех же лентах дала `−0.041420` относительно случайного прореживания. По поздним половинам `−0.039378` и `−0.043451`, парный 95%-й интервал `[−0.046129, −0.036766]`. В топ-10 стало на 851 положительный пример меньше.
- По двум ранним блокам для обеих смесей выбран вес `0`. У случайной ветки все ненулевые веса проиграли в первом блоке, у трудной ветки в обоих. Итоговые смеси воспроизвели NDCG `0.372957` и 4 864 положительных в топ-10. Критерий для новой попытки Public не выполнен. [Метрики, парные разницы и интервалы](../experiments/feed-width-comparison-048/assessment.json) и [ранний поиск весов](../experiments/feed-width-comparison-048/early_search.json).

Команды PowerShell из корня проекта. Первый отбор выполнен на Python 3.12 с NumPy `1.26.4`. Повторный отбор на Python `3.11.14` воспроизвёл SHA-256 обеих выборок и отчёта. Обучение и оценка выполнены на Python `3.11.14`, с пакетами из `$rankingPackages` и SciPy `1.17.1`. Локальный набор из 46 тестов прошёл.

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
uv run @rankingPackages --with scipy==1.17.1 python experiments/feed_width.py prepare experiments/cache-intent-048/train.npz experiments/time-train-048/random.npz experiments/repro-feed-width-048 --budget 3575775 --seed 2026
uv run @rankingPackages --with scipy==1.17.1 python experiments/ranking.py train experiments/cache-intent-024 experiments/repro-lgb-width-random-048 history-intent --objective binary --rounds 160 --threads 5 --train-file experiments/repro-feed-width-048/random.npz
uv run @rankingPackages --with scipy==1.17.1 python experiments/ranking.py train experiments/cache-intent-024 experiments/repro-lgb-width-hard-048 history-intent --objective binary --rounds 160 --threads 5 --train-file experiments/repro-feed-width-048/hard.npz
uv run @rankingPackages --with scipy==1.17.1 python experiments/feed_width.py assess experiments/cache-intent-024 experiments/nn-calibration-joint-024/val_predictions.parquet experiments/repro-feed-width-048 experiments/repro-feed-width-comparison-048 --random experiments/repro-lgb-width-random-048 --hard experiments/repro-lgb-width-hard-048 --whole-feed experiments/lgb-random-matched-048 --full experiments/lgb-binary-intent-048
```

## Шестнадцатый цикл. Поправка на вероятность отбора отрицательных

**Протокол до просмотра результатов.** Используем те же 3 575 775 строк из 295 376 лент случайной ветки пятнадцатого цикла. Контроль учился с единичными весами. В двух новых моделях положительные имеют исходный вес `1`. В первой вес каждого отрицательного равен отношению числа отрицательных в полном кэше к числу отобранных. Во второй вес отрицательного равен исходному числу отрицательных в его ленте, делённому на квоту этой ленты. Оба варианта восстанавливают общую долю положительных в полном кэше. Второй также восстанавливает вес отрицательных каждой ленты в сумме. После вычисления нормируем средний вес к единице в обоих вариантах. Сравнение с единичными весами измеряет эффект коррекции, а сравнение весов по лентам с общим весом показывает эффект распределения массы между лентами при одинаковой доле классов.

Фиксируем тот же бинарный LightGBM, 79 признаков, 160 деревьев, пять потоков и seed `2026`. Число деревьев выбираем на раннем val. Для смеси с отправленным ансамблем проверяем полные стандартизованные scores новых моделей и их разницу со стандартизованными scores контроля, отдельно и со средним двух веток. Сетка весов `0`, `0.025`, `0.05`, `0.10`, `0.20`, `0.35`, `0.50`. Максимизируем меньший прирост по двум ранним блокам, при равенстве берём больший общий ранний прирост. Ненулевой вес требует выигрыша в каждом раннем блоке. Для итоговой проверки используем поздний блок, две его временные половины, NDCG@10, NDCG@30 и число положительных в топ-10. Критерий попытки Public берём из пятнадцатого цикла.

Доля положительных по весам в обоих вариантах `2.2082%` вместо исходных для выборки `14.2722%`. Общий множитель для отрицательных до нормировки `7.37295`, множитель по лентам лежит в `[1.00, 7.94595]`. Максимальное отклонение восстановленной массы отрицательных от исходного числа в отдельной ленте составило `21.20` для общего множителя и `0.0000141` для весов по лентам. Эффективный размер выборки по весам `89.37%` и `88.28%` соответственно. Проверены ключи каждой строки весов, сохранение положительных и массы отрицательных в каждой ленте. [Аудит весов](../experiments/feed-width-weights-048/weight_audit.json).

Во всех трёх режимах выбрано 160 деревьев. Метрики посчитаны на полных лентах val.

| Вариант | Ранний NDCG | Поздний NDCG | Поздний NDCG@10 | Поздний NDCG@30 | Положительных в позднем топ-10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Случайное прореживание, единичные веса](../experiments/lgb-width-random-048/metrics.json) | 0.373807 | **0.368203** | 0.239348 | 0.308950 | 4 752 |
| [Общий вес отрицательных](../experiments/lgb-width-global-weight-048/metrics.json) | 0.374057 | 0.367136 | 0.238130 | 0.308038 | 4 749 |
| [Вес отрицательных по лентам](../experiments/lgb-width-feed-weight-048/metrics.json) | 0.373039 | 0.366855 | 0.238428 | 0.307483 | 4 765 |
| Смесь, выбранная на раннем val | 0.380098 | **0.372957** | 0.245700 | 0.315384 | 4 864 |

- Общий вес дал `−0.001067` полной NDCG к единичным весам. По поздним половинам `−0.002569` и `+0.000427`, парный 95%-й бутстрэп-интервал `[−0.002894, +0.000716]`.
- Вес по лентам дал `−0.001348` к единичным весам. По половинам `−0.000649` и `−0.002043`, интервал `[−0.003322, +0.000613]`. В топ-10 оказалось на 13 положительных больше, NDCG@10 снизился на `0.000920`. В прямом сравнении с общим весом поздняя разница `−0.000281`, в половинах `+0.001920` и `−0.002470`. [Метрики и парные сравнения](../experiments/feed-width-weights-comparison-048/diagnostics.json).
- Ранний поиск по 36 ненулевым конфигурациям выбрал вес `0`. У лучшей ненулевой поправки с весом `0.025` ранние приросты по блокам `+0.00004072` и `−0.00000039`. Поздний прирост выбранной смеси `0` относительно ансамбля с Public `0.3652139518`. Критерий сабмита не достигнут. [Сетка ранних весов](../experiments/feed-width-weights-comparison-048/early_search.json) и [выбор смеси](../experiments/feed-width-weights-comparison-048/assessment.json).

Повторная генерация воспроизвела SHA-256 двух файлов весов и отчёта. Локальный набор из 46 тестов прошёл.

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
uv run @rankingPackages --with scipy==1.17.1 python experiments/feed_width_weights.py prepare experiments/cache-intent-048/train.npz experiments/feed-width-048/random.npz experiments/repro-feed-width-weights-048
uv run @rankingPackages --with scipy==1.17.1 python experiments/ranking.py train experiments/cache-intent-024 experiments/repro-lgb-width-global-weight-048 history-intent --objective binary --rounds 160 --threads 5 --train-file experiments/feed-width-048/random.npz --sample-weights experiments/repro-feed-width-weights-048/global.npz
uv run @rankingPackages --with scipy==1.17.1 python experiments/ranking.py train experiments/cache-intent-024 experiments/repro-lgb-width-feed-weight-048 history-intent --objective binary --rounds 160 --threads 5 --train-file experiments/feed-width-048/random.npz --sample-weights experiments/repro-feed-width-weights-048/per_feed.npz
uv run @rankingPackages --with scipy==1.17.1 python experiments/feed_width_weights.py assess experiments/cache-intent-024 experiments/nn-calibration-joint-024/val_predictions.parquet experiments/repro-feed-width-weights-048 experiments/lgb-width-random-048 experiments/repro-lgb-width-global-weight-048 experiments/repro-lgb-width-feed-weight-048 experiments/repro-feed-width-weights-comparison-048
```

## Семнадцатый цикл. Целевая функция на широком train

**Протокол до просмотра результатов.** На тех же 295 376 лентах и 3 575 775 строках случайной выборки из пятнадцатого цикла сравниваем бинарный LightGBM, `rank_xendcg` и `lambdarank`. Оба новых запуска используют исходные 79 признаков, те же гиперпараметры деревьев, 160 раундов, пять потоков и seed `2026`. Для LambdaMART используем стандартное для проекта ограничение `lambdarank_truncation_level=40`. Позиция показа и веса строк не используются. Ранний val выбирает число деревьев из `40/80/120/160`. Две готовые модели на полном кэше из 23.1 млн строк служат справкой о разнице между бинарной задачей и XE-NDCG на целых лентах.

Проверяем на полных лентах val, в том числе разницу относительно бинарного контроля в двух поздних половинах и по длине ленты. Для смеси с отправленным ансамблем сравниваем полные стандартизованные scores и разницу со стандартизованными scores бинарного контроля, по одной ветке или со средним двух веток. Используем те же 36 ненулевых комбинаций и критерий выбора по двум ранним блокам, что в шестнадцатом цикле. Решение о подготовке сабмита принимаем по критерию пятнадцатого цикла.

На прореженном train медиана размера ленты `8`, 90-й перцентиль `29`. На полном кэше это `48` и `212`. Целевые функции ранжирования обучались на прореженных группах, а NDCG считали на полных группах val. В сравнении на одних строках сверены шарды, бюджет, seed, веса и настройки деревьев. [Аудит общего train](../experiments/feed-width-048/cohorts.json).

| Вариант | Деревьев | Ранний NDCG | Поздний NDCG | Поздний NDCG@10 | Поздний NDCG@30 | Положительных в позднем топ-10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Бинарный контроль](../experiments/lgb-width-random-048/metrics.json) | 160 | 0.373807 | **0.368203** | 0.239348 | 0.308950 | 4 752 |
| [XE-NDCG](../experiments/lgb-width-xendcg-048/metrics.json) | 160 | 0.372621 | 0.367108 | 0.237922 | 0.308397 | 4 740 |
| [LambdaMART](../experiments/lgb-width-lambdarank-048/metrics.json) | 80 | 0.374556 | 0.368112 | **0.240044** | **0.309151** | **4 797** |
| Отправленный ансамбль с Public `0.3652139518` | — | 0.380098 | **0.372957** | 0.245700 | 0.315384 | 4 864 |
| Выбранная смесь | — | 0.380215 | 0.372945 | 0.245671 | 0.315412 | 4 862 |

- XE-NDCG дал `−0.001095` к бинарному контролю. По поздним половинам `−0.002143` и `−0.000052`, парный 95%-й бутстрэп-интервал `[−0.003321, +0.001223]`. На полном кэше XE-NDCG давал `0.364272` против `0.368395` у бинарной модели. При прореживании этот разрыв сузился с `0.004122` до `0.001095`. [Полный XE-NDCG](../experiments/lgb-xendcg-intent-048/metrics.json) и [полный бинарный контроль](../experiments/lgb-binary-intent-048/metrics.json).
- LambdaMART дал `−0.000091` полной NDCG, `+0.000696` NDCG@10 и 45 дополнительных положительных в топ-10 относительно бинарного контроля. По поздним половинам полный NDCG изменился на `−0.001393` и `+0.001204`, интервал `[−0.002610, +0.002358]`. По длине 5–30, 31–100 и 101–300 прирост составил `−0.002626`, `+0.001259` и `+0.000522` соответственно. [Сравнение функций потерь](../experiments/feed-width-objective-comparison-048/assessment.json).
- Ранний поиск выбрал `2.5%` разницы между средним двух стандартизованных ранжировщиков и стандартизованным бинарным контролем. Прирост по ранним блокам `+0.000124` и `+0.000104`. На позднем val эффект `−0.000012`, по половинам `+0.0000025` и `−0.0000269`. Парный 95%-й интервал `[−0.000118, +0.000101]`. Критерий подготовки сабмита не выполнен. [Ранний поиск](../experiments/feed-width-objective-comparison-048/early_search.json) и [итог смеси](../experiments/feed-width-objective-comparison-048/comparison.json).

Прошли 49 локальных тестов, включая проверки совпадения параметров и состава обучающей выборки при сравнении целей.

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
uv run @rankingPackages --with scipy==1.17.1 python experiments/ranking.py train experiments/cache-intent-024 experiments/repro-lgb-width-xendcg-048 history-intent --objective rank_xendcg --rounds 160 --threads 5 --train-file experiments/feed-width-048/random.npz
uv run @rankingPackages --with scipy==1.17.1 python experiments/ranking.py train experiments/cache-intent-024 experiments/repro-lgb-width-lambdarank-048 history-intent --objective lambdarank --rounds 160 --threads 5 --train-file experiments/feed-width-048/random.npz
uv run @rankingPackages --with scipy==1.17.1 python experiments/compare_graph_rankers.py experiments/cache-intent-024 experiments/nn-calibration-joint-024/val_predictions.parquet experiments/repro-feed-width-objective-comparison-048 --pair experiments/repro-lgb-width-xendcg-048 experiments/lgb-width-random-048 --pair experiments/repro-lgb-width-lambdarank-048 experiments/lgb-width-random-048 --variant-label objective
```

## Восемнадцатый цикл. Отклики того же пользователя из полного train

**Протокол до просмотра результатов.** Читаем все 512 train-файлов. По `user_id` собираем более ранние ленты пользователей, встречающихся в val или test. Для этих пользователей используем все их train-показы и положительные отклики. В val и test профиль использует только train, и для каждой оцениваемой ленты проверяется, что время всех исходных показов строго меньше. Метки val нужны только для выбора малой поправки на 3 749 и 2 017 ранних лентах и итоговой оценки на 8 657 поздних.

Проверяем три сигнала: максимум косинуса кандидата к эмбеддингам положительных train-объявлений пользователя, тот же максимум после исключения товаров из доступной истории текущей ленты и сглаженный подъём категории. Для категории считаем долю среди train-показов пользователя с одним псевдонаблюдением на категорию. Долю среди его положительных сглаживаем тремя наблюдениями из этого распределения. Сигнал равен логарифму отношения этих двух долей. Диагностируем долю пользователей с профилем, число дополнительных положительных товаров и покрытие точными ID без использования точного совпадения как отдельной ветки.

Стандартизируем каждый доступный сигнал внутри полной ленты. Отсутствующий сигнал даёт нулевую поправку. Сравниваем три сигнала и среднее доступных сигналов с весами `0.025/0.05/0.10/0.20/0.35/0.50` поверх стандартизованных оценок отправленного ансамбля. Выбираем по меньшему приросту на двух ранних блоках, затем по полному раннему приросту. Для ненулевой поправки требуем положительный эффект на каждом раннем блоке. На позднем val сравниваем полный NDCG, NDCG@10, NDCG@30, топ-10 и эффект по двум временным половинам. Критерий подготовки Parquet берём из пятнадцатого цикла.

Перед итоговым запуском исключили `user_id=0` из персональных профилей. Первый проход обнаружил у этого кода 123 904 train-ленты и 224 974 положительных, при максимуме 103 положительных у остальных ID. Для персональной привязки считаем этот код отсутствующим идентификатором. Он встречается в 658 лентах val и 1 179 лентах test.

При итоговом проходе прочитано 3 151 208 лент и 246 745 836 кандидатов из 512 файлов. Для 12 577 персональных профилей использовано 20 374 train-ленты совпавших пользователей, 1 674 024 кандидата и 42 341 положительный отклик. Эмбеддинг доступен для каждого из них. Профиль есть у 4 386 из 14 423 val-лент (30.41%) и у 8 328 из 28 595 test-лент (29.12%). [Аудит профилей](../experiments/user-response-profile-fit/profile_audit.json).

Сверены ключи, порядок кандидатов и время по всем 14 423 val-лентам. Все профили строго предшествуют соответствующей ленте. На 8 275 парах одного и того же объявления в train и истории val средний косинус эмбеддингов `0.999110`. Для 2 595 из 8 657 поздних лент нашёлся профиль, у 901 ленты оказался доступен хотя бы один положительный train-товар вне текущей истории. Среди поздних кандидатов было 10 точных повторов предыдущего положительного товара пользователя, у всех десяти метка `0`.

Среди 24 ненулевых вариантов ни один не улучшил оба ранних блока. Наибольший минимум по блокам у веса `0.025` для сходства с положительными товарами вне истории. Ранние изменения NDCG `−0.00007773` и `−0.00002115`. С выбранным нулевым весом поздний NDCG составил `0.372957`, NDCG@10 `0.245700`, NDCG@30 `0.315384`, положительных в топ-10 `4 864`. Прирост в обеих поздних половинах `0`. Критерий подготовки сабмита не выполнен. [Оценка и покрытие](../experiments/user-response-profile-val/assessment.json) и [поиск на раннем val](../experiments/user-response-profile-val/early_search.json).

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
uv run @rankingPackages --with scipy==1.17.1 python experiments/user_response_profiles.py fit experiments/repro-user-response-profile-fit
uv run @rankingPackages --with scipy==1.17.1 python experiments/user_response_profiles.py select experiments/cache-intent-024 experiments/repro-user-response-profile-fit/profiles.npz experiments/nn-calibration-joint-024/val_predictions.parquet experiments/repro-user-response-profile-val
```

## Девятнадцатый цикл. Отклик объявления сверх ожидаемого на позиции показа

**Протокол до просмотра результатов.** Из всех 512 train-файлов оцениваем ожидаемую долю положительных откликов. Сравниваем общий фон, фон по исходной позиции и фон по позиции, длине ленты и категории кандидата. Позиции разбиваем границами `1/2/3/5/8/13/21/34/55/89/144/233`, длины — `30/100`. Для полного фона сглаживаем ячейку `позиция × длина × категория` к фону `позиция × длина` на 200 показов, фон `позиция × длина` к позиции на 500, позицию к общей частоте на 500. Файлы по очереди относим к восьми фолдам. Ожидаемую частоту для каждого train-показа считаем без его фолда. Время всех использованных train-показов должно предшествовать val и test.

Для каждого объявления из val/test суммируем фактические положительные отклики `O` и ожидаемое число `E` по всем показам объявления в train. Поправка для кандидата — `log((O + k) / (E + k))`, ограниченная `±3`. Объявления без train-показов получают ноль. Позиция из train используется только в ожидаемом числе. Сравниваем три фона, сглаживание `k = 1/3/10` и веса `0.025/0.05/0.10/0.20/0.35/0.50` поверх стандартизованных внутри ленты scores отправленного ансамбля. Всего 54 ненулевых сочетания. Для ненулевого веса требуется выигрыш в обоих ранних блоках, выбираем по меньшему из двух приростов и затем по общей ранней NDCG. Отдельно показываем раннего победителя каждого фона на позднем блоке как диагностику. Для отправки применяем критерий пятнадцатого цикла к смеси, выбранной только по ранним блокам.

Для оценки фона использовано 246 745 836 показов и 5 446 872 положительных из всех 512 файлов. На 3 401 538 уникальных товаров из val/test пришлось 45 393 591 train-показ и 955 780 положительных. История показов покрывает 62.77% кандидатов позднего val, 67.41% его положительных кандидатов и 53.82% кандидатов test. Последний train-показ предшествует первому val. [Аудит ожидаемых частот и фолдов](../experiments/position-item-priors-fit/fit.json).

| Фон раннего победителя | Поздний NDCG | Δ NDCG | Δ по поздним половинам | Δ положительных в топ-10 |
| --- | ---: | ---: | ---: | ---: |
| Общая частота, диагностика | 0.373296 | +0.000339 | +0.000220 / +0.000457 | −3 |
| Позиция, диагностика | 0.373288 | +0.000331 | +0.000432 / +0.000231 | −1 |
| Позиция, длина и категория, выбранный вариант | **0.373285** | **+0.000328** | **+0.000427 / +0.000229** | **+4** |

У выбранного сигнала `k=10`, вес `0.20`. Ранние приросты `+0.000283` и `+0.000279`. Поздний NDCG@10 вырос на `+0.000364`, NDCG@30 на `+0.000337`. Парный 95%-й бутстрэп-интервал полной NDCG `[−0.000160, +0.000845]`. Общая частота дала сопоставимый поздний прирост, поэтому преимущество самой поправки на позицию пока не подтверждено. Критерий подготовки сабмита не выполнен. [Оценка и покрытие](../experiments/position-item-priors-val/assessment.json) и [ранний поиск](../experiments/position-item-priors-val/early_search.json).

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
uv run @rankingPackages --with scipy==1.17.1 python experiments/position_item_priors.py fit experiments/cache-intent-024 experiments/repro-position-item-priors-fit
uv run @rankingPackages --with scipy==1.17.1 python experiments/position_item_priors.py select experiments/cache-intent-024 experiments/repro-position-item-priors-fit/items.npz experiments/nn-calibration-joint-024/val_predictions.parquet experiments/repro-position-item-priors-val
```

Для дополнительной попытки Public подготовили вариант, зафиксированный на раннем val. Применили поправку `position_length_category` с `k=10` и весом `0.20` к стандартизованным внутри ленты scores из `nn-calibration-joint-024/submission-reserve.parquet`. Ничьи разрешили по исходному score и `item_id`. Перед расчётом test воспроизвели все 1 389 971 сохранённых оценок val. `ranking.py audit` подтвердил все 2 904 806 пары шаблона, конечные scores и отсутствие ничьих. [Parquet для загрузки](../experiments/position-item-priors-val/submission-position-priors.parquet) и [аудит](../experiments/position-item-priors-val/submission_audit.json).

Пользователь загрузил этот файл и сообщил **Public NDCG `0.3649194661`**. Разница с `nn-calibration-joint-024` составила `−0.0002944857` на Public при `+0.0003281490` на позднем val. Для следующих экспериментов основной контроль — точные прогнозы `nn-calibration-joint-024`, а этот запуск показывает, что согласованного выигрыша порядка `0.0003` в двух локальных половинах оказалось недостаточно.

## Двадцатый цикл. История скорректированного отклика как признаки LightGBM

**Протокол до просмотра результатов.** Добавляем три признака: `log1p(E)` и `log((O + k)/(E + k))` с ограничением `±3` для `k=3` и `k=10`. Ожидаемую вероятность отклика на исторический показ оцениваем по дню этого показа и более ранним дням, исключая его фолд файлов. Для каждой train-ленты суммируем ожидаемые и фактические отклики объявления только по полностью завершившимся ранее московским дням. Для val и test используем весь train. Если показов раньше не было, все три признака равны нулю.

Сравниваем новую ветку с `lgb-binary-intent-024` на 24 целых файлах, 147 684 лентах и 11 566 844 кандидатах. Это готовый контроль с теми же 79 базовыми признаками и совпадающими ключами. Три статистических признака собираются из всех 512 файлов train. Оба запуска используют бинарный LightGBM, те же параметры, 160 раундов и шесть потоков. Сравниваем самостоятельные модели и два способа добавить сигнал к отправленному ансамблю, его полные стандартизованные scores и разницу с контролем. Веса и критерий Public те же, что в шестнадцатом цикле.

Для 9 536 549 товаров из кэшей train/val/test использовано 153 148 632 показа из полного train. Статистика доступна для 68.33% строк train, 65.43% val и 53.82% test. Все 1 105 615 строк первого дня train получили нулевые признаки. На всех строках проверены `feed`, `item`, `counts` и `timestamp` после записи. [Аудит временных префиксов](../experiments/position-prior-features-024/prefix_audit.json).

| Вариант | Деревьев | Ранний NDCG | Поздний NDCG | Поздний NDCG@10 | Поздний NDCG@30 | Положительных в позднем топ-10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Контроль, 79 признаков](../experiments/lgb-binary-intent-024/metrics.json) | 160 | 0.371735 | **0.367480** | 0.239934 | 0.308053 | 4 789 |
| [Ветка с 82 признаками](../experiments/lgb-binary-position-priors-024/metrics.json) | 120 | 0.372448 | 0.365923 | 0.236875 | 0.305763 | 4 717 |
| Отправленный ансамбль | — | 0.380098 | 0.372957 | 0.245700 | 0.315384 | 4 864 |
| Выбранная смесь | — | 0.380152 | 0.372963 | 0.245745 | 0.315362 | 4 866 |

У новой модели позднее изменение полной NDCG относительно бинарного контроля `−0.001558`, в половинах `−0.000586` и `−0.002525`. На раннем val выбрано `2.5%` полных стандартизованных scores новой модели. На позднем val смесь дала `+0.0000054`, по половинам `+0.0000444` и `−0.0000334`, парный 95%-й интервал `[−0.0000522, +0.0000763]`. Критерий подготовки сабмита не выполнен. [Сравнение веток и смеси](../experiments/position-prior-comparison-024/assessment.json) и [ранний поиск](../experiments/position-prior-comparison-024/early_search.json).

Для прямой поправки сверены расчёт ожидаемой частоты вне фолда, нейтральность для новых товаров и оценки всех кандидатов val. Для признаков LightGBM подтверждены нули на первом дне, совпадение масок покрытия в val/test с независимым полным подсчётом и границы завершённых дней. Воспроизведены оценки итоговой смеси на всех кандидатах val. Прошли все 49 локальных тестов.

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
uv run @rankingPackages --with scipy==1.17.1 python experiments/position_prior_features.py experiments/cache-intent-024 experiments/repro-position-prior-features-024
uv run @rankingPackages --with scipy==1.17.1 python experiments/ranking.py train experiments/cache-intent-024 experiments/repro-lgb-binary-position-priors-024 history-intent --objective binary --rounds 160 --threads 6 --extra-features experiments/repro-position-prior-features-024
uv run @rankingPackages --with scipy==1.17.1 python experiments/compare_graph_rankers.py experiments/cache-intent-024 experiments/nn-calibration-joint-024/val_predictions.parquet experiments/repro-position-prior-comparison-024 --pair experiments/repro-lgb-binary-position-priors-024 experiments/lgb-binary-intent-024 --variant-label position-prior
```

## Двадцать первый цикл. Предобучение GRU на следующем товаре

**Протокол до оценки позднего val.** В каждой train-истории выбираем одну допустимую цель равновероятно среди первых четырёх событий. У цели должны быть положительный ID, эмбеддинг и хотя бы одно более раннее событие. После цели в исходном порядке берём до 12 более ранних событий, разворачиваем их в хронологический порядок и добавляем паддинг справа. При пропуске эмбеддинга у события сохраняется тип, а вектор заполнен нулями. В батчевом softmax одинаковые ID исключаются из отрицательных. Повтор товара в прошлом разрешён. Векторы товаров из `data.memmap` проецируются из 128 в 32 признака и нормируются. GRU с 32 скрытыми признаками получает эту проекцию и восьмимерный вектор типа события. Логит следующего товара равен косинусу состояния и проекции товара, умноженному на 10. Два train-шарда отложены для диагностики предобучения. Предобучение использует только train.

К прежнему `FeedRanker` ширины 48 с 79 признаками и веткой векторов добавляем косинус состояния GRU по 12 последним событиям и проекции кандидата с обучаемым коэффициентом. В контрольной ветке GRU и проекция имеют ту же случайную инициализацию, с которой началось предобучение. Остальные параметры совпадают при старте. Маски dropout и порядок батчей совпадают на соответствующих шагах. Обе ветки используют seed `2026` и одинаковые полные размеченные ленты из кэша на 48 шардах. Дообучаем все параметры шесть эпох по 36 000 разных лент, батч до 96, AdamW с начальным LR `0.0015`, cosine decay до `0.00045` и weight decay `0.03`, прежний listwise loss с бинарным членом `0.25`. Прогнозы берём с шестой эпохи. Исходная позиция показа отсутствует среди входов.

Сравниваем обе модели отдельно, обе их смеси с точными прогнозами `nn-calibration-joint-024` и поправку к ансамблю по разности между ветками. Стандартизация внутри ленты, веса `0/0.025/0.05/0.10/0.20/0.35/0.50`. На двух ранних блоках максимизируем меньший прирост полной NDCG, при равенстве выбираем по общему раннему приросту. Решение о Parquet принимаем для выбранной по раннему val смеси по критерию пятнадцатого цикла. Поздняя оценка включает обе временные половины, парный бутстрэп и ленты на 101–300 кандидатов.


Пилот на четырёх шардах дал 24 231 пример. На двух отложенных train-шардах Recall@1 среди целей батча вырос с `0.002308` до `0.436614`, cross-entropy снизилась с `7.17480` до `3.14976`. Холодное чтение эмбеддингов с диска ограничило скорость пилота до 1 626 примеров/с. Поэтому для основного прогона таблица из 10 439 416 832 байт один раз прочитана в память за 26,38 с. Истории читались батчами по 384 ленты.

Основной проход обработал 3 138 896 лент из 510 шардов и использовал 3 095 400 примеров за 8 489 обновлений AdamW с LR `0.002` и weight decay `0.01`. Проход занял 200,48 с или 15 440 примеров/с. Отложенные два шарда содержат 12 132 примера из 12 312 лент. На них Recall@1 вырос до `0.659413`, cross-entropy снизилась до `1.90404`. В 28,62% отложенных контекстов уже встречается индекс целевого товара. Recall@1 здесь измеряет вспомогательную задачу внутри батча. [Пилот](../experiments/sequence-smoke-004/pretrain.json) и [аудит прохода](../experiments/sequence-pretrain-510/pretrain.json).


Для каждой ветки дообучение использовало 216 000 целых лент, 16 903 568 кандидатов, 373 607 положительных и 2 277 одинаковых по составу батчей. Шесть эпох обеих моделей заняли 208,43 с. Для кэша выровнены истории всех 295 376 train-лент и всех 14 423 val-лент. На 1 280 исходных историях сверены ID, индексы, типы событий, хронология и паддинг. Ключи прогнозов отправленного ансамбля совпали по порядку с полными кэшами val и test для всех 1 389 971 и 2 904 806 пар соответственно. Обе ветки и прогнозы сохранены в [sequence-finetune-048](../experiments/sequence-finetune-048), фактические параметры и объёмы по эпохам — в [tune.json](../experiments/sequence-finetune-048/tune.json).

| Модель | Ранний 1 | Ранний 2 | Поздний 1 | Поздний 2 | Поздний NDCG | Полный val |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Отправленный ансамбль | 0.382012 | 0.376541 | 0.367838 | 0.378050 | 0.372957 | 0.375812 |
| GRU со случайной инициализацией | 0.376717 | 0.368833 | 0.362661 | 0.374405 | 0.368549 | 0.370712 |
| GRU с предобучением | 0.377761 | 0.371074 | 0.364167 | 0.376621 | 0.370411 | 0.372414 |
| Ансамбль + 10% контроля, ранний выбор | 0.382156 | 0.376648 | 0.367675 | 0.378384 | 0.373044 | 0.375916 |
| Ансамбль + 10% предобученной ветки, диагностика | 0.381931 | 0.376669 | 0.367660 | 0.378435 | 0.373062 | 0.375872 |

| Модель | Поздний NDCG@10 | Поздний NDCG@30 | Положительных в топ-10 | NDCG@10, 101–300 | Положительных в топ-10, 101–300 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Отправленный ансамбль | 0.245700 | 0.315384 | 4 864 | 0.097326 | 963 |
| GRU со случайной инициализацией | 0.239498 | 0.309309 | 4 755 | 0.095559 | 951 |
| GRU с предобучением | 0.242055 | 0.311915 | 4 778 | 0.095932 | 944 |
| Ансамбль + 10% контроля | 0.245569 | 0.315390 | 4 855 | 0.097225 | 955 |
| Ансамбль + 10% предобученной ветки | 0.245650 | 0.315499 | 4 857 | 0.097395 | 958 |

Полная NDCG отдельной модели выросла на `+0.001862` к случайной инициализации. По поздним половинам `+0.001506` и `+0.002216`, парный 95%-й бутстрэп-интервал `[+0.000510, +0.003267]`. На 3 245 длинных лентах NDCG@10 выросла на `+0.000372`, число положительных в топ-10 снизилось с 951 до 944. Предобученная модель уступила отправленному ансамблю на `0.002546` полной поздней NDCG.

По ранним блокам выбран вес `0.10` для контроля с приростами `+0.000145` и `+0.000107`. Для предобученной ветки и разности веток выбран вес `0`. Выбранная смесь прибавила поздние `+0.000087`, по половинам `−0.000162` и `+0.000334`, 95%-й интервал `[−0.000250, +0.000443]`. После основной оценки для прямого сравнения посчитали смесь предобученной ветки с тем же весом `0.10`, зафиксированным ранним отбором контроля. Её выигрыш к ансамблю `+0.000104`, по половинам `−0.000178` и `+0.000385`, интервал `[−0.000239, +0.000442]`. В длинных лентах получено на пять положительных в топ-10 меньше, чем у ансамбля. Критерий подготовки нового Parquet не выполнен. [Исходный ранний отбор](../experiments/sequence-assessment-048/early_selection.json) и [метрики, парные интервалы и диагностическое сравнение](../experiments/sequence-assessment-048-matched/assessment.json).

Воспроизведение из корня проекта. Использованы доступные локально Python `3.12.0`, PyTorch `2.6.0+cpu`, NumPy `1.26.4`, Polars `2.1.0`, PyArrow `17.0.0`. Предобучение работало на четырёх потоках, дообучение на шести. Прошли 54 локальных теста.

```powershell
$sequencePython = "$env:LOCALAPPDATA/Programs/Python/Python312/python.exe"
$env:OPENBLAS_NUM_THREADS = '1'
& $sequencePython -B -m unittest discover -s experiments
& $sequencePython -B -u experiments/sequence_ranker.py pretrain experiments/repro-sequence-smoke-004 --shards 4
& $sequencePython -B -u experiments/sequence_ranker.py pretrain experiments/repro-sequence-pretrain-510 --stage-table
& $sequencePython -B -u experiments/sequence_ranker.py tune experiments/cache-intent-024 experiments/embedding-cache-024 experiments/cache-intent-048 experiments/embedding-cache-048 experiments/repro-sequence-pretrain-510/encoder.pt experiments/repro-sequence-finetune-048
& $sequencePython -B -u experiments/sequence_ranker.py assess experiments/cache-intent-024 experiments/repro-sequence-finetune-048 experiments/nn-calibration-joint-024/val_predictions.parquet experiments/repro-sequence-assessment-048
```


## Двадцать второй цикл. Предобучение на следующем новом товаре

**Протокол до оценки нового позднего val.** В предыдущем цикле в 28,62% отложенных примеров индекс эмбеддинга цели уже был среди 12 входных событий. В 845 021 кандидатах позднего val точных совпадений ID с опубликованной историей было 0, в том числе среди 13 168 положительных. Проверяем, улучшит ли перенос модель, если убрать точные повторы из целей предобучения.

Оставляем тот же GRU, 12 входных событий, четыре возможные позиции цели, батчи до 384, optimizer, seed `2026` и разбиение train на 510 шардов обучения и два диагностических. Для каждой позиции цели проверяем весь строго более ранний участок опубликованной истории по `item_id`, включая позиции за пределами окна GRU. Более новые события при проверке не учитываем. Ленту без допустимой цели пропускаем. Обе версии энкодера сравниваем на одинаковых новых целях двух диагностических шардов.

Дообучаем в условиях 21-го цикла и повторяем контроль со случайной инициализацией. Для раннего отбора используем прежние сетку весов, стандартизацию, временные блоки и критерий Parquet. Предыдущий энкодер участвует во внешнем парном сравнении после фиксации раннего выбора. Сравниваем оба энкодера отдельно и при общем весе, выбранном для контроля на раннем val. Совпадение выборок проверяем по числу лент, кандидатов, меток и шагов, контрольные прогнозы сравниваем с предыдущим прогоном.


На 5 036 возможных целях из восьми шардов сверили маску с прямой проверкой ID. В выборке нашлось 127 повторов за пределами окна из 12 событий. Пилот на четырёх шардах использовал 22 741 пример и достиг Recall@1 `0.290927` на отложенных шардах. Исходные Parquet позднего val независимо подтвердили нулевое число повторов среди кандидатов.

Полный проход обработал 3 138 896 лент и использовал 2 904 091 пример за 8 489 обновлений. У предыдущей версии было 3 095 400 примеров и столько же обновлений. Новый проход занял 202,52 с или 14 340 примеров/с, после чтения таблицы эмбеддингов в память за 12,33 с. Совпадений с индексом вектора цели во входном окне не встретилось. На одних и тех же 11 429 отложенных новых целях Recall@1 прежнего энкодера `0.546067`, нового `0.571704`. Cross-entropy снизилась с `2.368459` до `2.208805`. [Пилот](../experiments/sequence-novel-smoke-004/pretrain.json) и [полный проход](../experiments/sequence-novel-pretrain-510/pretrain.json).


На дообучении повторились все шесть бюджетов и потери контрольной ветки. На всех 1 389 971 парах val контрольные scores побитово совпали с предыдущим запуском. Каждая ветка использовала 216 000 лент, 16 903 568 кандидатов, 373 607 положительных и 2 277 шагов. Шесть эпох заняли 189,10 с. [Параметры и лог](../experiments/sequence-novel-finetune-048/tune.json).

| Модель | Ранний 1 | Ранний 2 | Поздний 1 | Поздний 2 | Поздний NDCG | Полный val |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Отправленный ансамбль | 0.382012 | 0.376541 | 0.367838 | 0.378050 | 0.372957 | 0.375812 |
| GRU без предобучения | 0.376717 | 0.368833 | 0.362661 | 0.374405 | 0.368549 | 0.370712 |
| GRU с предыдущим предобучением | 0.377761 | 0.371074 | 0.364167 | 0.376621 | 0.370411 | 0.372414 |
| GRU с предобучением на новых целях | 0.377024 | 0.369538 | 0.363557 | 0.376695 | 0.370144 | 0.371847 |
| Ансамбль + 10% контроля, ранний выбор | 0.382156 | 0.376648 | 0.367675 | 0.378384 | 0.373044 | 0.375916 |
| Ансамбль + 10% предыдущей GRU | 0.381931 | 0.376669 | 0.367660 | 0.378435 | 0.373062 | 0.375872 |
| Ансамбль + 10% новой GRU | 0.381576 | 0.377281 | 0.367826 | 0.378399 | 0.373126 | 0.375904 |

| Модель | Поздний NDCG@10 | Поздний NDCG@30 | Положительных в топ-10 | NDCG@10, 101–300 | Положительных в топ-10, 101–300 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Отправленный ансамбль | 0.245700 | 0.315384 | 4 864 | 0.097326 | 963 |
| GRU без предобучения | 0.239498 | 0.309309 | 4 755 | 0.095559 | 951 |
| GRU с предыдущим предобучением | 0.242055 | 0.311915 | 4 778 | 0.095932 | 944 |
| GRU с предобучением на новых целях | 0.242244 | 0.311742 | 4 808 | 0.097112 | 955 |
| Ансамбль + 10% контроля | 0.245569 | 0.315390 | 4 855 | 0.097225 | 955 |
| Ансамбль + 10% предыдущей GRU | 0.245650 | 0.315499 | 4 857 | 0.097395 | 958 |
| Ансамбль + 10% новой GRU | 0.245688 | 0.315603 | 4 856 | 0.097549 | 958 |

Новая модель дала `+0.001595` полной поздней NDCG к случайной инициализации. По половинам `+0.000896` и `+0.002291`, парный 95%-й интервал `[+0.000221, +0.002965]`. К предыдущему предобучению разница составила `−0.000267`, интервал `[−0.001270, +0.000750]`. На длинных лентах относительно него прибавили `+0.001181` NDCG@10 и 11 положительных в топ-10. Проверка выполнена с одним seed. Отбор новых целей изменил распределение и число примеров предобучения при прежнем числе обновлений.

Ранний отбор снова выбрал 10% контроля, с поздними `+0.000087` и интервалом `[−0.000250, +0.000443]`. Для новой модели и разности с контролем выбран нулевой вес. Диагностические 10% новой модели дали `+0.000169` к отправленному ансамблю, по поздним половинам `−0.000012` и `+0.000349`, парный 95%-й интервал `[−0.000204, +0.000532]`. К смеси с предыдущей моделью при том же весе разница `+0.000065`, интервал `[−0.000112, +0.000254]`. Критерий подготовки Parquet не выполнен. [Ранний отбор](../experiments/sequence-novel-assessment-048/early_selection.json) и [полные метрики](../experiments/sequence-novel-assessment-048/assessment.json).

Воспроизведение на том же Python 3.12 и PyTorch 2.6 CPU. Прошли 55 локальных тестов. Прогнозы и модели лежат в [sequence-novel-finetune-048](../experiments/sequence-novel-finetune-048), смеси в [sequence-novel-assessment-048](../experiments/sequence-novel-assessment-048).

```powershell
$sequencePython = "$env:LOCALAPPDATA/Programs/Python/Python312/python.exe"
$env:OPENBLAS_NUM_THREADS = '1'
& $sequencePython -B -m unittest discover -s experiments
& $sequencePython -B -u experiments/sequence_ranker.py pretrain experiments/repro-sequence-novel-smoke-004 --shards 4 --novel-targets
& $sequencePython -B -u experiments/sequence_ranker.py pretrain experiments/repro-sequence-novel-pretrain-510 --stage-table --novel-targets --compare-encoder experiments/sequence-pretrain-510/encoder.pt
& $sequencePython -B -u experiments/sequence_ranker.py tune experiments/cache-intent-024 experiments/embedding-cache-024 experiments/cache-intent-048 experiments/embedding-cache-048 experiments/repro-sequence-novel-pretrain-510/encoder.pt experiments/repro-sequence-novel-finetune-048
& $sequencePython -B -u experiments/sequence_ranker.py assess experiments/cache-intent-024 experiments/repro-sequence-novel-finetune-048 experiments/nn-calibration-joint-024/val_predictions.parquet experiments/repro-sequence-novel-assessment-048 --prior-pretrained experiments/sequence-finetune-048/pretrained/val_predictions.parquet
```


## Двадцать третий цикл. Сравнения на сопоставимой глубине показа

**Протокол до оценки нового позднего val.** Модель видит признаки кандидатов, по которым частично восстанавливается исходный порядок train. Метки train при этом зависят от видимости на разных позициях, а в val/test позиции показа перемешаны равномерно. Проверяем, поможет ли дополнение к listwise loss, которое сравнивает кандидатов внутри одной ленты и одного диапазона исходных позиций.

Границы диапазонов `2/4/8/16/32/64/128/256`, левая граница включается. Для каждого положительного кандидата считаем `−log softmax` среди кандидатов его диапазона. Используем его, если в диапазоне есть отрицательный кандидат. Усредняем по подходящим положительным внутри ленты, затем по всем лентам батча. К прежнему loss добавляем этот член с весом `0.5`. Позиция используется в loss на train. Входы ранжировщика и расчёт val/test совпадают с 21-м циклом.

Берём тот же предобученный энкодер 21-го цикла и шесть эпох на тех же 216 000 полных лентах. Сравниваем с обычным дообучением именно этого энкодера из 21-го цикла. В новом запуске также повторяем ветку со случайной инициализацией для проверки бюджетов и воспроизводимости. Используем прежние ранние блоки и сетку весов смеси с отправленным ансамблем. Оцениваем полную NDCG, NDCG@10, @30, топ-10 на длинных лентах и парные интервалы по поздним лентам. Критерий Parquet берём из 15-го цикла.


В полном кэше train на позициях 0–1 положительных 49 071 из 506 438 (9.69%), а на позициях 128–255 — 50 620 из 5 232 663 (0.97%). В используемой выборке в 210 423 из 216 000 лент есть хотя бы одно подходящее сравнение. В него попадает 359 962 из 373 607 положительных. На 256 исходных лентах из восьми шардов сверили тройки `(item_id, target, pos)` и независимо воспроизвели расчёт нового loss. В двух ветках совпали состав кандидатов, число меток и 2 277 шагов. Обе ветки обучались 208,59 с. [Параметры и журнал эпох](../experiments/sequence-position-finetune-048/tune.json).

| Модель | Ранний 1 | Ранний 2 | Поздний 1 | Поздний 2 | Поздний NDCG | Полный val |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Отправленный ансамбль | 0.382012 | 0.376541 | 0.367838 | 0.378050 | 0.372957 | 0.375812 |
| GRU без предобучения | 0.376717 | 0.368833 | 0.362661 | 0.374405 | 0.368549 | 0.370712 |
| GRU с прежним loss | 0.377761 | 0.371074 | 0.364167 | 0.376621 | 0.370411 | 0.372414 |
| GRU со сравнениями по диапазонам | 0.377242 | 0.370714 | 0.363965 | 0.376866 | 0.370433 | 0.372242 |
| Ансамбль + 10% GRU без предобучения, ранний выбор | 0.382156 | 0.376648 | 0.367675 | 0.378384 | 0.373044 | 0.375916 |
| Ансамбль + 10% GRU с прежним loss | 0.381931 | 0.376669 | 0.367660 | 0.378435 | 0.373062 | 0.375872 |
| Ансамбль + 10% GRU со сравнениями по диапазонам | 0.382251 | 0.376563 | 0.367717 | 0.378602 | 0.373174 | 0.376007 |

| Модель | Поздний NDCG@10 | Поздний NDCG@30 | Положительных в топ-10 | NDCG@10, 101–300 | Положительных в топ-10, 101–300 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Отправленный ансамбль | 0.245700 | 0.315384 | 4 864 | 0.097326 | 963 |
| GRU с прежним loss | 0.242055 | 0.311915 | 4 778 | 0.095932 | 944 |
| GRU со сравнениями по диапазонам | 0.242254 | 0.312057 | 4 781 | 0.096550 | 940 |
| Ансамбль + 10% GRU с прежним loss | 0.245650 | 0.315499 | 4 857 | 0.097395 | 958 |
| Ансамбль + 10% GRU со сравнениями по диапазонам | 0.245923 | 0.315630 | 4 865 | 0.098101 | 964 |

При одинаковом предобучении новая функция потерь дала `+0.000022` полной поздней NDCG. В половинах `−0.000203` и `+0.000245`, парный 95%-й интервал `[−0.000849, +0.000889]`. Прирост нового ранжировщика к ветке без предобучения `+0.001884` включает эффект обоих изменений.

Для новой ветки оба ранних блока допустили вес `0.10`, с приростами `+0.000240` и `+0.000022`. Общий ранний отбор выбрал 10% ветки со случайной инициализацией по большему минимуму двух приростов. Смесь с новой GRU дала `+0.000217` к отправленному ансамблю, по поздним половинам `−0.000120` и `+0.000552`, парный 95%-й интервал `[−0.000136, +0.000551]`. В длинных лентах её прирост NDCG@10 `+0.000775`, в топ-10 на один положительный больше. При одинаковых весах 10% новая смесь прибавила `+0.000112` к смеси с прежней GRU, интервал `[−0.000050, +0.000273]`. [Ранний отбор](../experiments/sequence-position-assessment-048/early_selection.json) и [точные метрики](../experiments/sequence-position-assessment-048/assessment.json).

Отдельно проверили смесь поверх зафиксированных прогнозов поправки из 19-го цикла, ранее подготовленной для Public. При той же сетке весов обе ранние части не поддержали ни одну ненулевую добавку. Выбран нулевой вес, поздний NDCG воспроизвёл `0.373285`. Критерий подготовки нового Parquet не выполнен. [Сравнение с поправкой 19-го цикла](../experiments/sequence-position-with-priors-048/assessment.json).

Воспроизведение на Python 3.12 и PyTorch 2.6 CPU с seed `2026`. Прошли 56 локальных тестов.

```powershell
$sequencePython = "$env:LOCALAPPDATA/Programs/Python/Python312/python.exe"
$env:OPENBLAS_NUM_THREADS = '1'
& $sequencePython -B -m unittest discover -s experiments
& $sequencePython -B -u experiments/sequence_ranker.py tune experiments/cache-intent-024 experiments/embedding-cache-024 experiments/cache-intent-048 experiments/embedding-cache-048 experiments/sequence-pretrain-510/encoder.pt experiments/repro-sequence-position-finetune-048 --position-conditioned
& $sequencePython -B -u experiments/sequence_ranker.py assess experiments/cache-intent-024 experiments/repro-sequence-position-finetune-048 experiments/nn-calibration-joint-024/val_predictions.parquet experiments/repro-sequence-position-assessment-048 --prior-pretrained experiments/sequence-finetune-048/pretrained/val_predictions.parquet
& $sequencePython -B -u experiments/sequence_ranker.py assess experiments/cache-intent-024 experiments/repro-sequence-position-finetune-048 experiments/position-item-priors-val/val_predictions.parquet experiments/repro-sequence-position-with-priors-048 --prior-pretrained experiments/sequence-finetune-048/pretrained/val_predictions.parquet
```


## Двадцать четвёртый цикл. Четыре раза больше разных лент при равном числе обновлений

**Протокол до оценки позднего val.** Проверяем охват train отдельно от числа градиентных шагов и баланса классов. Берём исходные 48 шардов из `cache-intent-048` для контроля и добавляем 144 других в широкую ветку. На каждой ленте сохраняем все положительные и до 12 равновероятно выбранных отрицательных без возвращения. У одинаковой ленты отбор одинаков в обеих ветках. Seed `2026`. Нормализация, архитектура, начальные веса и маски dropout общие.

Ленты сопоставляются по тройке «исходная длина, число отобранных кандидатов, число положительных». В широкой ветке каждую ленту используем один раз. В контроле повторяем перестановки соответствующей группы из 48 шардов. Широкая ветка пропускает редкие группы без пары. На каждом шаге совпадают число лент и кандидатов, число положительных, форма маски и распределение исходной длины. Обе ветки обучаем один проход за четыре равные по числу лент части, батч до 192, AdamW, LR `0.0015 → 0.00045`, weight decay `0.03`, прежний listwise loss с бинарным членом `0.25`. Оцениваем финальные веса.

Используем `FeedRanker` ширины 48 с 29 признаками и проекциями векторов товара, пользователя и недавней истории. Часть признаков, включая ранги и z-score пяти исходных значений и размер ленты, считается по полным кандидатам до отбора. Остальные включают категорию, полноту векторов, косинус с пользователем, совпадение категорий и максимальный косинус с 4/16/48 недавними событиями. Отключаем объединение скрытых состояний кандидатов внутри ленты, чтобы число кандидатов после отбора не меняло контекст при предсказании. В val/test используем все строки. Модели и смеси сравниваем с `nn-calibration-joint-024`, Public `0.3652139518`. Веса и критерий подготовки Parquet прежние.


Пилот на четырёх шардах обработал 24 544 ленты. Подготовка всего набора использовала четыре потока и общую таблицу эмбеддингов в памяти. Независимая сверка обнаружила погрешность накопления `float32` при расчёте z-score в лентах с постоянным признаком. Исправили суммирование на `float64` и пересчитали все шарды и обе модели. Предварительную пару исключили из оценки. После исправления сверены 128 исходных лент, 1 669 отобранных кандидатов, все положительные ключи, числовые ранги, z-score и косинусы истории. В исправленном кэше 192 шарда, 1 181 490 лент и 15 291 684 отобранных кандидата. Подготовка заняла 170,7 с. [Состав и объём кэша](../experiments/wide-rank-data/prepare.json).


Из широкой ветки исключили 3 683 ленты без совпадающей тройки в контроле. Использовано 1 177 807 разных лент из всех 192 шардов против 295 376 из 48. У каждой ветки ровно 1 177 807 посещений лент, 15 183 107 кандидатов, 1 979 784 положительных и 6 136 обновлений. Обучение пары заняло 183,21 с. Исходные ключи и метки val сверены для всех 1 389 971 кандидатов. [Отчёт обучения](../experiments/wide-rank-finetune-192-corrected/tune.json).

| Модель | Ранний 1 | Ранний 2 | Поздний 1 | Поздний 2 | Поздний NDCG | Полный val |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Отправленный ансамбль | 0.382012 | 0.376541 | 0.367838 | 0.378050 | 0.372957 | 0.375812 |
| Контроль, 48 шардов | 0.369993 | 0.369952 | 0.360972 | 0.372298 | 0.366650 | 0.367981 |
| Широкая ветка, 192 шарда | 0.369363 | 0.366159 | 0.361520 | 0.369930 | 0.365736 | 0.366738 |
| Ансамбль + 2.5% контроля | 0.382031 | 0.376650 | 0.367747 | 0.378167 | 0.372971 | 0.375841 |
| Ансамбль + 5% широкой ветки, ранний выбор | 0.382132 | 0.377057 | 0.367720 | 0.378032 | 0.372890 | 0.375875 |
| Ансамбль + 2.5% широкой ветки, общий вес | 0.382032 | 0.376646 | 0.367902 | 0.378052 | 0.372991 | 0.375852 |

| Модель | Поздний NDCG@10 | Поздний NDCG@30 | Положительных в топ-10 | NDCG@10, 101–300 | Положительных в топ-10, 101–300 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Отправленный ансамбль | 0.245700 | 0.315384 | 4 864 | 0.097326 | 963 |
| Контроль, 48 шардов | 0.236908 | 0.306470 | 4 688 | 0.091058 | 909 |
| Широкая ветка, 192 шарда | 0.235776 | 0.305479 | 4 694 | 0.090826 | 915 |
| Ансамбль + 2.5% контроля | 0.245757 | 0.315391 | 4 866 | 0.097479 | 965 |
| Ансамбль + 5% широкой ветки | 0.245754 | 0.315356 | 4 874 | 0.097734 | 971 |
| Ансамбль + 2.5% широкой ветки | 0.245755 | 0.315395 | 4 865 | 0.097543 | 964 |

Широкая ветка получила `−0.000914` полной поздней NDCG к контролю. Разница по половинам `+0.000548` и `−0.002368`. Парный 95%-й бутстрэп-интервал `[−0.003086, +0.001150]`. При этой архитектуре и фиксированном бюджете обновлений выигрыш от охвата не подтвердился.

Ранний отбор выбрал 5% широкой ветки, с приростами `+0.000121` и `+0.000516`. На позднем val смесь уступила отправленному ансамблю на `0.000067` и проиграла в обеих половинах. Парный интервал `[−0.000402, +0.000264]`. В топ-10 стало на 10 положительных больше, из них восемь в лентах на 101–300 кандидатов. Для разности между ветками выбран нулевой вес. Критерий подготовки нового Parquet не выполнен. [Ранний отбор](../experiments/wide-rank-assessment-192/early_selection.json) и [метрики с парными интервалами](../experiments/wide-rank-assessment-192/assessment.json).

Воспроизведение на Python 3.12, PyTorch 2.6 CPU и SciPy 1.16.3. Прошли 60 тестов. После расширения общего кода оценки все значения прежнего отчёта GRU и раннего выбора воспроизвелись точно.

```powershell
$sequencePython = "$env:LOCALAPPDATA/Programs/Python/Python312/python.exe"
$env:OPENBLAS_NUM_THREADS = '1'
& $sequencePython -B -m unittest discover -s experiments
& $sequencePython -B -u experiments/wide_ranker.py prepare experiments/cache-intent-048/train.npz experiments/repro-wide-rank-data --shards 4
& $sequencePython -B -u experiments/wide_ranker.py prepare experiments/cache-intent-048/train.npz experiments/repro-wide-rank-data --shards 192 --workers 4
& $sequencePython -B -u experiments/wide_ranker.py tune experiments/repro-wide-rank-data experiments/cache-intent-024 experiments/repro-wide-rank-finetune-192 --threads 6
& $sequencePython -B -u experiments/wide_ranker.py assess experiments/cache-intent-024 experiments/repro-wide-rank-finetune-192 experiments/nn-calibration-joint-024/val_predictions.parquet experiments/repro-wide-rank-assessment-192 --test-anchor experiments/nn-calibration-joint-024/submission-reserve.parquet
```


## Двадцать пятый цикл. Нелинейное переранжирование на случайных показах

**Протокол до оценки позднего val.** Используем прежние прогнозы ансамбля LightGBM и трёх векторных нейросетей, а также исходный ранг смеси из 7-го цикла. Веса этих моделей обучены на train, часть гиперпараметров и весов смесей ранее выбрана на раннем val. Прогнозы калибровок из 9-го цикла служат только частью сравниваемого ансамбля. Прямых обучающих признаков из его прогнозов на раннем val нет.

Строим линейный контроль с listwise softmax по целым лентам и штрафом `0.03`. Признаки из `score_calibration.design` дополняем разбросом и максимумом стандартизованных прогнозов моделей, голосами за топ-10, длиной ленты, пересечением топ-10, отступом первого кандидата от второго и максимальным косинусом к недавней истории. К замороженному линейному score добавляем нейросетевую поправку с одним слоем на 24 скрытых признака, `SiLU` и dropout `0.15`. Она начинается с нуля, ограничена `±1`, использует listwise loss с квадратичным штрафом `0.04` за величину поправки. Обучаем 12 эпох, батч до 96 лент, AdamW `lr=0.002`, weight decay `0.03`, seed `2026`.

Для выбора веса первые 3 749 ранних лент разрезаем по времени пополам. Первая половина служит стартовым train. Для каждого из двух следующих блоков обе новые модели и калибровки сравниваемого ансамбля обучаются только на предшествующем префиксе. Архитектура, признаки и сетка весов фиксированы. Выбираем среди полной линейной и нелинейной оценок и разности между ними. Стандартизация внутри ленты, вес `0/0.025/0.05/0.10/0.20/0.35/0.50`, выбор по меньшему приросту NDCG из двух блоков. После фиксации переобучаем на 5 766 ранних лентах, оцениваем на 8 657 поздних против точных scores сабмита с Public `0.3652139518`. Критерий Parquet прежний.


**Дополнительная модель с тем же временным протоколом.** Проверяем `HistGradientBoostingClassifier` с 64 итерациями, глубиной до 4, до 11 листьев, минимальным листом 120 строк, 127 бинами, LR `0.06` и L2 `5`. Модель получает ту же матрицу и линейный score с монотонностью по этому столбцу. Pointwise log loss даёт одинаковую суммарную массу положительным и отрицательным каждой ленты с обоими классами. Линейный контроль, кандидаты, временные блоки, веса смеси и критерий Parquet прежние.

При реконструкции сравниваемого ансамбля по префиксам переобучаются именно коэффициенты калибровок. Исторически выбранные гиперпараметры, состав моделей и веса по длине ленты сохраняются. Часть этих настроек в предыдущих циклах выбиралась на тех же ранних лентах. Поздние метки не участвуют в новых настройках.


Первый временной блок оценки — 1 879 лент после префикса из 1 870 лент и 176 417 строк. Второй — 2 017 лент после префикса из 3 749 лент и 347 742 строк. Итоговые модели видели 544 950 строк из 5 766 ранних лент. Всего 34 признака. Для признаков уверенности использовали z-score с ограничением `±5`. Проверка полного val обнаружила верхнее ограничение у деревьев в 71 из 14 423 лент, в четырёх из них оно затронуло обе лидирующие оценки. Состав топ-10 ни в одной ленте из-за этого ограничения не менялся.

| Модель | Поздний 1 | Поздний 2 | Поздний NDCG | NDCG@10 | NDCG@30 | Положительных в топ-10 | NDCG@10, 101–300 | Положительных в топ-10, 101–300 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Отправленный ансамбль | 0.367838 | 0.378050 | 0.372957 | 0.245700 | 0.315384 | 4 864 | 0.097326 | 963 |
| Линейный контроль | 0.367525 | 0.377521 | 0.372536 | 0.245624 | 0.314866 | 4 882 | 0.100256 | 992 |
| Линейный score с нейросетевой поправкой | 0.367544 | 0.378408 | 0.372991 | 0.245735 | 0.315648 | 4 879 | 0.099373 | 987 |
| Гистограммный бустинг | 0.365646 | 0.376961 | 0.371319 | 0.243827 | 0.313615 | 4 865 | 0.097805 | 981 |
| Ансамбль + 5% линейного контроля, ранний выбор | 0.367846 | 0.378202 | 0.373038 | 0.245845 | 0.315464 | 4 867 | 0.097713 | 966 |

Нейросетевая поправка дала `+0.000455` к линейной модели с парным 95%-м интервалом `[−0.000322, +0.001267]`. К отправленному ансамблю разница `+0.000033` и разные знаки по поздним половинам. У бустинга `−0.001218` к линейному контролю и `−0.001639` к отправленному ансамблю. На длинных лентах отдельная линейная модель подняла NDCG@10 на `+0.002929` и число положительных в топ-10 с 963 до 992. Это диагностическое сравнение по длине.

Обе проверки выбрали 5% линейного контроля с ранними приростами `+0.000067` и `+0.000188`. Поздний прирост `+0.000080`, по половинам `+0.000009` и `+0.000152`, парный интервал `[+0.000003, +0.000200]`. Для нелинейных прогнозов и разностей с линейным контролем ранний отбор выбрал нулевой вес. Критерий подготовки Parquet не выполнен. [Нейросетевая поправка](../experiments/residual-rerank-024/metrics.json) и [бустинг](../experiments/residual-tree-rerank-024/metrics.json).

Воспроизведение на Python 3.12, PyTorch 2.6 CPU, SciPy 1.16.3 и scikit-learn 1.8.0. Прошли 62 теста.

```powershell
$sequencePython = "$env:LOCALAPPDATA/Programs/Python/Python312/python.exe"
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '4'
& $sequencePython -B -m unittest discover -s experiments
& $sequencePython -B -u experiments/residual_rerank.py select experiments experiments/repro-residual-rerank-024
& $sequencePython -B -u experiments/residual_rerank.py select experiments experiments/repro-residual-tree-rerank-024 --trees
```


## Двадцать шестой цикл. Отдельная калибровка для лент на 101–300 кандидатов

**Протокол до расчёта нового позднего результата.** Отталкиваемся от диагностики 25-го цикла, где общий линейный прогноз улучшил верх длинных лент. Используем те же 34 признака и общий listwise контроль. Вторую модель обучаем на полных лентах со 101–300 кандидатами. Коэффициенты регуляризуем к общему контролю с L2 `0.10`. Метки коротких лент влияют на неё через общий контроль.

Для временного отбора берём те же два блока, что в 25-м цикле, и используем только их длинные ленты. Общую модель, специализированную модель и коэффициенты калибровок ансамбля обучаем на соответствующих предшествующих префиксах. Исторические параметры замороженных моделей и веса смеси по длине сохраняются. Сравниваем стандартизованные полные scores обеих моделей и их разность. Веса `0/0.05/0.10/0.20/0.35/0.50/0.75/1`. Выбираем по меньшему из двух приростов полной NDCG длинных лент. Поправка применяется только на 101–300 кандидатах.

После фиксации выбора переобучаем коэффициенты на ранних лентах. Сравниваем с точными прогнозами отправленного `nn-calibration-joint-024`, Public `0.3652139518`. Считаем полную NDCG, NDCG@10, @30, положительных в топ-10, парные интервалы и метрики длинных лент. Это проверка гипотезы, возникшей после просмотра поздней диагностики предыдущего цикла. При решении о новом Parquet применяем прежний критерий по полному позднему val.


**Вторая область применения.** На тех же двух ранних блоках проверяем глобальную и специализированную модели на объединении двух списков топ-20. За пределами объединения score остаётся исходным. Кандидатам в объединении присваиваем тот же набор scores ансамбля в новом порядке, заданном смесью. Это позволяет поднять кандидата из-за границы исходного топ-20. Сетка весов и критерий прежние, разность между ветками здесь не участвует. Область — те же ленты на 101–300 кандидатов.


В двух блоках оценки оказалось 651 и 749 длинных лент. Их модели обучены на префиксах с 661 и 1 312 длинными лентами. Итоговый специалист использовал 2 061 длинную ленту и 393 980 кандидатов, общий контроль — 5 766 лент. Ключи совпали с полным val на всех 1 389 971 парах.

| Вариант | Поздний 1 | Поздний 2 | Поздний NDCG | NDCG@10 | NDCG@30 | Положительных в топ-10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Отправленный ансамбль | 0.367838 | 0.378050 | 0.372957 | 0.245700 | 0.315384 | 4 864 |
| 5% общей модели в длинных лентах, ранний выбор | 0.367846 | 0.378202 | 0.373038 | 0.245845 | 0.315464 | 4 867 |
| 5% специалиста в длинных лентах | 0.367687 | 0.378185 | 0.372950 | 0.245709 | 0.315369 | 4 865 |
| 35% разности специалиста и общей модели | 0.367699 | 0.376908 | 0.372316 | 0.244753 | 0.314683 | 4 849 |
| 100% общей модели в длинных лентах, диагностика | 0.368376 | 0.378586 | 0.373494 | 0.246798 | 0.316023 | 4 893 |
| 100% специалиста в длинных лентах, диагностика | 0.368469 | 0.378472 | 0.373484 | 0.246704 | 0.316105 | 4 883 |
| 5% общей модели в объединении топ-20, ранний выбор | 0.367845 | 0.378202 | 0.373037 | 0.245845 | 0.315464 | 4 867 |
| 100% общей модели в объединении топ-20, диагностика | 0.368377 | 0.378561 | 0.373482 | 0.246798 | 0.315808 | 4 893 |

На длинных лентах отправленный ансамбль дал NDCG@10 `0.097326` и 963 положительных в топ-10. У раннего выбора `0.097713` и 966, у полной замены на общий прогноз `0.100256` и 992, у полной замены на специалиста `0.100005` и 982. На позднем val полный специалист уступил полному общему прогнозу на `0.000029` NDCG длинных лент, парный интервал `[−0.001322, +0.001295]`.

В обоих режимах ранний отбор выбрал `0.05` общей модели. Она прибавила примерно `+0.000080` полной поздней NDCG, с парным интервалом `[+0.000003, +0.000200]` при перестановке по всей длинной ленте. Полная замена общей моделью дала диагностические `+0.000537` с выигрышем в обеих поздних половинах и парным интервалом `[+0.000054, +0.001054]`. В обеих ранних оценках длинных лент полная замена проиграла `−0.001497` и `−0.002462`. Оценка использовала те же поздние ленты, на которых заметили исходный сигнал. Критерий подготовки нового Parquet не выполнен. [Оценка по целым длинным лентам](../experiments/long-feed-calibration-024/metrics.json) и [объединение топ-20](../experiments/long-feed-calibration-head-024/metrics.json).

Воспроизведение на том же окружении, что в 25-м цикле. Прошли 64 теста.

```powershell
$sequencePython = "$env:LOCALAPPDATA/Programs/Python/Python312/python.exe"
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '4'
& $sequencePython -B -m unittest discover -s experiments
& $sequencePython -B -u experiments/long_feed_calibration.py select experiments experiments/repro-long-feed-calibration-024
& $sequencePython -B -u experiments/long_feed_calibration.py select experiments experiments/repro-long-feed-calibration-head-024 --head-union 20
```


## Двадцать седьмой цикл. Внимание к группам кандидатов внутри ленты

**Протокол до оценки позднего val.** Поверх `SequenceRanker` из 21-го цикла добавляем восемь обучаемых запросов. Каждый собирает взвешенное представление всех кандидатов, игнорируя паддинг. Из этих восьми представлений второй проход внимания строит поправку к скрытому вектору кандидата. Размер проекций 24. Выходная проекция начинается с нуля.

В обеих ветках одинаковые параметры, входные признаки, общий mean/max контекст и предобученный GRU. Контроль получает один контекст для всей ленты, используя запрос из среднего скрытого вектора кандидатов. Во второй ветке запрос строится по собственному скрытому вектору каждого кандидата. Это единственное различие архитектур. Исходная позиция показа в модели отсутствует.

Обучаем на тех же 216 000 полных лентах, что и 21-й цикл. Один seed `2026`, шесть эпох, батчи до 96, идентичные порядок и маски dropout. Инициализация обеих веток совпадает. Используем прежний listwise loss с бинарным членом, одинаковый AdamW и cosine decay. Берём финальные веса. Сравниваем две модели и их смеси с точным отправленным ансамблем, прежний GRU без этой ветки служит дополнительным контролем. Для смесей используем два ранних блока и прежнюю сетку, для решения о Parquet — прежний критерий по позднему val. Положение кандидата в исходном списке и паддинг проверяются отдельно.


После обучения отдельно проверяем, насколько модель использует разные запросы кандидатов. У замороженного чекпойнта переключаем второй проход на один усреднённый запрос для ленты. Сравниваем его с исходным чекпойнтом на тех же val-лентах без дообучения. Эта проверка показывает чувствительность прогноза и не участвует в выборе смеси.


В каждой ветке 47 530 параметров. У прежнего GRU без добавленной ветки 42 154. Обе модели использовали 216 000 лент, 16 903 568 кандидатов, 373 607 положительных и 2 277 шагов. Обучение пары заняло 319,30 с. Прежний чекпойнт GRU вновь воспроизвёл свои оценки на всех 1 389 971 парах val. [Параметры и лог](../experiments/candidate-attention-finetune-048/tune.json).

| Модель | Ранний 1 | Ранний 2 | Поздний 1 | Поздний 2 | Поздний NDCG | Полный val |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Отправленный ансамбль | 0.382012 | 0.376541 | 0.367838 | 0.378050 | 0.372957 | 0.375812 |
| Прежний GRU | 0.377761 | 0.371074 | 0.364167 | 0.376621 | 0.370411 | 0.372414 |
| Один запрос на ленту, контроль | 0.378156 | 0.369932 | 0.364290 | 0.376463 | 0.370393 | 0.372346 |
| Индивидуальный запрос кандидата | 0.378725 | 0.370253 | 0.363709 | 0.376148 | 0.369945 | 0.372270 |
| Ансамбль + 2.5% контроля, ранний выбор | 0.382014 | 0.376545 | 0.367761 | 0.378009 | 0.372898 | 0.375778 |
| Ансамбль + 2.5% индивидуального внимания, общий вес | 0.382014 | 0.376540 | 0.367762 | 0.378009 | 0.372899 | 0.375777 |

| Модель | Поздний NDCG@10 | Поздний NDCG@30 | Положительных в топ-10 | NDCG@10, 101–300 | Положительных в топ-10, 101–300 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Отправленный ансамбль | 0.245700 | 0.315384 | 4 864 | 0.097326 | 963 |
| Прежний GRU | 0.242055 | 0.311915 | 4 778 | 0.095932 | 944 |
| Один запрос на ленту | 0.241877 | 0.311908 | 4 773 | 0.095542 | 937 |
| Индивидуальный запрос кандидата | 0.241790 | 0.311467 | 4 784 | 0.095521 | 943 |
| Ансамбль + 2.5% контроля | 0.245639 | 0.315325 | 4 864 | 0.097164 | 963 |
| Ансамбль + 2.5% индивидуального внимания | 0.245639 | 0.315326 | 4 864 | 0.097164 | 963 |

Индивидуальные запросы дали `−0.000448` поздней NDCG к контролю, по половинам `−0.000581` и `−0.000316`, парный интервал `[−0.000965, +0.000048]`. При переключении замороженного чекпойнта на общий запрос меняется первый кандидат в 287 из 8 657 поздних лент. Средняя абсолютная разница logits `0.0130`. NDCG после переключения выросла на `+0.000463`, парный интервал `[−0.000067, +0.001062]`. Это показывает, что индивидуальные запросы меняют ранжирование, а польза для этой метрики не подтвердилась. [Проверка одного запроса на замороженных весах](../experiments/candidate-attention-finetune-048/attention_ablation.json).

Ранний отбор выбрал `2.5%` контроля и нулевые веса для индивидуального внимания и разности веток. Поздний прирост выбранной смеси `−0.000059`, обе половины отрицательны. Критерий подготовки Parquet не выполнен. [Ранний выбор](../experiments/candidate-attention-assessment-048/early_selection.json) и [полные метрики](../experiments/candidate-attention-assessment-048/assessment.json).

Воспроизведение на Python 3.12 и PyTorch 2.6 CPU. Прошли 67 тестов, включая восстановление чекпойнта с индивидуальными запросами.

```powershell
$sequencePython = "$env:LOCALAPPDATA/Programs/Python/Python312/python.exe"
$env:OPENBLAS_NUM_THREADS = '1'
& $sequencePython -B -m unittest discover -s experiments
& $sequencePython -B -u experiments/sequence_ranker.py tune experiments/cache-intent-024 experiments/embedding-cache-024 experiments/cache-intent-048 experiments/embedding-cache-048 experiments/sequence-pretrain-510/encoder.pt experiments/repro-candidate-attention-finetune-048 --candidate-attention
& $sequencePython -B -u experiments/sequence_ranker.py assess experiments/cache-intent-024 experiments/repro-candidate-attention-finetune-048 experiments/nn-calibration-joint-024/val_predictions.parquet experiments/repro-candidate-attention-assessment-048 --candidate-attention --prior-pretrained experiments/sequence-finetune-048/pretrained/val_predictions.parquet
& $sequencePython -B -u experiments/sequence_ranker.py ablate-attention experiments/cache-intent-024 experiments/embedding-cache-024 experiments/repro-candidate-attention-finetune-048/attention/model.pt experiments/repro-candidate-attention-finetune-048/attention_ablation.json
```

## Двадцать восьмой цикл. Совместная адаптация скрытого блока к случайным показам

**16 ноября 2026. Протокол зафиксирован до обучения и новой поздней оценки.** Проверяем, помогают ли размеченные train-ленты сохранить полезное представление при адаптации к откликам из случайно перемешанных показов. Все ветки начинаются с одного `sequence-finetune-048/pretrained/model.pt`, сохранённого после фиксированной шестой эпохи обучения на train. Нового предобучения нет. Сначала восстановлены все **1 389 971** сохранённых val-score: побитовое совпадение, максимальная ошибка `0`. [Проверка источника и SHA-256](../experiments/randomized-adaptation-048/reproduction.json), [предварительный протокол](../experiments/randomized_adaptation_plan.md).

Сравниваем замороженный источник, адаптацию только на ранних случайных показах и совместную адаптацию с train. Переиспользованы `SequenceFeeds`, `SequenceRanker`, 79 признаков, кэши векторов и исходный listwise loss с бинарным членом `0.25`. Открыты остаточный скрытый блок `local` (два Linear) и выходная голова: **4 753 общих обучаемых параметра**. Входной слой, эмбеддинги, векторные проекции, GRU, его коэффициент и веса общего контекста заморожены. Операции общего контекста продолжают передавать градиент в `local`.

В совместной ветке отдельная train-голова начинается с копии исходной головы, получает то же скрытое представление и обучаемое смещение диапазона `pos`. Границы диапазонов `2/4/8/16/32/64/128/256`, граница относится к следующему диапазону. Дополнительных параметров 58: 49 в голове и 9 смещений глубины. Замороженный вклад GRU сохранён в обеих головах. Train-голова и `pos` не используются при прогнозе val/test. На шаге `loss = random_loss + 0.5 * train_loss + 0.1 * sum((shared_weights - original_weights)^2)`. В random-only ветке train-члена нет. Регуляризуются одинаковые исходные параметры `local` и случайной головы; дополнительная train-голова к источнику не притягивается. AdamW без weight decay, постоянный LR, отдельный фиксированный LR train-головы `0.001`, без clipping. Seed `2026`, шесть CPU-потоков.

Обе адаптируемые ветки получают одинаковые случайные батчи, число шагов, маски dropout и регуляризацию. На соответствующих шагах train-батчи также одинаковы между конфигурациями. Батчи содержат **до 64 полных лент**, кандидаты не отбираются; существующий загрузчик группирует ленты по длине. У финального обучения девять неполных случайных батчей из 100, минимальный размер 1, train-батчи по 64. Средний loss неполного батча получает целый шаг оптимизатора. Это ограничение схемы, одинаковое для сравниваемых веток; после результатов её не меняли.

Пилот пары на 20 шагах занял `2.107` с. Загрузка полного train-кэша заняла `43.72` с; доступно 295 376 лент и 23 111 632 кандидата из 48 шардов. Пик памяти процесса при загрузке `20.87 GiB`, рабочая память после пилота `6.48 GiB`. Зафиксирована небольшая сетка LR `3e-5 / 1e-4`, контрольные точки `100 / 300` обновлений. Обучение пилота, четырёх пар для временных проверок и финальной пары с промежуточными проверками заняло суммарно `107.62` с; финальная пара — `7.35` с. [Бюджет](../experiments/randomized-adaptation-048/budget_lock.json).

Для первого блока обучаемся на первых 1 870 лентах, оцениваем следующие 1 879; для второго обучаемся на первых 3 749, оцениваем следующие 2 017. Выбираем **одну общую конфигурацию** по максимальному минимуму двух приростов совместной ветки к замороженному источнику; при равенстве — по среднему с весом числа лент, затем меньшему бюджету. Random-only получает ту же конфигурацию для сопоставимого основного сравнения. Смеси в этом опыте не подбирались. Точный отправленный ансамбль — внешний контроль: его ранние калибровки видели метки этих блоков, поэтому он не участвует в новом выборе.

| LR | Шагов | Δ random-only, блок 1 | Δ random-only, блок 2 | Δ joint, блок 1 | Δ joint, блок 2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.00003 | 100 | −0.000301225 | −0.000108171 | −0.000219192 | +0.000076074 |
| 0.00003 | 300 | +0.000020906 | +0.000140098 | −0.000223335 | +0.000183026 |
| **0.00010** | **100** | +0.000227324 | −0.000149959 | **+0.000012250** | **+0.000026806** |
| 0.00010 | 300 | −0.000130881 | +0.000188138 | −0.000129276 | −0.000126818 |

NDCG замороженного источника в этих проверках `0.3845841013 / 0.3710735718`. Выбран LR `0.0001`, 100 обновлений. У random-only лучший собственный минимум был при `0.00003 / 300`, но в основном сравнении намеренно оставлен общий режим, чтобы не смешивать эффект train с разным бюджетом. [Полный ранний выбор](../experiments/randomized-adaptation-048/selection.json).

После фиксации выбора обе ветки заново загружены из исходного чекпойнта и адаптированы на **всех 5 766 ранних лентах**. За 100 шагов каждая получила 6 086 предъявлений ранних лент с учётом повторов, 562 164 кандидата и 9 245 положительных. Совместная ветка дополнительно получила 6 400 разных train-лент, 595 159 кандидатов и 11 413 положительных. Сохранены индексы всех батчей, их размеры и хеши кандидатов/меток; проверены границы временных префиксов. Поздние метки не участвуют в градиентах, подборе или остановке. [Фактический объём финального обучения](../experiments/randomized-adaptation-048/final/step-100/budget.json).

| Самостоятельная модель | Поздний 1 | Поздний 2 | Поздний NDCG | NDCG@10 | NDCG@30 | Положительных в топ-10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Замороженный исходный чекпойнт | 0.364167396 | 0.376621266 | 0.370410875 | 0.242054666 | 0.311915314 | 4 778 |
| Адаптация только на случайных показах | 0.363431582 | 0.376501126 | 0.369983716 | 0.241532288 | 0.311679875 | 4 780 |
| Совместная адаптация с train | 0.363238349 | 0.376414578 | 0.369843967 | 0.241409135 | 0.311486438 | 4 780 |
| Точный отправленный ансамбль | **0.367837522** | **0.378049720** | **0.372957187** | **0.245699852** | **0.315384208** | **4 864** |

Поздние половины содержат 4 317 и 4 340 лент. Интервалы ниже — парный бутстрэп по 8 657 поздним лентам, 2 000 повторов, seed `2026`; это интервалы по лентам для одного обучающего seed.

| Сравнение | Δ полной поздней NDCG | Парный 95%-й интервал |
| --- | ---: | --- |
| Joint − random-only | **−0.000139749** | **[−0.000272162, −0.000037068]** |
| Random-only − frozen | −0.000427159 | [−0.000889700, +0.000014583] |
| Joint − frozen | −0.000566908 | [−0.001036520, −0.000146708] |
| Frozen − отправленный | −0.002546312 | [−0.004745424, −0.000300256] |
| Random-only − отправленный | −0.002973471 | [−0.005116742, −0.000770488] |
| Joint − отправленный | −0.003113220 | [−0.005268061, −0.000921723] |

Совместное обучение уступило random-only в обеих половинах: `−0.000193233 / −0.000086548`. Его разность NDCG@10 `−0.000123153`, NDCG@30 `−0.000193437`, число положительных в топ-10 одинаково. При проверенном бюджете train **не сохранил перенос лучше**, чем адаптация только на случайных показах. Это вывод о данном открытом блоке, loss, весе `0.5`, регуляризации и бюджете, а не опровержение любых способов совместного обучения.

| Модель, 3 245 поздних лент со 101–300 кандидатами | NDCG | NDCG@10 | NDCG@30 | Положительных в топ-10 |
| --- | ---: | ---: | ---: | ---: |
| Frozen | 0.270440415 | 0.095931583 | 0.154259694 | 944 |
| Random-only | 0.270506322 | 0.096004988 | 0.154825948 | 946 |
| Joint | 0.270355924 | 0.096026249 | 0.154529759 | 948 |
| Отправленный ансамбль | 0.271898074 | 0.097326481 | 0.157142775 | 963 |

На длинных лентах joint дал к random-only `+0.000021261` NDCG@10 и два дополнительных положительных в топ-10. Парный 95%-й интервал этой разности NDCG@10 `[−0.000358714, +0.000397622]`; улучшение верха здесь не подтверждено. [Все метрики, половины и интервалы для NDCG/@10/@30](../experiments/randomized-adaptation-048/assessment.json).

**Проверки.** Найдено и исправлено различие при загрузке адаптированного чекпойнта: `state_dict` не сохраняет `requires_grad`, а восстановление всех параметров как обучаемых меняет низкоуровневые вычисления CPU. До исправления 198 score random-only отличались максимум на `2.3841858e-7`; веса совпадали. Новый `load_adapted` восстанавливает флаги заморозки и обе головы. Оценка остановилась до расчёта поздних метрик; обучение и ранний выбор не повторялись. После исправления восстановленные прогнозы обеих веток побитово воспроизводятся на всех 1 389 971 парах, а random-only также точно совпал с файлом до остановки. [Проверка повторной загрузки](../experiments/randomized-adaptation-048/checkpoint_roundtrip.json).

Прошли **72 теста**, включая независимость val от train-головы/позиции, градиент auxiliary-loss в скрытый блок и его отсутствие в случайной голове, совпадение обновлений при нулевом train-весе, регуляризацию к исходным весам и восстановление заморозки. Проверка сохранённых весов подтвердила изменение только `local.*` и `output.*` исходного ранжировщика. Независимый скрипт заново сгруппировал сохранённые Parquet по ключам и посчитал полную NDCG/@10/@30 по NumPy-формуле из официального `baseline/calculate_metrics.py`; все значения полного позднего блока и обеих половин совпали с отчётом в пределах `1e-10`. Проверка через CatBoost не запускалась: он отсутствует в локальном Python 3.12. Ключи, конечность score, точность отправленного контроля, ранний выбор и хронологические границы всех батчей проверены отдельно. [Независимая сверка](../experiments/randomized-adaptation-048/independent_verification.json).

Критерий второго seed и нового сабмита не выполнен: обе адаптации проиграли отправленному ансамблю. Test-прогнозы новых моделей и новый Parquet для Public не создавались. По запросу повторно запущен `ranking.py audit` **существующего лучшего сабмита** `nn-calibration-joint-024/submission-reserve.parquet`: ровно **2 904 806** пар официального шаблона, 28 595 лент, конечные float32 scores, **0 лент с ничьими**. [Аудит существующего контроля](../experiments/randomized-adaptation-048/control_submission_audit.json). Лучший подтверждённый Public остаётся `0.3652139518`; новых оценок Public в этом цикле нет.

Артефакты: [каталог прогона](../experiments/randomized-adaptation-048), [random-only val](../experiments/randomized-adaptation-048/random_only_val_predictions.parquet), [joint val](../experiments/randomized-adaptation-048/joint_val_predictions.parquet), [random-only checkpoint](../experiments/randomized-adaptation-048/final/step-100/random_only.pt), [joint checkpoint](../experiments/randomized-adaptation-048/final/step-100/joint.pt). В каталогах `forward-*` сохранены обе контрольные точки, прогнозы блоков, параметры, потери и расписания батчей. Полный val-файл финальной модели включает ранние ленты, использованные для обучения; их метрики не выдаются за отложенную оценку.

Воспроизведение: Python `3.12.0`, PyTorch `2.6.0+cpu`, NumPy `1.26.4`. Использован абсолютный путь: в sandbox `%LOCALAPPDATA%` указывает на отдельный каталог и не на установленный Python.

```powershell
$sequencePython = 'C:\Users\bikra\AppData\Local\Programs\Python\Python312\python.exe'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '6'
& $sequencePython -B -m unittest discover -s experiments
# Новый полный прогон; исправленный загрузчик уже включён.
& $sequencePython -B -u experiments/randomized_adaptation.py run experiments/repro-randomized-adaptation-048
& $sequencePython -B -u experiments/verify_randomized_adaptation.py experiments/repro-randomized-adaptation-048
& $sequencePython -B experiments/ranking.py audit experiments/nn-calibration-joint-024/submission-reserve.parquet
# В фактическом прогоне после исправления загрузчика выполнена только оценка сохранённых весов:
# & $sequencePython -B -u experiments/randomized_adaptation.py assess-saved experiments/randomized-adaptation-048
```

## Двадцать девятый цикл. История аудитории объявления

Проверяем, несёт ли состав откликнувшейся аудитории информацию помимо счётчика откликов. Для каждого объявления сравниваем косинус эмбеддинга текущего пользователя со средним нормированных векторов откликнувшихся пользователей и со средним вектором аудитории показов этого объявления. Для второго среднего вес каждого показа равен ожидаемой вероятности отклика по позиции, длине ленты и категории из фоновой модели цикла 19. Фон строится вне текущего шарда с восемью фолдами. Из обеих аудиторий исключили `user_id=0` и все известные ID пользователей val/test. Все показы заканчиваются раньше val/test.

Прочитаны все 512 шардов, 3 151 208 лент, 246 745 836 кандидатов и 5 446 872 отклика. Для аудитории объявлений val/test использованы 28 487 837 показов и 883 288 откликов. Векторы откликнувшейся аудитории есть у 414 810 объявлений. Подготовка заняла 112 с. [Объём и покрытие](../experiments/item-audience-512/cohorts.json).

Сравниваем сходство с откликнувшейся аудиторией и разность двух сходств. Сжимаем сигнал к нулю множителем `n/(n+k)` при `k=1/4`, где `n` — число откликов с доступным вектором. Для смеси в стандартизованной шкале отправленного ансамбля умножаем сигнал на 10, веса `0.05/0.10/0.20/0.35/0.50/1`. Из 24 вариантов выбираем максимальный минимум прироста полной NDCG на первых 3 749 и следующих 2 017 ранних лентах. Для ненулевого веса нужен выигрыш в обоих блоках. Лучший ненулевой вариант использовал разность косинусов, `k=4`, вес `0.05` и дал `−0.000125 / −0.000187`. Выбран вес 0. На позднем val точный контроль воспроизвёл `0.3729571873`, NDCG@10 `0.2456998515` и NDCG@30 `0.3153842081`. [Ранний отбор](../experiments/item-audience-512/early_selection.json), [итоговая оценка](../experiments/item-audience-512/assessment.json).

Хотя бы один подходящий train-отклик нашёлся для 21.63% положительных кандидатов позднего val. Не менее четырёх откликов есть у 5.27% положительных. Доля кандидатов с хотя бы одним откликом `17.92%` на позднем val и `14.66%` на test. Данный способ переносить информацию об аудитории не показал прироста. Сверены все ключи сигналов val/test, конечность оценок, нули у отсутствующей аудитории и совпадение всех 1 389 971 val-score с отправленным контролем. Полная NDCG/@10/@30 позднего блока и обеих половин дополнительно совпали с независимой NumPy-формулой в пределах `1e-10`.

Диагностика текущего ранжировщика на 5 766 ранних лентах. У отправленного ансамбля ниже позиции 30 находятся 3 094 из 8 815 положительных кандидатов (`35.10%`). В 2 061 ленте со 101–300 кандидатами это 2 450 из 3 916 (`62.56%`). Перестановка внутри топ-30 не может их поднять. Для следующей гипотезы с заметным потенциалом стоит измерять способность модели находить положительных по всей ленте, отдельно на длинных лентах. Для проверки качества короткого списка дополнительно считать долю всех положительных, найденных в топ-30.

```powershell
$sequencePython = 'C:\Users\bikra\AppData\Local\Programs\Python\Python312\python.exe'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '4'
& $sequencePython -B -u experiments/item_audience.py experiments/repro-item-audience-512
```

## Тридцатый цикл. Трудные отрицательные на сопоставимой глубине

**Протокол до итоговой оценки.** На раннем val отправленный ансамбль оставляет 62.56% положительных кандидатов длинных лент за топ-30. Проверяем, можно ли поднять часть этих кандидатов обучением ранжировщика на полных train-лентах. Обе ветки стартуют с одного `sequence-finetune-048/pretrained/model.pt` после фиксированной шестой эпохи. Перед обучением воспроизводим его сохранённые val-score. Все слои `SequenceRanker` открыты. Одинаковые 79 признаков, эмбеддинги, целые ленты, порядок кандидатов и лент, маски dropout и оптимизатор.

Контроль продолжает прежний listwise loss с бинарным членом `0.25`. Вторая ветка добавляет попарный член с коэффициентом `0.5`. Для положительного на позиции `p` кандидаты с меткой 0 допустимы при `pos <= p` и `2 * (pos + 2) >= p + 2`. Из них берём до четырёх с наибольшим score замороженного исходного ранжировщика. Loss пары `softplus(score_negative - score_positive)`. Положительные вне топ-30 исходного ранжировщика получают множитель 2. Усредняем по допустимым отрицательным каждого положительного, по положительным с парами в ленте, затем по батчу. Для лент без пар этот член равен нулю. Неизвестная глубина просмотра остаётся ограничением. Сопоставление использует только опубликованную `pos` в train.

Один проход по 295 376 полным лентам из 48 шардов. Seed `2027`, батч до 96, AdamW, weight decay `0.03`, LR по косинусу `0.0003 → 0.00009`, clipping нормы градиента 5, шесть CPU-потоков. Контрольные точки `8 192 / 36 000 / 108 000 / 216 000 / 295 376` лент, первая служит пилотом. Состав первой и следующих частей не пересекается. Новый ранжировщик оценивает всех кандидатов без `pos`.

На ранних блоках из 3 749 и 2 017 лент выбираем checkpoint совместного сравнения по максимальному минимуму двух приростов полной NDCG новой модели относительно замороженного источника. Контроль сравниваем при том же числе лент; отдельно сохраняем лучший по ранним блокам checkpoint контроля. Из стандартизованных scores проверяем смеси с отправленным ансамблем при весах `0.025/0.05/0.10/0.20/0.35/0.50` для замороженного источника и обеих веток. Смесь должна дать положительный прирост на обоих ранних блоках. Исторические калибровки отправленного ансамбля использовали эти ранние метки. После фиксации выбора смотрим полную NDCG, @10, @30, долю положительных в топ-30, временные половины и парные интервалы. Основная диагностика поиска пропущенных кандидатов — топ-30 лент со 101–300 кандидатами.

**Результат.** Исходный чекпойнт точно воспроизвёл все 1 389 971 сохранённых val-score. Обе ветки прошли 295 376 разных лент, 23 111 632 кандидата и 510 341 положительный отклик за 3 099 одинаковых шагов. Найдено 1 712 626 пар для 473 611 положительных в 278 247 лентах. Пары нашлись у 166 609 из 169 242 положительных, которые исходный GRU поставил ниже топ-30. Пилот занял 9.95 с, проход с промежуточными прогнозами 376.27 с. Пиковый RSS процесса 20.96 GiB. На 280 лентах сверены позиции и метки всех 22 990 кандидатов по исходным Parquet, 1 744 выбранные пары совпали с независимым перебором. При равных scores rank для удвоения штрафа считает кандидатов с большей оценкой и отрицательных с равной оценкой. Равные оценки двух положительных дают им один rank. На раннем val точных ничьих положительных через границу топ-30 не встретилось. [Параметры](../experiments/depth-hard-ranker-048/config.json), [лог обучения](../experiments/depth-hard-ranker-048/training.json), [сверка пар](../experiments/depth-hard-ranker-048/mining_audit.txt).

| Лент | Δ контроля к замороженному, ранний 1 / 2 | Δ трудных пар к замороженному, ранний 1 / 2 |
| ---: | ---: | ---: |
| 8 192 | +0.000583 / −0.001744 | −0.003155 / −0.003650 |
| 36 000 | −0.000851 / −0.001064 | −0.005932 / −0.005035 |
| 108 000 | +0.000289 / +0.000395 | −0.010293 / −0.008722 |
| 216 000 | +0.000622 / +0.001198 | −0.014607 / −0.010787 |
| 295 376 | −0.000391 / +0.000004 | −0.015327 / −0.010551 |

Выбранная контрольная точка новой отдельной модели — 8 192 ленты, лучшая отдельная контрольная точка обычного дообучения — 216 000. Для смеси выбраны `2.5%` модели с трудными парами после полного прохода, ранние приросты `+0.000046 / +0.000081`. Выбор зафиксирован до вычисления поздних метрик. [Критерий и вся сетка](../experiments/depth-hard-ranker-048/selection.json).

| Модель | Поздняя NDCG | NDCG@10 | NDCG@30 | Положительных в топ-30 длинных | Доля положительных в топ-30 длинных |
| --- | ---: | ---: | ---: | ---: | ---: |
| Замороженный GRU | 0.370411 | 0.242055 | 0.311915 | 2 247 | 36.49% |
| Контроль, 8 192 | 0.368834 | 0.240302 | 0.310025 | 2 238 | 36.34% |
| Трудные пары, 8 192 | 0.367977 | 0.238571 | 0.308155 | 2 206 | 35.82% |
| Контроль, 216 000 | 0.370227 | 0.241286 | 0.311652 | 2 234 | 36.28% |
| Отправленный ансамбль | 0.372957 | 0.245700 | 0.315384 | 2 307 | 37.46% |
| Ансамбль + 2.5% новой модели | 0.373124 | 0.245810 | 0.315551 | 2 306 | 37.45% |

На одинаковых 8 192 train-лентах разность поздней NDCG трудных пар и контроля `−0.000856`, по половинам `+0.000609 / −0.002314`, парный 95%-й интервал `[−0.002829, +0.001143]`. В 3 245 длинных поздних лентах из 6 158 положительных на 32 меньше попали в топ-30. Разница полноты топ-30 `−0.52 п.п.`, бутстрэп-интервал `[−1.30, +0.27] п.п.`. На ранних лентах рост объёма усилил ухудшение. Например, при 108 000 train-лент контроль нашёл 1 431 положительный в топ-30 длинных лент, ветка с трудными парами — 1 161.

У смеси поздний прирост `+0.000167`, по половинам `+0.000375 / −0.000041`, 95%-й интервал `[−0.000027, +0.000392]`. В топ-30 длинных лент на один положительный меньше. Проверенный способ с сопоставимой позицией не повысил способность самостоятельной модели находить пропущенных кандидатов. Малый эффект смеси не поддержала вторая поздняя половина, поэтому второго seed и нового сабмита нет. Public остаётся `0.3652139518`. [Все метрики, половины и интервалы](../experiments/depth-hard-ranker-048/assessment.json).

Независимо подтверждены метрики на 8 657 поздних лентах, обеих половинах и длинных лентах. Проверены конечность десяти чекпойнтов, ключи всех 1 389 971 val-пар и полный однократный проход по train. Прошли 74 теста. В прежнем тесте изоляции голов цикла 28 учтена погрешность CPU при сравнении моделей с разными `requires_grad` (`2.98e-8`). Проверка неизменности score той же модели после изменения train-головы по-прежнему требует точного равенства. [Независимая сверка](../experiments/depth-hard-ranker-048/verification.txt), [каталог с чекпойнтами и прогнозами](../experiments/depth-hard-ranker-048).

```powershell
$sequencePython = 'C:\Users\bikra\AppData\Local\Programs\Python\Python312\python.exe'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '6'
& $sequencePython -B -m unittest discover -s experiments
& $sequencePython -B -u experiments/depth_hard_ranker.py train experiments/repro-depth-hard-ranker-048 --seed 2027 --threads 6
& $sequencePython -B -u experiments/depth_hard_ranker.py assess experiments/repro-depth-hard-ranker-048
```

## Тридцать первый цикл. Смена источника обучающих примеров

[Проект нового подхода](HISTORY_RETRIEVAL.md). Обе проверки в этом цикле используют быстрые пилоты из исходных данных и оценивают перенос на случайные показы до решения о масштабировании.

**Поиск по скрытым действиям.** На входе 24 более ранних события, цели — новые объявления с вектором среди четырёх последних событий train-истории. Маска новизны использует всю доступную старшую историю. Каждому кандидату соответствует оценка сходства с четырьмя векторами интересов. У контроля общий список отрицательных по батчу. Во второй ветке до 32 близких по косинусу отрицательных той же категории получают вес 4 в знаменателе softmax. Исключены товары из опубликованной истории пользователя, цели других лент того же пользователя и известные ID пользователей val/test. Вход использует историю и эмбеддинги, без `pos` и меток отклика текущих лент. У повторяющихся отрицательных ID сохраняются несколько строк. Типы исторических событий не интерпретируются как подтверждённые целевые действия конкурса.

Прочитано 96 train-шардов с 590 850 лентами, получено 491 609 контекстов и 1 453 711 целей за 3 130 шагов. Seed `2026`, AdamW `lr=0.001`, weight decay `0.01`, батч до 192 историй, шесть CPU-потоков. На 1 264 контекстах отдельных пользователей из четырёх отложенных train-шардов полнота @5 среди скрытых целей и близких конкурентов `89.13% / 89.48%`, для обычных и трудных примеров соответственно. На 32 шардах `86.96% / 88.23%`. Это оценка по мишеням других историй в батче, не по каталогу. [Лог и синтетические метрики](../experiments/history-retrieval-096/training.json).

| Модель | Поздняя NDCG | Положительных в топ-30 длинных | Из них ниже топ-30 отправленного |
| --- | ---: | ---: | ---: |
| Отправленный ансамбль | 0.372957 | 2 307 | 0 |
| Отдельный GRU цикла 22 | 0.341869 | 1 887 | 571 |
| Поиск по истории, обычные отрицательные, 96 | 0.342975 | 1 877 | 533 |
| Поиск по истории, трудные отрицательные, 96 | 0.342190 | 1 891 | 564 |

Сильнейший ненулевой вариант из новых веток — 5% обычной модели на 96 шардах — проиграл `−0.000760 / −0.001062` на ранних временных блоках. У лучшего варианта с трудными отрицательными `−0.000987 / −0.001012`. Выбор зафиксирован на нулевом весе до поздней оценки. В длинных поздних лентах трудная модель подняла 564 пропущенных положительных и вытеснила 980 найденных. На обеих ранних частях проверка фиксированного бюджета 30 через 20 кандидатов ансамбля и 10 из модели истории тоже теряла отклики. На 32 шардах с трудными примерами получено 905 против 963 и 491 против 503. Новое обучение улучшает искусственную задачу, однако его отдельный список по полноте уступает текущему ансамблю. Масштабирование по синтетической метрике не обосновано. [Ранний отбор](../experiments/history-retrieval-096/selection.json), [ранняя диагностическая замена](../experiments/history-retrieval-096/early_long_diagnostics.json), [поздние метрики](../experiments/history-retrieval-096/assessment.json).

Оба выбранных чекпойнта модели истории побитово воспроизвели по 1 389 971 сохранённому score. Независимо проверены ключи, конечность и все три NDCG на позднем блоке и половинах. После добавления тестов хронологии и маски прошло 76 тестов. Проверка прежнего GRU служит внешним сравнением, поскольку различаются обучение, архитектура и бюджет. [Сверка прогнозов](../experiments/history-retrieval-096/verification.txt).

**Восстановление исходного порядка.** Отдельный пилот обучает HistGradientBoostingRegressor с 80 деревьями предсказывать `log1p(pos)` по исходным пяти признакам либо по 79 признакам. Для обучения взяты 15 000 полных лент (1 168 652 кандидата) из первых 85% по времени кэша 24 шардов. Проверка имитации на 3 000 более поздних train-лентах (228 055 кандидатов). Ни одна train-метка `target` не читалась в обучении. По местам исходного топ-10 полная NDCG `0.688092 / 0.708616`, MAE логарифма позиции `0.911392 / 0.668882`. Признаки восстанавливают часть исходного порядка, при этом отдельные поздние NDCG по реальным откликам случайных val-лент `0.290246 / 0.322124`. Лучший ненулевой вариант смешивания дал `−0.001066 / −0.000676` на ранних половинах. Выбран вес 0. Сохранённые модели побитово воспроизвели val-score. Проверены ключи и независимые метрики. [Параметры](../experiments/logged-order-probe-024/config.json), [имитация](../experiments/logged-order-probe-024/order_imitation.json), [отбор](../experiments/logged-order-probe-024/selection.json), [итог](../experiments/logged-order-probe-024/assessment.json), [сверка](../experiments/logged-order-probe-024/verification.txt).

**Совпадения в следующем снимке истории.** Выбраны 98 474 исходные ленты из 16 случайных шардов с seed `2026`. Для их пользователей среди всех 512 train-шардов нашли 4 528 строго последующих снимков в течение суток. Исключены `user_id=0` и пользователи с совпадающими timestamp. Для склейки требовали уникального совпадения четырёх первых событий старой истории с точным перекрытием остальных доступных событий. После склейки известные в исходной истории ID удалены из новых событий. Получена 501 выровненная лента. Из 838 положительных и 36 542 кандидатов с `target=0` в свежей части следующей истории обнаружены соответственно 828 и 56. Хотя история хорошо подтверждает `target` на этой подвыборке, в этих условиях она почти не добавляет других наблюдаемых действий. Сами совпадения не доказывают источник взаимодействия, и отсутствие события не доказывает отсутствия просмотра. Синтетические проверки склейки и сверка сумм по глубине/типу события прошли. [Отчёт со сверкой по глубине и типам](../experiments/history-snapshot-overlap-016/report.json).

Новых подтверждённых улучшений нет. Обе смеси воспроизводят поздний val отправленного ансамбля `0.3729571873`. Второй seed и новый сабмит не требуются. В проекте описан следующий вариант с выбором отдельных замен между кандидатами двух моделей и фиксированным бюджетом топ-30. [Проект и критерии следующей проверки](HISTORY_RETRIEVAL.md).

```powershell
$sequencePython = 'C:\Users\bikra\AppData\Local\Programs\Python\Python312\python.exe'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '6'
& $sequencePython -B -m unittest discover -s experiments
& $sequencePython -B -u experiments/history_retrieval.py experiments/repro-history-retrieval-096 --shards 96
& $sequencePython -B -u experiments/logged_order_probe.py experiments/repro-logged-order-probe-024
& $sequencePython -B -u experiments/history_snapshot_overlap.py experiments/repro-history-snapshot-overlap-016
```

## Тридцать второй цикл. Выбор отдельных замен в топ-30

**Протокол до метрик.** Отправленный ансамбль определяет исходный порядок. Сеть `hard_96` из цикла 31 предлагает десять кандидатов с наибольшим score по истории среди мест 31 и ниже. Возможные замены только мест 21–30. Рассматриваем все 100 пар в каждой ленте на 101–300 кандидатов. Заменяемые кандидаты обмениваются исходными scores, сохраняя число мест в топ-30 и исходный топ-20. Набор значений scores и ключи каждой ленты остаются теми же. При равных прогнозах истории порядок определяет `item_id`.

На пару строим 12 признаков из стандартизованных scores ансамбля и модели истории, логарифмов их рангов, размера ленты, длины истории, разностей scores и взаимодействия разности истории с её длиной. Регрессия с L2 оценивает ожидаемую разность бинарных меток кандидатов. Стандартизация признаков регрессии использует только обучающий префикс. При применении умножаем оценённую разность меток на известную разность скидок DCG между местами. Среди пар с оценённой разностью выше порога берём наибольший ожидаемый выигрыш без использования кандидата дважды.

Сетка L2-штрафа в среднеквадратичном loss `0.01/0.1/1`, минимальной ожидаемой разности меток `0/0.005/0.01/0.02/0.04` и максимума замен на ленту `1/2/4`, всего 45 настроек. Первая модель учится на префиксе 1 870 и прогнозирует следующие 1 879 лент. Вторая учится на префиксе 3 749 и прогнозирует следующие 2 017. В каждом префиксе участвуют только длинные ленты. Ненулевой вариант должен увеличить и полную NDCG на всех лентах, и число положительных в топ-30 на обеих проверках. Из допустимых выбираем по наибольшему минимуму приростов NDCG, затем по взвешенному среднему. Выбор сохраняем до поздней оценки. Переобучаем на 5 766 ранних лентах и проверяем 8 657 поздних. Исторические коэффициенты отправленного ансамбля уже использовали ранние метки, модели выбора замен используют только свои префиксы.

**Результат.** У моделей первого и второго префиксов соответственно 661 и 1 312 длинных обучающих лент. В первом префиксе из 66 100 пар 951 возвращает положительного взамен отрицательного и 1 211 теряют положительного. Во втором из 131 200 пар соответственно 1 824 и 2 524. Одна лента даёт до 100 зависимых пар. При переобучении на 5 766 ранних лентах регрессия использует 206 100 пар из 2 061 длинной ленты, включая 2 800 полезных и 3 790 вредных замен. [Параметры и обучающие объёмы](../experiments/history-swaps-096/model.json).

| Лучший ненулевой вариант по минимуму прироста NDCG | Замен | Поднято положительных | Вытеснено положительных | Чистый эффект в топ-30 | Δ полной NDCG |
| --- | ---: | ---: | ---: | ---: | ---: |
| Блок 1, после префикса в 1 870 лент | 756 | 12 | 15 | −3 | +0.00004805 |
| Блок 2, после префикса в 3 749 лент | 256 | 6 | 2 | +4 | +0.00002527 |

Лучший по NDCG активный вариант использовал `L2=1`, порог `0` и до четырёх замен. Из 1 012 замен 977 обменяли кандидатов с одинаковыми метками. Ни одна из 45 настроек не увеличила число положительных в топ-30 в обоих блоках. Выбран нулевой максимум замен. На поздней части все 1 389 971 score по ключам полного val совпали с исходным файлом ансамбля. На 8 657 поздних лентах NDCG `0.3729571873`, @10 `0.2456998515`, @30 `0.3153842081`. В топ-30 длинных поздних лент 2 307 положительных. Разница с отправленным ансамблем по обеим половинам и парный интервал равны нулю при выбранном правиле. Новый test-файл не создавался. [Ранний отбор](../experiments/history-swaps-096/selection.json), [итог](../experiments/history-swaps-096/assessment.json).

Независимо воспроизведены все ненулевые замены выбранной для диагностики настройки на ранних блоках. Сверены исходный топ-20, невзаимное использование кандидатов, перестановка scores и совпадение вычисленных приростов с полным перерасчётом NDCG. Поздние метрики и половины сверены NumPy-формулой официального вычислителя. Прошли 78 тестов. [Сверка](../experiments/history-swaps-096/verification.txt), [прогнозы](../experiments/history-swaps-096/val_predictions.parquet).

```powershell
$sequencePython = 'C:\Users\bikra\AppData\Local\Programs\Python\Python312\python.exe'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '4'
& $sequencePython -B -m unittest discover -s experiments
& $sequencePython -B -u experiments/history_swaps.py experiments/repro-history-swaps-096
```

## Тридцать третий цикл. Полное внимание по кандидатам и истории на GPU

**Протокол до оценки.** На этой машине AMD Radeon 780M. В исходной среде PyTorch `2.6.0+cpu`, CUDA недоступна. Для GPU создана отдельная `.venv-avito-directml` с PyTorch `2.4.1` и `torch-directml==0.2.5.dev240914`. Пробный шаг оптимизации двух слоёв полного внимания на этом GPU меняет веса и снижает синтетический loss. При батче 32 и 96/300 кандидатах измерено 437/155 лент в секунду, включая синхронизацию с устройством. DirectML сообщил о CPU fallback операции `aten::lerp.Scalar_out` внутри AdamW. Эти цифры измерены без чтения реальных батчей. [Проба GPU](../experiments/gpu-training-probe-directml.log).

Одна случайная инициализация, два слоя полного внимания шириной 64 и четыре головы. Все кандидаты, вектор пользователя с агрегатом недавней истории и до 12 событий входят в один набор токенов. Входы кандидата используют 79 признаков с индикаторами пропусков и эмбеддинг. Событие представляет эмбеддинг, тип и относительную давность. Кандидатская позиция отсутствует, паддинг исключён из ключей внимания и функции потерь. Прежний listwise loss с бинарным членом `0.25`. Seed `2026`, AdamW, LR `0.0012`, weight decay `0.03`. Выбираем без возвращения до 72 000 целых лент из кэша 24 шардов, батч до 32. Контрольные точки на 12 000, 36 000, 72 000 лент и в момент достижения лимита в 240 секунд обучения, если он сработает раньше. Время чтения кэшей и оценка чекпойнтов измеряются отдельно.

Сравниваем с точными прогнозами отправленного `nn-calibration-joint-024` на первых 3 749 и следующих 2 017 ранних лентах. Вес модели в смеси после стандартизации внутри ленты `0.10/0.25/0.50`; вес `1` означает самостоятельную модель. Выбираем по максимальному минимуму прироста полной NDCG и затем по объединённой ранней NDCG. Масштабирование допускается при приросте не менее `+0.001` в каждом раннем блоке. Поздние метки в пилоте не используются.

**Совместимость.** Первый запуск остановился до первого обновления весов. В сборке DirectML операция `float * bool` с градиентом вызывает внутреннее исключение PyTorch. Подтверждено на двух тензорах без данных соревнования. После преобразования положительной и отрицательной масок в `float32` сохранены прежние формула, значение и градиенты loss. Изолированная сборка пути `privateuseone:0` использовалась для обучения и прогноза. В логе рабочего прогона указан CPU fallback `lerp` в оптимизаторе. [Рабочий лог](../experiments/full-attention-directml-024-run.log), [лог до первого обновления](../experiments/full-attention-directml-024.log).

**Результат.** Обучено 111 169 параметров. На 64 724 разных целых лентах из 24 шардов обработано 5 050 395 кандидатов, 112 394 положительных и 2 035 шагов за 240.08 с. Лимит проверяется между батчами. Средняя скорость 270 лент/с с подготовкой батчей. Отправленный ансамбль дал `0.382011703 / 0.376541439` полной NDCG в двух ранних блоках.

| Train-лент | Секунды обучения | Ранняя NDCG отдельной модели | Δ смеси с весом 10%, блок 1 | Δ смеси с весом 10%, блок 2 |
| ---: | ---: | ---: | ---: | ---: |
| 12 000 | 53.21 | 0.362425 | −0.000278 | −0.000113 |
| 36 000 | 138.80 | 0.365241 | −0.000556 | −0.000149 |
| 64 724 | 240.08 | 0.365887 | −0.000020 | −0.000243 |

Лучшая комбинация использует 10% Transformer с финальной контрольной точки и даёт объединённую раннюю NDCG `0.3800001602`. Она уступает отправленному ансамблю в обоих блоках. Наблюдение ограничено одной инициализацией и бюджетом на этом устройстве. Заранее принятый порог масштабирования не выполнен. Поздние метрики, test-прогнозы и новый сабмит в этом цикле не рассчитывались. Подтверждённый Public `0.3652139518`. [Параметры](../experiments/full-attention-directml-024-run/config.json), [вся сетка](../experiments/full-attention-directml-024-run/selection.json), [чекпойнты и прогнозы](../experiments/full-attention-directml-024-run).

Все три чекпойнта содержат конечные веса. Для каждого сверили 544 950 ранних пар, четыре веса и метрики двух блоков независимой NumPy-формулой. Прошли 80 тестов, включая инвариантность к перестановке кандидатов и паддингу, а также сохранение значений и градиентов функции потерь. [Независимая сверка](../experiments/full-attention-directml-024-run/verification.txt).

```powershell
$py = 'C:\Users\bikra\AppData\Local\Programs\Python\Python312\python.exe'
# Окружение уже создано для фактического прогона.
# & $py -m venv --system-site-packages --without-pip .venv-avito-directml
# uv pip install --python .venv-avito-directml/Scripts/python.exe --index-url https://pypi.org/simple torch-directml==0.2.5.dev240914
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '6'
$gpuPython = '.\.venv-avito-directml\Scripts\python.exe'
& $gpuPython -B -u experiments/gpu_training_probe.py --backend directml
& $gpuPython -B -u experiments/full_attention_ranker.py experiments/repro-full-attention-directml-024 --backend directml
& $py -B -m unittest discover -s experiments
```

## Тридцать четвёртый цикл. Продолжение GPU Transformer на новых лентах

**Протокол до метрик.** По запросу пользователя продолжаем тот же Transformer ещё одним ограниченным проходом. Стартуем с финального чекпойнта после 64 724 лент из цикла 33 и сверяем все его сохранённые ранние прогнозы. Сохраняем нормализацию, архитектуру и loss. Состояние AdamW в исходном файле отсутствует, поэтому новый оптимизатор начинает с пустых моментов. LR `0.0006`, weight decay `0.03`. Перестановка с seed `2027` по остатку того же кэша 24 шардов. Исключаем все 72 000 индексов из исходного пула, в том числе 7 276 не дошедших до батча из-за лимита. Для продолжения остаётся 75 684 ленты, гарантированно не использованные ранее. Точки оценки после 36 000 и 75 684 новых лент или после последнего полного батча при лимите 300 секунд. Новые сохранения содержат веса и состояние AdamW.

На прежних ранних блоках оцениваем отдельную модель и смеси с весами `0.10/0.25/0.50`. Контроль — точный отправленный ансамбль, для ориентира приводим итоговую точку предыдущего пилота. Критерий для открытия поздних меток остаётся `+0.001` полной NDCG в каждом раннем блоке. Продолжение запускает один дополнительный проход даже при прежнем отрицательном результате, в соответствии с явным запросом пользователя.

**Результат.** Все 544 950 ранних scores исходного чекпойнта воспроизведены побитово. На Radeon 780M выполнены все 75 684 новые ленты за 292.72 с, 2 374 дополнительных шага, 5 916 719 кандидатов и 131 058 положительных. Суммарно с исходным пилотом модель видела 140 408 разных лент, 10 967 114 кандидатов за 4 409 шагов. Независимо проверено отсутствие пересечения исходного и нового пулов. Первый подготовительный запуск остановлен до обновления параметров, чтобы экспорт состояния AdamW в CPU-чекпойнт создавал копию и не менял работающий оптимизатор. [Параметры](../experiments/full-attention-directml-024-extra/config.json), [фактические объёмы](../experiments/full-attention-directml-024-extra/training.json).

| Всего train-лент | Ранняя NDCG Transformer | Δ смеси с 10% Transformer, блок 1 | Δ смеси с 10% Transformer, блок 2 |
| ---: | ---: | ---: | ---: |
| 64 724, исходная точка | 0.365887371 | −0.000019777 | −0.000243377 |
| 100 724 | 0.369176225 | +0.000013348 | −0.000224165 |
| 140 408 | 0.369020102 | −0.000046005 | +0.000014433 |

Отдельная модель достигла лучшего показателя в середине продолжения, `+0.003289` к исходной точке. Лучшая смесь на последней точке дала объединённую раннюю NDCG `0.380073291`, на `−0.000024863` ниже точно отправленного ансамбля. На обеих точках один из ранних блоков отрицателен. Порог `+0.001` в каждой части не пройден. На этапе обучения и отбора поздние метки и test не использовались. [Ранние результаты и полный перебор](../experiments/full-attention-directml-024-extra/selection.json).

Оба чекпойнта содержат конечные веса и состояние оптимизатора на CPU. Метрики всех весов и ключи ранних прогнозов для обеих точек независимо пересчитаны. Прошли 80 тестов. [Сверка](../experiments/full-attention-directml-024-extra/verification.txt), [чекпойнты, параметры и прогнозы](../experiments/full-attention-directml-024-extra).

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '6'
$gpuPython = '.\.venv-avito-directml\Scripts\python.exe'
& $gpuPython -B -u experiments/full_attention_ranker.py experiments/repro-full-attention-directml-024-extra --backend directml --resume-from experiments/full-attention-directml-024-run
```

**Пробный сабмит по запросу пользователя.** Собран с заранее сохранённой конфигурацией `90% nn-calibration-joint-024 + 10% Transformer` на 140 408 лентах. Оба scores стандартизованы внутри ленты. Равные значения смеси упорядочены по исходному score ансамбля, затем по `item_id`; итог записан уникальным рангом. Зафиксированы SHA-256 выбора, чекпойнта и исходных прогнозов. Все 544 950 ранних scores Transformer воспроизвелись побитово. Новая поздняя оценка не меняла вес смеси или чекпойнт. [Зафиксированная конфигурация](../experiments/transformer10-submission/configuration.json).

Поздняя NDCG смеси `0.3728952521` против `0.3729571873` у контроля, разница `−0.0000619351`. По временным половинам `−0.000285376 / +0.000160321`, парный 95%-й интервал `[−0.000511574, +0.000388772]`. NDCG@10 `0.2454997611`, NDCG@30 `0.3153494348`. Независимый перерасчёт метрик по ключам подтвердил обе половины и весь поздний блок. Первый кандидат изменился в 2 033 из 28 595 test-лент. [Поздние метрики](../experiments/transformer10-submission/assessment.json), [сверка](../experiments/transformer10-submission/verification.txt).

`ranking.py audit` подтвердил точное совпадение со всеми 2 904 806 парами официального шаблона, конечные float32 scores и ноль лент с ничьими. [Parquet](../experiments/transformer10-submission/submission-transformer10.parquet), [аудит](../experiments/transformer10-submission/ranking_audit.json).

Пользователь загрузил файл и сообщил **Public NDCG `0.36487042`**. Это `−0.0003435318` к `nn-calibration-joint-024` при `−0.0000619351` на позднем val. Вариант с 10% Transformer не улучшил Public. Лучший подтверждённый результат остаётся `0.3652139518` у `nn-calibration-joint-024`.

```powershell
$sequencePython = 'C:\Users\bikra\AppData\Local\Programs\Python\Python312\python.exe'
$gpuPython = '.\.venv-avito-directml\Scripts\python.exe'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '6'
& $gpuPython -B -u experiments/submit_full_attention.py experiments/full-attention-directml-024-extra experiments/repro-transformer10-submission --backend directml
& $sequencePython -B experiments/ranking.py audit experiments/repro-transformer10-submission/submission-transformer10.parquet
```

## Тридцать пятый цикл. Предыдущие появления объявления в случайных лентах

**Протокол до метрик.** Измеряем сигнал из состава недавних лент. Он доступен до отклика на оцениваемую ленту и обновляется внутри каждого опубликованного сплита. Валидируем поток val и собираем те же признаки на отдельном потоке test. Для каждого `timestamp` сначала оцениваем весь блок, затем добавляем его показы в память. Учитываем исключительно более ранние ленты текущего сплита. Метки, порядок кандидатов и исходная `pos` в признаки не входят. Это проверка доступного по порядку лент сигнала, которая предполагает наличие более ранних лент текущего периода.

Делим ID пользователей на четыре группы по остатку от деления. Для текущего пользователя учитываем три другие группы. Это исключает его собственные ленты и часть лент других пользователей. `user_id=0` и нулевой вектор исключают ленту из источников и обнуляют сигнал при оценке. Сравниваем `log1p(n)`, где `n` — число предыдущих появлений объявления, и косинус текущего пользователя со средним вектором пользователей, которым оно ранее попало в кандидаты. Из второго вычитаем косинус со средней предыдущей аудиторией трёх групп и сжимаем результат множителем `n/(n+4)`. Третья ветка усредняет два стандартизованных сигнала.

Отдельные признаки стандартизируем внутри каждой ленты и ограничиваем диапазоном `[-3, 3]`. Прибавляем к стандартизованным оценкам отправленного ансамбля при весах `0.025/0.05/0.10/0.20/0.35/0.50`. Ненулевой вес требует роста полной NDCG на обоих ранних блоках `3 749 / 2 017`. Выбираем по наибольшему минимуму приростов. Только результат не ниже `+0.001` в каждом раннем блоке даёт основание считать дальнейшее масштабирование полезным. После фиксации веса один раз оцениваем поздний val. В test измеряем покрытие без меток; новый сабмит зависит от результата проверки.

**Результат.** Построение признаков заняло 5.10 с на 14 423 val-лентах и 9.78 с на 28 595 test-лентах. В позднем val хотя бы одно более раннее появление доступно у `9.92%` кандидатов и `9.38%` положительных кандидатов. На test покрытие `13.58%`. Для двух или более появлений соответственно `2.34%`, `2.16%` и `4.19%`; для четырёх `0.38%`, `0.30%` и `0.96%`. Эти значения учитывают фильтр пользователей и отсутствие будущих лент в признаках. Покрытие зависит от числа уже обработанных лент сплита. [Val](../experiments/causal-exposure-valtest/val_coverage.json), [test](../experiments/causal-exposure-valtest/test_coverage.json).

| Лучший вес по семейству | Вес | Δ полной NDCG на раннем блоке 1 | Δ на раннем блоке 2 |
| --- | ---: | ---: | ---: |
| Частота предыдущих появлений | 0.025 | +0.000117060 | −0.000542017 |
| Сходство с предыдущими пользователями | 0.025 | +0.000147168 | −0.000548599 |
| Среднее двух сигналов | 0.025 | +0.000080752 | −0.000213696 |

Ни один из 18 ненулевых вариантов не выиграл в обоих блоках. Выбран нулевой вес. Знание эмбеддингов ранее показанных пользователей не дало преимущества к простой частоте по худшему блоку. На 8 657 поздних лентах выбранное правило точно воспроизвело `0.3729571873` NDCG, `0.2456998515` @10 и `0.3153842081` @30. Новый сабмит не создан. [Ранний перебор](../experiments/causal-exposure-valtest/selection.json), [итог и покрытие](../experiments/causal-exposure-valtest/assessment.json).

Прошёл 81 тест. Для val и test сверены все ключи и по 200 кандидатов с прямым расчётом по предыдущим `timestamp`, включая случаи с нулевым покрытием. Все 1 389 971 сохранённых val-score совпадают с отправленным контролем. Поздние метрики подтверждены независимым вычислением. [Сверка](../experiments/causal-exposure-valtest/verification.txt).

```powershell
$sequencePython = 'C:\Users\bikra\AppData\Local\Programs\Python\Python312\python.exe'
$env:OPENBLAS_NUM_THREADS = '1'
& $sequencePython -B -m unittest discover -s experiments
& $sequencePython -B -u experiments/causal_exposure.py experiments/repro-causal-exposure-valtest
```

## Тридцать шестой цикл. Продолжение бустинга на втором блоке шардов

**Состояние.** Последний загруженный файл `transformer10-submission` получил Public `0.36487042` и уступил `nn-calibration-joint-024` на `0.0003435318`. Лучший подтверждённый Public остаётся `0.3652139518` при позднем val `0.3729571873`. Этот цикл проверяет не новый сигнал, а более полное использование уже обученной сильнейшей модели деревьев: продолжение `lgb-binary-intent-048` на 48 шардах, не встречавшихся при её обучении (`experiments/cache-intent-096`, seed `2027`, 295 247 лент, 23 119 839 кандидатов). Первые 160 деревьев заморожены, добавленные обучаются теми же параметрами и seed, число деревьев выбирается по полному NDCG раннего val, поздний val открывается один раз. Массивы, прогнозы и чекпойнты исключены из Git; в репозитории сохранены [параметры и лог](../experiments/warm-start-096/metrics.json), [модель](../experiments/warm-start-096/model.txt) и [сравнение](../experiments/warm-start-096-assessment/assessment.json).

**Измерено.** Лучшая точка — 120 добавленных деревьев из 160, `280` всего. Одиночная модель дала ранний NDCG `0.373398` и поздний `0.369146` против `0.372035` и `0.368395` у источника, то есть продолжила его на обоих блоках. Смесь с отправленным ансамблем выбиралась по двум ранним блокам и на сетке `0.025…0.50` остановилась на `0.05`: поздний прирост `+0.000133` при интервале `[-0.000087, +0.000390]`, но первый ранний блок `-0.0000955`. По принятому правилу вес не допущен, новый Parquet не создавался. [Полный перебор весов](../experiments/warm-start-096-assessment/assessment.json).

**Не проверенная постановка.** Добавление 5% поверх ансамбля тратит преимущество продолжения на почти ортогональную компоненту: при весе `0.2` внутри ансамбля деревьев модель-источник входит в те же стандартизованные scores. Поэтому следующая проверка — **подстановка**: пересобрать `lgb-ensemble-weighted-intent-048` с `warm-start-096` вместо `lgb-binary-intent-048` и прежними весами. Контроль известен: поздний `0.368731`, `NDCG@10` `0.240497`, `NDCG@30` `0.310109`. Возможный прирост ограничен весом компонента `0.2` и корреляцией с источником, поэтому ожидать `+0.001` здесь нельзя; смысл проверки в том, что подстановка не меняет состав и веса, а только заменяет компонент строго доминирующей моделью.

```powershell
$rankingPackages = @('--no-project', '--python', '3.11', '--with', 'numpy==1.26.4', '--with', 'polars==1.14.0', '--with', 'pyarrow==19.0.1', '--with', 'lightgbm==4.6.0')
uv run @rankingPackages python experiments/warm_start.py swap experiments/cache-intent-024 experiments/repro-warm-start-substitution `
  experiments/lgb-xendcg-intent-024 experiments/lgb-xendcg-priors-024 experiments/lgb-debiased-024 `
  experiments/lgb-binary-intent-024 experiments/lgb-binary-priors-024 experiments/lgb-xendcg-intent-048 `
  --weights .12 .12 .12 .12 .12 .2 `
  --replace experiments/lgb-binary-intent-048 --replacement experiments/warm-start-096
```

Команда пишет `swap.json` с приростами по двум ранним блокам, обеим поздним половинам, длинам лент и парным интервалом. Принятое правило: положительный прирост в обоих ранних блоках и обеих поздних половинах, поздний прирост не меньше `+0.001` либо положительная нижняя граница интервала. Только после этого имеет смысл пересобирать цепочку калибровок (`nn-calibration-extra` → `nn-calibration-stratified` → `nn-calibration-joint`) на новых прогнозах ансамбля деревьев. Если подстановка нейтральна, следующий дешёвый шаг той же линии — продолжить на блоке `2027` второй компонент на 48 частях, `lgb-xendcg-intent-048`, и повторить подстановку.

Сам `swap` в этом цикле не запускался: в репозитории нет датасета, кэшей `val.npz` и прогнозов компонентов. Тесты `experiments/test_warm_start.py` дополнены проверками нормализации весов, отказа при отрицательном весе и правила допуска; они проходят без LightGBM.
## Тридцать седьмой цикл. CatBoost как отдельное семейство деревьев

**Почему это новое.** Все деревья в отправленном ансамбле — LightGBM. Свой CatBoost в ансамбле не обучался ни разу: есть только официальный бейзлайн организаторов (`baseline/catboost_baseline.py`, YetiRank на девяти признаках и полном train), а в независимых сверках циклов 28–30 CatBoost не запускался из-за отсутствия пакета в локальном Python. Симметричные деревья, упорядоченные статистики категорий и другой поиск разбиений дают ошибки, слабо коррелирующие с LightGBM, — это самый дешёвый оставшийся источник разнообразия при тех же 79 признаках и том же train-кэше.

**Протокол до метрик.** Вход — существующий кэш (`x`, `target`, `counts`) и та же нарезка признаков `history-intent`, поэтому ни один признак не пересчитывается. Параметры и seed фиксированы: `Logloss`, глубина `8`, `lr 0.1`, `l2_leaf_reg 3`, `border_count 127`, Bernoulli-бутстрэп `0.9`, seed `2026`. Число деревьев выбирается по полному NDCG раннего val с шагом `200` (`--checkpoint`), поздний val открывается один раз для выбранной точки. Категориальные колонки (`category`, `intent_match_type`, `intent_match_source`) по умолчанию входят как числовые коды: массив кэша имеет `float32`, а CatBoost требует для категорий целочисленные колонки. Флаг `--categoricals` собирает `DataFrame` с `int32` в этих трёх колонках и включает обработку категорий, но требует копию матрицы (примерно +7,5 ГиБ на 48 частях).

Порядок проверки: сначала дешёвый прогон на готовой подвыборке `experiments/feed-width-048/random.npz` (3 575 775 строк, 295 376 лент), затем полные 48 частей. Вес смеси с отправленным ансамблем выбирается на двух ранних блоках (`warm_start.py assess`, сетка `0.025…0.5`); для тестового файла используется `neural_blend.py predict`, который смешивает уже готовые score-файлы. Критерий прежний: `+0.003` позднего NDCG, либо от `+0.001` с положительной нижней границей парного интервала и ростом в обеих ранних частях и обеих поздних половинах. `select` перебирает свою сетку `0.15…1.0` на первых ранних 3 749 лентах; решает более тонкая сетка `assess` на двух ранних блоках, и в `comparison.json` перед `predict` подставляются именно выбранные там веса (`weights`, `normalization: zscore`), иначе тестовый файл соберётся с другим весом, чем проверенный.

```powershell
$rankingPackages = @('--no-project', '--python', '3.11', '--with', 'numpy==1.26.4', '--with', 'polars==1.14.0', '--with', 'pyarrow==19.0.1', '--with', 'lightgbm==4.6.0', '--with', 'catboost==1.2.10')
# 1. Дешёвый прогон на подвыборке строк.
uv run @rankingPackages python experiments/catboost_ranker.py train experiments/cache-intent-024 experiments/repro-catboost-random-048 `
  --train-file experiments/feed-width-048/random.npz --rounds 800 --checkpoint 200 --threads 6
# 2. Полные 48 частей, только если первый прогон дал ненулевой вес.
uv run @rankingPackages python experiments/catboost_ranker.py train experiments/cache-intent-024 experiments/repro-catboost-048 --rounds 800 --checkpoint 200 --threads 6
# 3. Вес смеси по двум ранним блокам и парный интервал.
uv run @rankingPackages python experiments/warm_start.py assess experiments/repro-catboost-048 experiments/nn-calibration-joint-024/val_predictions.parquet experiments/repro-catboost-weight
# 4. Тестовые прогнозы члена и итоговая смесь, если вес допущен.
uv run @rankingPackages python experiments/catboost_ranker.py predict experiments/cache-intent-024 experiments/repro-catboost-048/model.cbm experiments/repro-catboost-048/test_predictions.parquet
uv run @rankingPackages python experiments/neural_blend.py select experiments/cache-intent-024 experiments/repro-catboost-blend --baselines experiments/nn-calibration-joint-024 --neural experiments/repro-catboost-048
uv run @rankingPackages python experiments/neural_blend.py predict experiments/cache-intent-024 experiments/repro-catboost-blend/comparison.json submission-catboost.parquet experiments/nn-calibration-joint-024/submission-reserve.parquet experiments/repro-catboost-048/test_predictions.parquet
uv run @rankingPackages python experiments/ranking.py audit submission-catboost.parquet
```

**Что проверено, а что нет.** Код `experiments/catboost_ranker.py` прогнан сквозным образом на синтетическом кэше (3 000 обучающих и 6 000 val-лент, 79 признаков, обе ветки — с категориями и без): обучение, выбор контрольной точки, `model.cbm`, val-прогнозы, `metrics.json`, тестовый прогноз и `ranking.py audit` с нулём ничьих. Юнит-тесты `experiments/test_catboost_ranker.py` проверяют нарезку признаков, индексы категорий, отказ при узком кэше и правило выбора контрольной точки; они проходят без установленного CatBoost. Реальные метрики NDCG не считались: в репозитории нет датасета, кэшей и прогнозов. Ожидаемый порядок величины для члена с весом около `0.05…0.2` — `+0.0005…+0.0015` позднего NDCG, то есть попытка Public оправдана только при выполнении общего критерия.

**Время и память.** Матрица 48 частей — 23,1 млн строк и 7,5 ГиБ в `float32`; CatBoost на шести потоках заметно медленнее LightGBM на то же число итераций, поэтому первый прогон делается на подвыборке. `--train-file` принимает любой кэш той же схемы, так что повтор эксперимента воспроизводим без пересчёта признаков.
