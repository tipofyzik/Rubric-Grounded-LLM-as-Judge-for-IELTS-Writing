---
title: IELTS Stage 1 Smoke
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
python_version: '3.10'
app_file: App.py
models:
  - Qwen/Qwen2.5-1.5B-Instruct
---

# Практическая проверка этапа 1

Этот технический прототип запускает настоящий Qwen через новый `Judge.predict`.
Он принимает редактируемые task/essay и показывает completion + diagnostics.
Полный hybrid, реальный AES, Supabase retrieval и строгая проверка scores здесь
не запускаются. Первые успешные hosted результаты нового пути ещё нужно получить.

## Как установить в существующий Space

1. Из корня проекта выполните `python -B code/offline_research/tools/BuildStage1Space.py`.
2. Распакуйте `TEMP/ielts_stage1_space.zip`. Та же готовая структура есть в
   `TEMP/ielts_stage1_space/`. Загружать нужно **всё содержимое**, сохраняя папки.
3. Откройте существующий private Space `tipofyzik/ielts-writing-evaluator`, вкладку
   **Files**. Загрузите файлы в корень Space: замените `App.py`, `requirements.txt`,
   `README.md`; добавьте `Probe.py`, `BUILD_MANIFEST.json` и вложенную папку `code/`.
   ZIP как один файл Space автоматически не распакует.
4. Сохраните Gradio SDK и ZeroGPU hardware. Дождитесь **Running** и откройте **App**.

Для этого probe не нужны новые secrets. Существующие Supabase variables/secrets
не используются. Сборщик включает только нужные исходники и YAML, без данных,
весов, паролей, research outputs и tests. Stage 0 сохранён локально в
`code/offline_research/legacy/stage0_space/` и прежнем `TEMP/ielts_stage0_space.zip`; его три файла можно
вернуть в корень Space для отката.

## Что проверить вручную

| Проверка | Действие | Ожидаемое наблюдение |
|---|---|---|
| Первый запрос | Оставить sample, fixture выключен; нажать Run Stage 1 judge | Непустой Raw completion; `generated_tokens > 0`, `context_mode: empty`, правильный `model_id`, model revision и GPU name |
| Содержимое ответа | Посмотреть Raw completion | Для удачного ответа — один JSON с `task_response`, `coherence_and_cohesion`, `lexical_resource`, `grammatical_range_and_accuracy`; failure/обрезанный JSON сохранить как неуспех |
| Повторный запрос | Нажать кнопку повторно | Новый `probe_run_id`, тот же `submission_sha256`; текст может совпасть при `do_sample=False` |
| Собственный ввод | Заменить task и essay своим текстом и запустить | Новый `submission_sha256` и новый run ID; старый результат очищается при редактировании |
| Другое задание | Оставить essay, заменить task на несвязанный вопрос | Меняется `submission_sha256`; генерация выполняется снова. Снижение TR желательно для качества, но не является гарантией этого smoke |
| Prepared context | Включить synthetic context и запустить | `context_mode: synthetic_fixture`, `reference_ids: [synthetic-reference-1]`; новый run ID. Это тест передачи искусственного контекста, не retrieval/AES |
| Непустой ввод | Удалить task или essay и запустить | Понятная ошибка непустого поля; прежний ответ не остаётся на экране |
| Token budget | Вставить очень длинное essay и запустить | Сообщение с фактическими токенами и `limit is 3840`; без скрытого обрезания. GPU allocation может произойти до этого отказа |
| После rebuild | Перестроить Space и повторить первый запрос | Генерация снова работает, `loaded_model_revision` совпадает с `model_revision`; сохранить новые diagnostics |

`stop_reason: max_new_tokens` не означает успешный полный ответ. Если JSON
обрезан, сохранить completion и diagnostics: дальше проверяются prompt, output
budget и stopping behavior. Сами scores не являются подтверждением качества IELTS.
Не требуйте обязательного изменения scores при любом изменении текста: технически
важны новый запуск и соответствующий вход; чувствительность оценок — quality test.

Сначала достаточно выполнить первый запрос, повтор, собственный ввод и fixture.
Остальные случаи — расширенная проверка, когда хватает visitor quota. Зафиксируйте
queue/allocation wait и показанный HF расход квоты отдельно; `gpu_work_seconds`
их не измеряет. Ограничение одной GPU функции — 60 секунд, output budget — 256
токенов, total context budget — 4096.

## Что сохранить для разбора

Для каждого сценария скопируйте **Raw completion** и весь **Diagnostics**, указав
название сценария. Для своего эссе сам текст передавать необязательно. Для ошибки
сохраните сообщение интерфейса; при startup/build failure — соответствующий
фрагмент build/runtime logs. Из логов исключите secrets, если добавляли их сами.

В diagnostics есть fingerprint исходников `source_sha256`, новая метка запуска,
hash task/essay, model/prompt/schema revisions, токены, stop reason, GPU metrics
и версии пакетов. Прототип не пишет пользовательские тексты в собственные
persistent logs и не создаёт историю оценок. Эти поля не заменяют независимую
проверку deployment и поведения самой платформы.

## Границы локальной проверки

Builder проверяет состав ZIP, CRC и SHA-256 manifest. Локальные tests проверяют
probe → настоящий Judge → общий generation path с контролируемой моделью и запуск
core из распакованного bundle. Gradio/spaces и настоящий hosted Qwen этим не
подтверждаются: они проверяются описанными выше действиями в Space.

События `.then`, `change` и формат label/value в dropdown сверены с исходниками
закреплённой версии: [Gradio 5.49.1 events](https://github.com/gradio-app/gradio/blob/gradio%405.49.1/gradio/events.py),
[Dropdown](https://github.com/gradio-app/gradio/blob/gradio%405.49.1/gradio/components/dropdown.py).
