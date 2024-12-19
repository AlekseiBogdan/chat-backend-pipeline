# City Assistant

**City Assistant** — это сервис для обработки сообщений граждан, касающихся инфраструктуры и коммунальных услуг города. Система позволяет анализировать текстовые и голосовые сообщения, классифицировать их, генерировать инструкции для решения проблем, а также предоставлять ответы на обращения и запросы.

## Описание

Система предоставляет следующие функциональности:
- Классификация сообщений (обращения, информация, инструкции)
- Обработка голосовых сообщений с использованием модели Whisper
- Генерация инструкций для пользователей на основе GigaChat от Сбербанка 
- Проверка существующих заявок в базе данных

```plaintext
+-----------------------------+
|     Входные данные:         |
|  Текстовое сообщение или    |
|  Голосовое сообщение        |
+-----------------------------+
     |                    |
     v                    v
Голосовое сообщение  Текстовое сообщение
     |                    |
     v                    |
+----------------+        |
| Преобразование |        | 
|    (Whisper)   |        |
+----------------+        |
     |                    |
     v                    v
+-----------------------------+
| Классификация сообщения:    |
| Обращение, Информация,      |
| Инструкция                  |
+-----------------------------+
             |
             v
+-----------------------------+
| Проверка существующих       |
| заявок в базе данных        |
+-----------------------------+
     |                    |
     v                    v
  Новая заявка      Существующая заявка
     |                    |
     v                    v
+----------------+    +------------------+
| Генерация      |    | Инструкция       |
| решения/ответа |    | из базы данных   |
| (GigaChat)     |    |                  |
+----------------+    +------------------+
             |          |
             v          v
+-----------------------------+
| Вывод: Текстовый ответ      |
| пользователю                |
+-----------------------------+
```
![Flowchart](https://github.com/user-attachments/assets/e22c0c90-cd85-4caf-bf79-e9f647d2a5fe)


## Зависимости:
- Flask - создания веб-сервера
- requests - HTTP-запросы
- whisper - преобразование аудио в текст
- psycopg2 - база данных PostgreSQL используется для хранения информации о проблемах и ответах, хотя база данных пока статична и требует дальнейшей настройки
## Основные компоненты кода:
Используется .env файл для загрузки ключа API (API_KEY), который используется в запросах к GigaChat.
Создается уникальный идентификатор сессии (xsession) с помощью UUID, который будет использоваться для уникальной идентификации сессий пользователя.
В конце кода запускается Flask приложение, которое слушает порт 5002 для входящих запросов.
### Функции для взаимодействия с внешними сервисами:
- get_creds() — делает POST-запрос к внешнему API для получения токена авторизации.
- infer() — взаимодействует с сервисом GigaChat (который используется для обработки текстовых сообщений), чтобы получать ответы на запросы пользователей.
### Функции для работы с базой данных PostgreSQL:
- get_known_problems()
- add_new_problem()
- get_known_info()
### Функции для анализа и классификации сообщений:
- define_importance() — определяет, является ли проблема экстренной или нет.
- define_message_type() — классифицирует сообщение как "обращение", "информация" или "инструкция".
- define_problem_category() — определяет тип проблемы, например, "Отопление", "Электроснабжение", "Благоустройство" и т.д.
- create_application() — создает заявку на основе сообщения пользователя, используя данные о ранее известных проблемах.
### Интерфейсы для работы с аудио:
- transform_audio_to_text() — использует модель Whisper для преобразования аудиофайлов в текст. Это позволяет пользователю отправлять аудиофайлы, которые потом обрабатываются как текстовые запросы.

## Пользовательский интерфейс
See https://github.com/AlekseiBogdan/chat-frontend-app

## Требования к запуску

1. Python 3.11.5
2. Установить библиотеки из requirements.txt (`pip install -r /path/to/requirements.txt`)
3. Получить API-ключ для использования GigaChat (https://developers.sber.ru/), инструкции на сайте. Используйте .env для записи вашего ключа 
4. Установить и настроить ngrok на вашем устройстве (в нашем случае — Windows ПК с 1660ti на борту), **запустить его для 5002 порта**
5. Скопировать адрес из ngrok и передать на фронтенд (каждый раз при запуске ngrok будет генерироваться новый адрес, нужно пробрасывать на фронт для адекватной работы связи фронт<->бэк)
6. Запустить `main_server.py`
7. При первом запуске и использовании голосовых сообщений будет скачен whisper, время выполнения запроса может быть сильно увеличено

