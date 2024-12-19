from os.path import split
from uuid import uuid4

from flask import Flask, jsonify, request

import json

import requests
import uuid
import psycopg2
import whisper
import torch
import secrets

from flask_cors import CORS
from dotenv import load_dotenv

import os

load_dotenv()

API_KEY = os.getenv("API_KEY")

xsession = str(uuid.uuid4())

pref_problem_description = ""

def get_creds():
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

    payload = 'scope=GIGACHAT_API_PERS'
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': f'{str(uuid.uuid4())}',
        'Authorization': f'Basic {API_KEY}'
    }

    response = requests.request("POST", url, headers=headers, data=payload, verify=False)

    creds = str(response.json().get('access_token'))

    return creds


def infer(messages):
    url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

    payload = json.dumps({
        "model": "GigaChat",
        "messages": messages,
        "stream": False,
        "repetition_penalty": 1,
        'temperature': 0.0
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'X-Session-ID': xsession,
        'Authorization': f'Bearer {get_creds()}'
    }

    response = requests.request("POST", url, headers=headers, data=payload, verify=False)
    print(response.json())
    return response.json().get('choices')[0]['message']['content']


def get_known_problems(category, address):
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="devpass",
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM problems WHERE category = %s AND address = %s", (category, address))
    records = cursor.fetchall()

    rows = []
    for record in records:
        rows.append({'id': record[0], 'description': record[1], 'answer': record[2], 'category': record[3],
                     'address': record[4]})

    return rows

def get_known_info(category, address): #TODO: make DB
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="devpass",
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM problems WHERE category = %s AND address = %s", (category, address))
    records = cursor.fetchall()

    rows = []
    for record in records:
        rows.append({'id': record[0], 'description': record[1], 'answer': record[2], 'category': record[3],
                     'address': record[4]})

    return rows


def add_new_problem(description, answer, category, address):
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="devpass",
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()
    id = secrets.randbelow(10**18)
    cursor.execute("INSERT INTO problems (id, description, answer, category, address) VALUES (%s, %s, %s, %s, %s)",
                   (secrets.randbelow(10**18), description, answer, category, address))

    conn.commit()
    cursor.close()
    conn.close()

    return id

def define_importance(system_prompt, message):
    prompt = [system_prompt,
              {
                  'role': 'user',
                  'content': 'При сильном ветре кусты бьют в окна 3 этажа. Заявка в ДУК на опил была продана 5 июля 2024 года, но по настоящий момент проблема не решена.',
              },
              {
                  'role': 'assistant',
                  'content': 'Плановая',
              },
              {
                  'role': 'user',
                  'content': 'Сильно течёт крыша после капитального ремонта, видно небо.  Течёт сильно над квартирой где сейчас идет капитальный ремонт. Просим в срочном порядке устранить проблему .'
              },
              {
                  'role': 'assistant',
                  'content': 'Экстренная'
              },
              {
                  'role': 'user',
                  'content': 'Тротуары почти не чистят от снега. И не посыпают. Недавние заморозки показали, что с прошлого сезона ничего не изменилось. Даже во дворах не посыпают. Про тротуары вдоль дорог вообще молчу. Как зима, невозможно передвигаться. Конкретного адреса нет, так как проблема системная по всему городу.'
              },
              {
                  'role': 'assistant',
                  'content': 'Плановая'
              },
              {
                  'role': 'user',
                  'content': 'Целый день затапливает подвал кипятком-ни одна служба не реагирует!вызывали аварийную службу-никакого толка!примите меры!'
              },
              {
                  'role': 'assistant',
                  'content': 'Экстренная'
              },
              {
                  'role': 'user',
                  'content': 'Прошу вас разъяснить жителям кстовского района необходимость присоединения его к Нижнему Новгороду. Люди недовольны, не видят плюсов. Для чего это было сделано? Весь ли кстовский район присоединится или только часть?'
              },
              {
                  'role': 'assistant',
                  'content': 'Плановая'
              },
              {
                  'role': 'user',
                  'content': 'Три дня назад в доме  прорвало  общедомовую канализационную трубу между потолком музея и полом квартиры, стояк прогнил, ДУК на обращения не реагирует, воду перекрыли, три дня без воды'
              },
              {
                  'role': 'assistant',
                  'content': 'Экстренная'
              },
              {
                  'role': 'user',
                  'content': f"""Есть такое обращение "'{message}'"\n Как и в обращениях выше определи тип обращения (экстренная заявка или же плановая), учитывая, необходимо ли быстрое решение проблемы. Если быстрое решение необходимо, то напиши 'Экстренная' иначе 'Плановая'"""
              },
              ]
    return infer(prompt)

def define_message_type(system_prompt, message):
    prompt = [system_prompt,
              {
                  'role': 'user',
                  'content': f"""Вы аналитик обращений граждан, связанных с инфраструктурой и вопросами городской администрации. Ваша задача — классифицировать каждое сообщение пользователя в одну из двух категорий:
            1.	Обращение — сообщение описывает конкретную проблему, связанную с инфраструктурой, коммунальными службами или качеством жизни, которую необходимо решить. Примеры: аварии, неисправности, отсутствие услуг (воды, электричества и т. д.), проблемы с дорогами, жилищным фондом или коммунальными организациями.
            2.	Информация — сообщение содержит просьбу разъяснить или уточнить информацию по действиям властей, планам, событиям или другим вопросам без указания конкретной проблемы, требующей устранения.
            3.  Инструкция — сообщение содержит просьбу предоставить план действий для достижения какого-либо результата.

          Прочитайте сообщение пользователя и определите, к какой категории оно относится. При необходимости дайте краткое обоснование вашего выбора.

          Примеры:
            1.	Сообщение: “Три дня назад в доме прорвало общедомовую канализационную трубу между потолком музея и полом квартиры, стояк прогнил, ДУК на обращения не реагирует, воду перекрыли, три дня без воды.”
          Категория: Обращение
          Обоснование: Сообщается о конкретной аварии, связанной с канализацией, которая требует действий.
            2.	Сообщение: “Прошу вас разъяснить жителям Кстовского района необходимость присоединения его к Нижнему Новгороду. Люди недовольны, не видят плюсов. Для чего это было сделано? Весь ли Кстовский район присоединится или только часть?”
          Категория: Информация
          Обоснование: Пользователь интересуется действиями властей и их обоснованием, не указывает на проблему, требующую устранения.
            3.  Сообщение: "Как мне оформить проездной?"
          Категория: Инструкция
          Обоснование: Пользователь хочет получить план действий для получения чего-то, не указывает на проблему, требующуюустранения и не просит разъяснить ему действия властей.
            4.	Сообщение: “На улице Лесной уже две недели не работает освещение, в темноте небезопасно ходить. Просим срочно починить.”
          Категория: Обращение
          Обоснование: Сообщается о неисправности освещения, требующей ремонта.
            5.	Сообщение: “Когда планируется завершение ремонта дороги на улице Горького? Какие участки еще будут закрыты в ближайшие месяцы?”
          Категория: Информация
          Обоснование: Пользователь запрашивает информацию о сроках и планах, не сообщает о проблеме.
            6. Сообщение: "Что нужно сделать, чтобы оформить инвалидность?"
          Категория: Инструкция
          Обоснование: Пользователю интересно узнать последовательность действий для получения желаемого результата.

          Сообщение для анализа:
          “{message}”

          Требуемый результат:
          Укажите категорию: 'Обращение', 'Информация' или 'Инструкция' без слова категория (это важно!).
          """,
              },
              ]
    return infer(prompt)

def create_application(message, system_prompt, address):
    problem_category = define_problem_category(system_prompt, message)

    known_problems = get_known_problems(problem_category, address)
    known_problems_descriptions = [known_problem['description'] for known_problem in known_problems]
    prompt = [system_prompt,
              {
                  'role': 'user',
                  'content': f"""Есть проблема: {message}\nНайди похожую проблему в моём доме из списка известных проблем: {known_problems_descriptions}.
            Если нашлась близкая по смыслу проблема, ответь текстом этой проблемы, иначе ответь 'Нет совпадений'"""
              }
              ]  # Проверить, что ответит, если будут похожие заявления известные

    known_problem = infer(prompt)

    if known_problem == 'Нет совпадений':
        prompt = [system_prompt,
                  {
                      'role': 'user',
                      'content': f'Есть запрос пользователя: {message}\nСформулируй на его основе заголовок, состоящий не более чем из 5 слов и отражающий суть этого заявления. Заголовок составляй только на основе указанной в заявлении информации.'
                  }]
        return infer(prompt), problem_category

    known_problems_answers = [known_problem['answer'] for known_problem in known_problems]
    answer_to_problem = known_problems_answers[known_problems_descriptions.index(known_problem)]

    prompt = [system_prompt,
              {
                  'role': 'user',
                  'content': f"""Сформулируй следующий ответ простыми словами: {answer_to_problem}"""
              }]

    return infer(prompt)

def create_instruction(system_prompt, message):
    prompt = [system_prompt,
              {
                  'role': 'user',
                  'content': f"""
                    Вы — эксперт, помогающий гражданам Нижнего Новгорода решить проблемы, связанные с городской инфраструктурой и коммунальными услугами. На вход поступает сообщение от пользователя, описывающее конкретную ситуацию или проблему. Ваша задача — выдать четкую и понятную инструкцию (план действий), которая поможет пользователю решить описанную проблему.

                    При составлении инструкции:
                      1.	Учитывайте специфику проблемы, указанной пользователем.
                      2.	Если требуется обращение в соответствующие службы или органы, укажите их наименование, контактные данные (если известны), а также форму обращения (звонок, заявление, онлайн-запрос и т. д.).
                      3.	Добавьте шаги, которые пользователь может предпринять самостоятельно (например, сбор документов, фотографирование проблемы).
                      4.	Предложите временные или альтернативные меры, если проблема не может быть решена немедленно.
                      5.	Будьте лаконичны, но максимально конкретны.
    
                    Пример:
                      1.	Сообщение: “Три дня назад в доме прорвало общедомовую канализационную трубу между потолком музея и полом квартиры, стояк прогнил, ДУК на обращения не реагирует, воду перекрыли, три дня без воды.”
                    Инструкция:
                      •	Сделайте фотографии повреждений, зафиксируйте проблему на видео, если возможно.
                      •	Обратитесь в ДУК повторно. Направьте письменное заявление через их официальный сайт или почтой с описанием проблемы и приложением фотографий. Сохраните копию обращения.
                      •	Если реакции от ДУК нет в течение суток, подайте жалобу в Госжилинспекцию Нижнего Новгорода. Это можно сделать онлайн через портал Госуслуг или отправить заявление в бумажном виде.
                      •	Сообщите о проблеме в аварийную службу города по телефону (укажите телефон).
                      •	Если ситуация угрожает безопасности, вызовите представителей МЧС или местной администрации для оперативного решения проблемы.
                      2.	Сообщение: “На улице Лесной уже две недели не работает освещение, в темноте небезопасно ходить. Просим срочно починить.”
                    Инструкция:
                      •	Подайте обращение в администрацию вашего района через официальный сайт или по телефону (укажите контакты). В заявлении опишите проблему и точное местоположение.
                      •	Сообщите о проблеме в муниципальное предприятие, ответственное за освещение (укажите контакты).
                      •	Если проблема не решается в течение недели, направьте жалобу в управление благоустройства города.
                      •	Для быстрого контроля ситуации рекомендуем подключиться к группе вашего района в соцсетях и оставить сообщение там — это может ускорить решение проблемы.
    
                    Сообщение для анализа:
                    “{message}”
    
                    Требуемый результат:
                    Составьте подробную инструкцию (план действий), подходящую для решения указанной проблемы.
                    """
              }]
    return infer(prompt)

def define_problem_category(system_prompt, message):
    possible_problem_types = ['Отопление', 'Электроснабжение', 'Благоустройство', 'Водоснабжение',
                              'Дворы и территории общего пользования']
    prompt = [system_prompt,
              {
                  'role': 'user',
                  'content': f"""{message}\nОпредели тип проблемы, о которой
          я хочу сообщить - она относится к следующему списку возможных типов проблем: {possible_problem_types}.
    В качестве ответа приведи значение, как оно указано в списке без кавычек."""
              }
              ]
    return infer(prompt)

# def create_information(system_prompt, message, address): #TODO: create information
#     category = define_problem_category(system_prompt, message)
#
#     prompt = [system_prompt,
#               {
#                   'role': 'user',
#                   'content': f"""Имеется список со справочной информацией о предпринимаемых городской организацией мероприятиях: {}.
#                   Найди в этом списке наиболее релевантную для пользователя информацию, основываясь на его запросе: {message}.
#                   Дай пользователю ответ на основе обнаруженной тобой информации.'"""
#               }
#               ]
#     return infer(prompt)


def analize_message(message, address):
    system_prompt = {
        'role': 'system',
        'content': f"""Ты - личный ассистент жителя Нижнего Новгорода. К тебе можно обратиться, чтобы пожаловаться на проблему по месту жительства."""
    }

    importance = define_importance(system_prompt, message)
    print(importance)

    if importance == "Экстренная":
        prompt = [system_prompt,
                  {
                      'role': 'user',
                      'content': f"""
                                  Вы — эксперт, помогающий гражданам Нижнего Новгорода решить проблемы, связанные с городской инфраструктурой и коммунальными услугами. На вход поступает сообщение от пользователя, описывающее конкретную ситуацию или проблему. Ваша задача — выдать четкую и понятную инструкцию (план действий), которая поможет пользователю решить описанную проблему.

                                  При составлении инструкции:
                                    1.	Учитывайте специфику проблемы, указанной пользователем.
                                    2.	Если требуется обращение в соответствующие службы или органы, укажите их наименование, контактные данные (если известны), а также форму обращения (звонок, заявление, онлайн-запрос и т. д.).
                                    3.	Добавьте шаги, которые пользователь может предпринять самостоятельно (например, сбор документов, фотографирование проблемы).
                                    4.	Предложите временные или альтернативные меры, если проблема не может быть решена немедленно.
                                    5.	Будьте лаконичны, но максимально конкретны.

                                  Пример:
                                    1.	Сообщение: “Три дня назад в доме прорвало общедомовую канализационную трубу между потолком музея и полом квартиры, стояк прогнил, ДУК на обращения не реагирует, воду перекрыли, три дня без воды.”
                                  Инструкция:
                                    •	Сделайте фотографии повреждений, зафиксируйте проблему на видео, если возможно.
                                    •	Обратитесь в ДУК повторно. Направьте письменное заявление через их официальный сайт или почтой с описанием проблемы и приложением фотографий. Сохраните копию обращения.
                                    •	Если реакции от ДУК нет в течение суток, подайте жалобу в Госжилинспекцию Нижнего Новгорода. Это можно сделать онлайн через портал Госуслуг или отправить заявление в бумажном виде.
                                    •	Сообщите о проблеме в аварийную службу города по телефону (укажите телефон).
                                    •	Если ситуация угрожает безопасности, вызовите представителей МЧС или местной администрации для оперативного решения проблемы.
                                    2.	Сообщение: “На улице Лесной уже две недели не работает освещение, в темноте небезопасно ходить. Просим срочно починить.”
                                  Инструкция:
                                    •	Подайте обращение в администрацию вашего района через официальный сайт или по телефону (укажите контакты). В заявлении опишите проблему и точное местоположение.
                                    •	Сообщите о проблеме в муниципальное предприятие, ответственное за освещение (укажите контакты).
                                    •	Если проблема не решается в течение недели, направьте жалобу в управление благоустройства города.
                                    •	Для быстрого контроля ситуации рекомендуем подключиться к группе вашего района в соцсетях и оставить сообщение там — это может ускорить решение проблемы.

                                  Сообщение для анализа:
                                  “{message}”

                                  Требуемый результат:
                                  Составьте подробную инструкцию (план действий), подходящую для решения указанной проблемы.
                                  """
                  }]
        return {'message':infer(prompt)}

    message_type = define_message_type(system_prompt, message).lower()
    print("MESSAGE TYPE", message_type)

    if "обращение" in message_type:
        data, problem_category = create_application(message, system_prompt, address)
        return {'message': data, 'problem_category':problem_category}
    # elif "информация" in message_type:
    #     return {'message': create_information(system_prompt, message)}
    else:
        return {'message': create_instruction(system_prompt, message)}

def transform_audio_to_text(audio_file_path):
    whisper_model = whisper.load_model('medium',
                                       device='cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_sil, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                             model='silero_tts',
                                             language='ru',
                                             speaker='v3_1_ru')
    model_sil.to(device)
    transcription = whisper_model.transcribe(audio_file_path)
    return transcription["text"]

app = Flask(__name__)

CORS(app)

@app.route('/api/send_message', methods=['POST'])
def send_message():
    content = request.get_json()
    return jsonify({'data':analize_message(content['message'],content['address'])}), 200


@app.route('/api/save_application', methods=['POST'])
def save_application():
    content = request.get_json()

    return jsonify({'id': add_new_problem(content['description'], content['answer'], content['category'], content['address'])}), 200

@app.route('/api/send_audio', methods=['POST'])
def send_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file part"}), 400

    audio_file = request.files['audio']

    audio_file_path = r"C:\Users\1\Desktop\voice\gorkycode\audio\test.wav"
    audio_file.save(r"C:\Users\1\Desktop\voice\gorkycode\audio\test.wav")

    message = transform_audio_to_text(audio_file_path)

    content = request.form.get('address')

    return jsonify({'data':''}, 200)

if __name__ == '__main__':
    app.run(debug=True, port=5002)