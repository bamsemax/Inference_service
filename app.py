from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import time

MODEL_NAME = 'sergeyzh/rubert-mini-frida'
app = Flask(__name__)
try:
    model = SentenceTransformer(
        MODEL_NAME,
        default_prompt_name=None
    )
except Exception as e:
    model = None


@app.route('/embed', methods=['POST'])
def get_text():
    """
    Эндпоинт для получения эмбеддинга текста
    Input:
        POST запрос с JSON вида
        {"text": "Строка для получения эмбеддинга"}
    Output:
        JSON вида
        {
        result: [1,2,...] - эмбеддинг
        "inference_time": время работы модели
        }
    Коды:
        200 - успех
        400 - неверный запрос
        500 - ошибка загрузки модели
    """
    data = request.get_json()
    if model is None:
        return jsonify({'error': 'model not found.'}), 500
    if data is None:
        return jsonify({'error': 'json expected, but not received'}), 400
    if 'text' not in data:
        return jsonify({'error': 'There is no text_field in json'}), 400
    text = data.get('text')
    if not isinstance(text, str):
        return jsonify({'error': 'str expected in text field'}), 400
    if not text or len(text.strip()) == 0:
        return jsonify({'error': 'Text cannot be empty'}), 400
    try:
        start_time = time.perf_counter()
        answer = model.encode(text)
        end_time = time.perf_counter()
        return jsonify({'result': answer.tolist(),
                        'inference_time': (end_time - start_time) * 1000
                        }), 200
    except Exception as e:
        return jsonify({'error': f'Inference failed: {str(e)}'}), 500


@app.route('/health')
def check_health():
    """
    Эндпоинт для проверки работоспособности сервиса.
    Проверяет отвечает ли сервис на запросы. Загружена ли модель
    Output:
    JSON вида
        {
        "status": 'ok'/'unavailable'
        "model_loaded": bool; Зашружена ли модель
        }
    Коды:
        200 - сервис отвечает, модель загружена
        503 - сервис недоступен
    """
    if model is None:
        return jsonify({
            'status': 'unavailable',
            'model_loaded': False
        }), 503

    return jsonify({
        'status': 'ok',
        'model_loaded': True
    }), 200


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
