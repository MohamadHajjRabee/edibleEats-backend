from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from io import BytesIO
import numpy as np
import base64

app = Flask(__name__)


@app.route('/')
def index():
    return 'Hello from Flask!'


interpreter = tf.lite.Interpreter(
    model_path="plant_classification_model8.tflite")
interpreter.allocate_tensors()


@app.route('/predict', methods=['POST'])
def predict():
    try:
        request_data = request.json
        base64_image = request_data['image']

        image_bytes = base64.b64decode(base64_image)
        # image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # img_array = np.array(image)
        # img = Image.fromarray(img_array)
        # resized_image = img.resize((224, 224))
        # img_array = tf.keras.utils.img_to_array(resized_image)
        # img_array = np.expand_dims(img_array, axis=0)
        # img_array /= 255.0

        # tensor = tf.convert_to_tensor(img_array)
        img = image.load_img(BytesIO(image_bytes), target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        interpreter.set_tensor(interpreter.get_input_details()[0]["index"],
                               img_array)

        interpreter.invoke()

        output = interpreter.tensor(interpreter.get_output_details()[0]['index'])
        result = output()[0]
        predicted_class = np.argmax(result)
        print(f'Predicted class: {predicted_class}')
        print(f'Confidence: {result[predicted_class]:.2f}')

        return jsonify({"prediction": predicted_class})

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": "Internal Server Error"}), 500


if __name__ == '__main__':
    app.run()
