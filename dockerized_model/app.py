from flask import Flask, request
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import pickle
import tensorflow as tf
import json

version = '0.0.1'

app = Flask(__name__)

model = TFAutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased", num_labels=41)
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

# Load the saved weights into the model
saved_weights_path = './model-004.h5' 
model.load_weights(saved_weights_path)

#Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

#Load the encoder
pkl_file = open('label_encoder.pkl', 'rb')
encoder = pickle.load(pkl_file) 
pkl_file.close()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    
    # Handling unexpected input
    try:
        data = request.get_json()
        payload = data["text"].lower()
    except:
        my_dict = {"label": "Unknown", "confidence": 1, "version": version}
        return json.dumps(my_dict)
    if payload == "":
        return json.dumps(my_dict)
    
    # Expected input: {"text": "loren ipsum wahtever"}
    payload = data["text"].lower()
    
    # Get label
    inputs = tokenizer(payload, return_tensors="tf")
    logits = model(**inputs).logits
    predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
    label = encoder.classes_[predicted_class_id]
    
    # Get confidence
    probabilities = tf.nn.softmax(logits)
    conf = max(probabilities[0]).numpy()
    
    # Create output dictionary
    my_dict = {"label": label, "confidence": float(conf), "version": version}
    
    return json.dumps(my_dict)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug = False)