from flask import Flask, request, jsonify
from openvino.runtime import Core
from transformers import AutoTokenizer
import numpy as np
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Initialize OpenVINO runtime and load the model
core = Core()
model_path = "./model_ir/model.xml"  # Ensure this path is correct
compiled_model = core.compile_model(model_path, device_name="CPU")

# Initialize the tokenizer for BLOOM
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")

@app.route('/')
def index():
    return "Welcome to the BLOOM model API. Use the /predict endpoint to get predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input text from the request
        data = request.json
        text = data['text']
        logging.debug(f"Received text for prediction: {text}")
        
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="np")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        logging.debug(f"Tokenized input: {input_ids}, {attention_mask}")
        
        # Convert input tensors to the required format for OpenVINO
        input_ids = np.array(input_ids, dtype=np.int32)
        attention_mask = np.array(attention_mask, dtype=np.int32)
        
        # Run inference
        results = compiled_model({compiled_model.inputs[0].any_name: input_ids, compiled_model.inputs[1].any_name: attention_mask})
        logging.debug(f"Inference results: {results}")
        
        # Extract the output
        output_data = results[compiled_model.outputs[0].any_name]
        logging.debug(f"Output data: {output_data}")
        
        # Return the results as JSON
        return jsonify(output_data.tolist())

    except Exception as e:
        # Log the error
        logging.error("Error during prediction: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
