# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch

# Hugging Face model and tokenizer initialization
model_name = "bigscience/bloom"
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom")


# Example input
text = "Sample input text"
inputs = tokenizer(text, return_tensors="pt")

# Perform inference to get model outputs
outputs = model(**inputs)

# Export the model to ONNX format with opset version 14
onnx_model_path = "model.onnx"
torch.onnx.export(
    model,
    (inputs['input_ids'],),
    onnx_model_path,
    input_names=["input_ids"],
    output_names=["output"],
    dynamic_axes={"input_ids": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=14
)
