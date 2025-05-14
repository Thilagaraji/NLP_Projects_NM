from flask import Flask, request, jsonify
from transformers import DialoGPT-medium
import torch

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained DialoGPT model
model = DialoGPT-medium.from_pretrained("microsoft/DialoGPT-medium")
tokenizer = DialoGPT-medium.from_pretrained("microsoft/DialoGPT-medium")

# Route to handle chat requests
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = new_user_input_ids

    # Generate a response from the model
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode the response
    bot_output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    return jsonify({"response": bot_output})

if __name__ == "__main__":
    app.run(debug=True)
