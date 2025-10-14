from flask import Flask, render_template, request, jsonify, session
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your_fallback_secret_key_here')

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Load model and data
try:
    model = load_model('chatbot_model.h5')
    from tensorflow.keras.optimizers import SGD
    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))

    with open('intents.json') as file:
        intents = json.load(file)

    model_loaded = True
    print("Model and data loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False

def clean_up_sentence(sentence):
    """Tokenize and lemmatize the sentence"""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    """Create bag of words array"""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model):
    """Predict the intent class of the sentence"""
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.1
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    """Get random response from matched intent"""
    if ints:
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
    else:
        result = "I'm not sure I understand. Could you please rephrase that?"
    
    return result

def chatbot_response(msg, history=None):
    """Generate chatbot response using Gemini API"""
    try:
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        base_prompt = f"""You are a compassionate mental health support chatbot. Respond to the user's message: '{msg}' with empathy, validation, and helpful suggestions.

Guidelines:
- Validate their feelings: Acknowledge emotions without judgment.
- Use active listening: Reflect back what they said to show understanding.
- Ask open-ended questions: Encourage deeper sharing if appropriate.
- Suggest coping strategies: Offer simple, evidence-based techniques like deep breathing, mindfulness, or self-care.
- Emphasize self-compassion: Remind them to be kind to themselves.
- Avoid diagnosis or medical advice: This is not professional therapy.
- Always remind: Suggest seeking professional help for serious issues.
- Keep responses concise, caring, and supportive (under 150 words).

End with a caring sign-off if suitable."""

        if history:
            history_str = "\n".join([f"{h['role'].capitalize()}: {h['content']}" for h in history[-10:]])
            prompt = f"Conversation history:\n{history_str}\n\nCurrent user message: '{msg}'\n\n{base_prompt}"
        else:
            prompt = base_prompt

        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API error: {e}")
        # Fallback to original TensorFlow model
        if not model_loaded:
            return "Chatbot model is not loaded. Please train the model first."

        ints = predict_class(msg, model)
        res = get_response(ints, intents)
        return res

# Add safety disclaimer
SAFETY_DISCLAIMER = """
<div class="disclaimer">
    <strong>Important Disclaimer:</strong> This chatbot is for educational purposes only and is not a substitute for professional mental health care. If you're in crisis, please contact:
    <ul>
        <li>National Suicide Prevention Lifeline: 91-9820466726</li>
        <li>Crisis Text Line: Text HOME to 741741</li>
        <li>Emergency Services: 112</li>
    </ul>
</div>
"""

@app.route("/")
def home():
    return render_template("index.html", disclaimer=SAFETY_DISCLAIMER)

@app.route("/get")
def get_bot_response():
    user_text = request.args.get('msg')
    if user_text.lower() in ['quit', 'exit', 'bye']:
        session.pop('history', None)
        return "Thank you for chatting. Take care of yourself! 👋"

    history = session.get('history', [])
    history.append({'role': 'user', 'content': user_text})
    response = chatbot_response(user_text, history)
    history.append({'role': 'assistant', 'content': response})
    session['history'] = history
    return response

@app.route("/chat", methods=["POST"])
def chat_api():
    """API endpoint for chatbot"""
    data = request.get_json()
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    history = session.get('history', [])
    history.append({'role': 'user', 'content': user_message})
    response = chatbot_response(user_message, history)
    history.append({'role': 'assistant', 'content': response})
    session['history'] = history

    return jsonify({
        'response': response,
        'status': 'success'
    })

@app.route("/restart", methods=["POST"])
def restart():
    """Clear conversation history"""
    session.pop('history', None)
    return jsonify({'status': 'restarted'})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
