# Mental Health Support Chatbot

A web-based chatbot designed to provide empathetic mental health support. It uses natural language processing (NLP) with a TensorFlow model for intent classification and Google Gemini AI for generating compassionate responses. The chatbot addresses common mental health concerns like stress, anxiety, depression, loneliness, anger, and sleep issues.

**Important Disclaimer:** This chatbot is for educational purposes only and is not a substitute for professional mental health care. If you're in crisis, please contact:
- National Suicide Prevention Lifeline: 91-9820466726
- Crisis Text Line: Text HOME to 741741
- Emergency Services: 112

## Features

- **Empathetic Responses:** Leverages Google Gemini API to generate supportive, validating responses with coping strategies.
- **Web Interface:** Simple Flask-based web app with a home page for language selection and a chat interface.
- **Language Support:** Supports both English and Hinglish (mix of Hindi and English) for broader accessibility.
- **Conversation Memory:** Maintains conversation history for context-aware responses.
- **Restart Functionality:** Allows users to restart the conversation and clear history.
- **Dark Mode Toggle:** Includes a theme toggle for better user experience.
- **Supported Intents:** Greeting, goodbye, thanks, stress, anxiety, depression, loneliness, anger, sleep issues.
- **Safety Focus:** Includes disclaimers and encourages seeking professional help.
## Project Structure

- `app.py`: Main Flask application with chatbot logic.
- `train.py`: Script to train the TensorFlow model.
- `model.py`: Defines the MentalHealthChatbotModel class.
- `intents.json`: Training data with patterns and responses for intents.
- `chatbot_model.h5`: Pre-trained model file.
- `words.pkl`, `classes.pkl`: Pickled vocabulary and class labels.
- `templates/home.html`: HTML template for the home page with language selection.
- `templates/index.html`: HTML template for the chat interface.
- `requirements.txt`: Python dependencies.
- `generate_pickles.py`: Utility to generate pickle files (if needed).

## Installation

1. **Clone or Download the Project:**
   ```
   git clone <repository-url>
   cd Mental-Health-Support-Chatbot
   ```

2. **Set Up Virtual Environment (Recommended):**
   ```
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables:**
   Create a `.env` file in the root directory:
   ```
   GOOGLE_API_KEY=your_google_gemini_api_key_here
   ```
   Obtain a free API key from [Google AI Studio](https://aistudio.google.com/).

5. **Download NLTK Data:**
   The app will automatically download required NLTK data (punkt, wordnet) on first run.

6. **Train the Model (If Needed):**
   If you want to retrain or update the model:
   ```
   python train.py
   ```
   This generates `chatbot_model.h5`, `words.pkl`, and `classes.pkl`.

## Usage

1. **Run the Application:**
   ```
   python app.py
   ```
   The server starts at `http://0.0.0.0:8080` (or `http://localhost:8080`).

2. **Access the Chatbot:**
   Open your browser and navigate to `http://localhost:8080`.
   - On the home page, select your preferred language (English or Hinglish).
   - Proceed to the chat interface.
   - Type messages in the chat interface.
   - The bot responds with empathetic support in the selected language.
   - Use the "Restart" button to clear conversation history and start over.
   - Use commands like "quit", "exit", or "bye" to end the conversation.

3. **API Usage:**
   - **Chat Endpoint (POST `/chat`):**
     Send JSON: `{"message": "user input"}`
     Response: `{"response": "bot reply", "status": "success"}`
   - **Get Response (GET `/get?msg=user input`):** Simple query parameter for responses.
   - **Restart Endpoint (POST `/restart`):** Clears conversation history.

## Training the Model

The model is a neural network for intent classification based on bag-of-words features.

1. Update `intents.json` with new patterns/responses if desired.
2. Run `python train.py` (defaults to 200 epochs; adjust as needed).
3. The trained model and data files are saved automatically.

**Note:** Training requires TensorFlow and may take several minutes depending on your hardware.

## Customization

- **Add New Intents:** Edit `intents.json` and retrain the model.
- **Response Generation:** Modify the Gemini prompt in `app.py` for different response styles.
- **UI Improvements:** Update `templates/index.html` for better styling.
- **Fallback Responses:** Enhance rule-based responses in `intents.json`.

## Limitations

- Relies on Google Gemini API (requires API key and may have rate limits).
- Intent classification accuracy depends on training data quality.
- Not suitable for clinical use; always direct users to professionals.
- Supports English and Hinglish only.

## Contributing

Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch.
3. Make changes and add tests.
4. Submit a pull request.

Focus areas: Improving response quality, adding more intents, enhancing UI, or optimizing the model.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Flask, TensorFlow, NLTK, and Google Gemini.
- Intents inspired by common mental health support patterns.
- Safety guidelines based on best practices for AI chatbots in mental health.

For support or questions, feel free to open an issue.
