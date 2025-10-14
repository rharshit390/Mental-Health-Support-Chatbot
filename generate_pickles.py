from model import MentalHealthChatbotModel
import pickle

# Create model instance
chatbot_model = MentalHealthChatbotModel()

# Prepare data
words, classes, documents = chatbot_model.prepare_data()

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

print("words.pkl and classes.pkl generated successfully!")
