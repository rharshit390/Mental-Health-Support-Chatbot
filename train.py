import numpy as np
import tensorflow as tf
import json
import pickle
import random
from model import MentalHealthChatbotModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import nltk
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

class ChatbotTrainer:
    def __init__(self):
        self.chatbot_model = MentalHealthChatbotModel()
        self.tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
        
    def prepare_training_data(self):
        """Prepare training data"""
        words, classes, documents = self.chatbot_model.prepare_data()
        
        # Create training data
        training = []
        output_empty = [0] * len(classes)
        
        # Create bag of words for each pattern
        for doc in documents:
            bag = []
            pattern_words = doc[0]
            pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
            
            for w in words:
                bag.append(1) if w in pattern_words else bag.append(0)
            
            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1
            
            training.append([bag, output_row])
        
        # Shuffle features and convert to numpy array
        random.shuffle(training)
        training = np.array(training, dtype=object)
        
        train_x = list(training[:, 0])
        train_y = list(training[:, 1])
        
        return train_x, train_y, words, classes
    
    def train_model(self, epochs=200):
        """Train the chatbot model"""
        train_x, train_y, words, classes = self.prepare_training_data()
        
        # Convert to numpy arrays
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        
        print(f"Training data shape: {train_x.shape}")
        print(f"Training labels shape: {train_y.shape}")
        print(f"Number of classes: {len(classes)}")
        
        # Create model
        input_shape = len(train_x[0])
        output_shape = len(train_y[0])
        
        model = self.chatbot_model.create_model(input_shape, output_shape)
        
        # Train model
        history = model.fit(train_x, train_y, epochs=epochs, batch_size=8, verbose=1)
        
        # Save model
        model.save('chatbot_model.h5')
        print("Model saved as 'chatbot_model.h5'")
        
        # Save words and classes
        pickle.dump(words, open('words.pkl', 'wb'))
        pickle.dump(classes, open('classes.pkl', 'wb'))
        
        print("Training completed successfully!")
        return history

if __name__ == "__main__":
    trainer = ChatbotTrainer()
    trainer.train_model(epochs=200)