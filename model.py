import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM, Embedding
from tensorflow.keras.optimizers import SGD
import numpy as np
import random
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

class MentalHealthChatbotModel:
    def __init__(self):
        self.model = None
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_words = ['?', '!', '.', ',']
        
    def create_model(self, input_shape, output_shape):
        """Create the neural network model"""
        model = Sequential()
        model.add(Dense(128, input_shape=(input_shape,), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(output_shape, activation='softmax'))

        # Compile model
        sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        self.model = model
        return model
    
    def prepare_data(self, intents_file='intents.json'):
        """Prepare training data from intents"""
        # Load intents file
        with open(intents_file) as file:
            intents = json.load(file)
        
        # Process patterns and create documents
        for intent in intents['intents']:
            for pattern in intent['patterns']:
                # Tokenize each word
                w = nltk.word_tokenize(pattern)
                self.words.extend(w)
                # Add documents in the corpus
                self.documents.append((w, intent['tag']))
                # Add to classes list
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])
        
        # Lemmatize and lower each word and remove duplicates
        self.words = [lemmatizer.lemmatize(w.lower()) for w in self.words if w not in self.ignore_words]
        self.words = sorted(list(set(self.words)))
        
        # Sort classes
        self.classes = sorted(list(set(self.classes)))
        
        return self.words, self.classes, self.documents
    