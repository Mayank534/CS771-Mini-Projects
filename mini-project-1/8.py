import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

import logging
import json

class EmoticonClassifier:
    def __init__(self):
        self.config = {
            'lstm_units': 10,
            'dropout_rate': 0.3,
            'learning_rate': 0.005,
            'epochs': 20,
            'batch_size': 32
        }
        
        self.tokenizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.max_len = None
        self.vocab_size = None
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def preprocess_text(self, text):
        return text if isinstance(text, str) else text

    def build_model(self):
        model = Sequential([
            LSTM(
                units=self.config['lstm_units'],
                input_shape=(self.max_len, self.vocab_size),
                return_sequences=False,
                recurrent_dropout=0.2
            ),
            Dropout(self.config['dropout_rate']),
            Dense(2, activation='softmax')  # Binary classification (2 classes)
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        return model

    def preprocess_data(self, texts, fit_tokenizer=False):
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        if fit_tokenizer:
            # Initialize character-level tokenizer
            self.tokenizer = Tokenizer(char_level=True)
            self.tokenizer.fit_on_texts(processed_texts)
            
            # Set vocabulary size
            self.vocab_size = len(self.tokenizer.word_index) + 1  # Adding 1 for padding token
            self.max_len = max(len(seq) for seq in self.tokenizer.texts_to_sequences(processed_texts))
            self.logger.info(f"Max sequence length: {self.max_len}")
            self.logger.info(f"Vocabulary size: {self.vocab_size}")
            
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call fit() first or set fit_tokenizer=True")
            
        # Convert texts to padded sequences
        sequences = self.tokenizer.texts_to_sequences(processed_texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post')
        
        # One-hot encode the padded sequences
        one_hot_data = tf.keras.utils.to_categorical(padded_sequences, num_classes=self.vocab_size)
        
        return one_hot_data

    def train(self, X_train, y_train, X_val, y_val):
        try:
            # Preprocess data
            X_train_processed = self.preprocess_data(X_train, fit_tokenizer=True)
            X_val_processed = self.preprocess_data(X_val, fit_tokenizer=False)
            
            # Encode labels
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            y_val_encoded = self.label_encoder.transform(y_val)
            
            # Build model
            self.model = self.build_model()
            
            # Early stopping callback
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                min_delta=0.001,
                mode='min',
                restore_best_weights=True,
                verbose=1
            )
            
            # Train model
            history = self.model.fit(
                X_train_processed,
                y_train_encoded,
                batch_size=self.config['batch_size'],
                epochs=self.config['epochs'],
                validation_data=(X_val_processed, y_val_encoded),
                callbacks=[early_stopping],
                verbose=1
            )
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

    def predict(self, texts):
        try:
            X_processed = self.preprocess_data(texts, fit_tokenizer=False)
            predictions = self.model.predict(X_processed)
            return self.label_encoder.inverse_transform(np.argmax(predictions, axis=1))
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise

class FeatureModel:
    def __init__(self):
        self.config = {
            'lstm_units': 3,
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 30
        }
        self.model = None
        self.results = {
            'train_samples': None,
            'accuracy': None,
            'loss': None
        }

    def create_model(self, input_shape, num_classes):
        """Create and compile the LSTM model with the best parameters."""
        model = Sequential([
            LSTM(self.config['lstm_units'], 
                 input_shape=input_shape, 
                 return_sequences=False, 
                 recurrent_dropout=0.2),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        
        self.model = model
        
    def train_model(self, train_feat_X, train_feat_Y):
        """Train the model using 100% of the training data."""
        input_shape = (train_feat_X.shape[1], train_feat_X.shape[2])
        num_classes = len(np.unique(train_feat_Y))

        # Set the number of training samples
        self.results['train_samples'] = len(train_feat_X)

        # Create and compile the model
        self.create_model(input_shape, num_classes)

        # Callbacks
        early_stopping = EarlyStopping(
            monitor='loss',
            patience=10,
            min_delta=0.001,
            mode='min',
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train the model
        history = self.model.fit(
            train_feat_X,
            train_feat_Y,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Store final training accuracy and loss
        self.results['accuracy'] = history.history['accuracy'][-1]
        self.results['loss'] = history.history['loss'][-1]
        
        print(f"\nFinal Training Accuracy: {self.results['accuracy']*100:.2f}%")
        print(f"Final Training Loss: {self.results['loss']:.4f}")
    
    def predict(self, test_feat_X):
        """Predict on the test dataset."""
        predictions = self.model.predict(test_feat_X)
        return np.argmax(predictions, axis=1)

    def save_results(self, predictions, filename='pred_feature.txt'):
        """Save the predictions to a file."""
        np.savetxt(filename, predictions, fmt='%d')
        print(f"\nPredictions saved to '{filename}'")
    
    def save_training_metrics(self, filename='training_metrics.json'):
        """Save the training metrics (accuracy, loss) to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"\nTraining metrics saved to '{filename}'")



class TextSeqModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.label_encoder = LabelEncoder()
        self.max_len = None
        self.vocab_size = None

    def load_data(self):
        # Load training and validation datasets
        train_seq_df = pd.read_csv("datasets/train/train_text_seq.csv")
        valid_seq_df = pd.read_csv("datasets/valid/valid_text_seq.csv")

        train_seq_X = train_seq_df['input_str'].tolist()
        train_seq_Y = train_seq_df['label'].tolist()
        valid_seq_X = valid_seq_df['input_str'].tolist()
        valid_seq_Y = valid_seq_df['label'].tolist()

        return train_seq_X, train_seq_Y, valid_seq_X, valid_seq_Y

    def preprocess_data(self, train_seq_X, train_seq_Y, valid_seq_X, valid_seq_Y):
        # Character-level tokenization
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
        self.tokenizer.fit_on_texts(train_seq_X)  # Fit only on training data

        # Convert sequences to integers
        train_sequences = self.tokenizer.texts_to_sequences(train_seq_X)
        valid_sequences = self.tokenizer.texts_to_sequences(valid_seq_X)

        # Get max length from training data
        self.max_len = max(len(seq) for seq in train_sequences)

        # Pad sequences
        X_train_padded = pad_sequences(train_sequences, maxlen=self.max_len, padding='post')
        X_valid_padded = pad_sequences(valid_sequences, maxlen=self.max_len, padding='post')

        # One-hot encode sequences
        self.vocab_size = len(self.tokenizer.word_index) + 1
        X_train = tf.keras.utils.to_categorical(X_train_padded, num_classes=self.vocab_size)
        X_valid = tf.keras.utils.to_categorical(X_valid_padded, num_classes=self.vocab_size)

        # Encode labels
        y_train = self.label_encoder.fit_transform(train_seq_Y)
        y_valid = self.label_encoder.transform(valid_seq_Y)

        return X_train, y_train, X_valid, y_valid

    def train_model(self, X_train, y_train, X_valid, y_valid):
        # Build the model
        self.model = Sequential([
            LSTM(44, input_shape=(self.max_len, self.vocab_size), recurrent_dropout=0.2),
            Dropout(0.3),
            Dense(len(self.label_encoder.classes_), activation='softmax')
        ])

        # Compile the model
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy']
        )

        # Train the model
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        self.model.fit(
            X_train,
            y_train,
            epochs=150,
            batch_size=32,
            validation_data=(X_valid, y_valid),
            callbacks=[early_stopping],
            verbose=1
        )

    def predict(self, X):
        predictions = self.model.predict(X)
        return self.label_encoder.inverse_transform(np.argmax(predictions, axis=1))

    def load_test_data(self, test_path):
        test_df = pd.read_csv(test_path)
        return test_df['input_str'].tolist()




class CombinedModel:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=400)
        self.model = DecisionTreeClassifier(
            criterion='gini',
            max_depth=10,
            min_samples_leaf=10,
            min_samples_split=20
        )

    def load_data(self):
        # Load emoticon dataset
        train_emoticon_df = pd.read_csv("datasets/train/train_emoticon.csv")
        self.train_emoticon_X = train_emoticon_df['input_emoticon'].tolist()
        self.train_emoticon_Y = train_emoticon_df['label'].tolist()
        
        self.test_emoticon_X = pd.read_csv("datasets/test/test_emoticon.csv")['input_emoticon'].tolist()
        
        # Load text sequence dataset
        train_seq_df = pd.read_csv("datasets/train/train_text_seq.csv")
        self.train_seq_X = train_seq_df['input_str'].tolist()
        self.train_seq_Y = train_seq_df['label'].tolist()
        
        self.test_seq_X = pd.read_csv("datasets/test/test_text_seq.csv")['input_str'].tolist()
        
        # Load feature dataset
        train_feat = np.load("datasets/train/train_feature.npz", allow_pickle=True)
        self.train_feat_X = train_feat['features']
        
        self.test_feat_X = np.load("datasets/test/test_feature.npz", allow_pickle=True)['features']

    def preprocess_data(self):
        # Encode emoticons
        self.train_emoticon_X_encoded = self.encode_emoticons(self.train_emoticon_X)
        self.test_emoticon_X_encoded = self.encode_emoticons(self.test_emoticon_X)
        
        # TF-IDF on text sequences
        self.train_seq_X_tfidf = self.tfidf_vectorizer.fit_transform(self.train_seq_X).toarray()
        self.test_seq_X_tfidf = self.tfidf_vectorizer.transform(self.test_seq_X).toarray()

        # Flatten feature data
        self.train_feat_X_flattened = self.train_feat_X.reshape(self.train_feat_X.shape[0], -1)
        self.test_feat_X_flattened = self.test_feat_X.reshape(self.test_feat_X.shape[0], -1)

        # Combine training data
        self.train_X_combined = np.hstack([
            self.train_emoticon_X_encoded,
            self.train_seq_X_tfidf,
            self.train_feat_X_flattened
        ])

        # Combine test data
        self.test_X_combined = np.hstack([
            self.test_emoticon_X_encoded,
            self.test_seq_X_tfidf,
            self.test_feat_X_flattened
        ])

    def encode_emoticons(self, emoticon_list):
        # Flatten emoticons
        emoticon_flattened = [list(emoticon) for emoticon in emoticon_list]
        encoded_emoticons = [self.label_encoder.fit_transform(emoticon) for emoticon in emoticon_flattened]
        return np.array(encoded_emoticons)

    def train(self):
        # Train the model
        self.model.fit(self.train_X_combined, self.train_emoticon_Y)

    def evaluate(self):
        # Make predictions and evaluate on the validation set
        valid_predictions = self.model.predict(self.train_X_combined)
        valid_accuracy = accuracy_score(self.train_emoticon_Y, valid_predictions)
        print(f"Validation Accuracy: {valid_accuracy}")

        # Detailed performance metrics
        print("\nClassification Report:\n", classification_report(self.train_emoticon_Y, valid_predictions))
        print("\nConfusion Matrix:\n", confusion_matrix(self.train_emoticon_Y, valid_predictions))

    def predict(self):
        # Make predictions on the test dataset
        test_predictions = self.model.predict(self.test_X_combined)
        return test_predictions

    def save_predictions(self, predictions, filename="predictions.txt"):
        # Save predicted labels to a file
        with open(filename, 'w') as f:
            for pred in predictions:
                f.write(f"{pred}\n")

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Load data
        train_data = pd.read_csv('datasets/train/train_emoticon.csv')
        test_data = pd.read_csv('datasets/test/test_emoticon.csv')
        
        # Split training data into train and validation
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_data['input_emoticon'].tolist(),
            train_data['label'].values,
            test_size=0.2,
            stratify=train_data['label'].values,
        )
        
        # Initialize and train classifier
        classifier = EmoticonClassifier()
        history = classifier.train(train_texts, train_labels, val_texts, val_labels)
        
        # Make predictions on test set
        predictions = classifier.predict(test_data['input_emoticon'].tolist())
        
        # Save predictions
        output_file = "pred_emoticon.txt"
        np.savetxt(output_file, predictions, fmt='%d')
        logger.info(f"Predictions saved to {output_file}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise
    
    train_feat = np.load("datasets/train/train_feature.npz", allow_pickle=True)
    train_feat_X = train_feat['features']
    train_feat_Y = train_feat['label']

    test_feat = np.load("datasets/test/test_feature.npz", allow_pickle=True)
    test_feat_X = test_feat['features']

    # Initialize and train the feature model
    feature_model = FeatureModel()
    feature_model.train_model(train_feat_X, train_feat_Y)
    
    # Predict on the test set
    predictions = feature_model.predict(test_feat_X)
    
    # Save predictions and training metrics
    feature_model.save_results(predictions, filename="pred_feature.txt")
    feature_model.save_training_metrics()

    
    # Initialize the text sequence model
    text_seq_model = TextSeqModel()
    
    # Load and preprocess training and validation data
    train_seq_X, train_seq_Y, valid_seq_X, valid_seq_Y = text_seq_model.load_data()
    X_train, y_train, X_valid, y_valid = text_seq_model.preprocess_data(train_seq_X, train_seq_Y, valid_seq_X, valid_seq_Y)

    # Train the model
    text_seq_model.train_model(X_train, y_train, X_valid, y_valid)

    # Load test data
    test_X = text_seq_model.load_test_data("datasets/test/test_text_seq.csv")
    test_sequences = text_seq_model.tokenizer.texts_to_sequences(test_X)
    test_padded = pad_sequences(test_sequences, maxlen=text_seq_model.max_len, padding='post')
    X_test = tf.keras.utils.to_categorical(test_padded, num_classes=text_seq_model.vocab_size)

    # Make predictions on test set
    predicted_labels = text_seq_model.predict(X_test)

    # Save predicted labels to a text file
    output_file = "pred_textseq.txt"
    with open(output_file, 'w') as f:
        for label in predicted_labels:
            f.write(f"{label}\n")
    
    print(f"Predicted labels saved to '{output_file}'")
    
    
     # Create an instance of the CombinedModel
    combined_model = CombinedModel()
    
    # Load the datasets
    combined_model.load_data()
    
    # Preprocess the data
    combined_model.preprocess_data()
    
    # Train the model
    combined_model.train()
    
    # Evaluate the model on the training data
    combined_model.evaluate()
    
    # Make predictions on the test dataset
    test_predictions = combined_model.predict()
    
    # Save the predicted labels to a file
    combined_model.save_predictions(test_predictions, filename="pred_combined.txt")

if __name__ == '__main__':
    main()
