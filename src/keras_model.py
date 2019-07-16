import pickle
from keras.models import Sequential
from keras.layers import *
from keras import *
import numpy
from scipy import sparse

def CNN_model(input_dim, num_classes):
  model = Sequential()
  #input_dim = X_train.shape[1]  # Number of features
  model = Sequential()
  model.add(layers.Dense(128, input_dim=input_dim, activation='relu'))
  model.add(layers.Dense(1, activation='sigmoid'))
  return model
  
def main():
  with open (r'../preprocess_data/tfidf_vect.pkl', 'rb') as file:
    tfidf_vect = pickle.load(file)
  with open (r'../preprocess_data/train/X_train.pkl', 'rb') as file:
    X_train = pickle.load(file)
  with open (r'../preprocess_data/train/y_train.pkl', 'rb') as file:
    y_train = pickle.load(file)
  with open (r'../preprocess_data/test/X_test.pkl', 'rb') as file:
    X_test = pickle.load(file)
  with open (r'../preprocess_data/test/y_test.pkl', 'rb') as file:
    y_test = pickle.load(file)
  input_dim = X_train.shape[1]
  num_class = 2
  model = CNN_model(input_dim, num_class)
  model.compile(loss='binary_crossentropy', 
               optimizer='adam', 
               metrics=['accuracy'])
  history = model.fit(X_train, y_train,
                     epochs=10,
                     verbose=False,
                     validation_data=(X_test, y_test),
                     batch_size=128)
  loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
  print("Training Accuracy: {:.4f}".format(accuracy*100))
  loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
  print("Testing Accuracy:  {:.4f}".format(accuracy*100))
  
if __name__ == '__main__':
  main()