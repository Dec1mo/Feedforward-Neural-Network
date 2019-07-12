import pickle
import numpy as np
from sklearn.metrics import accuracy_score
import time
from scipy import sparse
	
def ReLU(X):
	#print (X.shape[1])
	for i in range (len(X)):
		for j in range (X.shape[1]):
			if X[i][j] < 0:
				X[i][j] = 0
	return X

def softmax(X):
	exp_X = np.exp(X - np.max(X, axis = 0, keepdims = True))
	return exp_X / exp_X.sum(axis = 0)

#One-hot
def one_hot(y, class_num = 2):
	Y = sparse.coo_matrix((np.ones_like(y),
		(y, np.arange(len(y)))), shape = (class_num, len(y))).toarray()
	return Y

def feedforward(X, W, b, L): # L la so layers
	#Dau ra la A
	A = np.array([None for l in range(L)])
	A[0] = X.T
	Z = None
	for l in range(1,L-1):
		Z = W[l].T @ A[l-1] + b[l]
		A[l] = ReLU(Z)
	
	#input()
	Z = W[L-1].T @ A[L-2] + b[L-1]
	A[L-1] = softmax(Z)	
	print (A[-1])
	return A

def backpropagation(predict_label, true_label, L, A, W, b, N):
	rate = 1 # learning rate
	print (predict_label)
	#print (true_label)
	E = (predict_label - true_label)/N
	dW = np.dot(A[L-2], E.T)
	db = np.sum(E, axis = 1, keepdims = True)
	W[L-1] += -rate*dW 
	b[L-1] += -rate*db
	for i in range (L-2, 0):
		E = np.dot(W[i+1], E.T)
		dW = np.dot(A[i-1], E.T)
		db = np.sum(E, axis = 1, keepdims = True)
		W[l] += -rate*dW 
		b[l] += -rate*db
	return W, b

def neu_net(X, y, L, W_init, b_init, epochs = 1000):
	W = W_init
	b = b_init
	N = len(y)
	y = one_hot(y)
	for epoch in range (epochs):
		print (epoch)
		A = feedforward(X, W, b, L)
		W, b = backpropagation(A[-1], y, L, A, W, b, N)
		#print (W)
	return W, b

def predict(W, b, L, X_test):
	A = feedforward(X_test, W, b, L)
	return np.argmax(A[-1], axis=0)

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
	
	sizes = [len(tfidf_vect.vocabulary_), 1024, 256, 128, 2]
	L = len(sizes)
	W_init = [None for i in range (L)]
	b_init = [None for i in range (L)]
	for i in range (0, len(sizes)-1):
		w_init = 0.01*np.random.randn(sizes[i], sizes[i+1])
		W_init[i+1] = w_init
		b = 0.01*np.random.randn(sizes[i+1], 1)
		b_init[i+1] = b
		
	start = time.time()
	W, b = neu_net(X_train, y_train, L, W_init, b_init)	
	end = time.time()
	with open (r'../model/W.pkl', 'wb') as file:
		pickle.dump(W, file)
	with open (r'../model/b.pkl', 'wb') as file:
		pickle.dump(b, file)
		
	predict_label = predict(W, b, L, X_test)
	
	accuracy = 100*accuracy_score(predict_label, y_test)
	print ('Accurancy of Feedforward Neural Network = %.2f%%' %  accuracy)
	print ('Time = ', end - start)
	
if __name__ == '__main__':
	main()

	

