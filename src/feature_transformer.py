import os
import io
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 

def all_file_to_text(path):
	list_of_text = []
	labels = []
	current_label = -1
	for path, subdirs, files in os.walk(path):
		for name in files:
			labels.append(current_label)
			with open(os.path.join(path, name), 'r+') as f:
				text = f.read()
				list_of_text.append(text)
		current_label += 1
	return list_of_text, labels

def clean_data(list_of_text):
	new_list_of_text = []
	lemmatizer = WordNetLemmatizer()
	set_stopwords = set(stopwords.words('english'))
    #clear unsuitable characters
	for data in list_of_text:
		#Remove http
		data = re.sub(r'http\S+', " ", data)
		#Remove html tags
		data = re.sub(r'<.*?>', " ", data)
		#Remove all www.
		data = re.sub(r'www\.*?', " ", data)
		#Convert all text to lower cases:
		data = data.lower()
		#Remove all non characters a-z and A-Z
		data = re.sub(r'([^a-zA-Z\s]+)', " ", data)
		words = data.split()
		data = [word for word in words if word not in set_stopwords and len(word) >= 2]
		data = " ".join(data)
		new_list_of_text.append(data)
	return new_list_of_text

def tfidf(list_of_text):
	tfidf_vect = TfidfVectorizer() 
	tfidf_vect.fit(list_of_text)	
	return tfidf_vect

def main():
	train_path = r'../raw_data/train'
	test_path = r'../raw_data/test'
	
	print ('Transforming files to texts')
	list_of_train_text, y_train = all_file_to_text(train_path)
	list_of_test_text, y_test = all_file_to_text(test_path)	
	print ('Transformed files to texts successfully')
	
	print ('Cleaning data')
	list_of_train_text = clean_data(list_of_train_text)
	list_of_test_text = clean_data(list_of_test_text)
	print ('Cleaned data successfully')
	
	print ('Tfidf Vectorizing')
	tfidf_vect = tfidf(list_of_train_text)
	X_train = tfidf_vect.transform(list_of_train_text)
	X_test = tfidf_vect.transform(list_of_test_text)
	print ('Tfidf Vectorized successfully')
	print ('Dimension of a vector: ', len(tfidf_vect.vocabulary_))
	print ('Train size: ', X_train.shape[0])
	print ('Train size: ', X_test.shape[0])
	
	print ('Pickling preprocessed data')
	with open (r'../preprocess_data/train/X_train.pkl', 'wb') as file:
		pickle.dump(X_train, file)
	with open (r'../preprocess_data/train/y_train.pkl', 'wb') as file:
		pickle.dump(y_train, file)
	with open (r'../preprocess_data/test/X_test.pkl', 'wb') as file:
		pickle.dump(X_test, file)
	with open (r'../preprocess_data/test/y_test.pkl', 'wb') as file:
		pickle.dump(y_test, file)	
	with open (r'../preprocess_data/tfidf_vect.pkl', 'wb') as file:
		pickle.dump(tfidf_vect, file)
	print ('Pickled preprocessed data successfully')
	
if __name__ == '__main__':
	main()