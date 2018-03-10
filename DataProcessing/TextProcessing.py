import pandas as pd
import numpy as np
import re

class BagOfWords:

	# My implementation of Bag of Words
	# X should be an array of text documents
	def bag_of_words(X):
		processed = []
		for doc in X:
			processed.append(BagOfWords.process(doc))
		processed = np.array(processed)
		vocab = BagOfWords.vocab(processed)
		return vocab, np.array(BagOfWords.word_counts(vocab, processed))


	# X should be an array of lists of words
	def word_counts(vocab, X):
		X_counts = []
		for doc in X:
			counts = {key: 0 for key in vocab}
			for word in doc:
				counts[word] = counts[word] + 1
			X_counts.append(list(counts.values()))
		return X_counts
    
	# Processes text
	# Removes non-letters / converts to lowercase
	def process(text):
		text = text[0]
		text = re.sub("[^a-zA-Z]", " ", text)
		text = text.lower()
		text = text.split(' ')
		text = BagOfWords.remove_empty_str(text)
		return text

	def remove_empty_str(text):
		new_list = []
		for t in text:
			if t != '':
				new_list.append(t)
		return new_list

	# X should be an array of lists of words
	def vocab(X):
		words = set()
		for doc in X:
			words = words | set(doc)
		return list(words)