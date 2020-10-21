#Category Predictor
#Input: Corpus [categorywise docs], [categorywise labels]
#Output : Predicts category of unseen data

import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

class CategoryPredictor:
	def __init__(self, docs, labels):
		if not isinstance(docs, list):
			raise Exception('docs must be a list of FileNames')
		if not isinstance(labels, list):
			raise Exception('labels must be a list')

		allNames = []
		for fn,lbl in zip(docs,labels):
			allNames.append(self.__load_dataset__(fn,lbl))

		random.shuffle(allNames)
		random.shuffle(allNames)
		random.shuffle(allNames)

		names = []
		genders = []
		for category in allNames:
			for x, label in category:
				names.append(x[-4:].lower())
				genders.append(label)

		self.trainNames, self.testNames, self.trainGenders, self.testGenders = train_test_split(names, genders, test_size=0.2)


	def __load_dataset__(self, fileName, label):
		fh = open(fileName)
		names = []
		for x in fh:
			names.append((x.strip(), label))
		return names

	def getTestSets(self):
		return self.testNames, self.testGenders

	def train(self):
		#create a vectorizer
		self.vectorizer = CountVectorizer()
		#create a vocabulary from the training set
		self.vectorizer.fit(self.trainNames)
		#create a bow of the training set
		bow = self.vectorizer.transform(self.trainNames)
		#algo
		self.algo = MultinomialNB()
		#train it
		self.algo.fit(bow, self.trainGenders)


	def predict(self, name):
		name_bow = self.vectorizer.transform([name])
		result = self.algo.predict(name_bow)
		return result[0]

def main():
	predictor = CategoryPredictor(['male.txt', 'female.txt'], ['M','F'])
	#test the algorithm
	predictor.train()
	testNames, testGenders = predictor.getTestSets()
	errcount = 0
	for nm, gen in zip(testNames, testGenders):
		result = predictor.predict(nm)
		#output the result
		print(nm, gen, result, sep='--->')
		if gen != result:
			errcount+=1
	print('Error rate : ', errcount , '/', len(testNames))
main()

