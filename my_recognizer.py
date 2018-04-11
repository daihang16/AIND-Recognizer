import warnings
from asl_data import SinglesData

def recognize(models: dict, test_set: SinglesData):
	""" Recognize test word sequences from word models set

	:param models: dict of trained models
		{'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
	:param test_set: SinglesData object
	:return: (list, list)  as probabilities, guesses
		both lists are ordered by the test set word_id
		probabilities is a list of dictionaries where each key a word and value is Log Liklihood
			[{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
			{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... }]
		guesses is a list of the best guess words ordered by the test set word_id
			['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
	"""

	warnings.filterwarnings("ignore", category=DeprecationWarning)
	probabilities = [] #dict of {possible_word: logL}
	guesses = [] #best guesses
	# TODO implement the recognizer
	for word_id in range(test_set.num_items):
		word_logL_dict = {} #dict
		X, lengths = test_set.get_all_Xlengths()[word_id]
		for word in models:
			hmm_model = models[word]
			try: #if the hmmlearn library can score the model
				logL = hmm_model.score(X, lengths)
			except: #if the hmmlearn library cannot score the model
				logL = float('-inf')
			word_logL_dict[word] = logL
		probabilities.append(word_logL_dict)
		guesses.append(max(word_logL_dict, key = lambda k: word_logL_dict[k])) #best guess according to logL

	return probabilities, guesses