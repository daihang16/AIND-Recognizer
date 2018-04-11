import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        BIC = [] #track the BIC
        hidden_states = [] #track the number of hidden_states
        n = len(self.sequences) #number of sequences
        for num_hidden_states in range(self.min_n_components, self.max_n_components + 1): #for each possible number of hidden states
            try: #if the hmmlearn library can train or score the model
                hmm_model = self.base_model(num_states = num_hidden_states)
                logL = hmm_model.score(self.X, self.lengths)
                #each state has mean and Var for each of the features:
                #2 * num_hidden_states * num_features
                #Initial state occupation probabilities: num_hidden_states - 1
                #Transition probabilities: num_hidden_states * (num_hidden_states - 1)
                #https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/12
                #https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/17
                p = num_hidden_states * (num_hidden_states - 1) + num_hidden_states - 1 + 2 * num_hidden_states * hmm_model.n_features
                BIC.append(-2 * logL + p * math.log(n))
                hidden_states.append(num_hidden_states)
            except: #if the hmmlearn library cannot train or score the model
                pass
        #now see which number of hidden states gave the smallest BIC
        optimal_num_hidden_states = hidden_states[BIC.index(min(BIC))]
        optimal_hmm_model = GaussianHMM(n_components = optimal_num_hidden_states, covariance_type="diag", n_iter=1000,
                                random_state = self.random_state, verbose = False).fit(self.X, self.lengths)
        return optimal_hmm_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        DIC = [] #track the DIC
        hidden_states = [] #track the number of hidden_states
        rest_words = list(self.words) #list
        rest_words.remove(self.this_word)
        print(self.this_word)
        for num_hidden_states in range(self.min_n_components, self.max_n_components + 1): #for each possible number of hidden states
            try: #if the hmmlearn library can train or score the model
                rest_logL = 0
                hmm_model = self.base_model(num_states = num_hidden_states)
                logL = hmm_model.score(self.X, self.lengths)
                rest_num_scorable_words = 0
                for word in rest_words:
                    X, lengths = self.hwords[word]
                    try: #if the hmmlearn library can score the model
                        rest_logL = rest_logL + hmm_model.score(X, lengths)
                        rest_num_scorable_words = rest_num_scorable_words + 1
                    except: #if the hmmlearn library cannot score the model
                        print('{0} is not scorable!'.format(word))
                DIC.append(logL - rest_logL / rest_num_scorable_words)
                hidden_states.append(num_hidden_states)
            except: #if the hmmlearn library cannot train or score the model
                print(num_hidden_states)
        #now see which number of hidden states gave the largest DIC
        optimal_num_hidden_states = hidden_states[DIC.index(max(DIC))]
        optimal_hmm_model = GaussianHMM(n_components = optimal_num_hidden_states, covariance_type="diag", n_iter=1000,
                                random_state = self.random_state, verbose = False).fit(self.X, self.lengths)
        return optimal_hmm_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        likelihoods = [] #track the likelihoods
        hidden_states = [] #track the number of hidden_states
        k = len(self.sequences) #number of sequences
        k = min(k, 5) #if k<=5, do leave-one-out cross-validation; otherwise, do 5-fold cross-validation
        for num_hidden_states in range(self.min_n_components, self.max_n_components + 1): #for each possible number of hidden states
            try: #if the hmmlearn library can train or score the model
                if k > 1: #only then do cross-validation
                    ave_logL = 0
                    split_method = KFold(n_splits = k)
                    for training_idx, testing_idx in split_method.split(self.sequences):
                        training_X, training_lengths = combine_sequences(training_idx, self.sequences) #list, list
                        training_X = np.asarray(training_X)
                        testing_X, testing_lengths = combine_sequences(testing_idx, self.sequences) #list, list
                        testing_X = np.asarray(testing_X)
                        hmm_model = GaussianHMM(n_components = num_hidden_states, covariance_type="diag", n_iter=1000,
                                        random_state = self.random_state, verbose = False).fit(training_X, training_lengths)
                        logL = hmm_model.score(testing_X, testing_lengths)
                        ave_logL = ave_logL + logL
                    ave_logL = ave_logL / k
                else:
                    hmm_model = GaussianHMM(n_components = num_hidden_states, covariance_type="diag", n_iter=1000,
                                    random_state = self.random_state, verbose = False).fit(self.X, self.lengths)
                    ave_logL = hmm_model.score(self.X, self.lengths)
                likelihoods.append(ave_logL)
                hidden_states.append(num_hidden_states)
            except: #if the hmmlearn library cannot train or score the model
                pass
        #now see which number of hidden states gave the largest likelihood
        optimal_num_hidden_states = hidden_states[likelihoods.index(max(likelihoods))]
        optimal_hmm_model = GaussianHMM(n_components = optimal_num_hidden_states, covariance_type="diag", n_iter=1000,
                                random_state = self.random_state, verbose = False).fit(self.X, self.lengths)
        return optimal_hmm_model
