#!/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import collections
from math import log
import sys
import numpy as np
import random
import copy


# Python 3 backwards compatibility tricks
if sys.version_info.major > 2:

    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

    def unicode(*args, **kwargs):
        return str(*args, **kwargs)

class LangModel:
    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        for s in corpus:
            self.fit_sentence(s)
        self.norm()

    def perplexity(self, corpus):
        """Computes the perplexity of the corpus by the model.

        Assumes the model uses an EOS symbol at the end of each sentence.
        """
        return pow(2.0, self.entropy(corpus))

    def entropy(self, corpus):
        num_words = 0.0
        sum_logprob = 0.0
        for s in corpus:
            num_words += len(s) + 1 # for EOS
            sum_logprob += self.logprob_sentence(s)
        return -(1.0/num_words)*(sum_logprob)

    def logprob_sentence(self, sentence):
        p = 0.0
        for i in xrange(len(sentence)):
            p += self.cond_logprob(sentence[i], sentence[:i])
        p += self.cond_logprob('END_OF_SENTENCE', sentence)
        return p

    # required, update the model when a sentence is observed
    def fit_sentence(self, sentence): pass
    # optional, if there are any post-training steps (such as normalizing probabilities)
    def norm(self): pass
    # required, return the log2 of the conditional prob of word, given previous words
    def cond_logprob(self, word, previous): pass
    # required, the list of words the language model suports (including EOS)
    def vocab(self): pass

class Unigram(LangModel):
    def __init__(self, backoff = 0.000001):
        self.model = dict()
        self.lbackoff = log(backoff, 2)

    def inc_word(self, w):
        if w in self.model:
            self.model[w] += 1.0
        else:
            self.model[w] = 1.0

    def fit_sentence(self, sentence):
        for w in sentence:
            self.inc_word(w)
        self.inc_word('END_OF_SENTENCE')

    def norm(self):
        """Normalize and convert to log2-probs."""
        tot = 0.0
        for word in self.model:
            tot += self.model[word]
        ltot = log(tot, 2)
        for word in self.model:
            self.model[word] = log(self.model[word], 2) - ltot

    def cond_logprob(self, word, previous):
        if word in self.model:
            return self.model[word]
        else:
            return self.lbackoff

    def vocab(self):
        return self.model.keys()

class TrigramLaplace(LangModel):
    def __init__(self):
        self.n = 3
        self.unk_cutoff = 2.0
        self.rare_words = dict()
        self.rare_words_sum = 0
        self.ngram_models = dict()
        for i in range(self.n):
            self.ngram_models[i] = dict()
    
    # creating unigram
    def inc_word_unigram(self, w):
        if w in self.ngram_models[0]:
            self.ngram_models[0][w] += 1.0
        else:
            self.ngram_models[0][w] = 1.0

    # fitting sentence for unigram
    def fit_sentence_unigram(self, sentence):
        for w in sentence:
            self.inc_word_unigram(w)
        self.inc_word_unigram('END_OF_SENTENCE')

    def get_rare_words(self):
        for w in self.ngram_models[0].keys():
            if self.ngram_models[0][w] <= self.unk_cutoff:
                rand_remove_prob = random.uniform(0, 1)
                if rand_remove_prob >= 0.5:
                    self.rare_words[w] = True
                    self.rare_words_sum += self.ngram_models[0][w]
        
    
    def remove_rare_set_unknown(self):
        for w in self.rare_words:
            self.ngram_models[0].pop(w, None)
        self.ngram_models[0]["UNK"] = self.rare_words_sum

    def inc_ngram_count(self, words):
        word_2 = words[0]
        if word_2 in self.rare_words:
            word_2 = "UNK"
        word_1 = words[1]
        if word_1 in self.rare_words:
            word_1 = "UNK"
        word_0 = words[2]
        if word_0 in self.rare_words:
            word_0 = "UNK"
        
        if word_1 not in self.ngram_models[1]:
            self.ngram_models[1][word_1] = dict()
        
        if word_0 not in self.ngram_models[1][word_1]:
            self.ngram_models[1][word_1][word_0] = 0.0

        self.ngram_models[1][word_1][word_0] += 1.0

        if word_2 not in self.ngram_models[2]:
            self.ngram_models[2][word_2] = dict()

        if word_1 not in self.ngram_models[2][word_2]:
            self.ngram_models[2][word_2][word_1] = dict()
        
        if word_0 not in self.ngram_models[2][word_2][word_1]:
            self.ngram_models[2][word_2][word_1][word_0] = 0.0

        self.ngram_models[2][word_2][word_1][word_0] += 1.0

    def fit_sentence(self, sentence):
        num_words = len(sentence)
        modified_sentence = ["*"]*(self.n - 1)
        modified_sentence.extend(sentence)
        modified_sentence.append('END_OF_SENTENCE')
        if "*" not in self.ngram_models[1]:
            self.ngram_models[1]["*"] = dict()
        if "*" not in self.ngram_models[1]["*"]:
            self.ngram_models[1]["*"]["*"] = 0.0

        self.ngram_models[1]["*"]["*"] += 1.0
        for i in range(num_words + 1):
            self.inc_ngram_count(modified_sentence[i:i+self.n])

    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        for s in corpus:
            self.fit_sentence_unigram(s)
            
        self.get_rare_words()
        self.remove_rare_set_unknown()
        for s in corpus:
            self.fit_sentence(s)

        self.norm()

    def norm(self):
        """Normalize and convert to log2-probs."""
        self.ngram_models_counts = copy.deepcopy(self.ngram_models)
        vocab_size = len(self.ngram_models[0])

        for w_2 in self.ngram_models[2]:
            for w_1 in self.ngram_models[2][w_2]:
                bigram_count = self.ngram_models[1][w_2][w_1]
                laplace_tot = bigram_count + vocab_size
                log_tot_k = log(laplace_tot, 2)
                 
                for w_0 in self.ngram_models[2][w_2][w_1]:
                    self.ngram_models[2][w_2][w_1][w_0] = log(self.ngram_models[2][w_2][w_1][w_0] + 1, 2) - log_tot_k
                
                if "UNK" not in self.ngram_models[2][w_2][w_1]:
                    num_words_present_in_bigram = len(self.ngram_models[2][w_2][w_1])
                    num_unk = vocab_size - num_words_present_in_bigram
                    self.ngram_models[2][w_2][w_1]["UNK"] = log(num_unk, 2) - log_tot_k

    def cond_logprob(self, word, previous):
        context_present = True
        prvs_context = []
        if len(previous) < self.n - 1:
            prvs_context.extend(["*"]*(self.n - 1 - len(previous)))
        prvs_context.extend(previous[-(self.n - 1):])
        
        mod = self.ngram_models[2]
        for prev in prvs_context:
            if prev not in mod:
                context_present = False
                break
            else:
                mod = mod[prev]  
        if context_present:
            if word in mod:
                return mod[word]
            else:
                return mod["UNK"]
        else:
            return -log(float(len(self.ngram_models[0])), 2)

    def vocab(self):
        return self.ngram_models[0].keys()
    
    
