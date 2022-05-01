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

class BiGram(LangModel):
    def __init__(self):
        self.model = dict()
        self.vocabulary = dict()
        # self.vocabulary["UNK"] = 0.0
        self.n = 2

    def inc_ngram_count(self, words):
        rec_model = self.model
        for i in range(2):
            if i == 1:
                if words[i] in self.vocabulary: # maintaining Unigram model too
                    self.vocabulary[words[i]] += 1.0
                else:
                    self.vocabulary[words[i]] = 1.0
                if words[i] in rec_model:
                    rec_model[words[i]] += 1.0
                else:
                    rec_model[words[i]] = 1.0
            else:
                if words[i] not in rec_model:
                    rec_model[words[i]] = dict()
                rec_model = rec_model[words[i]]
                
    def fit_sentence(self, sentence):
        num_words = len(sentence)
        modified_sentence = ["*"]*(1)
        modified_sentence.extend(sentence)
        modified_sentence.append('END_OF_SENTENCE')
        for i in range(num_words + 1):
            self.inc_ngram_count(modified_sentence[i:i+2])

    def norm(self):
        """Normalize and convert to log2-probs."""
        self.model_cpy = copy.deepcopy(self.model)
        # print(self.model_cpy)
        for k in self.model.keys():
            tot_k = float(np.array(list(self.model[k].values())).sum())
            log_tot_k = log(tot_k, 2)
            for word in self.model[k].keys():
                self.model[k][word] = log(self.model[k][word], 2) - log_tot_k
                self.model_cpy[k][word] = float(self.model_cpy[k][word])/float(tot_k)
        # print(self.model_cpy)
        
    def cond_logprob(self, word, previous):
        # print(len(previous))
        context_present = True
        prvs_context = []
        if len(previous) < self.n - 1:
            prvs_context.extend(["*"]*(self.n - 1 - len(previous)))
        prvs_context.extend(previous[-(self.n - 1):])
        # print(prvs_context)
        mod = self.model
        mod_cpy = self.model_cpy
        for prev in prvs_context:
            if prev not in mod:
                context_present = False
                break
            else:
                mod = mod[prev]
                mod_cpy = mod_cpy[prev]
        
        if not context_present:
            return log(0.000001, 2)
        else:
            if word in mod.keys():
                return mod[word]
            else:
                return log(0.000001, 2)
                
    def vocab(self):
        return self.vocabulary.keys()


class TrigramLaplace(LangModel):
    def __init__(self):
        self.n = 3
        self.unk_cutoff = 5.0
        self.delta = 0.2
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
                laplace_tot = float(bigram_count) + (float(vocab_size) * self.delta)
                log_tot_k = log(laplace_tot, 2)
                 
                for w_0 in self.ngram_models[2][w_2][w_1]:
                    self.ngram_models[2][w_2][w_1][w_0] = log(float(self.ngram_models[2][w_2][w_1][w_0]) + self.delta, 2) - log_tot_k
                
                # if "UNK" not in self.ngram_models[2][w_2][w_1]:
                #     num_words_present_in_bigram = len(self.ngram_models[2][w_2][w_1])
                #     num_unk = vocab_size - num_words_present_in_bigram
                #     # print(num_unk)
                #     self.ngram_models[2][w_2][w_1]["UNK"] = log(float(num_unk) * self.delta, 2) - log_tot_k

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
        # print(prvs_context)
        if context_present:
            bigram_count = float(self.ngram_models_counts[1][prvs_context[0]][prvs_context[1]])
            if word in mod:
                return mod[word]
            else:
                if word not in self.ngram_models[0].keys():
                    if "UNK" in mod: # oov
                        return mod["UNK"]
                    else:
                        return log(float(self.delta)/(bigram_count + float(len(self.ngram_models[0]) * self.delta)), 2)
                else:
                    return log(float(self.delta)/(bigram_count + float(len(self.ngram_models[0]) * self.delta)), 2)
        else:
            return log(float(self.delta)/(float(len(self.ngram_models[0]) * self.delta)), 2)

    def vocab(self):
        return self.ngram_models[0].keys()
    
    

class TrigramLinInter(LangModel):
    def __init__(self, unk_cutoff=10.0, lamda=[0.2, 0.3, 0.5]):
        self.n = 3
        self.unk_cutoff = unk_cutoff
        self.rare_words = dict()
        self.rare_words_sum = 0
        self.ngram_models = dict()
        self.lamda = lamda
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
        modified_sentence = ["*"]*(self.n - 1)
        modified_sentence.extend(sentence)
        modified_sentence.append('END_OF_SENTENCE')
        for w in modified_sentence:
            self.inc_word_unigram(w)
        
    def get_rare_words(self):
        for w in self.ngram_models[0].keys():
            if self.ngram_models[0][w] <= self.unk_cutoff:
                rand_remove_prob = random.uniform(0, 1)
                if rand_remove_prob >= 0.5:
                    self.rare_words[w] = True
                    self.rare_words_sum += self.ngram_models[0][w]
        
    # remove rare words and give sum to UNK token
    def remove_rare_set_unknown(self):
        for w in self.rare_words:
            self.ngram_models[0].pop(w, None)
        self.ngram_models[0]["UNK"] = self.rare_words_sum

    # find ngram counts (bigram and trigram)
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

    # fit sentence for bigram and trigram
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

    # fit corpus overidden
    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        for s in corpus:
            self.fit_sentence_unigram(s)
            
        self.get_rare_words()
        self.remove_rare_set_unknown()

        self.unigram_all_sum = float(np.array(list(self.ngram_models[0].values())).sum())

        for s in corpus:
            self.fit_sentence(s)

        self.norm()

    def norm(self):
        """Normalize and convert to log2-probs."""
        return

    # finding cond log prob for a given context and word
    def cond_logprob(self, word, previous):
        unigram_prob = 0.0
        bigram_prob = 0.0
        trigram_prob = 0.0
        prvs_context = []
        if len(previous) < self.n - 1:
            prvs_context.extend(["*"]*(self.n - 1 - len(previous)))
        prvs_context.extend(previous[-(self.n - 1):])
        
        word_2 = prvs_context[0]
        word_1 = prvs_context[1]
        if word_2 not in self.ngram_models[0]:
            word_2 = "UNK"
        if word_1 not in self.ngram_models[0]:
            word_1 = "UNK"
        if word not in self.ngram_models[0]:
            word = "UNK"
        
        if word in self.ngram_models[0]:
            unigram_prob = float(self.ngram_models[0][word]) / self.unigram_all_sum
        else:
            unigram_prob = 0.0
        
        if word_1 in self.ngram_models[1]:
            if word in self.ngram_models[1][word_1]:
                bigram_prob = float(self.ngram_models[1][word_1][word]) / float(self.ngram_models[0][word_1])
            else:
                bigram_prob = 0.0
        else:
            bigram_prob = 0.0
        
        if word_2 in self.ngram_models[2]:
            if word_1 in self.ngram_models[2][word_2]:
                if word in self.ngram_models[2][word_2][word_1]:
                    trigram_prob = float(self.ngram_models[2][word_2][word_1][word]) / float(self.ngram_models[1][word_2][word_1])
                else:
                    bigram_prob = 0.0
            else:
                trigram_prob = 0.0
        else:
            trigram_prob = 0.0
        

        final_prob = (unigram_prob * self.lamda[0]) + (bigram_prob *self.lamda[1]) + (trigram_prob * self.lamda[2])
        
        return log(final_prob, 2)

    # getting the vocabulary of the model
    def vocab(self):
        return self.ngram_models[0].keys()