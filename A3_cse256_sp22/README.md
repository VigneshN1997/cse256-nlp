# cse256-nlp
CSE 256 NLP assignments

# The linear interpolation smoothing trigram class is defined in lm.py : TrigramLinInter(unk_cutoff=10.0, lamda=[0.2, 0.3, 0.5]) class

There are two hyperparameters unk_cutoff and lamda (lambda1, lambda2, lambda3 in that order) 
The algorithm is completely implemented in lm.py TrigramLinInter() class.

1)	Computing n-gram counts (inc_word_unigram, fit_sentence_unigram, get_rare_words, remove_rare_set_unknown, inc_ngram_count, fit_sentence, fit_corpus): I first compute the unigram counts to get the vocabulary. Using the unigram counts, I find out the rare words in the corpus (Hyperparameter: Unknown cut off). A word is considered rare if the frequency of the word is <= unknown cut off. Rare words are removed from the corpus and their counts are assigned to the “UNK” (unknown) word to handle OOV words. I tried randomly removing some rare words (using a coin flip strategy). After the vocabulary is set, I compute the bigram and trigram counts (adding start of sentence * and end of sentence words for simplifying counting). 
2)	Computing conditional probability (norm, cond_logprob): This function finds out the probability of a word given its past context (previous). Since the linear interpolation method is used, I compute the unigram, bigram and trigram probabilities first. The words and the context words are pre-processed to “UNK” if any word is OOV. For all n-gram probabilities, if the n-gram is present, the probability is non-zero else the probability will be 0. The final probability will be computed using the lambda values (hyperparameter) and is given using the below equation. This probability is given to log function for further use.

# In data.py the function learn_unigram can be replaced by learn_trigram function defined in data.py to train the trigram model

# model training (defined in jupyter notebook)
```
def fit_model(train_data, model_type="unigram", lamda=[0.2, 0.3, 0.5], unk_cutoff=10.0):
    if model_type == "unigram":
        mod = Unigram()
        mod.fit_corpus(train_data)
        return mod
    else:
        mod = TrigramLinInter(unk_cutoff, lamda)
        mod.fit_corpus(train_data)
        return mod
trigram_mod = fit_model(data.train, "trigram", unk_cutoff=unk_val, lamda=lamda_fix)
```

# generating sentences with model
```
prefix1 = "At the moment ".split()
prefix2 = "The company reported".split()
prefix3 = "Debt of gratitude".split()
sampler_b = Sampler(models_trigram[0])
sampler_r = Sampler(models_trigram[1])
sampler_g = Sampler(models_trigram[2])
' '.join(sampler_g.sample_sentence("Debt of gratitude".split()))
# finding log prob of sentence
print(models_trigram[0].logprob_sentence(a11.split()))

```
The jupyter notebook cse256_a3.ipynb contains all the experiments conducted for the assignment and the results for each experiment. It also contains the generated sentences and the code for adaptating model to other datasets.