Running the code: (in count_freqs.py)

rare_words_cutoff = 5 # the rare words cutoff
modified_inp_suff = "gene_mod_suff.train" # where to save the modified input with infrequent words replaced
rare_word_suffix_file = "rare_words.txt" # the file containing the rare word suffixes (this can be tweaked in the jupyter notebook: infreq_words_testing.ipynb)
rare_word_suffixes =  replace_rare_words_diff(input, modified_inp_suff, rare_word_suffix_file, rare_words_cutoff) # replace the rare words

mod_inp = open(modified_inp_suff)
# Initialize a trigram counter
counter = Hmm(3)
# Collect counts
counter.train(mod_inp)
# Write the counts
op = open("gene.counts", "w")
counter.write_counts(op)
op.close()
op_read = open("gene.counts", "r")

# for running the base tagger, the counter, the counts file, and the rare word suffixes dictionary is needed as input

# bt = BaseTagger(counter, op_read, rare_word_suffixes)
# bt.find_emission_probabilities()
# dev_in  = open("gene.dev")
# bt.tag_corpus(dev_in, "gene_dev.p1.out")

# for the Trigram HMM tagger with the Viterbi decoder: an extra argument needed here is the lamda values for smoothing ([lamda_uni, lamda_bi, lamda_tri])
tt = TrigramHMMTagger(counter, op_read, rare_word_suffixes, lamdas=[0.6, 0.3, 0.1])
tt.find_param_probabilities(smooth=True) # smooth = True will use smoothing
dev_in = open("gene.dev")
tt.tag_corpus(dev_in, "gene_dev.p1.out") # tagging done using the Viterbi algorithm

# then run: python3 count_freqs.py gene.train
# python eval_gene_tagger.py gene.key gene_dev.p1.out

In the jupyter notebook analysis was done to find the infrequent words and different word classes.

The main cell doing the word class processing for suffixes:

common_3suffixes = {}
common_2suffixes = {}
for w in rare_words:
    if len(w) >= 3:
        suffix2 = w[-2:]
        suffix3 = w[-3:]
        if suffix2 not in common_2suffixes:
            common_2suffixes[suffix2] = 0
        if suffix3 not in common_3suffixes:
            common_3suffixes[suffix3] = 0
        common_2suffixes[suffix2] += 1
        common_3suffixes[suffix3] += 1

three_suffixes = common_3suffixes.keys()
delete_suf = []
for suffix in three_suffixes:
    if common_3suffixes[suffix] < 30: # suffix cutoff
        delete_suf.append(suffix)
for suf in delete_suf:
    del common_3suffixes[suf]

three_suffixes = common_3suffixes.keys()
two_in_three_suff = set([x[-2:] for x in three_suffixes])
delete_suff = []
for s in common_2suffixes:
    if s in two_in_three_suff:
        delete_suff.append(s)

for suf in common_2suffixes:
    if common_2suffixes[suf] < 30 and suf not in delete_suff: # suffix cutoff
        delete_suff.append(suf)
# print(delete_suff)
for suf in delete_suff:
    del common_2suffixes[suf]

In both lines 55 and 68, the suffix cutoff can be tweaked to perform different experiments with different suffix cutoff values

And then cells below the above cell in the notebook file have to be run to generate the rare_words.txt file required for new word classes.