#! /usr/bin/python

from operator import mod
from sre_compile import MAXCODE
import sys
from collections import defaultdict
import math
from unittest.mock import Base

"""
Count n-gram frequencies in a data file and write counts to
stdout. 
"""

def simple_conll_corpus_iterator(corpus_file):
    """
    Get an iterator object over the corpus file. The elements of the
    iterator contain (word, ne_tag) tuples. Blank lines, indicating
    sentence boundaries return (None, None).
    """
    l = corpus_file.readline()
    while l:
        line = l.strip()
        if line: # Nonempty line
            # Extract information from line.
            # Each line has the format
            # word pos_tag phrase_tag ne_tag
            fields = line.split(" ")
            ne_tag = fields[-1]
            #phrase_tag = fields[-2] #Unused
            #pos_tag = fields[-3] #Unused
            word = " ".join(fields[:-1])
            yield word, ne_tag
        else: # Empty line
            yield (None, None)                        
        l = corpus_file.readline()

def simple_conll_corpus_iterator_dev(corpus_file):
    """
    Get an iterator object over the dev corpus file. The elements of the
    iterator contain words. Blank lines, indicating
    sentence boundaries return (None, None).
    """
    l = corpus_file.readline()
    while l:
        line = l.strip()
        if line: # Nonempty line
            # Extract information from line.
            # Each line has the format
            # word pos_tag phrase_tag ne_tag
            yield line
        else: # Empty line
            yield None                        
        l = corpus_file.readline()

def sentence_iterator(corpus_iterator):
    """
    Return an iterator object that yields one sentence at a time.
    Sentences are represented as lists of (word, ne_tag) tuples.
    """
    current_sentence = [] #Buffer for the current sentence
    for l in corpus_iterator:        
            if l==(None, None):
                if current_sentence:  #Reached the end of a sentence
                    yield current_sentence
                    current_sentence = [] #Reset buffer
                else: # Got empty input stream
                    sys.stderr.write("WARNING: Got empty input file/stream.\n")
                    raise StopIteration
            else:
                current_sentence.append(l) #Add token to the buffer

    if current_sentence: # If the last line was blank, we're done
        yield current_sentence  #Otherwise when there is no more token
                                # in the stream return the last sentence.

def sentence_iterator_dev(corpus_iterator):
    """
    Return an iterator object that yields one sentence at a time.
    Sentences are represented as lists of (word, ne_tag) tuples.
    """
    current_sentence = [] #Buffer for the current sentence
    for l in corpus_iterator:        
            if l==None:
                if current_sentence:  #Reached the end of a sentence
                    yield current_sentence
                    current_sentence = [] #Reset buffer
                else: # Got empty input stream
                    sys.stderr.write("WARNING: Got empty input file/stream.\n")
                    raise StopIteration
            else:
                current_sentence.append(l) #Add token to the buffer

    if current_sentence: # If the last line was blank, we're done
        yield current_sentence  #Otherwise when there is no more token
                                # in the stream return the last sentence.

def get_ngrams(sent_iterator, n):
    """
    Get a generator that returns n-grams over the entire corpus,
    respecting sentence boundaries and inserting boundary tokens.
    Sent_iterator is a generator object whose elements are lists
    of tokens.
    """
    for sent in sent_iterator:
         #Add boundary symbols to the sentence
         w_boundary = (n-1) * [(None, "*")]
         w_boundary.extend(sent)
         w_boundary.append((None, "STOP"))
         #Then extract n-grams
         ngrams = (tuple(w_boundary[i:i+n]) for i in range(len(w_boundary)-n+1))
         for n_gram in ngrams: #Return one n-gram at a time
            yield n_gram        


def write_sentences(sentences, output_file):
    with open(output_file, 'w') as op:
        for sent in sentences:
            for word, ne_tag in sent:
                op.write(word + " " + ne_tag + "\n")
            op.write("\n")
        
def replace_rare_words(corpus_file, output_corpus_file, rare_cutoff):
    sentence_itr = sentence_iterator(simple_conll_corpus_iterator(corpus_file))
    word_freqs = {}
    rare_words = {}
    new_sentences = []
    old_sentences = []
    for sent in sentence_itr:
        for word, _ in sent:
            if word not in word_freqs:
                word_freqs[word] = 0
            word_freqs[word] += 1
        old_sentences.append(sent)

    for word in word_freqs:
        if word_freqs[word] < rare_cutoff:
            rare_words[word] = True
    for sent in old_sentences:
        new_sent = []
        for word, ne_tag in sent:
            if word in rare_words:
                new_sent.append(("_RARE_", ne_tag))
            else:
                new_sent.append((word, ne_tag))
        new_sentences.append(new_sent)
    write_sentences(new_sentences, output_corpus_file)

class Hmm(object):
    """
    Stores counts for n-grams and emissions. 
    """

    def __init__(self, n=3):
        assert n>=2, "Expecting n>=2."
        self.n = n
        self.emission_counts = defaultdict(int)
        self.ngram_counts = [defaultdict(int) for i in range(self.n)]
        self.all_states = set()

    def train(self, corpus_file):
        """
        Count n-gram frequencies and emission probabilities from a corpus file.
        """
        ngram_iterator = \
            get_ngrams(sentence_iterator(simple_conll_corpus_iterator(corpus_file)), self.n)

        for ngram in ngram_iterator:
            #Sanity check: n-gram we get from the corpus stream needs to have the right length
            assert len(ngram) == self.n, "ngram in stream is %i, expected %i" % (len(ngram, self.n))
            tagsonly = tuple([ne_tag for word, ne_tag in ngram]) #retrieve only the tags            
            for i in range(2, self.n+1): #Count NE-tag 2-grams..n-grams
                self.ngram_counts[i-1][tagsonly[-i:]] += 1
            
            if ngram[-1][0] is not None: # If this is not the last word in a sentence
                self.ngram_counts[0][tagsonly[-1:]] += 1 # count 1-gram
                self.emission_counts[ngram[-1]] += 1 # and emission frequencies

            # Need to count a single n-1-gram of sentence start symbols per sentence
            if ngram[-2][0] is None: # this is the first n-gram in a sentence
                self.ngram_counts[self.n - 2][tuple((self.n - 1) * ["*"])] += 1


    def write_counts(self, output, printngrams=[1,2,3]):
        """
        Writes counts to the output file object.
        Format:

        """
        # First write counts for emissions
        for word, ne_tag in self.emission_counts:            
            output.write("%i WORDTAG %s %s\n" % (self.emission_counts[(word, ne_tag)], ne_tag, word))


        # Then write counts for all ngrams
        for n in printngrams:            
            for ngram in self.ngram_counts[n-1]:
                ngramstr = " ".join(ngram)
                output.write("%i %i-GRAM %s\n" %(self.ngram_counts[n-1][ngram], n, ngramstr))

    def read_counts(self, corpusfile):

        self.n = 3
        self.emission_counts = defaultdict(int)
        self.ngram_counts = [defaultdict(int) for i in range(self.n)]
        self.all_states = set()
        nl = 0
        for line in corpusfile:
            nl += 1
            parts = line.strip().split(" ")
            count = float(parts[0])
            if parts[1] == "WORDTAG":
                ne_tag = parts[2]
                word = parts[3]
                self.emission_counts[(word, ne_tag)] = count
                self.all_states.add(ne_tag)
            elif parts[1].endswith("GRAM"):
                n = int(parts[1].replace("-GRAM",""))
                ngram = tuple(parts[2:])
                # if n == 3:
                #     print(ngram)
                self.ngram_counts[n-1][ngram] = count

class BaseTagger():
    def __init__(self, hmm, counts_file):
        self.hmm = hmm
        self.counts_file = counts_file
        self.emission_probs = {}
    
    def find_emission_probabilities(self):
        # emission_probabilities
        self.hmm.read_counts(self.counts_file)
        for word, ne_tag in self.hmm.emission_counts:
            if word not in self.emission_probs:
                self.emission_probs[word] = {}
            if ne_tag not in self.emission_probs[word]:
                self.emission_probs[word][ne_tag] = 0.0
            self.emission_probs[word][ne_tag] = float(self.hmm.emission_counts[(word, ne_tag)]) / float(self.hmm.ngram_counts[0][(ne_tag,)])

    def tag_word(self, word):
        max_prob = 0.0
        max_prob_tag = None
        for tag in self.emission_probs[word]:
            if self.emission_probs[word][tag] > max_prob:
                max_prob = self.emission_probs[word][tag]
                max_prob_tag = tag
        return max_prob_tag

    def tag_corpus(self, corpusfile, outfile):
        sentence_itr = sentence_iterator_dev(simple_conll_corpus_iterator_dev(corpusfile))
        new_sentences = []
        for sent in sentence_itr:
            new_sent = []
            for word in sent:
                if word not in self.emission_probs:
                    tag = self.tag_word("_RARE_")
                else:
                    tag = self.tag_word(word)
                
                new_sent.append((word, tag))
            new_sentences.append(new_sent)
        write_sentences(new_sentences, outfile)

class TrigramHMMTagger(BaseTagger):
    def __init__(self, hmm, counts_file):
        super().__init__(hmm, counts_file)
        self.param_probs = {}

    def find_param_probabilities(self):
        self.find_emission_probabilities()
        for (word_2, word_1, word_0) in self.hmm.ngram_counts[2]:
            trigram_count = self.hmm.ngram_counts[2][(word_2, word_1, word_0)]
            bigram_count = self.hmm.ngram_counts[1][(word_2, word_1)]
            q_prob = float(trigram_count) / float(bigram_count)
            if word_2 not in self.param_probs:
                self.param_probs[word_2] = {}
            if word_1 not in self.param_probs[word_2]:
                self.param_probs[word_2][word_1] = {}
            
            self.param_probs[word_2][word_1][word_0] = q_prob
            # print("word0 | word1, word2: " + str(word_0) + " " + str(word_1) + str(word_2))
            # print("prob:" + str(q_prob))
    
    def compute_prob(self, dp_mat, k, u, v, w, sentence, em_prob_compute=True):
        if k - 1 not in dp_mat:
            return 0.0
        if w not in dp_mat[k - 1]:
            return 0.0
        if u not in dp_mat[k - 1][w]:
            return 0.0
        
        mat_prob = dp_mat[k - 1][w][u]

        if w not in self.param_probs:
            return 0.0
        if u not in self.param_probs[w]:
            return 0.0
        if v not in self.param_probs[w][u]:
            return 0.0
        q_prob = self.param_probs[w][u][v]
        if em_prob_compute:
            if sentence[k - 1] not in self.emission_probs:
                return 0.0
            if v not in self.emission_probs[sentence[k - 1]]:
                return 0.0
            
            em_prob = self.emission_probs[sentence[k - 1]][v]
        else:
            em_prob = 1.0

        return mat_prob * q_prob * em_prob


    def tag_sentence_viterbi(self, sentence):
        dp_mat = {0:{"*":{"*":1.0}}}
        backing_mat = {}
        base_set = set(["*"])
        tag_set = set(self.hmm.ngram_counts[0].keys())
        n = len(sentence)
        for k in range(1, n + 1):
            tag_set_k_2 = tag_set
            if k <= 2:
                tag_set_k_2 = base_set
            tag_set_k_1 = tag_set
            if k <= 1:
                tag_set_k_1 = base_set
            tag_set_k = tag_set
            for u in tag_set_k_1:
                for v in tag_set_k:
                    max_prob = -0.01
                    max_prob_tag = None
                    for w in tag_set_k_2:
                        prob = self.compute_prob(dp_mat, k, u, v, w, sentence)
                        if prob > max_prob:
                            max_prob = prob
                            max_prob_tag = sentence[k - 1]
                    
                    if k not in dp_mat:
                        dp_mat[k] = {}
                    if u not in dp_mat[k]:
                        dp_mat[k][u] = {}
                    dp_mat[k][u][v] = max_prob

                    if k not in backing_mat:
                        backing_mat[k] = {}
                    if u not in backing_mat[k]:
                        backing_mat[k][u] = {}
                    backing_mat[k][u][v] = max_prob_tag

        tag_n_1 = None
        tag_n = None
        max_prob = -0.01
        for u in tag_set:
            for v in tag_set:
                prob = self.compute_prob(dp_mat, n, v, "STOP", u, sentence)
                if prob > max_prob:
                    max_prob = prob
                    tag_n_1 = u
                    tag_n = v
        tags = [None for i in range(n)]
        tags[n - 2] = tag_n_1
        tags[n - 1] = tag_n
        for k in range(n - 3, -1, -1):
            tags[k] = backing_mat[k + 2][tags[k + 1]][tags[k + 2]]

        tagged_sentence = []
        for i in range(n):
            tagged_sentence.append((sentence[i], tags[i]))
        return tagged_sentence

    def tag_corpus(self, corpusfile, outfile):
        sentence_itr = sentence_iterator_dev(simple_conll_corpus_iterator_dev(corpusfile))
        new_sentences = []
        for sent in sentence_itr:
            tagged_sentence = self.tag_sentence_viterbi(sent)
            new_sentences.append(tagged_sentence)
        write_sentences(new_sentences, outfile)

def usage():
    print ("""
    python count_freqs.py [input_file] > [output_file]
        Read in a gene tagged training input file and produce counts.
    """)

if __name__ == "__main__":
    if len(sys.argv)!=2: # Expect exactly one argument: the training data file
        usage()
        sys.exit(2)

    try:
        input = open(sys.argv[1],"r")
    except IOError:
        sys.stderr.write("ERROR: Cannot read inputfile %s.\n" % arg)
        sys.exit(1)
    
    modified_inp = "gene_mod.train"
    rare_words_cutoff = 5
    replace_rare_words(input, modified_inp, rare_words_cutoff)
    
    mod_inp = open(modified_inp)
    # Initialize a trigram counter
    counter = Hmm(3)
    # Collect counts
    counter.train(mod_inp)
    # Write the counts
    op = open("gene.counts", "w")
    counter.write_counts(op)
    op.close()
    op_read = open("gene.counts", "r")

    # bt = BaseTagger(counter, op_read)
    # bt.find_emission_probabilities()
    # dev_in  = open("gene.dev")
    # bt.tag_corpus(dev_in, "gene_dev.p1.out")
    tt = TrigramHMMTagger(counter, op_read)
    tt.find_param_probabilities()
    dev_in = open("gene.dev")
    tt.tag_corpus(dev_in, "gene_dev.p1.out")

    input.close()
    mod_inp.close()
    op_read.close()
    dev_in.close()