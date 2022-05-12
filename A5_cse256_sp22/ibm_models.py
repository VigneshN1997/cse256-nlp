
class IBMModel1():
    def __init__(self, english_corpus, foreign_corpus):
        self.eng_corpus = english_corpus
        self.for_corpus = foreign_corpus
        self.read_corpus_addnull()
        self.calc_num_diff_words()
        self.translation_probs = {}
        self.counts_e_f = {}
        self.counts_e = {}
        self.counts_align = {}
        self.counts_align_f = {}
        

    def read_corpus_addnull(self):
        "Reads a corpus and adds in the NULL word."
        self.english = [["*"] + e_sent.split() for e_sent in open(self.eng_corpus)]
        self.foreign = [f_sent.split() for f_sent in open(self.for_corpus)]
        self.parallel_corpus = zip(self.english, self.foreign)

    def num_sentences(self):
        print(self.english[1000])
        return len(self.english)
    
    def calc_num_diff_words(self):
        self.n_e = {}
        wordpairs = set()
        for e, s in self.parallel_corpus:
            for e_j in e:
                for s_i in s:
                    wordpair = (e_j, s_i)
                    if wordpair not in wordpairs:
                        wordpairs.add(wordpair)
                        if not e_j in self.n_e:
                            self.n_e[e_j] = 0
                        self.n_e[e_j]+=1

english_corpus = "corpus.en"
spanish_corpus = "corpus.es"
mod = IBMModel1(english_corpus, spanish_corpus)
print(mod.num_sentences())
