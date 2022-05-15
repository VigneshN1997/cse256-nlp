
class IBMModel1():
    def __init__(self, english_corpus, foreign_corpus):
        self.eng_corpus = english_corpus
        self.for_corpus = foreign_corpus
        self.counts_e_f = {}
        self.counts_e = {}
        self.counts_align = {}
        self.counts_align_f = {}
        self.read_corpus_addnull()
        self.calc_num_diff_words()
        self.init_translation_probs()
        self.params_file = "translation.params"

    def read_corpus_addnull(self):
        "Reads a corpus and adds in the NULL word."
        self.english = [["*"] + e_sent.split() for e_sent in open(self.eng_corpus)]
        self.foreign = [f_sent.split() for f_sent in open(self.for_corpus)]
        self.n_sentences = len(self.english)
        # print(len(self.english))


    def calc_num_diff_words(self):
        self.n_e = {}
        wordpairs = set()
        for k in range(self.n_sentences):
            e = self.english[k]
            s = self.foreign[k]
            for e_j in e:
                for s_i in s:
                    wordpair = (e_j, s_i)
                    if wordpair not in wordpairs:
                        wordpairs.add(wordpair)
                        if not e_j in self.n_e:
                            self.n_e[e_j] = 0
                        self.n_e[e_j]+=1
    
    def init_translation_probs(self):
        # print("here")
        self.translation_probs = {} # t(f | e)
        self.deltas = {}
        for k in range(self.n_sentences):
            eng_sent = self.english[k]
            for_sent = self.foreign[k]
            l = len(eng_sent)
            m = len(for_sent)
            # print(l, m)
            for i in range(m):
                f = for_sent[i]
                if f not in self.translation_probs:
                    self.translation_probs[f] = {}
                if i not in self.counts_align_f:
                    self.counts_align_f[i] = {}
                if l not in self.counts_align_f[i]:
                    self.counts_align_f[i][l] = {}
                if m not in self.counts_align_f[i][l]:
                    self.counts_align_f[i][l][m] = 0
                for j in range(l):
                    e = eng_sent[j]
                    self.translation_probs[f][e] = 1 / self.n_e[e]
                    # print(f, e, str(self.translation_probs[f][e]))
                    if e not in self.counts_e_f:
                        self.counts_e_f[e] = {}
                    self.counts_e_f[e][f] = 0
                    self.counts_e[e] = 0
                    if j not in self.counts_align:
                        self.counts_align[j] = {}
                    if i not in self.counts_align[j]:
                        self.counts_align[j][i] = {}
                    if l not in self.counts_align[j][i]:
                        self.counts_align[j][i][l] = {}
                    if m not in self.counts_align[j][i][l]:
                        self.counts_align[j][i][l][m] = 0

                    if k not in self.deltas:
                        self.deltas[k] = {}
                    if i not in self.deltas[k]:
                        self.deltas[k][i] = {}
                    self.deltas[k][i][j] = 0

    def find_new_translations(self):
        for k in range(self.n_sentences):
            eng_sent = self.english[k]
            for_sent = self.foreign[k]
            l = len(eng_sent)
            m = len(for_sent)
            for i in range(m):
                f = for_sent[i]
                for j in range(l):
                    e = eng_sent[j]
                    self.translation_probs[f][e] = self.counts_e_f[e][f] / self.counts_e[e]

    def set_counts_zero(self):
        for k in range(self.n_sentences):
            eng_sent = self.english[k]
            for_sent = self.foreign[k]
            l = len(eng_sent)
            m = len(for_sent)
            for i in range(m):
                f = for_sent[i]
                self.counts_align_f[i][l][m] = 0
                for j in range(l):
                    e = eng_sent[j]
                    self.counts_e_f[e][f] = 0
                    self.counts_e[e] = 0
                    self.counts_align[j][i][l][m] = 0

    def write_params_to_file(self):
        with open(self.params_file, "w") as pf:
            lines = []
            for k in range(self.n_sentences):
                eng_sent = self.english[k]
                for_sent = self.foreign[k]
                l = len(eng_sent)
                m = len(for_sent)
                for i in range(m):
                    f = for_sent[i]
                    for j in range(l):
                        e = eng_sent[j]
                        lines.append(str(f) + " " + str(e) + " " + str(self.translation_probs[f][e]) + "\n")
            pf.writelines(lines)

    def estimate_params(self, n_iterations):
        n_sentences = len(self.english)
        for s in range(n_iterations):
            print("iteration:" + str(s))
            self.set_counts_zero()
            for k in range(n_sentences):
                eng_sent = self.english[k]
                for_sent = self.foreign[k]
                nwords_eng = len(eng_sent)
                nwords_for = len(for_sent)
                l = nwords_eng
                m = nwords_for
                for i in range(m):
                    norm_den = 0
                    f = for_sent[i]
                    for j in range(l):
                        norm_den += self.translation_probs[f][eng_sent[j]]
                    for j in range(l):
                        e = eng_sent[j]
                        delta_k_i_j = self.translation_probs[f][e] / norm_den
                        self.counts_e_f[e][f] += delta_k_i_j
                        self.counts_e[e] += delta_k_i_j
                        self.counts_align[j][i][l][m] += delta_k_i_j
                        self.counts_align_f[i][l][m] += delta_k_i_j
            self.find_new_translations()
            print("done iteration:" + str(s))
        self.write_params_to_file()
    
    def read_params_from_file(self):
        with open(self.params_file) as pf:
            lines = [line[:-1].split() for line in pf.readlines()]
            for line in lines:
                self.translation_probs[line[0]][line[1]] = float(line[2])
                # print(line[0], line[1], str(self.translation_probs[line[0]][line[1]]))

    def find_alignment_sentence(self, eng_sent, for_sent, sentence_num):
        alignments = []
        l = len(eng_sent)
        m = len(for_sent)
        for i in range(m):
            max_prob = 0.0
            max_prob_index = -1
            f = for_sent[i]
            a = [sentence_num, 0, i + 1]
            if f in self.translation_probs:
                for j in range(l):
                    e = eng_sent[j]
                    if e not in self.translation_probs[f]:
                        continue
                    # if sentence_num == 2 and i+1==3:
                    #     print(j,f,e,self.translation_probs[f][e])

                    if self.translation_probs[f][e] > max_prob:
                        max_prob = self.translation_probs[f][e]
                        max_prob_index = j
                a[1] = max_prob_index
            else:
                continue
            if max_prob_index == -1:
                continue
            alignments.append(str(a[0]) + " " + str(a[1]) + " " + str(a[2]) + "\n")
        return alignments
        

    def find_alignments(self, dev_eng_corpus, dev_for_corpus, out_key_file):
        dev_eng = [["*"] + e_sent.split() for e_sent in open(dev_eng_corpus)]
        dev_for = [f_sent.split() for f_sent in open(dev_for_corpus)]
        parallel_dev_corpus = zip(dev_eng, dev_for)
        # self.read_params_from_file()
        sentence_num = 1
        all_alignments = []
        for eng_sent, for_sent in parallel_dev_corpus:
            alignments = self.find_alignment_sentence(eng_sent, for_sent, sentence_num)
            all_alignments.extend(alignments)
            sentence_num += 1
        with open(out_key_file, "w")as of:
            of.writelines(all_alignments)

class IBMModel2(IBMModel1):
    def __init__(self, english_corpus, foreign_corpus):
        super().__init__(english_corpus, foreign_corpus)
        self.read_params_from_file()
        self.init_soft_alignment_params()
    
    def init_soft_alignment_params(self):
        self.q_params = {}
        for k in range(self.n_sentences):
            eng_sent = self.english[k]
            for_sent = self.foreign[k]
            l = len(eng_sent)
            m = len(for_sent)
            for i in range(m):
                for j in range(l):
                    if j not in self.q_params:
                        self.q_params[j] = {}
                    if i not in self.q_params[j]:
                        self.q_params[j][i] = {}
                    if l not in self.q_params[j][i]:
                        self.q_params[j][i][l] = {}
                    if m not in self.q_params[j][i][l]:
                        self.q_params[j][i][l][m] = 1.0/float(l + 1)
    
    def find_new_translations2(self):
        for k in range(self.n_sentences):
            eng_sent = self.english[k]
            for_sent = self.foreign[k]
            l = len(eng_sent)
            m = len(for_sent)
            for i in range(m):
                f = for_sent[i]
                for j in range(l):
                    e = eng_sent[j]
                    self.translation_probs[f][e] = self.counts_e_f[e][f] / self.counts_e[e]
                    self.q_params[j][i][l][m] = self.counts_align[j][i][l][m] / self.counts_align_f[i][l][m]


    def estimate_params(self, n_iterations):
        n_sentences = len(self.english)
        for s in range(n_iterations):
            print("model 2 iteration:" + str(s))
            self.set_counts_zero()
            for k in range(n_sentences):
                eng_sent = self.english[k]
                for_sent = self.foreign[k]
                nwords_eng = len(eng_sent)
                nwords_for = len(for_sent)
                l = nwords_eng
                m = nwords_for
                for i in range(m):
                    norm_den = 0.0
                    f = for_sent[i]
                    for j in range(l):
                        norm_den +=  (self.q_params[j][i][l][m] * self.translation_probs[f][eng_sent[j]])
                    for j in range(l):
                        e = eng_sent[j]
                        delta_k_i_j = (self.q_params[j][i][l][m] * self.translation_probs[f][e]) / norm_den
                        self.counts_e_f[e][f] += delta_k_i_j
                        self.counts_e[e] += delta_k_i_j
                        self.counts_align[j][i][l][m] += delta_k_i_j
                        self.counts_align_f[i][l][m] += delta_k_i_j
            self.find_new_translations2()
            print("done iteration:" + str(s))
    
    def find_alignment_sentence(self, eng_sent, for_sent, sentence_num):
        alignments = []
        l = len(eng_sent)
        m = len(for_sent)
        for i in range(m):
            max_prob = 0.0
            max_prob_index = -1
            f = for_sent[i]
            a = [sentence_num, 0, i + 1]
            if f not in self.translation_probs:
                continue

            if f in self.translation_probs:
                for j in range(l):
                    e = eng_sent[j]
                    if e not in self.translation_probs[f]:
                        continue
                    if j not in self.q_params:
                        continue
                    if i not in self.q_params[j]:
                        continue
                    if l not in self.q_params[j][i]:
                        continue
                    if m not in self.q_params[j][i][l]:
                        continue
                    val = self.q_params[j][i][l][m] * self.translation_probs[f][e]
                    if val > max_prob:
                        max_prob = val
                        max_prob_index = j
                if max_prob_index == -1:
                    continue
                a[1] = max_prob_index
            else:
                continue
            alignments.append(str(a[0]) + " " + str(a[1]) + " " + str(a[2]) + "\n")
        return alignments


english_corpus = "corpus.en"
spanish_corpus = "corpus.es"
# mod = IBMModel1(english_corpus, spanish_corpus)
# mod.estimate_params(5)
# mod.find_alignments("dev.en", "dev.es", "dev.out")

mod2 = IBMModel2(english_corpus, spanish_corpus)
mod2.estimate_params(5)
mod2.find_alignments("dev.en", "dev.es", "dev2.out")