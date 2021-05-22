# Splits the sentences using newline
import os
import sys
import pickle


# input - training set
# ouput - list of sentences from input file
def text_retrieve(file_name):
    with open(file_name, 'r') as f:
        text = f.read()
    f.close()
    return text.split('\n')


# input - list of sentences from input file
# output - sentences without POS tags, Vocab(V), Total No of words(N)
def data_cleaning(lines):
    new_lines = []
    vocab = []
    count = 0
    for i in lines:
        i = i.lower()
        words = i.split(' ')
        new_words = []
        for j in words:
            count += 1
            word_split = j.split('_')
            if len(word_split) > 2:
                word = '_'.join(word_split[:-1])
            else:
                word = word_split[0]
            if word == ' ' or word == '':
                continue
            else:
                new_words.append(word)
            if word not in vocab:
                vocab.append(word)
        new_lines.append(' '.join(new_words))
    return new_lines, vocab, count

# input - Sentences without POS tag
# output - Unigram count & bigram count in the training set
def createBigrams (new_lines):
    count_bigrams = {}
    count_unigrams = {}

    for l in new_lines:
        l = l.lower()
        l_words = l.split(' ')
        for j in range(len(l_words) - 1):
            if l_words[j] not in count_unigrams:
                count_unigrams[l_words[j]] = 1
            else:
                count_unigrams[l_words[j]] += 1
            if (l_words[j], l_words[j+1]) not in count_bigrams:
                count_bigrams[(l_words[j], l_words[j+1])] = 1
            else:
                count_bigrams[(l_words[j], l_words[j + 1])] += 1

    return count_unigrams, count_bigrams


# input - Unigram count, bigram count, total no of words i the training set
# output - Bigram & unigram Probability - No Smoothing
def noSmoothing_Prob(count_bigrams, count_unigrams, total_count):
    p_bigram = {}
    p_unigram = {}

    for unigram in count_unigrams:
        p_unigram[unigram] = count_unigrams[unigram]/total_count

    for bigram in count_bigrams.keys():
        p_bigram[bigram] = count_bigrams[bigram]/count_unigrams[bigram[0]]
    return p_bigram, p_unigram


# input - Unigram count & bigram count in the training set, total no of words i the training set, vocab
# output -
def addone_prob(count_bigrams, count_unigrams, vocab, total_count):
    unigramsProb_addone = {}                # (C(w(n)) + 1) / (N + V)
    bigramsProb_addone = {}
    numerator = {}                          # C(w(n-1) w(n)) + 1
    denominator = {}                        # C(w(n-1)) + v


    for unigram in count_unigrams.keys():
        denominator[unigram] = count_unigrams[unigram] + len(vocab)
        unigramsProb_addone[unigram] = (count_unigrams[unigram] + 1) * total_count / (total_count + len(vocab))

    x="x"
    for i in count_unigrams.keys():
        for j in count_unigrams.keys():
            if (i, j) in count_bigrams.keys():
                numerator[(i, j)] = count_bigrams[(i, j)] + 1
            if (i, x) not in numerator:
                numerator[(i, x)] = 1

    for i in count_unigrams.keys():
        for j in count_unigrams.keys():
            if (i, j) in count_bigrams.keys():
                bigramsProb_addone[(i,j)] = numerator[(i,j)] / denominator[i]
            if (i, x) not in bigramsProb_addone:
                bigramsProb_addone[(i, x)] = numerator[(i, x)] / denominator[i]

    return bigramsProb_addone, unigramsProb_addone

def goodTuring_prob(count):
    Nc = {}
    c_star = {}
    Prob_goodTuring = {}
    N = sum(count.values())

    for bigram in count.keys():
        if count[bigram] not in Nc.keys():
            Nc[count[bigram]] = 1
        else:
            Nc[count[bigram]] += 1

    for c in Nc.keys():
        if c+1 not in Nc.keys():
            c_star[c] = 0
        else:
            c_star[c] = (c + 1) * Nc[c+1] / Nc[c]

    Prob_goodTuring[0] = Nc[1] / N
    for c in Nc.keys():
        Prob_goodTuring[c] =  c_star[c] / N

    return Prob_goodTuring


def save_obj(filename, dict, type):
    print('Saving probabilities in a pickle file - ', type)
    pickle.dump(dict, open(filename + '.pkl', 'wb'))

def load_obj(filename):
    print('Retriving probabilities from pickle file for testing')
    return pickle.load(open(filename + '.pkl', 'rb'))

def noSmoothing_testing(l, bigram_prob, unigram_prob):
    unigrams, bigrams = createBigrams(l)
    prob = 1

    # probability of first word of the sentence - p(w(1))
    words = (l[0].lower()).split()
    if words[0] not in unigram_prob.keys():
        prob *= 0
    else:
        prob *= unigram_prob[words[0]]

    # probability of all bigrams of the sentence - p(w(n) / w(n-1))
    for b in bigrams.keys():
        if b not in bigram_prob.keys():
            prob *= 0
        else:
            prob *= bigram_prob[b]
    return prob


def addone_testing(l, bigram_prob, unigram_prob):
    unigrams, bigrams = createBigrams(l)
    prob = 1

    # probability of first word of the sentence - p(w(1))
    words = (l[0].lower()).split()
    if words[0] not in unigram_prob.keys():
        prob *= 0
    else:
        prob *= unigram_prob[words[0]]

    # probability of all bigrams of the sentence - p(w(n) / w(n-1))
    x = "x"
    for b in bigrams.keys():
        if b not in bigram_prob.keys():
            prob *= bigram_prob[(b[0],x)]
        else:
            prob *= bigram_prob[b]
    return prob

def goodTuring_testing(l, bigram_prob, unigram_prob, count_bigram, count_unigram):
    unigrams, bigrams = createBigrams(l)
    prob = 1

    # probability of first word of the sentence - p(w(1))
    words = (l[0].lower()).split()
    if count_unigram[words[0]] not in unigram_prob.keys():
        prob *= 0
    else:
        prob *= unigram_prob[count_unigram[words[0]]]

    # probability of all bigrams of the sentence - p(w(n) / w(n-1))
    for b in bigrams.keys():
        if count_bigram[b] not in bigram_prob.keys():
            prob *= 0
        else:
            prob *= bigram_prob[count_bigram[b]]
    return prob

def main():
    if not os.path.exists('noSmooth_Prob_b.pkl'):
        lines = text_retrieve('NLP6320_POSTaggedTrainingSet-Windows.txt')
        new_lines, vocab, total_count = data_cleaning(lines)
        count_unigrams, count_bigrams = createBigrams(new_lines)

        p_bigram, p_unigram = noSmoothing_Prob(count_bigrams, count_unigrams, total_count)
        save_obj('noSmooth_Prob_b', p_bigram, 'No Smoothing')
        save_obj('noSmooth_Prob_u', p_unigram, 'No Smoothing')

        p_bigram_addone, p_unigram_addone = addone_prob(count_bigrams, count_unigrams, vocab, total_count)
        save_obj('addone_Prob_b', p_bigram_addone, 'Add-One Smoothing')
        save_obj('addone_Prob_u', p_unigram_addone, 'Add-One Smoothing')

        p_bigram_goodTuring = goodTuring_prob(count_bigrams)
        p_unigram_goodTuring = goodTuring_prob(count_unigrams)
        save_obj('goodturing_Prob_b', p_bigram_goodTuring, 'Good-Turing Discounting')
        save_obj('goodturing_Prob_u', p_unigram_goodTuring, 'Good-Turing Discounting')
        save_obj('count_b', count_bigrams, 'Good-Turing Discounting')
        save_obj('count_u', count_unigrams, 'Good-Turing Discounting')

    else:

        noSmooth_Prob_b = load_obj('noSmooth_Prob_b')
        noSmooth_Prob_u = load_obj('noSmooth_Prob_u')
        addone_Prob_b = load_obj('addone_Prob_b')
        addone_Prob_u = load_obj('addone_Prob_u')
        goodturing_Prob_b = load_obj('goodturing_Prob_b')
        goodturing_Prob_u = load_obj('goodturing_Prob_u')
        count_b = load_obj('count_b')
        count_u = load_obj('count_u')
        print()

        s = sys.argv[1]
        l = []
        l.append(s)
        if len(sys.argv) == 3:
            if sys.argv[2] == 'noSmoothing':
                prob1 = noSmoothing_testing(l, noSmooth_Prob_b, noSmooth_Prob_u)
                print("Probability of the test Sentence - No Smoothing", prob1)
            if sys.argv[2] =='addOne':
                prob2 = addone_testing(l, addone_Prob_b, addone_Prob_u)
                print("Probability of the test Sentence - Add-One Smoothing", prob2)
            if sys.argv[2] =='goodTuring':
                prob3 = goodTuring_testing(l, goodturing_Prob_b, goodturing_Prob_u, count_b, count_u)
                print("Probability of the test Sentence - Good-Turing Smoothing", prob3)
        else:
            prob1 = noSmoothing_testing(l, noSmooth_Prob_b, noSmooth_Prob_u)
            print("Probability of the test Sentence - No Smoothing", prob1)
            prob2 = addone_testing(l, addone_Prob_b, addone_Prob_u)
            print("Probability of the test Sentence - Add-One Smoothing", prob2)
            prob3 = goodTuring_testing(l, goodturing_Prob_b, goodturing_Prob_u, count_b, count_u)
            print("Probability of the test Sentence - Good-Turing Smoothing", prob3)

main()
