# Splits the sentences using newline
import os
import sys
import pickle
import numpy as np

# input - training set
# ouput - list of sentences from input file
def text_retrieve(file_name):
    with open(file_name, 'r') as f:
        text = f.read()
    f.close()
    return text.split('\n')


# input - list of sentences from input file
# output - sentences without POS tags, list of POS tags corresponding to each sentence, Vocab(v)
def data_cleaning(lines):
    sentences = []
    tags = []
    vocab = []
    count = 0
    for i in lines:
        words = i.split(' ')
        new_words = []
        new_tags = []
        for j in words:
            count += 1
            word_split = j.split('_')
            if word_split[0] == '':
                continue
            else:
                if len(word_split) > 2:
                    word = '_'.join(word_split[:-1])
                    tag = word_split[-1]
                else:
                    word = word_split[0]
                    tag = word_split[1]
            new_words.append(word)
            new_tags.append(tag)
            if word not in vocab:
                vocab.append(word)
        sentences.append(' '.join(new_words))
        tags.append(' '.join(new_tags))
    return sentences, tags, vocab

# input - Sentences without POS tag, list of POS tags corresponding to each sentence
# output - bigram count of word & its tag, bigram count of a tag & previous tag, unigram count of each tag
def createBigrams (sentences, tags):
    count_bigrams_word = {}
    count_bigrams_tag = {}
    count_unigrams_tag = {}

    for s,t in zip(sentences,tags):
        word = s.split(' ')
        tag = t.split(' ')
        for i,j in zip(word,tag):
            if (i,j) not in count_bigrams_word:
                count_bigrams_word[(i,j)] = 1
            else:
                count_bigrams_word[(i, j)] += 1

    for t in tags:
        tag = t.split(' ')
        if (tag[0], '<s>') not in count_bigrams_tag:
            count_bigrams_tag[(tag[0], '<s>')] = 1
        else:
            count_bigrams_tag[(tag[0], '<s>')] += 1

        for i in range(1, len(tag)):
            if (tag[i-1],tag[i]) not in count_bigrams_tag:
                count_bigrams_tag[(tag[i-1],tag[i])] = 1
            else:
                count_bigrams_tag[(tag[i-1],tag[i])] += 1

        if ('</s>', tag[-1]) not in count_bigrams_tag:
            count_bigrams_tag[('</s>', tag[-1])] = 1
        else:
            count_bigrams_tag[('</s>', tag[-1])] += 1

    for t in tags:
        tag = t.split(' ')
        for i in range(len(tag)):
            if tag[i] not in count_unigrams_tag:
                count_unigrams_tag[tag[i]] = 1
            else:
                count_unigrams_tag[tag[i]] += 1
        count_unigrams_tag['<s>'] = len(sentences)
        count_unigrams_tag['</s>'] = len(sentences)

    return count_bigrams_word, count_bigrams_tag, count_unigrams_tag


# input - bigram count of word & its tag, bigram count of a tag & previous tag, unigram count of each tag
# output - Probability of word given tag, probability of tag given previous tag
def compute_Prob(bigrams_word, bigrams_tag, unigrams_tag):
    p_word = {}
    p_tag = {}

    for (i,j) in bigrams_word.keys():
        p_word[(i,j)] = bigrams_word[(i,j)]/unigrams_tag[j]

    for (i,j) in bigrams_tag.keys():
        p_tag[(i,j)] = bigrams_tag[(i,j)]/unigrams_tag[i]

    return p_word, p_tag

# Saves probabilities in pickle file
def save_obj(filename, dict, type):
    print('Saving probabilities in a pickle file - ', type)
    pickle.dump(dict, open(filename + '.pkl', 'wb'))

# Loads probabilites from pickle file
def load_obj(filename):
    print('Retriving probabilities from pickle file for testing')
    return pickle.load(open(filename + '.pkl', 'rb'))

# Input - Input sequence, model probabilities
# Output - Max probability and the sequence of tags for input sequence
def testing(l, p_word, p_tag):
    tags = []
    words = l.split(' ')
    for word in words:
        word_tags = []
        for (i,j) in p_word.keys():
            if word == i:
                word_tags.append(j)
        tags.append(word_tags)
    print('Tags each word in the input sequence can take')
    print(tags)

    if [] in tags:
        return 0, tags

    tag_list = tags[0]
    for i in range(1,len(tags)):
        new_list = []
        temp = []
        for t1 in tag_list:
            for t2 in tags[i]:
                temp = []
                if type(t1) == type(['0']):
                    temp = temp + t1
                else:
                    temp.append(t1)
                temp = temp + [t2]
                new_list.append(temp)
        tag_list =new_list
    print('All possible combination of tags for the input sequence')
    for sentence_tag in tag_list:
        print(sentence_tag)

    probability = []
    for tags in tag_list:
        prob = 1
        for word,tag in zip(words, tags):
            prob *= p_word[(word,tag)]
        if (tags[0],'<s>') in p_tag.keys():
            prob *= p_tag[(tags[0],'<s>')]
            if('</s>', tags[-1]) in p_tag.keys():
                prob *= p_tag[('</s>', tags[-1])]
                for i in range(1, len(tags)):
                    if (tags[i-1],tags[i]) in p_tag.keys():
                        prob *= p_tag[(tags[i-1],tags[i])]
        else:
            prob = 0
        probability.append(prob)

    max_prob = max(probability)
    max_index = probability.index(max_prob)
    return max_prob, tag_list[max_index]

def main():
    if not os.path.exists('p_word.pkl'):
        lines = text_retrieve('NLP6320_POSTaggedTrainingSet-Windows.txt')
        sentences, tags, vocab = data_cleaning(lines)

        count_bigrams_word, count_bigrams_tag, count_unigrams_tag = createBigrams(sentences, tags)
        p_word, p_tag = compute_Prob(count_bigrams_word, count_bigrams_tag, count_unigrams_tag)
        save_obj('p_word', p_word, 'P(w/t)')
        save_obj('p_tag', p_tag, 'P(t(i)/t(i-1))')

    else:

        p_word = load_obj('p_word')
        p_tag = load_obj('p_tag')

        l = sys.argv[1]
        prob, list = testing(l, p_word, p_tag)
        if prob == 0:
            print('One of hte words in the input sequence is unknown to the model')
            print(list)
        else:
            print('Max probability : ',prob)
            print('POS tags for the input sequence : ',list)

main()
