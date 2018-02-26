import os 
import numpy as np
import argparse
from create_counts import get_unique_tags_words, get_sentence_tag_pairs
from HMM import HMMTagger
import sys
import timeit
import h5py
parser = argparse.ArgumentParser(description='SEAE')
parser.add_argument('--mode', type=str, choices=['train', 'test'], 
                    default='train', help='wether to train or test')
parser.add_argument('--infile', type=str, default='pos_train.txt', 
                    help='Input file to train or test from. This file contains tagged sentences with one sentence per line.')
args = parser.parse_args()
border='#'*20
if args.mode == 'train':
    print('{} Reading input file {}'.format(border, border))
    print('')
    start = timeit.default_timer()
    tag_dict, word_dict, tag_given_tag_counts, word_given_tag_counts, word_seq_lst = get_unique_tags_words(args.infile)
    np.savez('word_count_dict.npz', word_count_dict=word_dict)
    #sys.exit()
    tag_lst = tag_dict.keys()
    unique_word_lst = word_dict.keys()
    end = timeit.default_timer()
    print('{} Done reading input file. Time taken = {} {}'.format(border, end-start, border))
    print('')
    '''
    cnt_threshold = 1
    cnt1_sum = 0
    total_cnt_sum = 0
    unique_word_lst = []
    vocab_size = 0
    for key, value in word_dict.items():
        if value > cnt_threshold:
            unique_word_lst.append(key)
            vocab_size += 1
    unique_word_lst.append('UNK')
    with open('vocab.txt', 'w')as f:
        for word in unique_word_lst:
            f.write(word)
            f.write('\n')
    #sys.exit()        
    vocab_size += 1
    print('Vocabulary size => {}').format(len(unique_word_lst))
    #print('Total Number of unique_words => {}'.format(total_cnt_sum))
    print('Total Number of unique_words => {}'.format(len(word_dict.keys())))
    start = timeit.default_timer()
    new_word_seq_lst = []
    for word_seq in word_seq_lst:
        new_word_seq = []
        for word in word_seq:
            if word_dict[word] <= cnt_threshold:
                new_word_seq.append('UNK')
            else:
                new_word_seq.append(word)
        new_word_seq_lst.append(new_word_seq)
    end=timeit.default_timer()
    print('new_word_seq_lst completed. total time taken => {:.4f}'.format(end-start))
    print('')
    print('{} Filling word_give_tag with UNK {}'.format(border, border))
    print('')
    for tag in word_given_tag_counts.keys():
        for word in word_given_tag_counts[tag].keys():
            word_cnt = word_dict[word]
            if word_cnt <= cnt_threshold:
                del word_given_tag_counts[tag][word]
                word_given_tag_counts[tag]['UNK'] = word_given_tag_counts[tag].get('UNK', 0) + word_cnt
    '''
    print('{} Initialising transition probabilities. {}'.format(border, border))
    print('')
    start = timeit.default_timer()
    A = dict()
    for curr_tag in tag_lst:
        if tag_given_tag_counts.has_key(curr_tag):
            A[curr_tag] = dict()
            count = 0.
            for next_tag in tag_lst:
                if tag_given_tag_counts[curr_tag].has_key(next_tag):
                    A[curr_tag][next_tag] = 1.
                    count += 1.
            for next_tag in A[curr_tag].keys():
                A[curr_tag][next_tag] = A[curr_tag][next_tag] / count
        else:
            continue
    end = timeit.default_timer()
    print('{} Done initialising transition probabilities. Time taken = {} {}'.format(border, end-start, border))
    print('')
    print('{} Initialising emission probabilities. {}'.format(border, border))
    print('')
    start = timeit.default_timer()
    B = dict()
    word_lst = []
    for curr_tag in tag_lst:
        if word_given_tag_counts.has_key(curr_tag):
            B[curr_tag] = dict()
            count = 0.
            for word in unique_word_lst:
                if word_given_tag_counts[curr_tag].has_key(word):
                    B[curr_tag][word] = 1.
                    count += 1.
                    word_lst.append(word)
            for key in B[curr_tag].keys():
                B[curr_tag][key] = B[curr_tag][key] / count
        else:
            continue
    end = timeit.default_timer()
    print('{} Done initialising emission probabilities. Time taken = {} {}'.format(border, end-start, border))
    print('')
    print('{} Initialising initial state probabilities. {}'.format(border, border))
    print('')
    start = timeit.default_timer()
    pi = {}
    count = 0.
    for tag in A.keys():
        pi[tag] = 1.
        count += 1
    for tag in A.keys():
        pi[tag] = 1. / count
    end = timeit.default_timer()
    print('{} Done initialising initial state probabilities. Time taken = {} {}'.format(border, end-start, border))
    print('')
    #print(word_lst)
    #sys.exit()
    '''
    for i, word_seq in enumerate(new_word_seq_lst):
        if 'UNK' in word_seq:
            print(word_seq)
            print('*'*10)
            print(i)
            print('*'*10)
            print(word_seq_lst[i])
            print('*'*10)
            break
    sys.exit()
    '''
    model  = HMMTagger(A_init=A, B_init=B, pi_init=pi, tags=A.keys(), words=word_lst, obs_lst=word_seq_lst, mode=args.mode)
    #model.load()
    model.train()
    model.save()
    model.load()
    _, path = model.viterbi_decoding(word_seq_lst[0])
    print(path)
else:
    model = HMMTagger(mode='test') 
    sent_lst, tag_seq_lst = get_sentence_tag_pairs(args.infile)
    print('{} Starting eveluation for sentences in {} {}'.format(border, args.infile, border))
    start = timeit.default_timer()
    print(model.logpi)
    #sys.exit()
    '''
    print('len_sent_lst', len(sent_lst))
    print('len_tag_lst', len(tag_seq_lst))
    print('sent_lst', sent_lst[0])
    print('tag_lst', tag_seq_lst[0])
    print('sent_lst', sent_lst[-1])
    print('tag_lst', tag_seq_lst[-1])
    '''
    file1 = np.load('word_count_dict.npz')
    word_count_dict = file1['word_count_dict'].item()
    new_sent_lst = []
    cnt_threshold = 0
    for word_seq in sent_lst:
        new_word_seq = []
        for word in word_seq:
            if word_count_dict.get(word, 0) <= cnt_threshold:
                new_word_seq.append('UNK')
            else:
                new_word_seq.append(word)
        new_sent_lst.append(new_word_seq)
    total_tag_cnt = 0
    correct_tag_cnt = 0
    for i in range(len(sent_lst)):#len(sent_lst)
        #if 'UNK' not in new_sent_lst[i]:
            #print('new_sent', new_sent_lst[i])
        prob, est_tag_seq = model.viterbi_decoding(new_sent_lst[i])
        print('*'*10)
        print('Input sentence => {}'.format(new_sent_lst[i]))
        print('')
        print('Estimated tags => {}'.format(est_tag_seq))
        print('')
        print('*'*10)
        print('Prob => {}'.format(prob))
        print('')
        print('*'*10)
        for est_tag, correct_tag  in zip(est_tag_seq, tag_seq_lst[i]):
            if est_tag in correct_tag:
                correct_tag_cnt += 1
            total_tag_cnt += 1
    accy = correct_tag_cnt / float(total_tag_cnt)
    print('Tagging Accuracy on this set is {:.2f} %'.format(accy*100))
    print('')
    end = timeit.default_timer()
    print('{} Total time taken = {} seconds {}'.format(border, end - start, border))
             