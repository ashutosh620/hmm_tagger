from collections import Counter
import numpy as np
import timeit

# this is an example of how to parse the POS tag file and get counts
# needed for a bigram tagger 
def get_unique_tags_words(infile):
    with open (infile) as f:
        tag_dict = {}
        word_dict = {}
        tag_given_tag_counts = {}
        word_given_tag_counts = {}
        obs_seq = []
        for line in f:
            wordtags = line.rstrip().split(" ")
            curr_wordtag = wordtags[0]
            curr_parts = curr_wordtag.split("/")
            curr_tagstring = curr_parts[-1]
            curr_tags = curr_tagstring.split("|")
            curr_word='/'.join(curr_parts[:-1])
            word_dict[curr_word] = word_dict.get(curr_word, 0) + 1
            for tag in curr_tags:
                tag_dict[tag] = tag_dict.get(tag, 0) + 1
            for tag in curr_tags:
                if not word_given_tag_counts.has_key(tag):
                    word_given_tag_counts[tag] = {}
                word_given_tag_counts[tag][curr_word] = word_given_tag_counts[tag].get(curr_word, 0) + 1
            word_seq = [curr_word]
            for i in range(1, len(wordtags)):
                next_wordtag = wordtags[i]
                if next_wordtag == "":
                    continue
                next_parts=next_wordtag.split("/")
              
                next_tagstring = next_parts[-1]
                next_tags = next_tagstring.split("|")
                next_word = '/'.join(next_parts[:-1])
                word_dict[next_word] = word_dict.get(next_word, 0) + 1
                for curr_tag in curr_tags:
                    if not tag_given_tag_counts.has_key(curr_tag):
                        tag_given_tag_counts[curr_tag] = {}
                    for next_tag in next_tags:
                        tag_given_tag_counts[curr_tag][next_tag] = tag_given_tag_counts[curr_tag].get(next_tag, 0) + 1
                for tag in next_tags:
                    tag_dict[tag] = tag_dict.get(tag, 0) + 1    
                for tag in next_tags:
                    if not word_given_tag_counts.has_key(tag):
                        word_given_tag_counts[tag] = {}
                    word_given_tag_counts[tag][next_word] = word_given_tag_counts[tag].get(next_word, 0) + 1
                word_seq.append(next_word)
                curr_tags = next_tags
            obs_seq.append(word_seq)
   
                
    return tag_dict, word_dict, tag_given_tag_counts, word_given_tag_counts, obs_seq
def get_sentence_tag_pairs(infile):
    sent_lst = []
    tag_seq_lst = []
    with open(infile) as f:
        for line in f:
            wordtags = line.rstrip().split(" ")
            tag_lst = []
            word_lst = []
            for i in range(len(wordtags)):
                wordtag = wordtags[i]
                if wordtag == "":
                    continue
                parts=wordtag.split("/")
                tagstring = parts[-1]
                tags = tagstring.split("|")
                word = '/'.join(parts[:-1])
                word_lst.append(word)
                tag_lst.append(tags)
            sent_lst.append(word_lst)
            tag_seq_lst.append(tag_lst)
        return sent_lst, tag_seq_lst
                
            
    

