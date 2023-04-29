'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

For implementation of this MP, You may use numpy (though it's not needed). You may not 
use other non-standard modules (including nltk). Some modules that might be helpful are 
already imported for you.
'''

import math
from collections import defaultdict, Counter
from math import log
import numpy as np
import copy

# define your epsilon for laplace smoothing here

def baseline(train, test):
    '''
    Implementation for the baseline tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    # generate a dict of counters that contains the tag frequency for each word
    word_tags_dict=dict()
    tags_counter=Counter()
    cnt=0
    for sentence in train:
        for word,tag in sentence:
#             print(word,tag)
#             return None
            if word not in word_tags_dict:
                word_tags_dict[word]=Counter()
            word_tags_dict[word].update([tag])
#             if cnt<100:
#                 print(tag)
#                 cnt+=1
            tags_counter.update([tag])
    print(tags_counter)
    sentence_list=[]
#     for sentence in test:
#         print(sentence)
#         return None
    for sentence in test:
        cur_sentence=[]
        for word in sentence:            
            if word in word_tags_dict:
                most_common_tag=word_tags_dict[word].most_common(1)[0][0]
                word_tag_pair=(word,most_common_tag)
                cur_sentence.append(word_tag_pair)
            else:
                most_common_tag=tags_counter.most_common(1)[0][0]
                word_tag_pair=(word,most_common_tag)
                cur_sentence.append(word_tag_pair)
        sentence_list.append(cur_sentence)
    return sentence_list
def Laplace_smooth(k,dict_transition):
    p_transition=copy.deepcopy(dict_transition)
    for tag_before in dict_transition:
        y=dict_transition[tag_before]
        num_types=len(y)
        #calculation the total number of possible transitions from tag_before
        num_transitions=0
        for i,j in y.items():
            num_transitions+=y[i]
        for tag_cur,tag_num in y.items():
            p_transition[tag_before][tag_cur]=(tag_num+k)/(num_transitions+k*(num_types+1))
        p_transition[tag_before]['UNKNOWN']=k/(num_transitions+k*(num_types+1))
    return p_transition
# def cal_hapax(dict_emission):
#     dict_hapax=dict()
#     hapax_word=[]
#     for tag in dict_emission:
#         y=dict_emission[tag]
#         for word,word_num in y.items():
#             if word_num==1:
#                 if tag not in dict_hapax:
#                     dict_hapax[tag]=0
#                 dict_hapax[tag]+=1
#     for 
            
# def p_hapax(word,hapax):
#     if word in hapax:
#         return hapax[word]
#     else:
#         return 1

def b_i(b,i,x):
    if x in b[i]:
        return b[i][x]
    else:
        return b[i]['UNKNOWN']
def a_i_j(a,i,j,d,t):
    if i in a:
        if j in a[i]:
            return a[i][j]
        else:
            return a[i]['UNKNOWN']
    else:
        print('find error!')
        print(d-t)
        print(i,j)
def viterbi(train, test):
    '''
    Implementation for the viterbi tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    k=1e-3
    dict_transition=dict()
    for sentence in train:
        for i in range(len(sentence)-1):
            tag_before=sentence[i][1]
            tag_cur = sentence[i+1][1]
            if tag_before not in dict_transition:
                dict_transition[tag_before]=dict()
            if tag_cur not in dict_transition[tag_before]:
                dict_transition[tag_before][tag_cur]=1
            dict_transition[tag_before][tag_cur]+=1
    #normalize p of tag_cur given tag_before with Laplace smoothing.
    a=Laplace_smooth(k,dict_transition)
    y_without_end=a.keys()

#     values=p_transition['START'].values()
#     print(sum(values))
    #record the recurrence of each word given the tag
    dict_emission = dict()
    for sentence in train:
        for word, tag in sentence:
            if tag not in dict_emission:
                dict_emission[tag] = dict()
            if word not in dict_emission[tag]:
                dict_emission[tag][word] = 0
            dict_emission[tag][word] += 1
    # smooth dict_emission with Laplace smoothing
    b=Laplace_smooth(k,dict_emission)
#     print(b.keys())
    #initialization
    y=b.keys()
    num_tags=len(y)
    #initialize silence distribution
    c=0.99
    pi=dict()
    for i in y:
        if i=='START':
            pi[i]=c
        else:
            pi[i]=(1-c)/(num_tags-1)
    output=[]
    for x in test:
        #initialize v_0
        v_0=dict()
        for i in y:
#             v_0[i]=pi[i]*b_i(b,i,x[0])
            v_0[i]=np.log(pi[i])+np.log(b_i(b,i,x[0]))
        #iteration
        d=len(x)
        v=list(range(d))
        psai=list(range(d))
        v[0]=v_0
        for t in range(1,d):
            v_t=dict()
            psai_t=dict()
            for j in y:
                max_num=float('-inf')
                #This is a very good convention for debugging!
                argmax='NULL'
                for i in y_without_end:                    
#                     temp=v[t-1][i]*a_i_j(a,i,j,d,t)*b_i(b,j,x[t])
                    temp=v[t-1][i]+np.log(a_i_j(a,i,j,d,t))+np.log(b_i(b,j,x[t]))
                    if temp>max_num:
                        max_num=temp
                        argmax=i
                v_t[j]=max_num
                psai_t[j]=argmax
            v[t]=v_t
            psai[t]=psai_t
        #termination
        max_num=float('-inf')
        argmax='NULL'
        Y=list(range(d))
        for i in y:
            temp=v[d-1][i]            
            if temp>max_num:
                max_num=temp
                argmax=i             
        Y[d-1]=argmax
        #Back-Trace
        for t in range(d-2,-1,-1):
            Y[t]=psai[t+1][Y[t+1]]
        sentence=[]
        #generate a list contains (word,tag) for each sentence
        for t in range(d):
            word_tag=(x[t],Y[t])
            sentence.append(word_tag)
        output.append(sentence)
    return output
# def Laplace_smooth_hapax(k,dict_emission,p_given_hapax,hatch_list):
#     p_emission=copy.deepcopy(dict_emission)
#     for tag in dict_emission:
#         y=dict_emission[tag]
#         num_types=len(y)
#         #calculation the total number of possible transitions from tag_before
#         num_words=0
#         for i,j in y.items():
#             num_words+=y[i]
#         for word,word_num in y.items():
#             if word in hatch_list:
#                 k_hapax = k * p_given_hapax[True][tag]
#             else:
#                 k_hapax = k
#             p_emission[tag][word]=(word_num+k_hapax)/(num_words+k_hapax*(num_types+1))
#         k_hapax_un=0.
# #         if (tag in p_given_hapax[False]):
# #              k_hapax_un=k*p_given_hapax[False][tag]
# #         else:
# #             k_hapax_un=k*p_given_hapax[True][tag]
# #         p_emission[tag]['UNKNOWN']=k_hapax_un/(word_num+k_hapax_un*(num_types+1))
#         p_emission[tag]['UNKNOWN']=k/(word_num+k*(num_types+1))
#     return p_emission
def Laplace_smooth_hapax(k,dict_emission,hapax):
    p_emission=copy.deepcopy(dict_emission)
    for tag in dict_emission:
        y=dict_emission[tag]
        num_types=len(y)
        k_hapax=k*hapax[tag]
        #calculation the total number of possible transitions from tag_before
        num_words=0
        for i,j in y.items():
            num_words+=y[i]
        for word,word_num in y.items():            
            p_emission[tag][word]=(word_num+k_hapax)/(num_words+k_hapax*(num_types+1))
        p_emission[tag]['UNKNOWN']=k_hapax/(word_num+k_hapax*(num_types+1))
    return p_emission
def viterbi_ec(train, test):
    '''
    Implementation for the improved viterbi tagger.
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    k=1e-5
    dict_transition=dict()
    for sentence in train:
        for i in range(len(sentence)-1):
            tag_before=sentence[i][1]
            tag_cur = sentence[i+1][1]
            if tag_before not in dict_transition:
                dict_transition[tag_before]=dict()
            if tag_cur not in dict_transition[tag_before]:
                dict_transition[tag_before][tag_cur]=1
            dict_transition[tag_before][tag_cur]+=1
    #normalize p of tag_cur given tag_before with Laplace smoothing.
    a=Laplace_smooth(k,dict_transition)
    y_without_end=a.keys()

#     values=p_transition['START'].values()
#     print(sum(values))
    #record the recurrence of each word given the tag
    dict_emission = dict()
    for sentence in train:
        for word, tag in sentence:
            if tag not in dict_emission:
                dict_emission[tag] = dict()
            if word not in dict_emission[tag]:
                dict_emission[tag][word] = 0
            dict_emission[tag][word] += 1
    #find hatch dictionary
    dict_for_word = dict()
    word_counter=Counter()
    for sentence in train:
        for word, tag in sentence:
            word_counter.update([word])
            if word not in dict_for_word:
                dict_for_word[word] = dict()
            if tag not in dict_for_word[word]:
                dict_for_word[word][tag]=0
            dict_for_word[word][tag] +=1
    
    k_hapax=1e-5
    hapax=dict()
    hapax_count = k_hapax
    for tag in dict_emission:
        for word in dict_emission[tag]:
            if  word_counter[word] == 1:
                hapax_count += (1+k_hapax)
            else:
                hapax_count += k_hapax
    for tag in dict_emission:
        hapax_count_for_T = k_hapax
        for word in dict_emission[tag]:
            if  word_counter[word] == 1:
                hapax_count_for_T += (1+k_hapax)
        hapax[tag] = hapax_count_for_T / hapax_count
#     hatch_list=[]
#     hatch_dict=dict()
#     hatch_dict[True]=dict()
#     hatch_dict[False]=dict()
#     for word in word_counter:
#         if word_counter[word]==1:
#             hatch_list.append(word)
#             for tag in dict_for_word[word]:
#                 if tag not in hatch_dict[True]:
#                     hatch_dict[True][tag]=0
#                 hatch_dict[True][tag]+=1
#         else:
# #             if(word=='START'):
# #                 print('get')
# #                 print(dict_for_word[word])
# #                 print(dict_for_word[word].keys())
#             for tag in dict_for_word[word].keys():
#                 if tag not in hatch_dict[False]:
#                     hatch_dict[False][tag]=0
#                 hatch_dict[False][tag]+=1
    
#     p_tag_given_hatch=copy.deepcopy(hatch_dict)
#     for hatch in hatch_dict:
#         total=sum(hatch_dict[hatch].values())
#         for tag in hatch_dict[hatch]:
# #             if tag=='START':
# #                 print('good')
#             p_tag_given_hatch[hatch][tag] = hatch_dict[hatch][tag]/(total)
#     print(p_tag_given_hatch[False])
    # smooth dict_emission with Laplace smoothing
#     b=Laplace_smooth_hapax(k,dict_emission,p_tag_given_hatch,hatch_list)
    b=Laplace_smooth_hapax(k,dict_emission,hapax)
    #initialization
    y=b.keys()
    num_tags=len(y)
    #initialize silence distribution
    c=0.99
    pi=dict()
    for i in y:
        if i=='START':
            pi[i]=c
        else:
            pi[i]=(1-c)/(num_tags-1)
    output=[]
    for x in test:
        #initialize v_0
        v_0=dict()
        for i in y:
#             v_0[i]=pi[i]*b_i(b,i,x[0])
            v_0[i]=np.log(pi[i])+np.log(b_i(b,i,x[0]))
        #iteration
        d=len(x)
        v=list(range(d))
        psai=list(range(d))
        v[0]=v_0
        for t in range(1,d):
            v_t=dict()
            psai_t=dict()
            for j in y:
                max_num=float('-inf')
                #This is a very good convention for debugging!
                argmax='NULL'
                for i in y_without_end:                    
#                     temp=v[t-1][i]*a_i_j(a,i,j,d,t)*b_i(b,j,x[t])
                    temp=v[t-1][i]+np.log(a_i_j(a,i,j,d,t))+np.log(b_i(b,j,x[t]))
                    if temp>max_num:
                        max_num=temp
                        argmax=i
                v_t[j]=max_num
                psai_t[j]=argmax
            v[t]=v_t
            psai[t]=psai_t
        #termination
        max_num=float('-inf')
        argmax='NULL'
        Y=list(range(d))
        for i in y:
            temp=v[d-1][i]            
            if temp>max_num:
                max_num=temp
                argmax=i             
        Y[d-1]=argmax
        #Back-Trace
        for t in range(d-2,-1,-1):
            Y[t]=psai[t+1][Y[t+1]]
        sentence=[]
        #generate a list contains (word,tag) for each sentence
        for t in range(d):
            word_tag=(x[t],Y[t])
            sentence.append(word_tag)
        output.append(sentence)
    return output


