#!/usr/bin/env python
# coding: utf-8

# # Testing ppgGA
# testing the path pattern generator using genetics algorithm with genetics algorithm aproach

# In[1]:


import ppg_ea
import random


# ## Testing
# 1. [Mutations](#Mutations)
#     1. `insert_mutation`
#     1. `scramble_mutation`
#     1. `inverse_mutation`
#     1. `swap_mutation`
# 2. [Recombinations](#Recombinations)
#     1. `order1_crossover`
#     1. `partially_mapped_crossover`
#     1. `cycle_crossover`
#     1. `edge_recombination`
# 3. [SSGA Selections](#Selections(SSGA))
#     1. [Parent Selection](#Parent-Selection)
#     2. [Survival Selection](#Survival-Selection)
#         1. `random_population_select`
#         1. `best_population_select`
#         1. `fitness_porportionate_select`
#         1. `linear_ranked_based_select`
#         1. `exponential_ranked_based_select`
# 4. [Population control functions](#MISC-and-population-related)

# In[2]:


TEST_NUM = 4


# In[3]:


def test_insert_mutation():
    test_child = list(range(10))
    # testing 10 times
    print('input before applying mutation:')
    print(test_child)
    print('-------------------------------')
    test_num = TEST_NUM
    for i in range(test_num):
        print('\n =================== i= ', i+1)
        ppg_ea.insert_mutation(test_child)
        print(test_child)


# In[4]:


def test_scramble_mutation():
    test_child = list(range(10))
    # testing 10 times
    print('input before applying mutation:')
    print(test_child)
    print('-------------------------------')
    test_num = TEST_NUM
    for i in range(test_num):
        print('\n =================== i= ', i+1)
        ppg_ea.scramble_mutation(test_child)
        print(test_child)


# In[5]:


def test_inverse_mutation():
    test_child = list(range(10))
    # testing 10 times
    print('input before applying mutation:')
    print(test_child)
    print('-------------------------------')
    test_num = TEST_NUM
    for i in range(test_num):
        print('\n =================== i= ', i+1)
        ppg_ea.inverse_mutation(test_child)
        print(test_child)


# In[6]:


def test_swap_mutation():
    test_child = list(range(10))
    # testing 10 times
    print('input before applying mutation:')
    print(test_child)
    print('-------------------------------')
    test_num = TEST_NUM
    for i in range(test_num):
        print('\n =================== i= ', i+1)
        ppg_ea.swap_mutation(test_child)
        print(test_child)


# In[7]:


# generating fake parents 
# out_option: if true will output the parents list.
# test_parents_num: the number of parents to be generated.
def test_generate_parents(test_parents_num=79, out_option=False):
    # making random parents
    test_parents = []
    for i in range(test_parnets_num):
        test_ind = list(range(10))
        random.shuffle(test_ind)
        test_parents.append(test_ind)
        
    if(out_option):
        for x in test_parents:
            print(x)
    return test_parents


# In[8]:


def test_duplocal_parentspair(test_parents, out_option=False):
    test_result_duplocal = ppg_ea.get_parents_pair_duplocal(test_parents)
    if(out_option):
        for x in test_result_duplocal:
            print(x)
    return test_result_duplocal


# In[9]:


def test_dupglobal_parentspair(test_parents, out_option=False):
    test_result_dupglobal = ppg_ea.get_parents_pair_dupglobal(test_parents)
    if(out_option):
        for x in test_result_dupglobal:
            print(x)
    return test_result_dupglobal


# In[10]:


def test_nodup_parentspair(test_parents, out_option=False):
    test_result_nodup = ppg_ea.get_parents_pair_nodup(test_parents)
    if(out_option):
        for x in test_result_nodup:
            print(x)
    return test_result_nodup


# In[11]:


def test_uniqpair_parentspair(test_parents, out_option=False, test_num=TEST_NUM):
    ppg_ea.DEBUGMODE = True
    okay_counter = 0
    for i in range(test_num):
        if(out_option):
            print('test', i, ':')
        test_result_uniqpair, test_result_uniqpair_i = ppg_ea.get_parents_pair_uniqpair(test_parents)
        if(out_option):
            set_test_result_uniqpair_i = set(test_result_uniqpair_i)
            if(len(set_test_result_uniqpair_i)==len(test_result_uniqpair_i)):
                print('okay!')
                okay_counter +=1
    if(out_option):
        print('out of', test_num, 'tests,', okay_counter, 'successfully had no duplicate pairs!')
    #     for x in test_result_uniqpair:
    #         print(x)
    ppg_ea.DEBUGMODE = False
    return test_result_uniqpair


# In[12]:


def test_primeorder_parentspair(test_parents, out_option=False):
    test_result_prime_order = ppg_ea.get_parents_pair_prime_order(test_parents)
    if(out_option):
        for x in test_result_prime_order:
            print(x)
    return test_result_prime_order


# In[13]:


# testing the order1 crossover module
# test_num: the number of trials for the module function call.
# parentspairs: the parents pairs obtained from get_parents_pair prefixed modules,
# can be of the following list:
# *random pairs, no duplication of parents in the parents pool
# *random pairs, no duplication of a parent in a pair
# *random pairs, no duplication of pairs
# *random pairs, no restriction on duplicates
# *unchanged order fo the pool paired.
def test_o1x(parentspairs, test_num=TEST_NUM, out_option=False):
    for i in range(test_num):
        test_o1x_children = ppg_ea.order1_crossover(parentspairs)
        if(out_option):
            print('resulted children from order 1 crossover:')
            for child in test_o1x_children:
                print(child)
    return test_o1x_children


# In[14]:


# testing the partially mapped crossover module
# test_num: the number of trials for the module function call.
# parentspairs: the parents pairs obtained from get_parents_pair prefixed modules,
# can be of the following list:
# *random pairs, no duplication of parents in the parents pool
# *random pairs, no duplication of a parent in a pair
# *random pairs, no duplication of pairs
# *random pairs, no restriction on duplicates
# *unchanged order fo the pool paired.
def test_pmx(parentspairs, test_num=TEST_NUM, out_option=False):
    for i in range(test_num):
        test_pmx_children = ppg_ea.partially_mapped_crossover(parentspairs)
        if(out_option):
            print('resulted children from partially mapped crossover:')
            for child in test_pmx_children:
                print(child)
    return test_pmx_children


# In[15]:


# testing the cycle crossover module
# test_num: the number of trials for the module function call.
# parentspairs: the parents pairs obtained from get_parents_pair prefixed modules,
# can be of the following list:
# *random pairs, no duplication of parents in the parents pool
# *random pairs, no duplication of a parent in a pair
# *random pairs, no duplication of pairs
# *random pairs, no restriction on duplicates
# *unchanged order fo the pool paired.
def test_cx(parentspairs, test_num=TEST_NUM, out_option=False):
    for i in range(test_num):
        test_cx_children = ppg_ea.cycle_crossover(parentspairs)
        if(out_option):
            print('resulted children from cycle crossover')
            for child in test_cx_children:
                print(child)
    return test_cx_children


# In[16]:


# testing the edge recombination module
# test_num: the number of trials for the module function call.
# parentspairs: the parents pairs obtained from get_parents_pair prefixed modules,
# can be of the following list:
# *random pairs, no duplication of parents in the parents pool
# *random pairs, no duplication of a parent in a pair
# *random pairs, no duplication of pairs
# *random pairs, no restriction on duplicates
# *unchanged order fo the pool paired.
def test_ejx(parentspairs, test_num=TEST_NUM, out_option=False):
    ppg_ea.DEBUGMODE = True
    for i in range(test_num):
        test_ejx_children = ppg_ea.edge_recombination(parentspairs)
        if(out_option):
            print('resulted children from edge recombination')
            for child in test_ejx_children:
                print(child)
    ppg_ea.DEBUGMODE = False
    return test_ejx_children


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




