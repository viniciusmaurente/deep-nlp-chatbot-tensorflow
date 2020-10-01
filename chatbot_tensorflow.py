#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020

@author: viniciusmaurente
"""

#create chatbot whith Deep NLP

#import library
import numpy as np
import tensorflow as tf
import re
import time

# -- first step 1 - pré processing data

#import database
linhas = open('movie_lines.txt', encoding ='utf-8', errors = 'ignore').read().split('\n')
conversas = open('movie_conversations.txt', encoding ='utf-8', errors = 'ignore').read().split('\n')

#criação de um dicionário para mapear cada linha com seu ID
id_para_linha = {}
for linha in linhas:
    #print (linha)
    _linha = linha.split(' +++$+++ ')
    #print (_linha)
    if len(_linha) == 5:
    #print(_linha[4])
        id_para_linha[_linha[0]] = _linha[4]
        
#criação de uma lista com todas as conversas