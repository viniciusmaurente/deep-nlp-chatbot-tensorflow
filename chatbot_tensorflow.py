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

# -- first step 1 - pre processing data

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
conversas_id = []
for conversa in conversas[:-1]:
    #print(conversa)
    _conversa = conversa.split(' +++$+++ ') [-1][1:-1].replace("'", "").replace(" ", "") #pegar somente a última coluna / replace para escluir as aspas e o espaço
    #print(_conversa)
    conversas_id.append(_conversa.split(','))
      
#separação das perguntas e respostas
perguntas = [] #criando a lista de perguntas
respostas = [] #criando a lista de respostas
for conversa in conversas_id:
    #print(conversa)
    for i in range (len(conversa) -1):
        perguntas.append(id_para_linha[conversa[i]]) # pegando o texto
        respostas.append(id_para_linha[conversa[i + 1]])
        