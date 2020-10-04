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
        
#função para limpar textos e subistiturir textos

def limpa_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"i'm", "i am", texto)
    texto = re.sub(r"he's", "he is", texto)
    texto = re.sub(r"she's", "she is", texto)
    texto = re.sub(r"that's", "that is", texto)
    texto = re.sub(r"what's", "what is", texto)
    texto = re.sub(r"where's", "where is", texto)
    texto = re.sub(r"\'ll", "will", texto)
    texto = re.sub(r"\'ve", "have", texto)
    texto = re.sub(r"\'re", "are", texto)
    texto = re.sub(r"\'d", "would", texto)
    texto = re.sub(r"won't", "will not", texto)
    texto = re.sub(r"can't", "cannot", texto)
    texto = re.sub(r"[-()#/@;:<>{}~+=?.,|]", "", texto)
    return texto

limpa_texto("exemplo i'm #@")

#limpeza das perguntas
perguntas_limpas = []
for pergunta in perguntas:
    perguntas_limpas.append(limpa_texto(pergunta))
        
 #limpeza das respostas
respostas_limpas = []
for resposta in respostas:
    respostas_limpas.append(limpa_texto(resposta))
    
#criação de um dicionário que mapeia cada palavra e o número de ocorrências
#codificação da contagem das palavras que não são frequentes -. biblioteca NLTK

palavras_contagem = {}  #inicializando com vazio
for pergunta in perguntas_limpas:
    #print(pergunta)
    for palavra in pergunta.split():  #verificando se a palavra existe
        if palavra not in palavras_contagem:
            palavras_contagem[palavra] = 1
        else:
            palavras_contagem[palavra] += 1
            
            
for resposta in respostas_limpas:
    for resposta in resposta.split():  
        if palavra not in palavras_contagem:
            palavras_contagem[palavra] = 1
        else:
            palavras_contagem[palavra] += 1
            

# Remoção de palavras não frequentes e tokenização (dois dicionários)
limite = 20
perguntas_palavras_int = {}
numero_palavra = 0
for palavra, contagem in palavras_contagem.items():
    #print(palavra)
    #print(contagem)
    if contagem >= limite:
        perguntas_palavras_int[palavra] = numero_palavra
        numero_palavra += 1

respostas_palavras_int = {}
numero_palavra = 0
for palavra, contagem in palavras_contagem.items():
    if contagem >= limite:
        respostas_palavras_int[palavra] = numero_palavra
        numero_palavra += 1


# Adição de tokens no dicionário
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    perguntas_palavras_int[token] = len(perguntas_palavras_int) + 1
for token in tokens:
    respostas_palavras_int[token] = len(respostas_palavras_int) + 1

# Criação do dicionário inverso com o dicionário de respostas
respostas_int_palavras = {p_i: p for p, p_i in respostas_palavras_int.items()}

# Adição do token final de string <EOS> para o final de cada resposta
for i in range(len(respostas_limpas)):
    respostas_limpas[i] += ' <EOS>'
    
# Tradução de todas as perguntas e respostas para inteiros
# Substituição das palavras menos frequentes para <OUT>
perguntas_para_int = []
for pergunta in perguntas_limpas:
    ints = []
    for palavra in pergunta.split():
        if palavra not in perguntas_palavras_int:
            ints.append(perguntas_palavras_int['<OUT>'])
        else:
            ints.append(perguntas_palavras_int[palavra])
    perguntas_para_int.append(ints)
        
respostas_para_int = []
for resposta in respostas_limpas:
    ints = []
    for palavra in resposta.split():
        if palavra not in respostas_palavras_int:
            ints.append(respostas_palavras_int['<OUT>'])
        else:
            ints.append(respostas_palavras_int[palavra])
    respostas_para_int.append(ints)
    
# Ordenação das perguntas e respostas pelo tamanho das perguntas
perguntas_limpas_ordenadas = []
respostas_limpas_ordenadas = []
for tamanho in range(1, 25 + 1):
    #print(tamanho)
    for i in enumerate(perguntas_para_int):
        #print(i[1])
        if len(i[1]) == tamanho:
            perguntas_limpas_ordenadas.append(perguntas_para_int[i[0]])
            respostas_limpas_ordenadas.append(respostas_para_int[i[0]])