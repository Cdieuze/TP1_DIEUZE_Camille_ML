#!/usr/bin/env python
# coding: utf-8

# In[727]:


# Importation des bibliothèques

import pickle # Fichiers
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plot

import nltk # preprocessing
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score 
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
import seaborn
from joblib import parallel_backend
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import GridSearchCV


# In[728]:


#Importation données

file_neg =  pd.read_pickle("C:\\Users\\dieuz\\Desktop\\Professionnels\\ESGF M2\\Machine_Learning\\TP1\\imdb_raw_neg.pickle")
file_pos = pd.read_pickle("C:\\Users\\dieuz\\Desktop\\Professionnels\\ESGF M2\\Machine_Learning\\TP1\\imdb_raw_pos.pickle")

len(file_neg) #taille_fichier
len(file_pos) 


# In[729]:


#ETAPE 1 : Création DataFrame + vecteur Y

positif ={'Avis' : file_pos}
negatif ={'Avis' : file_neg}

df_pos = pd.DataFrame(positif)
df1 =df_pos.assign(Label=1)

df_neg = pd.DataFrame(negatif)
df0 =df_neg.assign(Label=0)

df = pd.concat([df1,df0], axis=0)
df


# In[730]:


# ETAPE 2 : Préprocessing (polluants)

def preprocessing(corpus): # fonction pour nettoyage du text
    corpus = re.sub('<[^>]*>', '', corpus)
    ponctuation = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', corpus)
    corpus = re.sub('[\W]+', ' ', corpus.lower()) +        ' '.join(ponctuation).replace('-', '')
    return corpus

df['Avis'] = df['Avis'].apply(preprocessing)

df


# In[731]:


# ETAPE 3 :Initiation du modèle (données apprentissage 70-30)

X = df['Avis']
Y = df['Label']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=2) 


# In[732]:


stop = stopwords.words('english')


# In[733]:


#X_train

tfidf = TfidfVectorizer(min_df=4,stop_words=stop).fit(X_train) #min_df=4 > réduire le nombre d'occurence
X_train =tfidf.transform(X_train)
print(X_train.shape)


# In[734]:


#X_test

X_test =tfidf.transform(X_test)
print(X_test.shape)


# In[735]:


# ETAPE 4 Modèle RegressionLogistic
log_reg = LogisticRegression()

log_reg.fit(X_train, Y_train)


# In[736]:


# Entrainement RegressionLogistic

X_training_score = accuracy_score(log_reg.predict(X_train), (Y_train))
print(X_training_score)


# In[737]:


X_prediction_score = accuracy_score(log_reg.predict(X_test),(Y_test))
print(X_prediction_score)


# In[738]:


print(classification_report(Y_test, log_reg.predict(X_test)))


# In[739]:


# ETAPE 5 Top10_words pondérés par le modèle

coef = abs(log_reg.coef_)

index = coef.argsort()
top_index = index[0][::-1] # vecteur inversé

# Top10_coefficients
top10_words = top_index[:10]

words_index = []
for i in top10_words: # les indexs sous forme d'une liste
    words_index.append(i)
    
top_10_words = []
for i in words_indexs:
    top_10_words.append(list(count_vectorizer.vocabulary_.keys())[list(count_vectorizer.vocabulary_.values()).index(i)])

top_10_words


# In[740]:


cnf_matrix = metrics.confusion_matrix(Y_test,log_reg.predict(X_test))
cnf_matrix


# In[741]:


# matrice de confusion regression logistic

seaborn.heatmap(cnf_matrix/
            np.sum(cnf_matrix),
            annot=True, cmap='magma')


# In[742]:


#ETAPE 6 Amélioration modèle

# Il y d'autres modèles qui peuvent améliorer nos résultats mais concernant la regression logistique, il existe plusieurs solutions d'optimisation.


# In[743]:


# Gridsearchcv et hyperparamètre pour Gridsearchcv (tentative de coodage mais erreur/problème)

parameters_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l2'],
               'clf__C': [1.0, 10.0, 100.0]}]


gs = GridSearchCV(log_reg, parameters_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=10,
                           n_jobs=1)

gs.fit(X_train,Y_train)

gs_training = gs.score(X_train,Y_train)
gs_prediction = gs.score(X_test,Y_test)
cnf_matrix_gs = metrics.confusion_matrix(Y_test,gs.predict(X_test))

print(gs_training)
print(gs_prediction)
print(cnf_matrix_gs)


# In[ ]:


# Modèle Ridge
ridge_tfidf = RidgeClassifier()

ridge_tfidf.fit(X_train,Y_train)

ridge_training = ridge_tfidf.score(X_train,Y_train)
ridge_prediction = ridge_tfidf.score(X_test,Y_test)
cnf_matrix_ridge = metrics.confusion_matrix(Y_test,ridge_tfidf.predict(X_test))

print(ridge_training)
print(ridge_prediction)
print(cnf_matrix_ridge)


# In[ ]:


# matrice de confusion ridge

seaborn.heatmap(cnf_matrix_ridge/
            np.sum(cnf_matrix_ridge),
            annot=True, cmap='magma')


# In[ ]:


# Modèle Passif agressif
pa_tfidf = PassiveAggressiveClassifier()

pa_tfidf.fit(X_train,Y_train)

pa_training = pa_tfidf.score(X_train,Y_train)
pa_prediction = pa_tfidf.score(X_test,Y_test)
cnf_matrix_pa = metrics.confusion_matrix(Y_test,pa_tfidf.predict(X_test))

print(pa_training)
print(pa_prediction)
print(cnf_matrix_pa)


# In[ ]:


# matrice de confusion passif agressive

seaborn.heatmap(cnf_matrix_pa/
            np.sum(cnf_matrix_pa),
            annot=True, cmap='magma')


# In[ ]:


## DIEUZE Camille BDF5


# In[ ]:




