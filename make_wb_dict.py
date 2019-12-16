import os
import pandas as pd
import pickle
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import nltk
import re
from nltk.stem import WordNetLemmatizer 
  
 


#in out paths
dir_path = os.path.dirname(os.path.abspath(__file__))
input_files_path = dir_path + "\\news_files\\20061020_20131126_bloomberg_news"
model_path = os.path.join(dir_path,'wb_models\\_model.word2vec')

#lematizer
lemmatizer = WordNetLemmatizer()

#W2V model functions
def store_model(model,path):
    with open(path, 'wb') as p:
        pickle.dump(model, p)

def load_model(path):
    with open(path, 'rb') as model:
        print('loadind model from path: ')
        print(path,'...')
        mdl = pickle.load(model)    
        print('model loaded successfully')
    return mdl

def retrain_model(model,sentenses):
    print("Re-training model...")
    model.build_vocab(sentenses, update=True)
    model.train(sentenses, total_examples=model.corpus_count, epochs=model.iter)
    print("Model re-trained successfully")
    return model

def create_model():
    model = Word2Vec( "Ok i know. I need to know. I want to know. So I know",
                        min_count=2,   # Ignore words that appear less than this
                        size=100,      # Dimensionality of word embeddings
                        workers=2,     # Number of processors (parallelisation)
                        window=5,      # Context window for words during training
                        iter=30)
    
    print(model)
    return model


def clean_text(corpus):
    processed_article = corpus.lower()
    processed_article = re.sub('\d+', '#', processed_article)
    processed_article = re.sub('[^a-zA-Z#$%&]',' ', processed_article )
    return processed_article

def text_normalization(corpus):
    processed_article = clean_text(corpus)
    # Preparing the dataset
    all_sentences = nltk.sent_tokenize(processed_article)
    all_words = [nltk.word_tokenize(sent) for sent in all_sentences]
    for i in range(len(all_words)):
        all_words[i] = [w for w in all_words[i] if w not in stopwords.words('english')]
    return all_words


def list_files_train_model():
    #Listing data for training
    data=[]
    directories = []
    print("listing files in news dataset...")
    for path, subdirs, files in os.walk(input_files_path):
        for name in files:
            data.append(os.path.join(path, name))
        for name in subdirs:
            directories.append(os.path.join(path, name))

    print(len(directories))
    #If new model
    #mdl = create_model()
    #store_model(mdl,model_path)

    
    model = load_model(model_path)
    print(model)


    #retraining model, store it every batch size files
    batch_size=100
    for k in range(0,len(data)-batch_size,batch_size):
        print(k," out of ",len(data)," files")
        print("building corpus...")
        corpus=''
        for i in range(k,k+batch_size,1):
                
            file = open(data[i], "r",encoding="utf-8", errors='ignore')
            corpus = corpus + file.read().replace('\n', '')
            
        # Cleaning the text
        all_words = text_normalization(corpus)


        #for the new batch of files load and retrain model
        
        model = retrain_model(model,all_words)
        if (k%10000 == 0):
            print(model)
            store_model(model,model_path)
        

#store_model(create_model(),model_path)
#list_files_train_model()
