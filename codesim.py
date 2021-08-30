import numpy as np
import pandas as pd 
import gensim
import re
from gensim.parsing.preprocessing import remove_stopwords
from gensim import corpora
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import gensim.downloader as api
class qasystem:
    def __init__(self):
        self.data=self.process_data()
        self.cleaned_sentence=self.get_cleaned_sentences(stopwords=True)
        self.cleaned_sentence_sp=self.get_cleaned_sentences(stopwords=False)
        
    def load_w2vmodel(self):
        w2v_model=None
        try:
            w2v_model=gensim.models.KeyedVectors.load('./w2vecmodel.mod')
            print("Loaded Word2Vec model")
        except:
            w2v_model=api.load("word2vec-google-news-300")
            w2v_model.save('./w2vecmodel.mod')
            print("Saved Word2Vec model")
        return(w2v_model)

    def load_glovemodel(self):
        glove_model=None
        try:
            glove_model=gensim.models.KeyedVectors.load('./glovemodel.mod')
            print("Loaded glove model")
        except:
            glove_model=api.load("glove-twitter-25")
            glove_model.save('./glovemodel.mod')
            print("Saved glove model")
        return(glove_model)


    def process_data(self):
        df1 = pd.read_csv('S08_question_answer_pairs.txt', sep='\t')
        df2 = pd.read_csv('S09_question_answer_pairs.txt', sep='\t')
        df3 = pd.read_csv('S10_question_answer_pairs.txt', sep='\t', encoding = 'ISO-8859-1')
        self.data=df1.append([df2,df3])
        columns=['Question','Answer']
        self.data=self.data.loc[:,columns]
        #remove duplicate column 
        self.data =self.data.drop_duplicates(subset='Question')
        self.data = self.data.dropna(axis=0)
        return(self.data)

    def clean_sentence(self,sentence,stopwords=False):
        sentence=sentence.lower().strip()
        sentence=re.sub(r'[^a-z0-9\s]','',sentence)
        if (stopwords):
            sentence=remove_stopwords(sentence)
        return(sentence)

    def get_cleaned_sentences(self,stopwords=False):
        #sents=data[["Question"]]
        cleaned_sentences=[]
        for index,row in self.data.iterrows():
            cleaned=self.clean_sentence(row['Question'],stopwords)
            cleaned_sentences.append(cleaned)
        return(cleaned_sentences)

    def bow(self,original_ques):
        sentences=self.cleaned_sentence_sp            
        sentence_words=[[word for word in document.split()] for document in sentences]
        dictionary=corpora.Dictionary(sentence_words)
        bow_corpus=[dictionary.doc2bow(text) for text in sentence_words]
        question=self.clean_sentence(original_ques,stopwords=False)
        question_embedding=dictionary.doc2bow(question.split())
        ques,ans=self.retrieve_answer(question_embedding,bow_corpus,sentences)
        return(ques,ans)

    def retrieve_answer(self,question_embedding,sentence_embeddings,sentences):
        max_sim=-1
        index_sim=-1
        for index,faq_embedding in enumerate(sentence_embeddings):
            sim=cosine_similarity(faq_embedding,question_embedding)[0][0]
            print(index,sim,sentences[index])
            if (sim>max_sim):
                max_sim=sim
                index_sim=index
        #max_similarity=np.argmax(sim, axis=None)
        #print("Question :",question)
        print("Retreived Question:",self.data.iloc[index_sim,0])
        print("Retreived Answer:",self.data.iloc[index_sim,1])
        print("Max sim :",max_sim)
        return(self.data.iloc[index_sim,0],self.data.iloc[index_sim,1])

    def tfidf(self,question):
        sentences_tfidf=self.cleaned_sentence_sp 
        sentence_words_tfidf=[[word for word in document.split()] for document in sentences_tfidf]
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(tuple(self.data['Question']))
        ques_tfidf=tfidf_vectorizer.transform([question])
        similarity = cosine_similarity(ques_tfidf, tfidf_matrix)
        max_similarity = np.argmax(similarity, axis=None)
        return(self.data.iloc[max_similarity]['Question'],self.data.iloc[max_similarity]['Answer'])


    def getwordvec(self,word,model):
        samp=model['computer']
        vec=[0]*len(samp)
        try:
            vec=model[word]
        except:
            vec=[0]*len(samp)
        return(vec)

    def getphraseembedding(self,phrase,embeddingmodel):
        samp=self.getwordvec('computer',embeddingmodel)
        vec=np.array([0]*len(samp))
        den=0
        for word in phrase.split():
            den=den+1
            vec=vec+np.array(self.getwordvec(word,embeddingmodel))
        return(vec.reshape(1,-1))

    def glove_w2vec(self,sentence,question,embeddingmodel):
        sent_embeddings=[]
        for sent in sentence:
            sent_embeddings.append(self.getphraseembedding(sent,embeddingmodel))
        question_embedding=self.getphraseembedding(question,embeddingmodel)
        ques,ans=self.retrieve_answer(question_embedding,sent_embeddings,sentence)
        return(ques,ans)

    def glove(self,question):
        glove_model=self.load_glovemodel()
        sentences=self.cleaned_sentence_sp
        ques,ans=self.glove_w2vec(sentences,question,glove_model) 
        return(ques,ans)


    def w2vec(self,question):
        w2v_model=self.load_w2vmodel()
        sentences=self.cleaned_sentence
        ques,ans=self.glove_w2vec(sentences,question,w2v_model) 
        return(ques,ans)
