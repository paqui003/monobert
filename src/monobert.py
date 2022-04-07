from operator import itemgetter 

import pandas as pd
from gensim.summarization.bm25 import BM25, get_bm25_weights

import tensorflow as tf

from official.modeling import tf_utils
from official import nlp
from official.nlp import bert

# Load the required submodules
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization
import official.nlp.data.classifier_data_lib
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks


from abc import ABC, abstractmethod

class InitRetrievalSystem(ABC):
    @abstractmethod
    def __init__(self, docIds):
        self.docIds = docIds
        pass
    
    @abstractmethod
    def retrieve(self, query):
        pass
    
    @abstractmethod
    def retrieve_k(self, query, k):
        pass
    

class BM25Retrieval(InitRetrievalSystem):
    """
    Initial retrieval system based on bm25-scores retrieval.

    Attributes
    ----------
    docIds : list
        A list of document identifiers. 
        Example: ['MED-201', 'MED-987', ...]       
    bm25 : BM25 class provided by gensim=3.8.3
        Retrieval based on bm25 scoring

    Methods
    -------
    retrieve(query)
        Retrieve all documents given by a query ordered by the relevance score.
    retrieve_k(query, k=10)
        Retrieve top-k documents given by a query ordered by the relevance score.

    """
    def __init__(self, corpus, docids):
        """
        Initializes a BM25Retrieval class object
        
        Parameters
        ----------
        corpus : list of str
            Each item of list specifies a document. 
            list entries correspond to entries in docids. 
        docids : list
            A list of document identifiers. 
            Example: ['MED-201', 'MED-987', ...]
        """
        print("Initilizing BM25 based Retrieval System...")
        self.bm25 = BM25([i.split() for i in corpus]) 
        print("Finished.")
        print()
        print(f"Corpus size: {self.bm25.corpus_size}")
        print(f"Avg. document len: {self.bm25.avgdl:0.2f}")
        print(f"Avg. idf: {self.bm25.average_idf:0.2f}")
        
        self.docIds = docids
        
        
    def retrieve(self, query):
        """
        Retrieve each document relevant to query.
        
        Parameters
        ----------
        query : str
            A query. 
        
        Returns
        -------
        (ids, scores) : list of tuples
            A list of tuples. ids are given by self.docIds and scores are floats
            specifying the relevance of a document to the query.
        """
        query = query.split()
        ids, scores = zip(*sorted(enumerate(self.bm25.get_scores(query)), key=lambda x: x[1], reverse=True))
        ids = list(itemgetter(*ids)(self.docIds))
        return (ids, scores)
        
    def retrieve_k(self, query, k=10):
        """
        Retrieve top-k documents relevant to query.
        
        Parameters
        ----------
        query : str
            A query. 
        k : int
            Number of results to return.
        
        Returns
        -------
        (ids, scores) : list of tuples
            A list of k-tuples. ids are given by self.docIds and scores are floats
            specifying the relevance of a document to the query.
        """
        query = query.split()
        ids, scores = zip(*sorted(enumerate(self.bm25.get_scores(query)), key=lambda x: x[1], reverse=True)[:k])
        ids = list(itemgetter(*ids)(self.docIds))
        return (ids, scores)
        
        
class MonoBERT:
    """
    Retrieval system based on MonoBERT 
    (see [Nogueira, Rodrigo, and Kyunghyun Cho. "Passage Re-ranking with BERT." arXiv preprint arXiv:1901.04085 (2019)] for more
    information).

    Attributes
    ----------
    corpus_map : dict
        Maps doc_ids to documents.      
    initial_retriever : class object
        The initial retrieval system to pre-rank documents for BERT.
        The class must implement the following methods:
            retrieve(query)
            retrieve_k(query, k -> int)
        These methods must return a list of tuples of document identifiers and ranking scores. 
        (See class BM25Retrieval for reference).  
    intial_k : int
        Determines the initial number of retrieved documents given to BERT for re-ranking.
    bert : Finetuned BERT-Model
        A finetuned BERT-Model which outputs a single logit for each document

    Methods
    -------
    retrieve(query)
        Retrieve all documents given by a query ordered by the relevance score.
    retrieve_k(query, k=10)
        Retrieve top-k documents given by a query ordered by the relevance score.

    """
    def __init__(self, corpus, docids, bert_path, init_retrieval, initial_k):
        """
        Initializes a MonoBERT class object
        
        Parameters
        ----------
        corpus : list of str
            Each item of list specifies a document. 
            list entries correspond to entries in docids. 
        docids : list
            A list of document identifiers. 
            Example: ['MED-201', 'MED-987', ...]
        bert_path : str
            Path to a BERT-Model stored with tf.saved_model.
        initial_k : int
            Determines the initial number of retrieved documents given to BERT for re-ranking.
        """
        self.corpus_map = {doc_id : doc for (doc_id, doc) in zip(docids, corpus)}
        #self.initial_retriever = BM25_Retrieval(corpus, docids)
        self.initial_retriever = init_retrieval
        self.initial_k = initial_k
        
        print("Loading BERT model with tensorflow...")
        self.bert = tf.saved_model.load(bert_path)
        print("BERT model loaded!")
            
        
    def retrieve(self, query):
        """
        Retrieve each document relevant to query.
        
        Parameters
        ----------
        query : str
            A query. 
        
        Returns
        -------
        (ids, scores) : list of tuples
            A list of tuples. ids are given by self.docIds and scores are floats
            specifying the relevance of a document to the query.
        """
        docids, scores = self.initial_retriever.retrieve_k(query, self.initial_k)
        docSample = [self.corpus_map[docid] for docid in docids]
        print("Initial Retrieval...")
        print(docids, scores)
        queries = [query] * len(docSample)
        reloaded_results = tf.sigmoid(self.bert([queries, docSample]))
        scores = reloaded_results.numpy().flatten().tolist()
        ids, scores = zip(*sorted(zip(docids, scores), key=lambda x: x[1], reverse=True))
        print("After BERT...")
        print(ids, scores)
        return (ids, scores)
        
    def retrieve_k(self, query, k):
        """
        Retrieve top-k documents relevant to query.
        
        Parameters
        ----------
        query : str
            A query. 
        k : int
            Number of results to return.
        
        Returns
        -------
        (ids, scores) : list of tuples
            A list of k-tuples. ids are given by self.docIds and scores are floats
            specifying the relevance of a document to the query.
        """
        ids, scores = self.retrieve(query)
        return ids[:k], scores[:k]
        
