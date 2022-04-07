import numpy as np
import pandas as pd

def precision(y_true, y_pred):
    """
        Calculates precision score for a list of relevant documents and the groundtruth.
        
        Parameters
        ----------
        y_true : list
            List of known relevant documents for a given query.
        y_pred : list
            List of retrieved documents.  
        
        Returns
        -------
        Score: float
            Precision = TP / (TP + FP)
    """
    return len(set(y_true) & set(y_pred)) / len(y_pred)

def recall(y_true, y_pred):
    """
        Calculates recall score for a list of relevant documents and the groundtruth.
        
        Parameters
        ----------
        y_true : list
            List of known relevant documents for a given query.
        y_pred : list
            List of retrieved documents.  
        
        Returns
        -------
        Score: float
            Recall = TP / (TP + FN)
    """
    return len(set(y_true) & set(y_pred)) / len(y_true)

def fscore(y_true, y_pred, beta=1.0):
    """
        Calculates f-measure for a list of relevant documents and the groundtruth.
        
        Parameters
        ----------
        y_true : list
            List of known relevant documents for a given query.
        y_pred : list
            List of retrieved documents.  
        beta : float
            beta parameter weighting precision vs. recall
        
        Returns
        -------
        Score: float
            F-Measure = (1 + beta^2) \cdot \frac{Precision \cdot Recall}{beta^2 \cdot Precision+Recal}
    """
    betasquared = beta*beta
    pre = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    
    return (1+betasquared) * ((pre*rec)/(betasquared*pre + rec))

def precision_recall_fscore(y_true, y_pred, beta=1.0):
    """
        Convenience function, calculating precision, recall and f-measure.
        
        Parameters
        ----------
        y_true : list
            List of known relevant documents for a given query.
        y_pred : list
            List of retrieved documents.  
        beta : float
            beta parameter weighting precision vs. recall
        
        Returns
        -------
        Score: tuple
            (precision, recall, f-measure)
    """
    return precision(y_true, y_pred),  recall(y_true, y_pred),  fscore(y_true, y_pred, beta=beta)


'''def aveP(n, query, y_true):
    k = n
    print(query)
    ids, scores = retrieveBM25(query, k=k)
    retrdocids = docs_df.loc[ids, :]["docid"].tolist()
    
    pre_sum = 0.0
    #tmp1 = []
    #tmp2 = []
    
    for k in range(1, n):
        retrdocids_k = retrdocids[:k]
        docid_k = retrdocids[k]
        prec_k = precision(y_true, set(retrdocids_k))
        rec_k = recall(y_true, set(retrdocids_k))
        
        #tmp1.append(prec_k)
        #tmp2.append(rec_k)
        
        rel_k = (docid_k in y_true)
        pre_sum += (prec_k * rel_k)
        #print(prec_k, rel_k)
    return pre_sum/len(y_true)'''
            
class RetrievalScorer:
    """
    Retrieval score system. 
    Provides functions like RScore, Average Precision and Mean-Average-Precision. 

    Attributes
    ----------
    retrieval_system : class object
           A Retrieval system. Must implement the abstract class InitRetrievalSystem.
    Methods
    -------
    rPrecision(y_true, query)
        Calculate the RScore.
    aveP(query, groundtruth)
        Calculate the average precision score for a query.
    MAP(queries, groundtruths)
        Calculate the mean average precision for a list of queries.

    """
    def __init__(self, system):
        """
        Initializes a RetrievalScorer class object
        
        Parameters
        ----------
        system : class object
            A retrieval system that implements InitRetrievalSystem.
        """
        self.retrieval_system = system
    
    def rPrecision(self, y_true, query):
        """
        Calculates the precision at R where R denotes the number of all relevant
        documents to a given query.
        
        Parameters
        ----------
        y_true : list
            List of known relevant documents for a given query.
        query : str
            A query.

        Returns
        -------
        Score: float
            R-precision = TP / (TP + FN)
        """
        k = len(y_true)
        ids, scores = self.retrieval_system.retrieve_k(query, k=k)

        return precision(y_true, ids)

    def aveP(self, query, y_true):
        """
        Calculate the 11-point average precision score.
        
        Parameters
        ----------
        y_true : list
            List of known relevant documents for a given query.
        query : str
            A query.

        Returns
        -------
        Tuple: (float, list, list)
            (11-point average precision score, recall levels, precision levels).
        """
        recal_lvls = np.arange(0,1.1,0.1)
        precision_l = [0.0]
        recall_l = [0.0]

        precision_tmp = []
        l =[i for i in y_true]
        #ids, scores = self.retrieval_system.retrieve_k(query, k=len(self.retrieval_system.initialRetriever.docIds))
        ids, scores = self.retrieval_system.retrieve_k(query, k=len(y_true))
        retrdocids = [i for i, j in zip(ids, scores)]

        for j, _ in enumerate(retrdocids):
            if (len(l) == 0):
                break
            if retrdocids[j] in l:
                recall_at = recall(y_true, set(retrdocids[:j+1]))
                precision_at = precision(y_true, set(retrdocids[:j+1]))
                l.remove(retrdocids[j])
                precision_l.append(precision_at)
                recall_l.append(recall_at)


        df = pd.DataFrame(data={
                "recall": recall_l, 
                "precision": precision_l
                })
        l = []
        for r in recal_lvls:
            l.append(df[df["recall"]>=r]["precision"].max())

        _map = pd.Series(l).fillna(0.0).mean()    

        return _map, recal_lvls, pd.Series(l).fillna(0.0)

    def MAP(self, queries, groundtruths):
        """
        Calculate the mean average precision.
        
        Parameters
        ----------
        groundtruths : list(list)
            A double nested list. Each entry contains a list of known relevant documents for a given query.
        queries : list(str)
            A list of queries. Each query maps exactly to one groundtruth list in groundtruths.

        Returns
        -------
        Score: float
            MAP = frac{1}{|Q|} \cdot \sum_{q \in Q} AP(q).
        """
        n = len(queries)
        avep_score = 0.0
        for q, g in zip(queries, groundtruths):
            s, _, _ = self.aveP(q, g)
            avep_score += s
        return avep_score / n