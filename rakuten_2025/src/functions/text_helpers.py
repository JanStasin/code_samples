
import numpy as np
from gensim.models import Word2Vec

class Word2VecEmbedder:
    '''takes a list of tokenized texts and returns their embeddings using Word2Vec'''
    def __init__(self, vector_size=300, min_count=2, workers=4):
        self.vector_size = vector_size
        self.min_count = min_count
        self.workers = workers
        self.model = None
    
    def fit(self, token_lists):
        print('Training Word2Vec...')
        self.model = Word2Vec(token_lists, vector_size=self.vector_size, 
                             min_count=self.min_count, workers=self.workers)
        return self
    
    def transform(self, token_lists):
        if self.model is None:
            raise ValueError('Model not trained. Call fit() first.')
        
        def text_to_vector(token_lists):
            vectors = [self.model.wv[word] for word in token_lists if word in self.model.wv]
            return np.mean(vectors, axis=0) if vectors else np.zeros(self.vector_size)
        
        embeddings = np.array([text_to_vector(text) for text in token_lists])
        print(f'Word2Vec shape: {embeddings.shape}')
        return embeddings
    
    def fit_transform(self, token_lists):
        return self.fit(token_lists).transform(token_lists)





