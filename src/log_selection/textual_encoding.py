from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import gensim
import nltk
from sentence_transformers import SentenceTransformer, models

def text_encoding(text_list, model_type):
    if model_type == 'tfidf':
        model = TfidfVectorizer(stop_words='english')
        X = model.fit_transform(text_list)
    elif model_type == 'bow':
        model = CountVectorizer(stop_words='english')
        X = model.fit_transform(text_list)
    elif model_type == 'word2vec':
        model_path = 'path/to/word2vec/model'
        nltk.download('stopwords')
        stop_words = nltk.corpus.stopwords.words('english')
        model = gensim.models.Word2Vec.load(model_path)
        X = np.zeros((len(text_list), model.vector_size))
        for i, text in enumerate(text_list):
            tokens = [token for token in nltk.word_tokenize(text.lower()) if token not in stop_words]
            vectors = [model.wv[token] for token in tokens if token in model.wv.vocab]
            if vectors:
                X[i] = np.mean(vectors, axis=0)
    elif model_type in ['bert', 'roberta', 'codebert', 'unixcoder']:
        if model_type == 'bert':
            model_name = {'net': 'bert-base-uncased',
                          'local': './pretrained_models/bert-base', }
        elif model_type == 'roberta':
            model_name = {'net': 'roberta-base',
                          'local': './pretrained_models/roberta-base', }
        elif model_type == 'codebert':
            model_name = {'net': 'microsoft/codebert-base',
                          'local': './pretrained_models/codebert', }
        elif model_type == 'unixcoder':
            model_name = {'net': 'microsoft/unixcoder-base',
                          'local': './pretrained_models/unixcoder-base', }
        try:
            word_embedding_model = models.Transformer(model_name['net'])
        except:
            word_embedding_model = models.Transformer(model_name['local'])
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        X = np.array(model.encode(text_list))
    else:
        raise ValueError("Invalid model type.")

    return X
