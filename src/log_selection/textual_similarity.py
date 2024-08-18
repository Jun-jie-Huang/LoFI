import numpy as np
from log_selection.textual_encoding import text_encoding
def find_most_similar_embedding_Nd(text_list, text_idx, plm='bert', return_sim=False):
    X = text_encoding(text_list, model_type=plm)
    high_level_embedding = np.take(X, text_idx, axis=0)
    mask = np.ones([1, X.shape[0]], dtype=bool)
    mask[:, text_idx] = False
    similarities = cosine_similarity(high_level_embedding, X)
    sort_idx = np.argsort(-1 * np.max(similarities * mask, axis=0)).tolist()
    if return_sim:
        return sort_idx, np.max(similarities * mask, axis=0)
    return sort_idx


def cosine_similarity(v1, v2):
    # sim range [0, 1]
    num = np.dot(v1, np.array(v2).T)
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res

