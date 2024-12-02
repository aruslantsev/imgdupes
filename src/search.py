import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm


def search_batch(index, query, k, batch_size=8192):
    dist = np.empty((query.shape[0], k))
    ind = np.empty((query.shape[0], k))
    batch_count = int(np.ceil(query.shape[0] / batch_size))
    for n in tqdm(range(batch_count)):
        i0 = n * batch_size
        i1 = (n + 1) * batch_size
        dist_tmp, ind_tmp = index.search(query[i0:i1], k)
        dist[i0:i1, :], ind[i0:i1, :] = dist_tmp, ind_tmp
    return dist, ind


def match(embeddings, n_neighbors: int):
    index = faiss.IndexFlat(embeddings.shape[1], faiss.METRIC_INNER_PRODUCT)
    # index = faiss.index_factory(embeddings.shape[1], 'Flat')

    index.add(embeddings)
    D, I = search_batch(index, embeddings, n_neighbors, batch_size=16384)
    dist = D
    ind = I

    column = np.arange(ind.shape[0])
    r = np.repeat(column[:, np.newaxis], ind.shape[1], axis=1)
    matched = np.vstack([r.ravel(), ind.ravel(), dist.ravel()]).transpose()
    matched = pd.DataFrame(matched, columns=['pic_query', 'pic_index', 'distance'])
    matched['pic_query'] = matched['pic_query'].astype(np.int64)
    matched['pic_index'] = matched['pic_index'].astype(np.int64)
    return matched
