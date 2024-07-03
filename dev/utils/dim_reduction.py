import umap
from sklearn.manifold import TSNE


def calculate_umap(data):
    print('Calculating 2dim UMAP of ' + str(len(data)) + ' elements...')
    result = umap.UMAP(n_neighbors=5).fit_transform(data)
    return result[:, 0], result[:, 1]


def calculate_tsne(data):
    print('Calculating 2dim TSNE of ' + str(len(data)) + ' elements...')
    result = TSNE(learning_rate='auto', init='pca').fit_transform(data)
    return result[:, 0], result[:, 1]
