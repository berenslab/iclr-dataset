import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate


def knn_acc(embeddings, true_labels, test_size=0.1, k = 10, rs=42, metric="euclidean"):
    random_state = np.random.seed(rs)

    X_train, X_test, y_train, y_test = train_test_split(embeddings, true_labels, test_size=test_size, random_state = random_state)

    knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute', n_jobs=-1, metric=metric)
    knn = knn.fit(X_train, y_train)
    return knn.score(X_test, y_test)


def knn_accuracy(embeddings, true_labels, test_size=0.1, k = 10, rs=42, metric="euclidean"):
    """Calculates kNN accuracy.
    
    Parameters
    ----------
    embeddings : array or list of arrays
        List with the different datasets for which to calculate the kNN accuracy.
    true_labels : array-like
        Array with labels (colors).
    k : int, default=10
        Number of nearest neighbors to use.
    rs : int, default=42
        Random seed.
    
    Returns
    -------
    knn_accuracy : float or array
        kNN accuracy.
    
    """
    if not isinstance(embeddings, list):
        embeddings = [embeddings]
        
    knn_accuracy = []
    for embed in embeddings:
        knn_accuracy.append(knn_acc(embed, labels, test_size=test_size, k = k, rs=rs, metric=metric))
        
    knn_accuracy= np.array(knn_accuracy)
    
    return knn_accuracy



def knn_acc_cv(embeddings, labels, k=10, metric="euclidean"):
    clf = KNeighborsClassifier(
        n_neighbors=k, algorithm="brute", n_jobs=-1, metric=metric
    )
    cvresults = cross_validate(clf, embeddings, labels, cv=10)

    knn_accuracy = np.mean(cvresults["test_score"])

    return knn_accuracy


def knn_accuracy_cv(
    embeddings, labels, k=10, metric="euclidean"
):
    if not isinstance(embeddings, list):
        embeddings = [embeddings]
        
    knn_accuracy = []
    for embed in embeddings:
        knn_accuracy.append(knn_acc_cv(embed, labels, k=k, metric=metric))
        
    knn_accuracy= np.array(knn_accuracy)

    return knn_accuracy

