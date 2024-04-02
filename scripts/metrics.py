import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate


def knn_accuracy(embeddings, true_labels, test_size=0.1, k = 10, rs=42, set_numpy = True, metric="euclidean"):
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
    
    random_state = np.random.seed(rs)

    if type(embeddings) == list:
        knn_accuracy = []
        for embed in embeddings:
            X_train, X_test, y_train, y_test = train_test_split(embed, true_labels, test_size=test_size, random_state = random_state)
    
            knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute', n_jobs=-1, metric=metric)
            knn = knn.fit(X_train, y_train)
            knn_accuracy.append(knn.score(X_test, y_test))
        if set_numpy == True:
            knn_accuracy= np.array(knn_accuracy)
        
    else:
        X_train, X_test, y_train, y_test = train_test_split(embeddings, true_labels, test_size=test_size, random_state = random_state)
        knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute', n_jobs=-1, metric=metric)
        knn = knn.fit(X_train, y_train)
        knn_accuracy = knn.score(X_test, y_test)

    
    return knn_accuracy



def knn_accuracy_cv(
    embeddings, labels, k=10, set_numpy=True, metric="euclidean"
):
    if type(embeddings) == list:
        knn_accuracy = []
        for embed in embeddings:
            clf = KNeighborsClassifier(
                n_neighbors=k, algorithm="brute", n_jobs=-1, metric=metric
            )
            cvresults = cross_validate(clf, embed, labels, cv=10)

            knn_accuracy.append(np.mean(cvresults["test_score"]))

        if set_numpy == True:
            knn_accuracy = np.array(knn_accuracy)

    else:
        clf = KNeighborsClassifier(
            n_neighbors=k, algorithm="brute", n_jobs=-1, metric=metric
        )
        cvresults = cross_validate(clf, embeddings, labels, cv=10)

        knn_accuracy = np.mean(cvresults["test_score"])

    return knn_accuracy