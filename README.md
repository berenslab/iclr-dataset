# Learning representations of learning representations

The ICLR dataset is a complete scrape of ICLR submissions from OpenReview. It contains 24,445 ICLR submissions from 2017 to 2024.

![ICLR dataset, SBERT embedding](/results/figures/embedding.png)

The dataset is described in [González-Márquez & Kobak, Learning representations of learning representations, DMLR workshop at ICLR 2024](https://arxiv.org/abs/2404.08403). Please cite as follows:

```
@inproceedings{gonzalez2024learning,
  title={Learning representations of learning representations},
  author={Gonz{\'a}lez-M{\'a}rquez, Rita and Kobak, Dmitry},
  booktitle={Data-centric Machine Learning Research (DMLR) workshop at ICLR 2024},
  year={2024}
}
```

## Dataset description
Each sample corresponds to a **submitted** article to the ICLR conference and includes as features:
-  Year
-  OpenReview ID
-  Title
-  Abstract
-  Authors
-  Decision
-  Scores
-  Keywords
-  Labels
  
To label the dataset, we relied on the author-provided keywords and used them to assign papers to 45 non-overlapping classes. We combined some keywords together into one class (e.g. *attention* and *transformer*), disregarded very broad keywords (e.g. *deep learning*), and assigned papers to rarer classes first. Using this procedure, we ended up labeling 53.4% of all papers.

![ICLR dataset, dataframe screenshot](https://github.com/berenslab/iclr-dataset/assets/82372364/2fa62933-7a71-4231-b009-31ababd88a50)

Note that 26 submissions with placeholder abstracts (below 100 characters) are excluded.

## Descriptive statistics
- **Dataset:** Abstracts submitted to ICLR in 2017-2024 (24,445 papers).
- **Labels:** based on keywords, 45 classes, 53.4% labeled papers.
- **Reviewers:** Reviewed papers had on average 3.7 reviews, with 93% having either 3 or 4 reviews.
- **Scores:** Across all 244,226 possible pairs of reviews of the same paper, the correlation coefficient between scores was 0.40.
- **Basic statistics:**
![ICLR dataset, summary statistics](/results/figures/summary-stats.png)

## Benchmark
We propose to use the ICLR dataset as a benchmark for embedding quality. The ICLR dataset is not part of the training data of many of the existing off-the-shelf models, therefore it makes a good evaluation dataset. 
We found that on this dataset, bag-of-words representation outperforms most dedicated sentence transformer models in terms of kNN classification accuracy, and the top performing language models barely outperform TF-IDF. We
see this as a challenge for the NLP community: to train a language model without using the labels (self-supervised) that produces a sentence embedding that would substantially surpass a naive bag-of-words representation in kNN accuracy.
### Models performance

| **Model**        | **High-dim.** | **2D** |
|------------------|---------------|--------|
| TF-IDF           | 59.2%         | 52.0%  |
| SVD              | 58.9%         | 55.9%  |
| SVD, $L^2$ norm. | 60.7%         | 56.7%  |
| SimCSE           | 45.1%         | 36.3%  |
| DeCLUTR-sci      | 52.7%         | 47.1%  |
| SciNCL           | 58.8%         | 54.9%  |
| SPECTER2         | 58.8%         | 54.1%  |
| ST5              | 57.0%         | 52.6%  |
| SBERT            | 61.6%         | 56.8%  |
| Cohere v3        | 61.1%         | 56.4%  |
| OpenAI v3        | 62.3%         | 57.1%  |

### Evaluation code
Do you want to evaluate your model on the ICLR benchmark? Here is the code for it:
```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate

def knn_accuracy_cv(embeddings, labels):
    clf = KNeighborsClassifier(
        n_neighbors=10, algorithm="brute", n_jobs=-1, metric="euclidean"
    )
    cvresults = cross_validate(clf, embeddings, labels, cv=10)

    knn_accuracy = np.mean(cvresults["test_score"])

    return knn_accuracy

# load the dataset
iclr2024 = pd.read_parquet("path/to/file/iclr24v2.parquet")

# substitute for your embeddings
embeddings = TfidfVectorizer(sublinear_tf=True).fit_transform(
    iclr2024.abstract.to_list()
)

# compute the knn accuracy
knn_acc = knn_accuracy_cv(
    embeddings[iclr2024.labels != "unlabeled"],
    iclr2024.labels[iclr2024.labels != "unlabeled"],
)
```


## Data version and maintenance
The dataset will be updated yearly.

**Last Updated:** 10/2025: added submissions to ICLR 2025 and new labels.

Labels are the same as for the 2024 dataset, except for:
 
- class `contrastive learning` and `self-supervised learning` have been merged.
 
- keyword `semantic segmentation` has been added to the class `object detection`.
 
- keyword `multi-agent` has been added to the class `multi-agent RL`.
 
- keywords `bert` and `text generation` have been added to the class `LLMs`.
 
- For all keywords where it makes sense, plural has been aded (e.g. `adversarial attack` and `adversarial attacks`).

- 6 new classes have been added:
  - `safety` with keywords `ai safety` and `safety`.
  - `alignment` with keywords `alignment` and `rlhf`.
  - `code generation` with keywords `code generation` and `program synthesis`.
  - `autonomous driving`.
  - `knowledge graph`.
  - `neuroscience`.

