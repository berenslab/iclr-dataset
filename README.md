# Learning representations of learning representations

The ICLR dataset is a complete scrape of ICLR submissions from OpenReview. It contains 24,445 ICLR submissions from 2017 to 2024.

![ICLR dataset, SBERT embedding](/results/figures/embedding.png)


## Dataset snapshot
Each sample corresponds to a **submitted** article to the ICLR conference and includes as features:
-  Year
-  OpenReview Id
-  Title
-  Abstract
-  Authors
-  Decision
-  Scores
-  Keywords
-  Labels
  
To label the dataset, we relied on the author-provided keywords and used them to assign papers to 45 non-overlapping classes. We combined some keywords together into one class (e.g. *attention* and *transformer*), disregarded very broad keywords (e.g. *deep learning*), and assigned papers to rarer classes first. Using this procedure, we ended up labeling 53.4% of all papers.

![image](https://github.com/berenslab/iclr-dataset/assets/82372364/2fa62933-7a71-4231-b009-31ababd88a50)

Note that 26 submissions with placeholder abstracts (below 100 characters) are excluded.

### Descriptive statistics
![ICLR dataset, summary statistics](/results/figures/summary-stats.png)

## Data version and maintenance
The dataset will be updated yearly.

**Last Updated:** 04/2024

**Update:** Add deanonymized submissions.


## Citations

The dataset is described in González-Márquez & Kobak, González Márquez et al., Learning representations of learning representations, DMLR workshop at ICLR 2024. Please cite as follows:

```
@inproceedings{gonzalez2024learning,
  title={Learning representations of learning representations},
  author={Gonz{\'a}lez-M{\'a}rquez, Rita and Kobak, Dmitry},
  booktitle={Data-centric Machine Learning Research (DMLR) workshop at ICLR 2024},
  year={2024}
}
```


