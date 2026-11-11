# Model evaluation results

Time-based test set: 19,200 interactions, 903 evaluated users (relevant = test rating >= 4). Items seen in training are never recommended.

## Top-10

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage@10 | Diversity@10 | RMSE | MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Popularity (baseline) | 0.0683 | 0.0721 | 0.0871 | 0.0404 | 0.7499 | 1.0880 | 0.8734 |
| SVD matrix factorisation | 0.1208 | 0.1575 | 0.1659 | 0.2652 | 0.7154 | 0.9912 | 0.7817 |
| Hybrid (SVD + genre) | 0.1194 | 0.1567 | 0.1661 | 0.2687 | 0.6702 | 0.9912 | 0.7817 |

## Top-5

| Model | Precision@5 | Recall@5 | NDCG@5 | Coverage@5 | Diversity@5 | RMSE | MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Popularity (baseline) | 0.0678 | 0.0372 | 0.0792 | 0.0262 | 0.7819 | 1.0880 | 0.8734 |
| SVD matrix factorisation | 0.1417 | 0.0924 | 0.1589 | 0.2087 | 0.7039 | 0.9912 | 0.7817 |
| Hybrid (SVD + genre) | 0.1435 | 0.0966 | 0.1624 | 0.2093 | 0.6628 | 0.9912 | 0.7817 |

*Coverage = share of the catalog recommended to at least one user. Diversity = mean pairwise genre distance inside a recommendation list. RMSE/MAE = rating-prediction error on all test ratings.*
