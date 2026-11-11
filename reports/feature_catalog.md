# Feature catalog

All features are computed from the **training split only** (point-in-time correct) and are served by `src/features/store.py`, so training and prediction use identical code.

| Feature | Entity | Type | Description | Computation logic | Source | Used by |
|---|---|---|---|---|---|---|
| `user_activity_count` | user | int | How many items the user interacted with (train period). | count(interactions) per user | train | cold-start detection (== 0), analysis |
| `user_avg_rating` | user | float | The user's average given rating (rating generosity). | mean(rating) per user; global mean for cold-start users | train | pair feature avg_rating_gap, analysis |
| `user_rating_std` | user | float | Spread of the user's ratings. | std(rating) per user | train | analysis |
| `user_positive_rate` | user | float | Share of the user's ratings that are 'liked' (>= threshold). | mean(rating >= 4) per user | train | analysis |
| `user_active_days` | user | int | Number of distinct calendar days with activity. | nunique(floor(timestamp / 86400)) per user | train | analysis |
| `days_since_last_interaction` | user | float | Recency of the user (days). | (as_of - max(timestamp)) / 86400, as_of = last timestamp in train | train | analysis, cold-start rules |
| `user_preferred_category` | user | category | The genre the user liked most. | argmax of positive-interaction counts per genre (all interactions if no positives) | train + items | pair feature preferred_category_match, explanations |
| `user_preferred_category_share` | user | float | How dominant the preferred genre is (0-1). | max genre affinity | train + items | analysis |
| `user_age` | user | int | Age of the user (imputed with median if missing). | users_clean.age | users | analysis |
| `user_age_group` | user | category | Age bucket. | pd.cut(age, bins) | users | analysis |
| `user_gender` | user | category | Gender (M / F / U = unknown). | users_clean.gender | users | analysis |
| `user_occupation` | user | category | Occupation (lower-cased, 'unknown' if missing). | users_clean.occupation | users | analysis |
| `user_genre_affinity[g]` | user | float vector | Distribution over genres of the user's liked items (sums to 1). | L1-normalised (positive user x item matrix @ item x genre matrix) | train + items | Hybrid (content score) |
| `item_popularity_count` | item | int | Number of interactions the item received. | count(interactions) per item | train | Popularity model |
| `item_popularity_rank` | item | int | Rank by popularity (1 = most popular). | rank(item_popularity_count, descending, method=min) | train | analysis |
| `item_popularity_score` | item | float | Popularity scaled to 0-1, damped by a log. | log1p(count) / log1p(max count) | train | analysis |
| `item_avg_rating` | item | float | Mean rating of the item (global mean if never rated). | mean(rating) per item | train | pair feature avg_rating_gap |
| `item_bayes_avg_rating` | item | float | Mean rating shrunk to the global mean for items with few ratings. | (C * global_mean + sum(rating)) / (C + count), C = 10 | train | Popularity model (rating prediction + tie-break) |
| `item_rating_std` | item | float | Disagreement between users about the item. | std(rating) per item | train | analysis |
| `item_positive_rate` | item | float | Share of ratings that are 'liked'. | mean(rating >= 4) per item | train | analysis |
| `item_days_since_last_interaction` | item | float | How recently the item was interacted with. | (as_of - max(timestamp)) / 86400 | train | analysis |
| `item_age_years` | item | int | Age of the item at as_of time. | as_of year - release_year | items | analysis |
| `item_n_genres` | item | int | Number of genres of the item. | len(genres) | items | genre_affinity normalisation |
| `item_primary_genre` | item | category | First genre (alphabetical) of the item. | sorted(genres)[0] | items | analysis |
| `item_genres` | item | string | All genres, '\|' separated. | items_clean.genres | items | genre matrix (Hybrid, Diversity metric), explanations |
| `genre_affinity` | interaction | float | How well the item's genres match the user's genre taste. | mean over the item's genres of user_genre_affinity[g] | user + item features | Hybrid (computed at training AND prediction time) |
| `preferred_category_match` | interaction | int (0/1) | 1 if the item belongs to the user's preferred genre. | user_preferred_category in item_genres | user + item features | analysis, explanations |
| `avg_rating_gap` | interaction | float | Is the item better or worse rated than the user's usual rating? | item_avg_rating - user_avg_rating | user + item features | analysis |
| `interaction_age_days` | interaction | float | How old an observed interaction is (train rows only). | (as_of - timestamp) / 86400 | train | recency weighting (future work) |
