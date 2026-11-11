# Data validation report

*Generated: 2026-11-11T12:37:06Z  |  config hash: `f2571976c7d4`  |  data source: movielens*

## 1. Summary per table

| Table | Rows in | Valid rows | Rejected rows | Reject rate | Rows with warnings |
|---|---:|---:|---:|---:|---:|
| users | 946 | 943 | 3 | 0.317% | 21 |
| items | 1,685 | 1,682 | 3 | 0.178% | 9 |
| interactions | 100,400 | 98,400 | 2,000 | 1.992% | 0 |

**Quality gate:** interactions reject rate 1.992% (threshold 5.0%) -> **PASSED**

## 2. Rule results

| Table | Rule | Column | Severity | Failed | % | Injected (demo) | Action |
|---|---|---|---|---:|---:|---:|---|
| users | `missing_user_id` | user_id | error | 0 | 0.0 |  | reject row |
| users | `invalid_type_user_id` | user_id | error | 0 | 0.0 |  | reject row |
| users | `duplicate_user_id` | user_id | error | 3 | 0.317 | 3 | reject later duplicates |
| users | `missing_age` | age | warning | 5 | 0.529 | 5 | set NULL, impute (median) |
| users | `invalid_type_age` | age | warning | 2 | 0.211 | 2 | set NULL, impute (median) |
| users | `age_out_of_range` | age | warning | 4 | 0.423 | 4 | set NULL, impute (median) |
| users | `missing_gender` | gender | warning | 5 | 0.529 | 5 | set to 'U' (unknown) |
| users | `invalid_gender` | gender | warning | 2 | 0.211 | 2 | set to 'U' (unknown) |
| users | `missing_occupation` | occupation | warning | 3 | 0.317 | 3 | set to 'unknown' |
| items | `missing_item_id` | item_id | error | 0 | 0.0 |  | reject row |
| items | `invalid_type_item_id` | item_id | error | 0 | 0.0 |  | reject row |
| items | `duplicate_item_id` | item_id | error | 3 | 0.178 | 3 | reject later duplicates |
| items | `missing_title` | title | warning | 4 | 0.237 | 4 | set to 'Unknown title' |
| items | `invalid_release_date` | release_date | warning | 5 | 0.297 | 5 | set NULL, impute (median year) |
| items | `missing_genres` | genres | warning | 0 | 0.0 |  | set to 'unknown' |
| interactions | `missing_user_id` | user_id | error | 200 | 0.199 | 200 | reject row |
| interactions | `missing_item_id` | item_id | error | 0 | 0.0 |  | reject row |
| interactions | `missing_rating` | rating | error | 200 | 0.199 | 200 | reject row |
| interactions | `missing_timestamp` | timestamp | error | 0 | 0.0 |  | reject row |
| interactions | `invalid_type_user_id` | user_id | error | 0 | 0.0 |  | reject row |
| interactions | `invalid_type_item_id` | item_id | error | 0 | 0.0 |  | reject row |
| interactions | `invalid_type_rating` | rating | error | 200 | 0.199 | 200 | reject row |
| interactions | `invalid_type_timestamp` | timestamp | error | 200 | 0.199 | 200 | reject row |
| interactions | `rating_out_of_range` | rating | error | 200 | 0.199 | 200 | reject row |
| interactions | `timestamp_out_of_range` | timestamp | error | 400 | 0.398 | 400 | reject row |
| interactions | `duplicate_interaction` | user_id | error | 200 | 0.199 | 200 | reject later duplicates |
| interactions | `orphan_user_id` | user_id | error | 200 | 0.199 | 200 | reject row |
| interactions | `orphan_item_id` | item_id | error | 200 | 0.199 | 200 | reject row |

*'Injected (demo)' = number of corrupt records deliberately inserted by the ingestion stage (`ingestion.inject_faults`). Detected >= injected shows that the rule works.*

## 3. Where the bad records went

* `data/errors/rejected_users.csv`, `rejected_items.csv`, `rejected_interactions.csv` - rejected rows with the failed rule(s)
* `data/errors/validation_issues.csv` - one line per issue (errors and warnings)
* `data/processed/valid_*.parquet` - records that passed validation

## 4. Example rejected records

**duplicate_interaction**

| source_line | column | bad value |
|---:|---|---|
| 100002 | user_id | `527` |
| 100003 | user_id | `790` |
| 100004 | user_id | `833` |

**duplicate_item_id**

| source_line | column | bad value |
|---:|---|---|
| 1684 | item_id | `728` |
| 1685 | item_id | `886` |
| 1686 | item_id | `48` |

**duplicate_user_id**

| source_line | column | bad value |
|---:|---|---|
| 945 | user_id | `44` |
| 946 | user_id | `837` |
| 947 | user_id | `425` |

**invalid_type_rating**

| source_line | column | bad value |
|---:|---|---|
| 552 | rating | `good` |
| 1588 | rating | `good` |
| 2446 | rating | `five` |

**invalid_type_timestamp**

| source_line | column | bad value |
|---:|---|---|
| 533 | timestamp | `not_a_timestamp` |
| 1304 | timestamp | `yesterday` |
| 1977 | timestamp | `not_a_timestamp` |

**missing_rating**

| source_line | column | bad value |
|---:|---|---|
| 436 | rating | `<NULL>` |
| 494 | rating | `<NULL>` |
| 800 | rating | `<NULL>` |

**missing_user_id**

| source_line | column | bad value |
|---:|---|---|
| 121 | user_id | `<NULL>` |
| 233 | user_id | `<NULL>` |
| 564 | user_id | `<NULL>` |

**orphan_item_id**

| source_line | column | bad value |
|---:|---|---|
| 708 | item_id | `2734` |
| 733 | item_id | `2786` |
| 1359 | item_id | `2735` |

**orphan_user_id**

| source_line | column | bad value |
|---:|---|---|
| 181 | user_id | `2010` |
| 423 | user_id | `1976` |
| 541 | user_id | `2105` |

**rating_out_of_range**

| source_line | column | bad value |
|---:|---|---|
| 32 | rating | `6` |
| 735 | rating | `7` |
| 1516 | rating | `0` |

**timestamp_out_of_range**

| source_line | column | bad value |
|---:|---|---|
| 405 | timestamp | `4102444800` |
| 628 | timestamp | `4102444800` |
| 958 | timestamp | `4102444800` |
