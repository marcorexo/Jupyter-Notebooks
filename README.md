# Classification Trees: Building and Pruning a Decision Tree

This notebook uses a small loan dataset to show how a decision tree can overfit, and how pruning makes it simpler and better at handling new data.

## Dataset

`loan.csv` has 30 loans with two features, **Income** and **Loan Amount**, and one target, **Default** (Yes/No). The data is split 80/20 into 24 training rows and 6 test rows.

## What the notebook covers

1. Exploring the data with box plots and a scatter plot
2. Training an unrestricted classification tree
3. Visualising the tree and its decision boundaries
4. Checking feature importance
5. Pruning the tree with `GridSearchCV`, first by limiting depth, then by also tuning `min_samples_split` and `min_samples_leaf`

## Key results

| Model | Train accuracy | Test accuracy |
|---|---|---|
| Unpruned tree | 100% | 50% |
| `max_depth=3` | 95.8% | 33.3% |
| `max_depth=2`, `min_samples_leaf=6` | 87.5% | 83.3% |

The unpruned tree memorises the training data and does poorly on new data. Limiting depth alone doesn't solve this. Requiring at least 6 samples per leaf gives a much simpler tree that mostly comes down to one rule: **Income ≤ 14.5 → likely to default**.

With only 6 test rows, each prediction changes test accuracy by about 17 percentage points, so these results show the idea rather than prove it.

## Running it

The notebook was written in Google Colab and loads `loan.csv` from Google Drive. To run it elsewhere, change the file path in the data-loading cell to point to your copy of `loan.csv`.

Requirements: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
