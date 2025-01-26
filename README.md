# Random Forest Model
Random Forest is one of the ensemble techniques that uses the bagging method (Bootstrap Aggregating). Within bagging, there are two main techniques: bootstrap, which involves creating multiple datasets through sampling with replacement, and aggregating, which combines the results of multiple models to achieve more stable and accurate predictions.

## Configuration options:
- `max_depth`: Maximum depth of the tree (default 10).
- `min_samples_split`: Minimum number of samples required to split a node (default 5).
- `min_samples_leaf`: : Minimum number of samples required in a leaf node (default 1).
- `n_features`: Number of features to randomly sample at each node (default √number of features in the dataset).
- `n_trees`: Number of trees in the forest (default 10).
- `bootstrap_sample_percentage`: Percentage of data used in the bootstrap sample to create each tree (default 50%).

## Dependencies
This project depends on the RegressionTree class, which is implemented in another repository. To use the Random Forest, you need to download the Regression-Trees---From-Scratch repository and place its code in the appropriate folder.
