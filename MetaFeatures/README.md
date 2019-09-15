ip# Metafeatures

### Basic Features

<br/>

| Metafeatures | Description |
| ------------ | ----------- |
| log_number_of_features | common logarithm of number_of_features |
| log_number_of_instances | common logarithm of number_of_instances |
| number_of_features | number of features |
| number_of_instances | number of instances |

<br/>

<br/>

### Class Features

<br/>

| Metafeatures | Description |
| ------------ | ----------- |
| class_entropy | information entropy associated with each class probability |
| class_probability_max | maximum of class probabilities |
| class_probability_mean | mean of class probabilities |
| class_probability_min | minimum of class probabilities |
| class_probability_std | standard deviation of class probabilities |
| dataset_ratio | ratio of class_probability_max to sum of other class probabilities |
| inverse_dataset_ratio | inverse of dataset_ratio |
| log_dataset_ratio | common logarithm of dataset_ratio |
| log_inverse_dataset_ratio | common logarithm of inverse_dataset_ratio |
| number_of_classes | number of classes |

### Distribution Features

<br/>

| Metafeatures | Description |
| ------------ | ----------- |
| kurtosis_max | maximum of kurtosis values of all features |
| kurtosis_mean | mean of kurtosis valeus of all features |
| kurtosis_min | minimum of kurtosis values of all features |
| kurtosis_std | standard deviation of kurtosis values of all features |
| skewness_min | minimum of skewness values of all features |
| skewness_mean | mean of skewness values of all features |
| skewness_max | maximum of skewness values of all features |
| skewness_std | standard deviation of skewness values of all features |
| symbols_max | maximum of symbols, where symbols refer to the numbers of levels of categorical features |
| symbols_mean | mean of symbols, where symbols refer to the numbers of levels of categorical features |
| symbols_min | minimum of symbols, where symbols refer to the numbers of levels of categorical features |
| symbols_std | standard deviation of symbols, where symbols refer to the numbers of levels of categorical features |
| symbols_sum | sum of symbols, where symbols refer to the numbers of levels of categorical features

<br/>

<br/>

### Datatype Features

<br/>

| Metafeatures | Description |
| ------------ | ----------- |
| number_of_categorical_features | number of categorical features |
| ratio_categorical_to_numerical | ratio of categorical features to numerical features |
| ratio_numerical_to_categorical | ratio of numerical features to categorical features |

<br/>

<br/>

### Missing Data Features

<br/>

| Metafeatures | Description |
| ------------ | ----------- |
| number_of_instances_with_missing_values| number of instances with missing values |
| number_of_features_with_missing_values | number of features with missing values |
| number_of_missing_values | total number of missing values |
| percentage_of_instances_with_missing_values | percentage of instances with missing values |
| percentage_of_features_with_missing_values | percentage of features with missing values |
| percentage_of_missing_values | percentage of missing values |

<br/>

<br/>

### Clustering Features

<br/>

| Metafeatures | Description |
| ------------ | ----------- |
| landmark_1NN | Evaluate the performance of the 1-nearest neighbor classifier. It uses the euclidean distance of the nearest neighbor to determine how noisy is the data (multi-valued). |
| landmark_decision_node_learner | Construct a single decision tree node model induced by one single attribute. |
| landmark_decision_tree | Evaluate the performance of a single decision tree classifier. |
| landmark_lda | Evaluate the performance of the Latent Dirichlet Allocation classifier. |
| landmark_naive_bayes | Evaluate the performance of the Naive Bayes classifier. It assumes that the attributes are independent and each example belongs to a certain class based on the Bayes probability (multi-valued). |
| landmark_random_node_learner | Construct a single decision tree node model induced by a random attribute. |
| pca_95percent | Ratio of number of compenents to number of features. Components are selected so that variance explained by each of the selected components sum to over 0.95. |
| pca_kurtosis_first_pc | skewness of input data after being applied dimensionality reduction with regard to principal axes in feature space representing the directions of maximum variance in the data. |
| pca_skewness_first_pc | kurtosis of input data after being applied dimensionality reduction with regard to principal axes in feature space representing the directions of maximum variance in the data. |

<br/>

<br/>

### Added Features

<br/>

| Metafeatures | Description |
| ------------ | ----------- |
| attr_ent_mean | mean of numerical attributes' entropy values |
| attr_ent_std | standard deviation of numerical attributes' entropy values |
| inst_to_attr | ratio of number of instances to number of attributes |
| attr_to_inst | ratio of number of attributes to number of instances |
| attr_var_mean | mean of numerical attributes' variance values |
| attr_var_std | standard deviation of numerical attributes' variance values |
| per_of_norm_attr | percentage of attributes that follows a Gaussian distribution by Shapiro–Wilk test |

<br/>

<br/>

# References:

1. Balte, A., Pise, N., & Kulkarni, P. (2014). Meta-learning with landmarking: A survey. *International Journal of Computer Applications, 105*(8).
2. Feurer, M., Klein, A., Eggensperger, K., Springenberg, J., Blum, M., & Hutter, F. (2015). Efficient and robust automated machine learning. In *Advances in neural information processing systems* (pp. 2962-2970).
3. Jomaa, H. S., Grabocka, J., & Schmidt-Thieme, L. (2019). Dataset2Vec: Learning Dataset Meta-Features. *arXiv preprint arXiv:1905.11063*.
4. Rivolli, A., Garcia, L. P., Soares, C., Vanschoren, J., & de Carvalho, A. C. (2018). Towards Reproducible Empirical Research in Meta-Learning. *arXiv preprint arXiv:1808.10406*.
5. Wistuba, M., Schilling, N., & Schmidt-Thieme, L. (2015, September). Learning Data Set Similarities for Hyperparameter Optimization Initializations. In *Metasel@ pkdd/ecml* (pp. 15-26).
