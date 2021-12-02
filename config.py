col_name_path = "data/feature_names.csv"
feature_data_path = "data/data.csv"
label_data_path = "data/labels.csv"

n_folds = 5

filter_variance = 0.02
filter_correlation = 0.85
importance_threshold = 0.01

# name of models for feature selection:
# "F" means filtering method
# "E" means embedding method
# "FE" means method using F and E
# "W" means our proposed method"
# "O" means original data
feature_selection_names = ["F", "E", "W", "O", "FE"]

