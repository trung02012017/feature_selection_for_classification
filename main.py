import json

from datetime import datetime
from feature_selection import *
from config import *

from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report

col_df = pd.read_csv(col_name_path)
col_names = col_df.columns

data_df = pd.read_csv(feature_data_path, names=col_names)
label_df = pd.read_csv(label_data_path, names=['label'], dtype=np.int8)

data_df, label_df = shuffle(data_df, label_df, random_state=2021)

results = {}

for model_name in feature_selection_names:
    start_time = datetime.now()
    if model_name == "FE":
        print(f"Select features by FE scheme...")
        filter_x = filter_features(data_df,
                                   filter_variance,
                                   filter_correlation)
        final_x = embed_feature(filter_x,
                                label_df,
                                threshold=importance_threshold)

        results[model_name] = {
            "filter_dim": filter_x.shape[1],
            "embed_dim": final_x.shape[1]
        }

    if model_name == "E":
        print(f"Select features by Embedding scheme...")
        final_x = embed_feature(data_df,
                                label_df,
                                threshold=importance_threshold)
        results[model_name] = {
            "embed_dim": final_x.shape[1]
        }

    if model_name == "W":
        print(f"Select features by Filtering + Wrapper scheme...")
        filter_x = filter_features(data_df,
                                   filter_variance,
                                   filter_correlation)
        final_x = wrap_feature(filter_x,
                               label_df,
                               SVC(),
                               int(filter_x.shape[1] * 0.8),
                               "forward")
        results[model_name] = {
            "filter_dim": filter_x.shape[1],
            "wrapper_dim": final_x.shape[1]
        }

    if model_name == "F":
        print(f"Select features by Filtering scheme...")
        final_x = filter_features(data_df,
                                  filter_variance,
                                  filter_correlation)
        results[model_name] = {
            "filter_dim": final_x.shape[1]
        }

    if model_name == "O":
        final_x = data_df.values
        results[model_name] = {
            "origin_dim": final_x.shape[1]
        }

    results[model_name]['selection_time'] = round((datetime.now() - start_time).total_seconds(), 2)

    print("Training SVM classifier...")
    model_results = []
    kf = KFold(n_splits=n_folds)
    for train, test in kf.split(data_df):
        x_train, y_train, x_test, y_test = final_x[train], label_df.iloc[train], \
                                           final_x[test], label_df.iloc[test]

        clf = SVC()
        clf.fit(x_train, y_train.values[:, 0])
        y_pred = clf.predict(x_test)

        model_results.append(classification_report(y_test, y_pred, output_dict=True))

    results[model_name]['results'] = model_results
    print("Done training !")
    print("==================")

with open("results/evaluation.json", "w") as fp:
    json.dump(results, fp)
    fp.close()

print("Done saving results at results/evaluation.json")






