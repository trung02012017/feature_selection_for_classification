import json

with open("results/evaluation.json", "r") as fp:
    results = json.load(fp)
    fp.close()

for model_name, result_list in results.items():
    print(model_name)

    accuracy, recall_c1, recall_c0 = [], [], []
    for r in result_list['results']:
        accuracy.append(r['accuracy'])
        recall_c1.append(r["1"]["recall"])
        recall_c0.append(r["0"]["recall"])

    print(f"feature selection time: {result_list['selection_time']}")
    print(f"avg accuracy: {sum(accuracy) / len(accuracy)}")
    print(f"avg sensitivity: {sum(recall_c0) / len(recall_c0)}")
    print(f"avg specificity: {sum(recall_c1) / len(recall_c1)}")