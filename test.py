import os
import csv
from collections import defaultdict


def load_params(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        return dict(reader)


def load_stats(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        keys = next(reader)
        ret = defaultdict(list)
        for line in reader:
            for idx, key in enumerate(keys):
                ret[key].append(float(line[idx]))
    return dict(ret)

ans = []

for model_folder in os.listdir("./models"):
    if not os.path.isdir(os.path.join("./models", model_folder)):
        continue
    model_path = os.path.join("./models", model_folder)
    params = load_params(os.path.join(model_path, "params.csv"))
    stats = load_stats(os.path.join(model_path, "stats.csv"))
    results = params
    for key in stats.keys():
        results["{key}_max".format(key=key)] = max(stats[key])
        results["{key}_min".format(key=key)] = min(stats[key])
    print(results)
    ans.append(results)

with open("./models/summary.csv", "w", encoding="utf-8") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(ans[0].keys())
    for item in ans:
        writer.writerow(item.values())



