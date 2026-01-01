import json

with open("data_10.json") as f:
    data = json.load(f)

print("num samples:", len(data["mfcc"]))
print("num labels:", len(data["labels"]))
print("num genres:", len(data["mapping"]))
print("mfcc shape (one sample):", len(data["mfcc"][0]), "x", len(data["mfcc"][0][0]))