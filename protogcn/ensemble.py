import json
import pickle

from .utils import comb, load_label, top1


def load(path):
    if path.endswith((".pkl", ".pickle")):
        with open(path, "rb") as f:
            return pickle.load(f)
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"Unsupported file format: {path}")


j_1 = load("j_1/best_pred.pkl")
b_1 = load("b_1/best_pred.pkl")
k_1 = load("k_1/best_pred.pkl")
j_2 = load("j_2/best_pred.pkl")
b_2 = load("b_2/best_pred.pkl")
k_2 = load("k_2/best_pred.pkl")
jm = load("jm/best_pred.pkl")
bm = load("bm/best_pred.pkl")
km = load("km/best_pred.pkl")
label = load_label("/data/finegym/gym_hrnet.pkl", "test")


print("InfoGCN  v0:")
print("j jm b bm k km")
print("2S")
fused = comb([j_1, b_1], [1, 1])
print("Top-1", top1(fused, label))

print("4S")
fused = comb([j_1, b_1, jm, bm], [2, 2, 1, 1])
print("Top-1", top1(fused, label))


print("InfoGCN  v1:")
print("j j b b k k")
print("2S")
fused = comb([j_1, b_1], [1, 1])
print("Top-1", top1(fused, label))

print("4S")
fused = comb([j_1, b_1, j_2, b_2], [1, 1, 1, 1])
print("Top-1", top1(fused, label))

