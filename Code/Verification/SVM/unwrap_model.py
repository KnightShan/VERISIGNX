import pickle
import os
from pathlib import Path

class ModelWrapper:
    def __init__(self, clf=None, scaler=None, voc=None, k=None):
        self.clf = clf
        self.scaler = scaler
        self.voc = voc
        self.k = k

    def predict(self, X):
        return self.clf.predict(X)

def main():
    p = Path("model.pkl")
    if not p.exists():
        print("model.pkl not found in current folder:", os.getcwd())
        return

    with open(p, "rb") as f:
        wrapper = pickle.load(f)

    clf = getattr(wrapper, "clf", None)
    if clf is None:
        print("Could not find 'clf' attribute inside the unpickled object. Aborting.")
        return

    backup = Path("model_wrapper_backup.pkl")
    p.rename(backup)
    print(f"Backed up wrapper to {backup}")

    with open("model.pkl", "wb") as f:
        pickle.dump(clf, f)
    print("Saved classifier-only model.pkl (compatible with run.py).")

if __name__ == "__main__":
    main()
