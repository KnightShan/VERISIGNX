# unwrap_model.py
# Run this in the same folder where your current model.pkl (wrapper) is saved.

import pickle
import os
from pathlib import Path

# Re-declare ModelWrapper with the same name so pickle can unpickle the wrapper object.
# Minimal version matching what train.py saved (pickle uses the class name and module __main__).
class ModelWrapper:
    def __init__(self, clf=None, scaler=None, voc=None, k=None):
        self.clf = clf
        self.scaler = scaler
        self.voc = voc
        self.k = k

    def predict(self, X):
        # not needed for unwrapping
        return self.clf.predict(X)

def main():
    p = Path("model.pkl")
    if not p.exists():
        print("model.pkl not found in current folder:", os.getcwd())
        return

    # Load the wrapper (pickle looks up ModelWrapper in __main__ so we defined it above)
    with open(p, "rb") as f:
        wrapper = pickle.load(f)

    # wrapper should now be an instance of ModelWrapper with `.clf` attribute
    clf = getattr(wrapper, "clf", None)
    if clf is None:
        print("Could not find 'clf' attribute inside the unpickled object. Aborting.")
        return

    # Backup the original wrapper file
    backup = Path("model_wrapper_backup.pkl")
    p.rename(backup)
    print(f"Backed up wrapper to {backup}")

    # Save classifier-only model.pkl (overwrite the original name)
    with open("model.pkl", "wb") as f:
        pickle.dump(clf, f)
    print("Saved classifier-only model.pkl (compatible with run.py).")

if __name__ == "__main__":
    main()
