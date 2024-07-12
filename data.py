import pickle

def save(data, fn):
    with open(f"data/{fn}.pickle", 'wb') as f:
        pickle.dump(data, f)
    
def load(fn):
    try:
        with open(f"data/{fn}.pickle", 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return []
