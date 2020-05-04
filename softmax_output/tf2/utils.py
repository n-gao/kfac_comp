import pickle as pk


def load_pk(path):
    with open(path, 'rb') as f:
        x = pk.load(f)
    return x

def save_pk(x, path):
    with open(path, 'wb') as f:
        pk.dump(x, f)