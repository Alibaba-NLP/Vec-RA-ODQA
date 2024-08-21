import pickle


def load_vec_db(path):
    print(f"Loading the vector database from {path} ...")
    db_dict = pickle.load(open(path, 'rb'))
    # db_dict = {"key2index":..., "index2val": ...}
    print(f"Loading vector database finish.")
    return db_dict


def get_vec_from_db(db_dict, idx):
    if type(idx) == int:
        return db_dict['index2val'][idx]
    elif type(idx) == list:
        return [get_vec_from_db(db_dict, i) for i in idx]
