

def save_model(model, model_type, model_path) :
    savers= {
        "Random Forest Classifier" : save_pickle_model
    }
    return savers[model_type](model, model_path)

def save_pickle_model(model, model_path) :
    import pickle
    path = model_path / "model.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return None

def predict(model, data) :
    return model.predict(data['X_test']) # Cette fonction doit devenir dynamique selon le type de model

def save_model_type(model_type : str, path : str) :
    model_type_path = path / "model_type.txt"
    with open(model_type_path, "w") as f:
        f.write(model_type)
    return None