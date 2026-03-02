

def train(model_type, data, hyperparameters) :
    trainers = {
        "Random Forest Classifier" : train_rfc
    }
    return trainers[model_type](data, hyperparameters)


def train_rfc(data, hyperparameters) :
    from sklearn.ensemble import RandomForestClassifier
    X_train = data['X_train']
    y_train = data['y_train'].values.ravel()
    model = RandomForestClassifier(**hyperparameters)
    model.fit(X_train, y_train)
    return model