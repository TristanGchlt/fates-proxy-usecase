

def train(model_type, X_train, y_train, hyperparameters) :
    trainers = {
        "Random Forest Classifier" : train_rfc
    }
    return trainers[model_type](X_train, y_train, hyperparameters)


def train_rfc(X_train, y_train, hyperparameters) :
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(**hyperparameters)
    model.fit(X_train, y_train)
    return model