

def train(model_type, X_train, y_train, hyperparameters) :
    trainers = {
        "Random Forest Classifier" : train_rfc,
        "XGBoost" : train_xgboost
    }
    return trainers[model_type](X_train, y_train, hyperparameters)


def train_rfc(X_train, y_train, hyperparameters) :
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(**hyperparameters)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, hyperparameters) :
    from xgboost import XGBClassifier
    model = XGBClassifier(**hyperparameters)
    model.fit(X_train, y_train)
    return model