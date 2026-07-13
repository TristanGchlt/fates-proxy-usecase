from sklearn.model_selection import train_test_split
from .sampling import resample


def train_test(dataset, test_size, protected, strategy, seed) :
    splitter = {
        'random' : train_test_random,
        'stratified' : train_test_stratified
    }
    return splitter[strategy](dataset, test_size, protected, seed)

def train_test_random(dataset, test_size, protected, seed) :
    train, test = train_test_split(dataset, test_size=test_size, random_state=seed, shuffle=True)
    return train, test

def train_test_stratified(dataset, test_size, protected, seed) :
    train, test = train_test_split(dataset, test_size=test_size, random_state=seed, shuffle=True, stratify=dataset[protected])
    return train, test

def x_y_p(sample, target, protected) :
    X = sample.drop(target, axis=1)
    y = sample[target]
    p = sample[protected]
    return X, y, p

def hide_p(X, protected) :
    return X.drop(protected, axis=1)

def hide_p_proxies(dataset, protected, target, threshold=0.2, method="pearson") :
    numeric_data = dataset.select_dtypes(include=["number", "bool"])
    numeric_data.drop(target, axis=1)
    numeric_data = numeric_data.astype({col : int for col in numeric_data.select_dtypes("bool").columns})
    corr = numeric_data.corr(method=method)[protected].drop(protected)
    proxies = corr[corr.abs() > threshold].index.tolist()
    to_remove = [p for p in proxies if p not in [protected, target]]
    return dataset.drop(to_remove, axis=1)


def split(dataset, test_size, protected_feature, split_strategy, seed, target_feature, balance_strategy, balance_seed, hide_protected, hide_proxies):
    if hide_proxies :
        dataset = hide_p_proxies(dataset, target_feature, protected_feature)
    train, test = train_test(dataset, test_size, protected_feature, split_strategy, seed)
    X_train, y_train, p_train = x_y_p(train, target_feature, protected_feature)
    X_test, y_test, p_test = x_y_p(test, target_feature, protected_feature)
    X_train, y_train, p_train = resample(X_train, y_train, p_train, balance_strategy, balance_seed)
    if hide_protected : 
        X_train = hide_p(X_train, protected_feature)
        X_test = hide_p(X_test, protected_feature)
    return X_train, y_train, p_train, X_test, y_test, p_test