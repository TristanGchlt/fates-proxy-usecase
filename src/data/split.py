from sklearn.model_selection import train_test_split



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