from sklearn.model_selection import train_test_split

def train_test(dataset, test_size, seed) :

    train, test = train_test_split(dataset, test_size=test_size, random_state=seed, shuffle=True)
    return train, test

def x_y_p(sample, target, protected) :
    
    X = sample.drop(target, axis=1)
    y = sample[target]
    p = sample[protected]

    return X, y, p