from imblearn.under_sampling import RandomUnderSampler

def resample(X, y, p, balance_strategy, seed) :
    resampler = {
        'undersampling' : undersampling
    }
    if balance_strategy == 'none' :
        return X, y, p
    else :
        return resampler[balance_strategy](X, y, p, seed)

def undersampling(X, y, p, seed) :
    rus = RandomUnderSampler(random_state=seed)
    X, p = rus.fit_resample(X, p)
    y = y[rus.sample_indices_]
    return X, y, p

