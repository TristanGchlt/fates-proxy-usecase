from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
import pandas as pd

def resample(X, y, p, balance_strategy, seed) :
    resampler = {
        'undersampling' : undersampling,
        'oversampling' : oversampling
    }
    if balance_strategy == 'none' :
        return X, y, p
    else :
        return resampler[balance_strategy](X, y, p, seed)

def undersampling(X, y, p, seed) :
    rus = RandomUnderSampler(random_state=seed)
    X_res, p_res = rus.fit_resample(X, p)
    y_res = y.iloc[rus.sample_indices_]
    return X_res, y_res, p_res

def oversampling(X, y, p, seed):
    ros = RandomOverSampler(random_state=seed)
    X_res, p_res = ros.fit_resample(X, p)
    y_res = y.iloc[ros.sample_indices_]
    return X_res, y_res, p_res

def smote_sampling(X, y, p, seed):
    X_parts = []
    y_parts = []
    p_parts = []
    for target_value in y.unique():
        mask = (y == target_value)
        X_group = X.loc[mask]
        y_group = y.loc[mask]
        p_group = p.loc[mask]

        smote = SMOTE(random_state=seed)

        X_res, p_res = smote.fit_resample(
            X_group,
            p_group
        )
        
        y_res = pd.Series(
            target_value,
            index=range(len(X_res))
        )

        X_parts.append(X_res)
        y_parts.append(y_res)
        p_parts.append(p_res)

    X_res = pd.concat(X_parts, ignore_index=True)
    y_res = pd.concat(y_parts, ignore_index=True)
    p_res = pd.concat(
        [pd.Series(p) for p in p_parts],
        ignore_index=True
    )

    return X_res, y_res, p_res