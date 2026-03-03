from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

def demographic_parity(data):
    y_test = data['y_test']
    y_pred = data['y_pred']
    p_test = data['p_test']
    return demographic_parity_difference(y_test, y_pred, sensitive_features=p_test)

def equalized_odds(data):
    y_test = data['y_test']
    y_pred = data['y_pred']
    p_test = data['p_test']
    return equalized_odds_difference(y_test, y_pred, sensitive_features=p_test)