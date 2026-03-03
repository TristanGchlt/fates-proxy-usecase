from sklearn.metrics import accuracy_score, f1_score

def accuracy(data):
    y_test = data['y_test']
    y_pred = data['y_pred']
    return accuracy_score(y_test, y_pred)

def f1(data):
    y_test = data['y_test']
    y_pred = data['y_pred']
    return f1_score(y_test, y_pred)