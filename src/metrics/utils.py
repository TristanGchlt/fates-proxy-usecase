import src.metrics.predictive_perf as perf
import src.metrics.fairness as fairness
import json


METRICS_FUNC = {
    # "accuracy" : perf.accuracy,
}

def compute_measures(model, data, metrics) :
    measures = {}
    for metric in metrics :
        measures[metric] = METRICS_FUNC[metric](model, data)
    return measures

def save_measures(measures, model_path) :
    path = model_path / "measures.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(measures, f, indent=4, ensure_ascii=False)