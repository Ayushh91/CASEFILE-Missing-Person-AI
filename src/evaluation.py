import numpy as np
def top_k_accuracy(probabilities, actual, classes, k=3):
    labels=np.asarray(classes); return float(np.mean([a in labels[np.argsort(row)[-k:]] for row,a in zip(probabilities,actual)]))
