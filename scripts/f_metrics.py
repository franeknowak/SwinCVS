# Standard library imports
from copy import deepcopy

# Third-party imports
import numpy as np
from sklearn.metrics import average_precision_score



# FUNCTIONS


def get_balanced_accuracies(y_true, y_pred):
    true_labels = np.concatenate([np.array(x) for x in y_true])
    predicted_labels = np.concatenate([np.array(x) for x in y_pred])

    C1_true = deepcopy(true_labels)
    C1_predicted = deepcopy(predicted_labels)
    C1_true = np.delete(C1_true, [1,2], axis=1)
    C1_predicted = np.delete(C1_predicted, [1,2], axis=1)

    C2_true = deepcopy(true_labels)
    C2_true = np.delete(C2_true, [0,2], axis=1)
    C2_predicted = deepcopy(predicted_labels)
    C2_predicted = np.delete(C2_predicted, [0,2], axis=1)

    C3_true = deepcopy(true_labels)
    C3_predicted = deepcopy(predicted_labels)
    C3_true = np.delete(C3_true, [0, 1], axis=1)
    C3_predicted = np.delete(C3_predicted, [0,1], axis=1)

    C1_recall = get_recall(C1_true, C1_predicted)
    C2_recall = get_recall(C2_true, C2_predicted)
    C3_recall = get_recall(C3_true, C3_predicted)

    C1_specificity = get_specificity(C1_true, C1_predicted)
    C2_specificity = get_specificity(C2_true, C2_predicted)
    C3_specificity = get_specificity(C3_true, C3_predicted)

    # nan-aware means: a criterion with no positives (or no negatives) has an undefined
    # recall (or specificity); average over whatever is defined instead of dividing by zero.
    C1_balanced_accuracy = _nanmean([C1_recall, C1_specificity])
    C2_balanced_accuracy = _nanmean([C2_recall, C2_specificity])
    C3_balanced_accuracy = _nanmean([C3_recall, C3_specificity])
    total_balanced_accuracy = _nanmean([C1_recall, C2_recall, C3_recall])

    return C1_balanced_accuracy, C2_balanced_accuracy, C3_balanced_accuracy, total_balanced_accuracy

def get_recall(true, predicted):
    TP = np.sum((true == 1) & (predicted == 1))
    FN = np.sum((true == 1) & (predicted == 0))
    return TP / (TP + FN) if (TP + FN) > 0 else float("nan")


def get_specificity(true, predicted):
    TN = np.sum((true == 0) & (predicted == 0))
    FP = np.sum((true == 0) & (predicted == 1))
    return TN / (TN + FP) if (TN + FP) > 0 else float("nan")


def get_map(y_true, y_pred_probs_list):
    average_precisions = []

    true_labels = np.concatenate([np.array(x) for x in y_true])
    predicted_probabilities = np.concatenate([np.array(x) for x in y_pred_probs_list])
    for class_idx in range(true_labels.shape[1]):
        class_true = true_labels[:, class_idx]
        class_scores = predicted_probabilities[:, class_idx]
        if np.sum(class_true == 1) == 0:
            # AP is undefined without positives
            average_precisions.append(float("nan"))
            continue
        average_precision = average_precision_score(class_true, class_scores)
        average_precisions.append(average_precision)

    # Calculate the mean of the average precisions across all classes to obtain mAP
    # (over classes where AP is defined)
    mAP = _nanmean(average_precisions)
    C1_ap = average_precisions[0]
    C2_ap = average_precisions[1]
    C3_ap = average_precisions[2]
    
    return C1_ap, C2_ap, C3_ap, mAP

def _nanmean(values):
    values = np.asarray(values, dtype=float)
    if np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanmean(values))


def find_best_epoch(results_dict):
    best_result = 0
    for key, val in results_dict.items():
        map_score = results_dict[key]['avg_map']
        if map_score >= best_result:
            best_epoch = key
            best_result = map_score
    return best_epoch

