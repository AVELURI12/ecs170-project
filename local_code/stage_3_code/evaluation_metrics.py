'''
Evaluation metrics for stage 3 (multiclass classification)
'''

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from local_code.base_class.evaluate import evaluate


class Evaluate_Metrics(evaluate):
    def __init__(self, eName=None, eDescription=None):
        self.evaluate_name = eName
        self.evaluate_description = eDescription

    def evaluate(self):
        print('Evaluating Performance...')
        true_y = self.data['true_y']
        pred_y = self.data['pred_y']
        return {'Accuracy': round(accuracy_score(true_y, pred_y), 4),
                'Precision_Macro': round(precision_score(true_y, pred_y, average='macro', zero_division=0), 4),
                'Recall_Macro': round(recall_score(true_y, pred_y, average='macro', zero_division=0), 4),
                'F1_Macro': round(f1_score(true_y, pred_y, average='macro', zero_division=0), 4),
                'Precision_Weighted': round(precision_score(true_y, pred_y, average='weighted', zero_division=0), 4),
                'Recall_Weighted': round(recall_score(true_y, pred_y, average='weighted', zero_division=0), 4),
                'F1_Weighted': round(f1_score(true_y, pred_y, average='weighted', zero_division=0), 4),}
