from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import numpy as np

def arr2hot(arr, N):
    res = [0] * N
    for e in arr:
        res[e - 1] = 1
    return res

def evaluate(pred, trues, classes):
    count = pred.shape[0]
    preds = []
    lebels = [i for i in range(classes)]
    for i in range(count):
        preds.append(np.argmax(pred[i]))
    acc = accuracy_score(trues, preds)
    # micro-precision
    micro_p = precision_score(trues, preds, labels=lebels, average='micro')
    # micro-recall
    micro_r = recall_score(trues, preds, labels=lebels, average='micro')
    # micro f1-score
    micro_f1 = f1_score(trues, preds, labels=lebels, average='micro')

    # macro-precision
    macro_p = precision_score(trues, preds, average='macro')
    # macro-recall
    macro_r = recall_score(trues, preds, average='macro')
    # macro f1-score
    macro_f1 = f1_score(trues, preds, average='macro')

    return acc, macro_p, macro_r, macro_f1


# def evaluate(pred, y):
#     bs = pred.shape[0]
#     # auc = roc_auc_score(y, pred, multi_class='ovo')
#     auc = 0.5
#     # rmse = np.sqrt(np.mean((y - pred) ** 2))
#     rmse = 0.1
#     # pred[pred >= 0.5] = 1
#     # pred[pred < 0.5] = 0
#     TP, FP, TN, FN = 0, 0, 0, 0
#     for i in range(bs):
#         maxP = np.argmax(pred[i])
#         if maxP == y[i]:
#             if maxP == 1:
#                 TP += 1
#             else:
#                 TN += 1
#         elif maxP == 1:
#             FP += 1
#         else:
#             FN += 1
#     print('total predict num: {}, correct predict: {}, wrong predict: {}'.format(TP + FP + TN + FN, TP + TN, FP + FN))
#     print('TP: {}, TN: {}, FP: {}, FN: {}'.format(TP, TN, FP, FN))
#     acc = (TP + TN) / (TP + FP + TN + FN)
#     precision = TP / (TP + FP)
#     recall = TP / (TP + FN)
#     f1 = 2 * precision * recall / (precision + recall)
#     print('acc: {}, auc: {}, precision: {}, recall: {}, f1: {}, rmse: {}'.format(acc, auc, precision, recall, f1, rmse))
#     return acc, auc, precision, recall, f1, rmse