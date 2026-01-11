import torch 
import numpy as np
from sklearn.metrics import recall_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score
from collections import Counter

def eva_dim(Y_true, Y_pred):
    num_dim = Y_true.shape[1]  
    
    dim_accuracies = []
    dim_balanced_accuracies = []
    dim_f1_scores = []
    for j in range(num_dim):
        accuracy = accuracy_score(Y_true[:,j],Y_pred[:,j])
        balanced_accuracy = balanced_accuracy_score(Y_true[:,j],Y_pred[:,j])
        f1 = macro_f1_score(Y_true[:,j],Y_pred[:,j])

        dim_accuracies.append(accuracy)
        dim_balanced_accuracies.append(balanced_accuracy)
        dim_f1_scores.append(f1)

    return dim_accuracies, dim_balanced_accuracies, dim_f1_scores   

def eva_class(Y_true, Y_pred):
    num_dim = Y_true.shape[1]  
    
    recalls = {}
    f1 = {}
    for j in range(num_dim):
        recalls[j] = recall_score(
            Y_true[:, j], 
            Y_pred[:, j], 
            average=None, 
        )
        f1[j] = f1_score(
            Y_true[:, j], 
            Y_pred[:, j], 
            average=None,
        )

    return recalls, f1

def variance_eva(Y_true, Y_pred):
    variance_list = []
    for j in range(Y_true.shape[1]):
        variance_list.append(class_accuracy_variance(Y_true[:,j], Y_pred[:,j]))
    
    return variance_list

def class_accuracy_variance(y_true, y_pred, num_classes=None):
    """
    计算类别准确率的方差
    """
    if y_true.is_cuda:
        y_true = y_true.cpu()
        y_pred = y_pred.cpu()

    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()
    
    # 确保所有类别都包含在内
    labels = range(num_classes) if num_classes is not None else None
    
    # 1. 计算每个类别的独立 Recall
    # average=None 会返回一个 list/array: [recall_class_0, recall_class_1, ...]
    per_class_recall = recall_score(
        y_true_np, 
        y_pred_np, 
        labels=labels, 
        average=None, 
        zero_division=np.nan  # 注意这里：建议设为 nan 以区分“没样本”和“预测错”
    )
    
    # 2. 排除掉测试集中不存在的类别 (NaN)
    # 如果某个类别在当前测试 Batch/Set 中完全没出现，我们不应该把它计入“最差”
    valid_recalls = per_class_recall[~np.isnan(per_class_recall)]
    
    if len(valid_recalls) == 0:
        return 0.0

    variance = np.var(valid_recalls)

    return variance

def worst_class_eva(Y_true, Y_pred):
    worst_class_accuracy_list, worst_class_macro_f1_score_list = [], []
    for j in range(Y_true.shape[1]):
        worst_class_accuracy_list.append(worst_class_accuracy_score(Y_true[:,j], Y_pred[:,j]))
        worst_class_macro_f1_score_list.append(worst_class_macro_f1_score(Y_true[:,j], Y_pred[:,j]))
    
    return worst_class_accuracy_list, worst_class_macro_f1_score_list

def worst_class_macro_f1_score(y_true, y_pred, num_classes=None):
    """
    计算最差类别的宏观 F1 分数 (Worst-Class Macro F1-Score)。
    """
    if y_true.is_cuda:
        y_true = y_true.cpu()
        y_pred = y_pred.cpu()

    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()
    
    # 确保所有类别都包含在内
    labels = range(num_classes) if num_classes is not None else None
    
    # 1. 计算每个类别的独立 F1 分数
    # average=None 会返回一个 list/array: [f1_class_0, f1_class_1, ...]
    per_class_f1 = f1_score(
        y_true_np, 
        y_pred_np, 
        labels=labels, 
        average=None, 
        zero_division=np.nan  # 注意这里：建议设为 nan 以区分“没样本”和“预测错”
    )
    
    # 2. 排除掉测试集中不存在的类别 (NaN)
    # 如果某个类别在当前测试 Batch/Set 中完全没出现，我们不应该把它计入“最差”
    valid_f1s = per_class_f1[~np.isnan(per_class_f1)]
    
    if len(valid_f1s) == 0:
        return 0.0
        
    # 3. 取最小值，即为 Worst-Class Macro F1-Score
    worst_f1 = np.min(valid_f1s)
    
    return worst_f1

def worst_class_accuracy_score(y_true, y_pred, num_classes=None):
    """
    计算最差类别准确率 (Worst-Class Accuracy / Minimum Recall)。
    """
    if y_true.is_cuda:
        y_true = y_true.cpu()
        y_pred = y_pred.cpu()

    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()
    
    # 确保所有类别都包含在内
    labels = range(num_classes) if num_classes is not None else None
    
    # 1. 计算每个类别的独立 Recall
    # average=None 会返回一个 list/array: [recall_class_0, recall_class_1, ...]
    per_class_recall = recall_score(
        y_true_np, 
        y_pred_np, 
        labels=labels, 
        average=None, 
        zero_division=np.nan  # 注意这里：建议设为 nan 以区分“没样本”和“预测错”
    )
    
    # 2. 排除掉测试集中不存在的类别 (NaN)
    # 如果某个类别在当前测试 Batch/Set 中完全没出现，我们不应该把它计入“最差”
    valid_recalls = per_class_recall[~np.isnan(per_class_recall)]
    
    if len(valid_recalls) == 0:
        return 0.0
        
    # 3. 取最小值，即为 Worst-Class Accuracy
    worst_acc = np.min(valid_recalls)
    
    return worst_acc

def worst_dim_eva(Y_true, Y_pred):
    worst_accuracy = worst_dim_accuracy(Y_true, Y_pred)
    worst_balanced_accuracy = worst_dim_balanced_accuracy(Y_true, Y_pred)
    worst_f1_score = worst_dim_f1_score(Y_true, Y_pred)

    return worst_accuracy, worst_balanced_accuracy, worst_f1_score

def worst_dim_f1_score(Y_true, Y_pred):
    """
    计算每个维度的宏观 F1 分数，并返回最差维度的宏观 F1 分数。

    Args:
        y_true (torch.Tensor): 真实标签，形状为 (num_samples, num_dim)。
        y_pred (torch.Tensor): 预测标签，形状为 (num_samples, num_dim)。
    """
    num_dim = Y_true.shape[1]  
    
    dim_f1_scores = []
    for j in range(num_dim):
        f1 = macro_f1_score(Y_true[:,j],Y_pred[:,j])
        dim_f1_scores.append(f1)
    
    worst_f1_score = min(dim_f1_scores)
    # worst_dim = np.argmin(dim_f1_scores)
    
    return worst_f1_score

def worst_dim_balanced_accuracy(Y_true, Y_pred):
    """
    计算每个维度的平衡准确率，并返回最差维度的平衡准确率。

    Args:
        y_true (torch.Tensor): 真实标签，形状为 (num_samples, num_dim)。
        y_pred (torch.Tensor): 预测标签，形状为 (num_samples, num_dim)。
    """
    num_dim = Y_true.shape[1]  
    
    dim_balanced_accuracies = []
    for j in range(num_dim):
        balanced_accuracy = balanced_accuracy_score(Y_true[:,j],Y_pred[:,j])
        dim_balanced_accuracies.append(balanced_accuracy)
    
    worst_balanced_accuracy = min(dim_balanced_accuracies)
    # worst_dim = np.argmin(dim_balanced_accuracies)
    
    return worst_balanced_accuracy

def worst_dim_accuracy(Y_true, Y_pred):
    """
    计算每个维度的准确率，并返回最差维度的准确率。
    Args:
        y_true (torch.Tensor): 真实标签，形状为 (num_samples, num_dim)。
        y_pred (torch.Tensor): 预测标签，形状为 (num_samples, num_dim)。
    """
    num_dim = Y_true.shape[1]  
    
    dim_accuracies = []
    for j in range(num_dim):
        accuracy = accuracy_score(Y_true[:,j],Y_pred[:,j])
        dim_accuracies.append(accuracy)
    
    worst_accuracy = min(dim_accuracies)
    # worst_dim = np.argmin(dim_accuracies)

    return worst_accuracy

def accuracy_score(y_true, y_pred):
    """
    计算准确率 (Accuracy Score)。
    Args:
        y_true (torch.Tensor): 真实标签 (类别索引)
        y_pred (torch.Tensor): 预测标签 (类别索引)
    """
    if y_true.is_cuda:
        y_true = y_true.cpu()
        y_pred = y_pred.cpu()

    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()
    
    correct = np.sum(y_true_np == y_pred_np)
    total = y_true_np.shape[0]
    
    accuracy = correct / total if total > 0 else 0.0
    
    return accuracy
    

def balanced_accuracy_score(y_true, y_pred, num_classes=None):
    """
    计算平衡准确率 (Balanced Accuracy Score / Mean Recall)。
    Args:
        y_true (torch.Tensor): 真实标签 (类别索引)。
        y_pred (torch.Tensor): 预测标签 (类别索引)。
        num_classes (int, optional): 类别总数。如果为 None, 则从标签中推断。
    """
    if y_true.is_cuda:
        y_true = y_true.cpu()
        y_pred = y_pred.cpu()

    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()
    
    # 使用 sklearn 的 recall_score 计算每个类别的召回率，然后取平均
    # average='macro' 正是计算平均召回率
    
    # 确保所有类别都包含在内，即使某个 Batch 中缺少该类别 (防止训练时报错)
    labels = range(num_classes) if num_classes is not None else None
    
    try:
        # recall_score(average='macro') 等同于 Balanced Accuracy
        # zero_division=0: 确保如果某个类别没有样本，召回率记为 0
        balanced_acc = recall_score(y_true_np, y_pred_np, labels=labels, average='macro', zero_division=0)
    except ValueError:
        # 如果 labels 包含未出现在 y_true/y_pred 中的类别，可能会抛出 ValueError
        # 此时退化为只计算现有类别的平均召回率
        balanced_acc = recall_score(y_true_np, y_pred_np, average='macro', zero_division=0)
        
    return balanced_acc

def macro_f1_score(y_true, y_pred, num_classes=None):
    """
    计算宏观 F1 分数 (Macro F1-Score)。
    Args:
        y_true (torch.Tensor): 真实标签 (类别索引)。
        y_pred (torch.Tensor): 预测标签 (类别索引)。
        num_classes (int, optional): 类别总数。
    """
    if y_true.is_cuda:
        y_true = y_true.cpu()
        y_pred = y_pred.cpu()

    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()

    labels = range(num_classes) if num_classes is not None else None
    
    try:
        macro_f1 = f1_score(y_true_np, y_pred_np, labels=labels, average='macro', zero_division=0)
    except ValueError:
        macro_f1 = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
        
    return macro_f1


def tail_to_head_ratio(y_true_all, y_pred_all, train_labels_all, head_ratio=0.1, tail_ratio=0.5):
    """
    计算尾部到头部召回率比值 (R_Tail / R_Head)。
    
    Args:
        y_true_all (torch.Tensor): 整个测试集/验证集真实标签。
        y_pred_all (torch.Tensor): 整个测试集/验证集预测标签。
        train_labels_all (torch.Tensor): 整个训练集标签 (用于确定频率)。
        head_ratio (float): 定义头部类别的比例 (e.g., 0.1 表示频率前 10% 的类别)。
        tail_ratio (float): 定义尾部类别的比例 (e.g., 0.5 表示频率后 50% 的类别)。

    Returns:
        float: R_Tail / R_Head 比值。
    """
    if y_true_all.is_cuda:
        y_true_all = y_true_all.cpu()
        y_pred_all = y_pred_all.cpu()
        train_labels_all = train_labels_all.cpu()

    y_true_np = y_true_all.numpy()
    y_pred_np = y_pred_all.numpy()
    train_labels_np = train_labels_all.numpy()

    # 1. 确定类别频率排名 (基于训练集)
    counts = Counter(train_labels_np)
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    all_classes = [item[0] for item in sorted_counts]
    num_classes = len(all_classes)

    # 2. 划分头部和尾部
    # 头部类别 (Head): 频率最高的 top K 个类别
    num_head = int(num_classes * head_ratio)
    head_classes = all_classes[:num_head]

    # 尾部类别 (Tail): 频率最低的 bottom K 个类别 (通常从尾部开始取)
    num_tail = int(num_classes * tail_ratio)
    # 注意：为了避免 Head 和 Tail 重叠，通常 Tail 是指频率最低的 K 个类别
    tail_classes = all_classes[-num_tail:]

    # 3. 计算所有类别的召回率 (用于测试集)
    recalls = recall_score(y_true_np, y_pred_np, labels=all_classes, average=None, zero_division=0)
    class_to_recall = dict(zip(all_classes, recalls))

    # 4. 计算 R_Head (头部类别的平均召回率)
    head_recalls = [class_to_recall.get(c, 0.0) for c in head_classes]
    r_head = np.mean(head_recalls) if head_recalls else 0.0

    # 5. 计算 R_Tail (尾部类别的平均召回率)
    tail_recalls = [class_to_recall.get(c, 0.0) for c in tail_classes]
    r_tail = np.mean(tail_recalls) if tail_recalls else 0.0

    # 6. 计算比值
    ratio = r_tail / r_head if r_head > 1e-6 else 0.0
    
    return ratio, r_head, r_tail

def eva(Y,Y_result):
    '''
    Evaluations for MDC.
    '''
    num_testing = Y.shape[0]                     #number of training examples
    num_dim = Y.shape[1]                          #number of dimensions(class variables)
    num_correctdim = torch.sum(Y == Y_result,dim=1)  #number of correct dimmensions for each example
        
    #Hamming Score(or Class Accuracy)
    HammingScore = torch.sum(num_correctdim)/(num_dim*num_testing)    
    
    #Exact Match(or Example Accuracy or Subset Accuracy)
    ExactMatch = torch.sum(num_correctdim == num_dim)/num_testing
    
    #Sub-ExactMatch    
    SubExactMatch = torch.sum(num_correctdim >= num_dim-1)/num_testing

    return HammingScore,ExactMatch,SubExactMatch