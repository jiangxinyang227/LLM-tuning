from itertools import chain


def mean(item: list) -> float:
    """
    计算列表中元素的平均值
    :param item: 列表对象
    :return:
    """
    res = sum(item) / len(item) if len(item) > 0 else 0
    return res


def accuracy(pred_ys, true_ys):
    total = 0
    corr = 0

    for pred_y, true_y in zip(pred_ys, true_ys):
        # 做一层转换，让生成的结果对应上预测的结果，即前n-1个token预测第n个token
        pred_y = pred_y[:-1]
        true_y = true_y[1:]

        for p, t in zip(pred_y, true_y):
            if t != -100:
                total += 1
                if p == t:
                    corr += 1

    return corr / total if total > 0 else 0