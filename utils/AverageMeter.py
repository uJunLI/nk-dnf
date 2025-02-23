
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        """ 重置所有统计值 """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ 更新当前值、总和、样本计数和平均值 """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
