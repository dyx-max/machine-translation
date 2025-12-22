"""
学习率调度器
"""


class NoamOpt:
    """Noam学习率调度器（Transformer标准调度器）"""
    def __init__(self, model_size, factor, warmup, optimizer):
        """
        Args:
            model_size: 模型维度
            factor: 缩放因子
            warmup: warmup步数
            optimizer: 优化器
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size

    def step(self):
        """更新学习率并执行优化器step"""
        self._step += 1
        lr = self.factor * (self.model_size ** -0.5) * \
             min(self._step ** -0.5, self._step * (self.warmup ** -1.5))
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        self.optimizer.step()

    def zero_grad(self):
        """清零梯度"""
        self.optimizer.zero_grad()

