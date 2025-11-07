
class StepLR:
    def __init__(self, optimizer, learning_rate: float, total_epochs: int, gamma: float):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.base = learning_rate
        self.gamma = gamma

    def __call__(self, epoch):
        #if epoch < self.total_epochs * 3/10:
        #    lr = self.base
        #elif epoch < self.total_epochs * 6/10:
        #    lr = self.base * 0.2
        #elif epoch < self.total_epochs * 8/10:
        #    lr = self.base * 0.2 ** 2
        #else:
        #    lr = self.base * 0.2 ** 3
        lr = self.base*self.gamma

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]
