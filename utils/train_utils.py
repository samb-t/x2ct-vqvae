class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def optim_warmup(step, optim, lr, warmup_iters):
    lr = lr * float(step) / warmup_iters
    for param_group in optim.param_groups:
        param_group['lr'] = lr

def update_ema(model, ema_model, ema_rate):
    for p1, p2 in zip(model.parameters(), ema_model.parameters()):
        # Beta * previous ema weights + (1 - Beta) * current non ema weight
        p2.data.mul_(ema_rate)
        p2.data.add_(p1.data * (1 - ema_rate))