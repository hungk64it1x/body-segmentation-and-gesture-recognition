from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt
import torch


class GradualWarmupScheduler(_LRScheduler):
    # https://github.com/seominseok0429/pytorch-warmup-cosine-lr
    def __init__(self, optimizer, multiplier, total_warmup_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_warmup_epoch = total_warmup_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_warmup_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [
            base_lr
            * (
                (self.multiplier - 1.0) * self.last_epoch / self.total_warmup_epoch
                + 1.0
            )
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_warmup_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def cosine(optimizer, total_epoch, num_warmup_epoch):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, total_epoch, eta_min=0, last_epoch=-1
    )


def cosine_warmup(optimizer, total_epoch, num_warmup_epoch):
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, total_epoch, eta_min=0, last_epoch=-1
    )
    return GradualWarmupScheduler(
        optimizer,
        multiplier=8,
        total_warmup_epoch=num_warmup_epoch,
        after_scheduler=cosine_scheduler,
    )


if __name__ == "__main__":
    v = torch.zeros(10)
    optim = torch.optim.SGD([v], lr=0.0001)
    scheduler = cosine_warmup(optim, 200, 8)

    a = []
    b = []
    for epoch in range(0, 200):
        scheduler.step(epoch)
        a.append(epoch)
        b.append(optim.param_groups[0]["lr"])
        print(epoch, optim.param_groups[0]["lr"])

    plt.plot(a[:], b[:])
    plt.savefig("cosine_warmup.png")
