import torch


class MultiLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lambda_factories, last_epoch=-1, verbose=False):
        """
        MultiLR is a learning rate scheduler that allows for multiple learning rates to be scheduled simultaneously.
        """
        self.schedulers = []
        values = self._get_optimizer_lr(optimizer)
        for idx, factory in enumerate(lambda_factories):
            self.schedulers.append(factory(optimizer))
            values[idx] = self._get_optimizer_lr(optimizer)[idx]
            self._set_optimizer_lr(optimizer, values)
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """
        Returns the current learning rates of the scheduler.
        """
        result = []
        for idx, sched in enumerate(self.schedulers):
            result.append(sched.get_last_lr()[idx])
        return result

    @staticmethod
    def _set_optimizer_lr(optimizer, values):
        """
        Sets the learning rates of the optimizer to the given values.
        """
        for param_group, lr in zip(optimizer.param_groups, values):
            param_group["lr"] = lr

    @staticmethod
    def _get_optimizer_lr(optimizer):
        """
        Returns the learning rates of the optimizer.
        """
        return [group["lr"] for group in optimizer.param_groups]

    def step(self, epoch=None):
        """
        Steps each of the schedulers and sets the learning rates of the optimizer to the new values.
        """
        if self.last_epoch != -1:
            values = self._get_optimizer_lr(self.optimizer)
            for idx, sched in enumerate(self.schedulers):
                sched.step()
                values[idx] = self._get_optimizer_lr(self.optimizer)[idx]
                self._set_optimizer_lr(self.optimizer, values)
        super().step()
