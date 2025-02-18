# from timm.scheduler.cosine_lr import CosineLRScheduler
# from timm.scheduler.step_lr import StepLRScheduler

import jittor as jt
from jittor import nn

class CosineLRScheduler:
    def __init__(self, optimizer, t_initial, lr_min=0, warmup_lr_init=0, warmup_t=0, cycle_mul=1., cycle_limit=1, t_in_epochs=True):
        self.optimizer = optimizer
        self.t_initial = t_initial
        self.lr_min = lr_min
        self.warmup_lr_init = warmup_lr_init
        self.warmup_t = warmup_t
        self.cycle_mul = cycle_mul
        self.cycle_limit = cycle_limit
        self.t_in_epochs = t_in_epochs
        self.last_epoch = 0

    def step(self):
        # Current step in the training process
        current_step = self.last_epoch
        if current_step < self.warmup_t:
            lr = self.warmup_lr_init + (self.lr_min - self.warmup_lr_init) * (current_step / self.warmup_t)
        else:
            cosine_decay = 0.5 * (1 + jt.cos(jt.pi * (current_step - self.warmup_t) / (self.t_initial - self.warmup_t)))
            lr = self.lr_min + (1 - self.lr_min) * cosine_decay

        # Update learning rates for all parameters
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.last_epoch += 1

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class StepLRScheduler:
    def __init__(self, optimizer, decay_t, decay_rate=0.1, warmup_lr_init=0.0, warmup_t=0, t_in_epochs=True):
        self.optimizer = optimizer
        self.decay_t = decay_t
        self.decay_rate = decay_rate
        self.warmup_lr_init = warmup_lr_init
        self.warmup_t = warmup_t
        self.t_in_epochs = t_in_epochs
        self.last_epoch = 0

    def step(self):
        # Get the current step in the training
        current_step = self.last_epoch
        if current_step < self.warmup_t:
            lr = self.warmup_lr_init + (self.get_initial_lr() - self.warmup_lr_init) * (current_step / self.warmup_t)
        else:
            # Apply the step decay
            if current_step % self.decay_t == 0:
                lr = self.get_initial_lr() * (self.decay_rate ** (current_step // self.decay_t))
            else:
                lr = self.get_initial_lr()

        # Update learning rates for all parameters
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.last_epoch += 1

    def get_initial_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


def build_scheduler(config, optimizer, n_iter_per_epoch=1):
    if config['lr_scheduler']['t_in_epochs']:
        n_iter_per_epoch = 1
    num_steps = int(config['epochs'] * n_iter_per_epoch)
    warmup_steps = int(config['warmup_epochs'] * n_iter_per_epoch)
    lr_scheduler = None
    if config['lr_scheduler']['type'] == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            cycle_mul=1.,
            lr_min=config['min_lr'],
            warmup_lr_init=config.get('warmup_lr', 0.0),
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=config['lr_scheduler']['t_in_epochs'],
        )
    elif config['lr_scheduler']['type'] == 'step':
        decay_steps = int(config['lr_scheduler']['decay_epochs'] * n_iter_per_epoch)
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=config['lr_scheduler']['decay_rate'],
            warmup_lr_init=config.get('warmup_lr', 0.0),
            warmup_t=warmup_steps,
            t_in_epochs=config['lr_scheduler']['t_in_epochs'],
        )
    else:
        raise NotImplementedError()

    return lr_scheduler
