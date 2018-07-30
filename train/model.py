"""
Author: Yunpeng Chen
"""
import os
import time
import socket
import logging

import torch

from . import metric
from . import callback

"""
Static Model
"""
class static_model(object):

    def __init__(self,
                 net,
                 criterion=None,
                 model_prefix='',
                 **kwargs):
        if kwargs:
            logging.warning("Unknown kwargs: {}".format(kwargs))

        # init params
        self.net = net
        self.model_prefix = model_prefix
        self.criterion = criterion

    def load_state(self, state_dict, strict=False):
        if strict:
            self.net.load_state_dict(state_dict=state_dict)
        else:
            # customized partialy load function
            net_state_keys = list(self.net.state_dict().keys())
            for name, param in state_dict.items():
                if name in self.net.state_dict().keys():
                    dst_param_shape = self.net.state_dict()[name].shape
                    if param.shape == dst_param_shape:
                        self.net.state_dict()[name].copy_(param.view(dst_param_shape))
                        net_state_keys.remove(name)
            # indicating missed keys
            if net_state_keys:
                logging.warning(">> Failed to load: {}".format(net_state_keys))
                return False
        return True

    def get_checkpoint_path(self, epoch):
        assert self.model_prefix, "model_prefix undefined!"
        if torch.distributed._initialized:
            hostname = socket.gethostname()
            checkpoint_path = "{}_at-{}_ep-{:04d}.pth".format(self.model_prefix, hostname, epoch)
        else:
            checkpoint_path = "{}_ep-{:04d}.pth".format(self.model_prefix, epoch)
        return checkpoint_path

    def load_checkpoint(self, epoch, optimizer=None):

        load_path = self.get_checkpoint_path(epoch)
        assert os.path.exists(load_path), "Failed to load: {} (file not exist)".format(load_path)

        checkpoint = torch.load(load_path)

        all_params_matched = self.load_state(checkpoint['state_dict'], strict=False)

        if optimizer:
            if 'optimizer' in checkpoint.keys() and all_params_matched:
                optimizer.load_state_dict(checkpoint['optimizer'])
                logging.info("Model & Optimizer states are resumed from: `{}'".format(load_path))
            else:
                logging.warning(">> Failed to load optimizer state from: `{}'".format(load_path))
        else:
            logging.info("Only model state resumed from: `{}'".format(load_path))

        if 'epoch' in checkpoint.keys():
            if checkpoint['epoch'] != epoch:
                logging.warning(">> Epoch information inconsistant: {} vs {}".format(checkpoint['epoch'], epoch))

    def save_checkpoint(self, epoch, optimizer_state=None):

        save_path = self.get_checkpoint_path(epoch)
        save_folder = os.path.dirname(save_path)

        if not os.path.exists(save_folder):
            logging.debug("mkdir {}".format(save_folder))
            os.makedirs(save_folder)

        if not optimizer_state:
            torch.save({'epoch': epoch,
                        'state_dict': self.net.state_dict()},
                        save_path)
            logging.info("Checkpoint (only model) saved to: {}".format(save_path))
        else:
            torch.save({'epoch': epoch,
                        'state_dict': self.net.state_dict(),
                        'optimizer': optimizer_state},
                        save_path)
            logging.info("Checkpoint (model & optimizer) saved to: {}".format(save_path))


    def forward(self, data, target):
        """ typical forward function with:
            single output and single loss
        """
        # data = data.float().cuda(async=True)
        # target = target.cuda(async=True)
        data = data.float().cuda()
        target = target.cuda()
        if self.net.training:
            input_var = torch.autograd.Variable(data, requires_grad=False)
            target_var = torch.autograd.Variable(target, requires_grad=False)
        else:
            input_var = torch.autograd.Variable(data, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

        output = self.net(input_var)
        if hasattr(self, 'criterion') and self.criterion is not None \
            and target is not None:
            loss = self.criterion(output, target_var)
        else:
            loss = None
        return [output], [loss]


"""
Dynamic model that is able to update itself
"""
class model(static_model):

    def __init__(self,
                 net,
                 criterion,
                 model_prefix='',
                 step_callback=None,
                 step_callback_freq=50,
                 epoch_callback=None,
                 save_checkpoint_freq=1,
                 opt_batch_size=None,
                 **kwargs):

        # load parameters
        if kwargs:
            logging.warning("Unknown kwargs: {}".format(kwargs))

        super(model, self).__init__(net, criterion=criterion,
                                         model_prefix=model_prefix)

        # load optional arguments
        # - callbacks
        self.callback_kwargs = {'epoch': None,
                                'batch': None,
                                'sample_elapse': None,
                                'update_elapse': None,
                                'epoch_elapse': None,
                                'namevals': None,
                                'optimizer_dict': None,}

        if not step_callback:
            step_callback = callback.CallbackList(callback.SpeedMonitor(),
                                                  callback.MetricPrinter())
        if not epoch_callback:
            epoch_callback = (lambda **kwargs: None)

        self.step_callback = step_callback
        self.step_callback_freq = step_callback_freq
        self.epoch_callback = epoch_callback
        self.save_checkpoint_freq = save_checkpoint_freq
        self.batch_size=opt_batch_size


    """
    In order to customize the callback function,
    you will have to overwrite the functions below
    """
    def step_end_callback(self):
        # logging.debug("Step {} finished!".format(self.i_step))
        self.step_callback(**(self.callback_kwargs))

    def epoch_end_callback(self):
        self.epoch_callback(**(self.callback_kwargs))
        if self.callback_kwargs['epoch_elapse'] is not None:
            logging.info("Epoch [{:d}]   time cost: {:.2f} sec ({:.2f} h)".format(
                    self.callback_kwargs['epoch'],
                    self.callback_kwargs['epoch_elapse'],
                    self.callback_kwargs['epoch_elapse']/3600.))
        if self.callback_kwargs['epoch'] == 0 \
           or ((self.callback_kwargs['epoch']+1) % self.save_checkpoint_freq) == 0:
            self.save_checkpoint(epoch=self.callback_kwargs['epoch']+1,
                                 optimizer_state=self.callback_kwargs['optimizer_dict'])

    """
    Learning rate
    """
    def adjust_learning_rate(self, lr, optimizer):
        for param_group in optimizer.param_groups:
            if 'lr_mult' in param_group:
                lr_mult = param_group['lr_mult']
            else:
                lr_mult = 1.0
            param_group['lr'] = lr * lr_mult

    """
    Optimization
    """
    def fit(self, train_iter, optimizer, lr_scheduler,
            eval_iter=None,
            metrics=metric.Accuracy(topk=1),
            epoch_start=0,
            epoch_end=10000,
            **kwargs):

        """
        checking
        """
        if kwargs:
            logging.warning("Unknown kwargs: {}".format(kwargs))

        assert torch.cuda.is_available(), "only support GPU version"

        """
        start the main loop
        """
        pause_sec = 0.
        for i_epoch in range(epoch_start, epoch_end):
            self.callback_kwargs['epoch'] = i_epoch
            epoch_start_time = time.time()

            ###########
            # 1] TRAINING
            ###########
            metrics.reset()
            self.net.train()
            sum_sample_inst = 0
            sum_sample_elapse = 0.
            sum_update_elapse = 0
            batch_start_time = time.time()
            logging.info("Start epoch {:d}:".format(i_epoch))
            for i_batch, (data, target) in enumerate(train_iter):
                self.callback_kwargs['batch'] = i_batch

                update_start_time = time.time()

                # [forward] making next step
                outputs, losses = self.forward(data, target)

                # [backward]
                optimizer.zero_grad()
                for loss in losses: loss.backward()
                self.adjust_learning_rate(optimizer=optimizer,
                                          lr=lr_scheduler.update())
                optimizer.step()

                # [evaluation] update train metric
                metrics.update([output.data.cpu() for output in outputs],
                               target.cpu(),
                               [loss.data.cpu() for loss in losses])

                # timing each batch
                sum_sample_elapse += time.time() - batch_start_time
                sum_update_elapse += time.time() - update_start_time
                batch_start_time = time.time()
                sum_sample_inst += data.shape[0]

                if (i_batch % self.step_callback_freq) == 0:
                    # retrive eval results and reset metic
                    self.callback_kwargs['namevals'] = metrics.get_name_value()
                    metrics.reset()
                    # speed monitor
                    self.callback_kwargs['sample_elapse'] = sum_sample_elapse / sum_sample_inst
                    self.callback_kwargs['update_elapse'] = sum_update_elapse / sum_sample_inst
                    sum_update_elapse = 0
                    sum_sample_elapse = 0
                    sum_sample_inst = 0
                    # callbacks
                    self.step_end_callback()

            ###########
            # 2] END OF EPOCH
            ###########
            self.callback_kwargs['epoch_elapse'] = time.time() - epoch_start_time
            self.callback_kwargs['optimizer_dict'] = optimizer.state_dict()
            self.epoch_end_callback()

            ###########
            # 3] Evaluation
            ###########
            if (eval_iter is not None) \
                and ((i_epoch+1) % max(1, int(self.save_checkpoint_freq/2))) == 0:
                logging.info("Start evaluating epoch {:d}:".format(i_epoch))

                metrics.reset()
                self.net.eval()
                sum_sample_elapse = 0.
                sum_sample_inst = 0
                sum_forward_elapse = 0.
                batch_start_time = time.time()
                for i_batch, (data, target) in enumerate(eval_iter):
                    self.callback_kwargs['batch'] = i_batch

                    forward_start_time = time.time()

                    outputs, losses = self.forward(data, target)

                    metrics.update([output.data.cpu() for output in outputs],
                                    target.cpu(),
                                   [loss.data.cpu() for loss in losses])

                    sum_forward_elapse += time.time() - forward_start_time
                    sum_sample_elapse += time.time() - batch_start_time
                    batch_start_time = time.time()
                    sum_sample_inst += data.shape[0]

                # evaluation callbacks
                self.callback_kwargs['sample_elapse'] = sum_sample_elapse / sum_sample_inst
                self.callback_kwargs['update_elapse'] = sum_forward_elapse / sum_sample_inst
                self.callback_kwargs['namevals'] = metrics.get_name_value()
                self.step_end_callback()

        logging.info("Optimization done!")
