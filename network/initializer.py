import json
import logging

import numpy as np
import torch


def xavier(net):
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 and hasattr(m, 'weight'):
            torch.nn.init.xavier_uniform(m.weight.data, gain=1.)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform(m.weight.data, gain=1.)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname in ['Sequential', 'AvgPool3d', 'MaxPool3d', \
                           'Dropout', 'ReLU', 'Softmax', 'BnActConv3d'] \
             or 'Block' in classname:
            pass
        else:
            if classname != classname.upper():
                logging.warning("Initializer:: '{}' is uninitialized.".format(classname))
    net.apply(weights_init)



def init_from_dict(net, state_dict, strict=False):
    logging.debug("Initializer:: loading from `state_dic', strict = {} ...".format(strict))

    if strict:
        net.load_state_dict(state_dict=state_dict)
    else:
        # customized partialy load function
        net_state_keys = list(net.state_dict().keys())
        for name, param in state_dict.items():
            if name in net_state_keys:
                dst_param_shape = net.state_dict()[name].shape
                net.state_dict()[name].copy_(param.view(dst_param_shape))
                net_state_keys.remove(name)

        # indicating missed keys
        if net_state_keys:
            logging.info("Initializer:: failed to load: \n{}".format(
                         json.dumps(net_state_keys, indent=4, sort_keys=True)))


def init_3d_from_2d_dict(net, state_dict, method='inflation'):
    logging.debug("Initializer:: loading from 2D neural network, filling method: `{}' ...".format(method))

    # filling method
    def filling_kernel(src, dshape, method):
        assert method in ['inflation', 'random'], \
            "filling method: {} is unknown!".format(method)
        src_np = src.numpy()

        if method == 'inflation':
            dst = torch.FloatTensor(dshape)
            # normalize
            src = src/float(dshape[2])
            src = src.view(dshape[0],dshape[1], 1, dshape[3],dshape[4])
            dst.copy_(src, broadcast=True)
        elif method == 'random':
            dst = torch.FloatTensor(dshape)
            tmp = torch.FloatTensor(src.shape)
            # normalize
            src = src/float(dshape[2])
            # random range
            scale = src.abs().mean()
            # filling
            dst[:,:,0,:,:].copy_(src)
            i = 1
            while i < dshape[2]:
                if i+2 < dshape[2]:
                    torch.nn.init.uniform(tmp, a=-scale, b=scale)
                    dst[:,:,i,:,:].copy_(tmp)
                    dst[:,:,i+1,:,:].copy_(src)
                    dst[:,:,i+2,:,:].copy_(-tmp)
                    i += 3
                elif i+1 < dshape[2]:
                    torch.nn.init.uniform(tmp, a=-scale, b=scale)
                    dst[:,:,i,:,:].copy_(tmp)
                    dst[:,:,i+1,:,:].copy_(-tmp)
                    i += 2
                else:
                    dst[:,:,i,:,:].copy_(src)
                    i += 1
            # shuffle
            tmp = dst.numpy().swapaxes(2, -1)
            shp = tmp.shape[:-1]
            for ndx in np.ndindex(shp):
                np.random.shuffle(tmp[ndx])
            dst = torch.from_numpy(tmp)
        else:
            raise NotImplementedError

        return dst


    # customized partialy loading function
    src_state_keys = list(state_dict.keys())
    dst_state_keys = list(net.state_dict().keys())
    for name, param in state_dict.items():
        if name in dst_state_keys:
            src_param_shape = param.shape
            dst_param_shape = net.state_dict()[name].shape
            if src_param_shape != dst_param_shape:
                if name.startswith('classifier'):
                    continue
                assert len(src_param_shape) == 4 and len(dst_param_shape) == 5, "{} mismatch".format(name)
                if list(src_param_shape) == [dst_param_shape[i] for i in [0, 1, 3, 4]]:
                    if dst_param_shape[2] != 1:
                        param = filling_kernel(src=param, dshape=dst_param_shape, method=method)
                    else:
                        param = param.view(dst_param_shape)
                assert dst_param_shape == param.shape, \
                    "Initilizer:: error({}): {} != {}".format(name, dst_param_shape, param.shape)
            net.state_dict()[name].copy_(param, broadcast=False)
            src_state_keys.remove(name)
            dst_state_keys.remove(name)

    # indicat missing / ignored keys
    if src_state_keys:
        out = "[\'" + '\', \''.join(src_state_keys) + "\']"
        logging.info("Initializer:: >> {} params are unused: {}".format(len(src_state_keys),
                     out if len(out) < 300 else out[0:150] + " ... " + out[-150:]))
    if dst_state_keys:
        logging.info("Initializer:: >> failed to load: \n{}".format(
                     json.dumps(dst_state_keys, indent=4, sort_keys=True)))
