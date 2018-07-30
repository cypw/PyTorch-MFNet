import logging

from .mfnet_3d import MFNET_3D
from .config import get_config

def get_symbol(name, print_net=False, **kwargs):

    if name.upper() == "MFNET_3D":
        net = MFNET_3D(**kwargs)
    else:
        logging.error("network '{}'' not implemented".format(name))
        raise NotImplementedError()

    if print_net:
        logging.debug("Symbol:: Network Architecture:")
        logging.debug(net)

    input_conf = get_config(name, **kwargs)
    return net, input_conf
