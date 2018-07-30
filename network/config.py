import logging

def get_config(name, **kwargs):

    logging.debug("loading network configs of: {}".format(name.upper()))

    config = {}

    if "3D" in name.upper():
        logging.info("Preprocessing:: using MXNet default mean & std.")
        config['mean'] = [124 / 255, 117 / 255, 104 / 255]
        config['std'] = [1 / (.0167 * 255)] * 3
    else:
        config['mean'] = [0.485, 0.456, 0.406]
        config['std'] = [0.229, 0.224, 0.225]
    # else:
    #    raise NotImplemented("Configs for {} not implemented".format(name))

    logging.info("data:: {}".format(config))
    return config