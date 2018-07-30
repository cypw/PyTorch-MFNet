import sys
sys.path.append("..")

import os
import time
import json
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn

import dataset
from train.model import static_model
from train import metric
from data import video_sampler as sampler
from data import video_transforms as transforms
from data.video_iterator import VideoIter
from network.symbol_builder import get_symbol


parser = argparse.ArgumentParser(description="PyTorch Video Recognition Parser (Evaluation)")
# debug
parser.add_argument('--debug-mode', type=bool, default=True,
                    help="print all setting for debugging.")
# io
parser.add_argument('--clip-length', default=16,
                    help="define the length of each input sample.")
parser.add_argument('--log-file', type=str, default="./eval-speed.log",
                    help="set logging file.")
# device
parser.add_argument('--gpus', type=int, default=0,
                    help="define gpu id")
# algorithm
parser.add_argument('--network', type=str, default='MFNet_3D',
                    choices=['MFNet_3D'],
                    help="chose the base network")
# evaluation
parser.add_argument('--batch-size', type=int, default=128,
                    help="batch size")

def set_logger(log_file='', debug_mode=False):
    if log_file:
        if not os.path.exists("./"+os.path.dirname(log_file)):
            os.makedirs("./"+os.path.dirname(log_file))
        handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
    else:
        handlers = [logging.StreamHandler()]

    """ add '%(filename)s' to format show source file """
    logging.basicConfig(level=logging.DEBUG if debug_mode else logging.INFO,
                format='%(asctime)s %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                handlers = handlers)


if __name__ == '__main__':

    # set args
    args = parser.parse_args()

    set_logger(log_file=args.log_file, debug_mode=args.debug_mode)
    logging.info("Cudnn Version: {}".format(cudnn.version()))
    cudnn.benchmark = True
    logging.info("Start evaluation with args:\n" +
                 json.dumps(vars(args), indent=4, sort_keys=True))

    # set device states
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus) # before using torch
    assert torch.cuda.is_available(), "CUDA is not available"

    # creat model
    sym_net, input_config = get_symbol(name=args.network, num_classes=101)
    
    # network
    if torch.cuda.is_available():
        sym_net = sym_net.cuda()
    net = static_model(net=sym_net)

    # main loop
    with torch.no_grad():
        net.net.eval()
        sum_batch_elapse = 0.
        sum_batch_inst = 0
        if args.network == 'RESNET18_C3D':
            data = torch.autograd.Variable(torch.randn(args.batch_size,3,args.clip_length,112,112).float().cuda(), \
                                            requires_grad=False)
        else:
            data = torch.autograd.Variable(torch.randn(args.batch_size,3,args.clip_length,224,224).float().cuda(), \
                                            requires_grad=False)

        outputs = net.net(data) # ignore the first forward

        total_round = 15
        for i_round in range(total_round):

            batch_start_time = time.time()

            outputs = net.net(data)

            sum_batch_elapse += time.time() - batch_start_time
            sum_batch_inst += 1

            if i_round%2 == 0:
                logging.info("{}/{}: \t{:.2f} clips/sec ({:.2f}ms/clips), \t{:.2f} frames/sec ({:.2f}ms/frame)".format(i_round, \
                                                                             total_round, \
                                                                             sum_batch_inst*args.batch_size/float(sum_batch_elapse),
                                                                             1000./(sum_batch_inst*args.batch_size/float(sum_batch_elapse)),
                                                                             sum_batch_inst*args.batch_size*args.clip_length/float(sum_batch_elapse),
                                                                             1000./(sum_batch_inst*args.batch_size*args.clip_length/float(sum_batch_elapse)),
                                                                             ))
                sum_batch_inst = 0
                sum_batch_elapse = 0


    # finished
    logging.info("Evaluation Finished!")
