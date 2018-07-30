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
parser.add_argument('--dataset', default='UCF101', choices=['UCF101','Kinetics'],
                    help="path to dataset")
parser.add_argument('--clip-length', default=16,
                    help="define the length of each input sample.")    
parser.add_argument('--frame-interval', type=int, default=2,
                    help="define the sampling interval between frames.")    
parser.add_argument('--task-name', type=str, default='../exps/<your_tesk_name>',
                    help="name of current task, leave it empty for using folder name")
parser.add_argument('--model-dir', type=str, default="./",
                    help="set logging file.")
parser.add_argument('--log-file', type=str, default="./eval-ucf101-split1.log",
                    help="set logging file.")
# device
parser.add_argument('--gpus', type=int, default=1,
                    help="define gpu id")
# algorithm
parser.add_argument('--network', type=str, default='mfnet_3d',
                    choices=['mfnet_3d'],
                    help="chose the base network")
# evaluation
parser.add_argument('--load-epoch', type=int, default=0,
                    help="resume trained model")
parser.add_argument('--batch-size', type=int, default=8,
                    help="batch size")


def autofill(args):
    # customized
    if not args.task_name:
        args.task_name = os.path.basename(os.getcwd())
    # fixed
    args.model_prefix = os.path.join(args.model_dir, args.task_name)
    return args

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
    args = autofill(args)

    set_logger(log_file=args.log_file, debug_mode=args.debug_mode)
    logging.info("Start evaluation with args:\n" +
                 json.dumps(vars(args), indent=4, sort_keys=True))

    # set device states
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus) # before using torch
    assert torch.cuda.is_available(), "CUDA is not available"

    # load dataset related configuration
    dataset_cfg = dataset.get_config(name=args.dataset)

    # creat model
    sym_net, input_config = get_symbol(name=args.network, **dataset_cfg)
    
    # network
    if torch.cuda.is_available():
        cudnn.benchmark = True
        sym_net = torch.nn.DataParallel(sym_net).cuda()
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        sym_net = torch.nn.DataParallel(sym_net)
        criterion = torch.nn.CrossEntropyLoss()
    net = static_model(net=sym_net,
                       criterion=criterion,
                       model_prefix=args.model_prefix)
    net.load_checkpoint(epoch=args.load_epoch)
    
    # data iterator:
    data_root = "../dataset/{}".format(args.dataset)
    normalize = transforms.Normalize(mean=input_config['mean'], std=input_config['std'])
    val_sampler = sampler.RandomSampling(num=args.clip_length,
                                         interval=args.frame_interval,
                                         speed=[1.0, 1.0])
    val_loader = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'), # change this part accordingly
                      txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'testlist01.txt'), # change this part accordingly
                      sampler=val_sampler,
                      force_color=True,
                      video_transform=transforms.Compose([
                                         transforms.Resize((256,256)),
                                         transforms.RandomCrop((224,224)),
                                         # transforms.CenterCrop((224, 224)), # we did not use center crop in our paper
                                         # transforms.RandomHorizontalFlip(), # we did not use mirror in our paper
                                         transforms.ToTensor(),
                                         normalize,
                                      ]),
                      name='test',
                      return_item_subpath=True,
                      )
                      
    eval_iter = torch.utils.data.DataLoader(val_loader,
                      batch_size=args.batch_size,
                      shuffle=True,
                      num_workers=4, # change this part accordingly
                      pin_memory=True)

    # eval metrics
    metrics = metric.MetricList(metric.Loss(name="loss-ce"),
                                metric.Accuracy(topk=1, name="top1"),
                                metric.Accuracy(topk=5, name="top5"))
    metrics.reset()

    # main loop
    net.net.eval()
    avg_score = {}
    sum_batch_elapse = 0.
    sum_batch_inst = 0
    duplication = 1
    softmax = torch.nn.Softmax(dim=1)

    total_round = 999999999 # change this part accordingly if you do not want an inf loop
    for i_round in range(total_round):
        i_batch = 0
        logging.info("round #{}/{}".format(i_round, total_round))
        for data, target, video_subpath in eval_iter:
            batch_start_time = time.time()

            outputs, losses = net.forward(data, target)

            sum_batch_elapse += time.time() - batch_start_time
            sum_batch_inst += 1

            # recording
            output = softmax(outputs[0]).data.cpu()
            target = target.cpu()
            losses = losses[0].data.cpu()
            for i_item in range(0, output.shape[0]):
                output_i = output[i_item,:].view(1, -1)
                target_i = torch.LongTensor([target[i_item]])
                loss_i = losses
                video_subpath_i = video_subpath[i_item]
                if video_subpath_i in avg_score:
                    avg_score[video_subpath_i][2] += output_i
                    avg_score[video_subpath_i][3] += 1
                    duplication = 0.92 * duplication + 0.08 * avg_score[video_subpath_i][3]
                else:
                    avg_score[video_subpath_i] = [torch.LongTensor(target_i.numpy().copy()), 
                                                  torch.FloatTensor(loss_i.numpy().copy()), 
                                                  torch.FloatTensor(output_i.numpy().copy()),
                                                  1] # the last one is counter

            # show progress
            if (i_batch % 100) == 0:
                metrics.reset()
                for _, video_info in avg_score.items():
                    target, loss, pred, _ = video_info
                    metrics.update([pred], target, [loss])
                name_value = metrics.get_name_value()
                logging.info("{:.1f}%, {:.1f} \t| Batch [0,{}]    \tAvg: {} = {:.5f}, {} = {:.5f}, {} = {:.5f}".format(
                            float(100*i_batch) / eval_iter.__len__(), \
                            duplication, \
                            i_batch, \
                            name_value[0][0][0], name_value[0][0][1], \
                            name_value[1][0][0], name_value[1][0][1], \
                            name_value[2][0][0], name_value[2][0][1]))
            i_batch += 1


    # finished
    logging.info("Evaluation Finished!")

    metrics.reset()
    for _, video_info in avg_score.items():
        target, loss, pred, _ = video_info
        metrics.update([pred], target, [loss])

    logging.info("Total time cost: {:.1f} sec".format(sum_batch_elapse))
    logging.info("Speed: {:.4f} samples/sec".format(
            args.batch_size * sum_batch_inst / sum_batch_elapse ))
    logging.info("Accuracy:")
    logging.info(json.dumps(metrics.get_name_value(), indent=4, sort_keys=True))
