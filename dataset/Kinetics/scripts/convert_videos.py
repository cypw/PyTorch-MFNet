import os
import csv
import logging
import subprocess

from joblib import delayed
from joblib import Parallel

def decode_csv_list(csv_file, root_folder='', trim_format='%06d'):
    assert os.path.exists(csv_file)
    info = []
    f = open(csv_file, 'rt')
    reader = csv.reader(f, delimiter=',')
    for i, row in enumerate(reader):
        if i < 1:
            continue        
        category = row[0].replace(' ', '_')
        # item
        info_i = [category,         # label
                  row[1],           # youtube_id
                  int(row[2]),      # time_start
                  int(row[3]),      # time_end
                  row[4],           # split
                  bool(int(row[5]))]# is_cc

        dir_old = os.path.join(root_folder, category.replace('_', ' '))
        dir_new = os.path.join(root_folder, category)
        if dir_old != dir_new and os.path.exists(dir_old):
            move(dir_old, dir_new)
        basename = '%s_%s_%s.mp4' % (info_i[1],
                                     trim_format % info_i[2],
                                     trim_format % info_i[3])
        video_path = os.path.join(root_folder, category, basename)
        if not os.path.exists(video_path):
            continue

        info.append(info_i)
    f.close()

    logging.info("- found {} lines records on '{}'".format(
                 len(info), csv_file))

    return info 

def exe_cmd(cmd):
    try:
        dst_file = cmd.split()[-1]
        if os.path.exists(dst_file):
            return "exist"
        cmd = cmd.replace('(', '\(').replace(')', '\)').replace('\'', '\\\'')
        output = subprocess.check_output(cmd, shell=True, 
                                        stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        logging.warning("failed: {}".format(cmd))
        # logging.warning("failed: {}: {}".format(cmd, err.output.decode("utf-8"))) # detailed error
        return False
    return output

def convert_video_wapper(video_info, 
                         src_root, 
                         dst_root, 
                         cmd_format,
                         in_parallel=True,
                         trim_format='%06d'):
    commands = []
    for vid_info in video_info:
        category = vid_info[0]
        basename = '%s_%s_%s' % (vid_info[1],
                   trim_format % vid_info[2],
                   trim_format % vid_info[3])
        input_video_path = os.path.join(src_root, category, basename+'.mp4')
        output_video_dir = os.path.join(dst_root, category)
        if not os.path.exists(output_video_dir):
            os.makedirs(output_video_dir)
        output_video_prefix = os.path.join(output_video_dir, basename)
        cmd = cmd_format.format(input_video_path, output_video_prefix)
        commands.append(cmd)

    logging.info("- {} commonds to excute".format(len(commands)))

    if not in_parallel:
        for i, cmd in enumerate(commands):
            if i % 100 == 0:
                logging.info("{} / {}: '{}'".format(i, len(commands), cmd))
            exe_cmd(cmd=cmd)
    else:
        num_jobs = 24
        logging.info("processing videos in parallel, num_jobs={}".format(num_jobs))
        Parallel(n_jobs=num_jobs)(delayed(exe_cmd)(cmd) for cmd in commands)



if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    
    # resize to slen = x256
    cmd_format = 'ffmpeg -y -i {} -c:v mpeg4 -filter:v "scale=min(iw\,(256*iw)/min(iw\,ih)):-1" -b:v 512k -an {}.avi'

    src_roots = ['../raw/data/val',
                 '../raw/data/train', ]
                 # '../raw/data/test', ]

    dst_roots = ['../raw/data/val_avi-x256',
                 '../raw/data/train_avi-x256',]
                 # '../raw/data/test_avi-x256',]

    src_csv_list = ['../raw/list/kinetics_val.csv',
                    '../raw/list/kinetics_train.csv',]
                    # '../raw/list/kinetics_test.csv',]

    for (src_csv, src_root, dst_root) in zip(src_csv_list, src_roots, dst_roots):
        assert os.path.exists(src_csv) and os.path.exists(src_root)
        if not os.path.exists(dst_root):
            os.makedirs(dst_root)
        logging.info("decoding video information from: {}".format(src_csv))
        video_info = decode_csv_list(csv_file=src_csv, root_folder=src_root)
        convert_video_wapper(video_info=video_info, 
                             src_root=src_root, 
                             dst_root=dst_root, 
                             cmd_format=cmd_format)

    logging.info("- Done.")