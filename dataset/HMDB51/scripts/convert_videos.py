import os
import logging
import subprocess

from joblib import delayed
from joblib import Parallel

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
        # logging.warning("failed: {}: {}".format(cmd, err.output.decode("utf-8"))) # more details
        return False
    return output

def convert_video_wapper(src_videos, 
                         dst_videos, 
                         cmd_format,
                         in_parallel=True):
    commands = []
    for src, dst in zip(src_videos, dst_videos):
        cmd = cmd_format.format(src, dst)
        commands.append(cmd)

    logging.info("- {} commonds to excute".format(len(commands)))

    if not in_parallel:
        for i, cmd in enumerate(commands):
            # if i % 100 == 0:
            #     logging.info("{} / {}: '{}'".format(i, len(commands), cmd))
            exe_cmd(cmd=cmd)
    else:
        num_jobs = 24
        logging.info("processing videos in parallel, num_jobs={}".format(num_jobs))
        Parallel(n_jobs=num_jobs)(delayed(exe_cmd)(cmd) for cmd in commands)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    
    # resize to slen = x360
    cmd_format = 'ffmpeg -y -i {} -c:v mpeg4 -filter:v "scale=min(iw\,(360*iw)/min(iw\,ih)):-1" -b:v 640k -an {}'

    src_root = '../raw/data'
    dst_root = '../raw/data-x360'
    assert os.path.exists(dst_root), "cannot locate `{}'".format(dst_root)

    classname = [name for name in os.listdir(src_root) \
                    if os.path.isdir(os.path.join(src_root,name))]
    classname.sort()

    for cls_name in classname:
        src_folder = os.path.join(src_root, cls_name)
        dst_folder = os.path.join(dst_root, cls_name)
        assert os.path.exists(src_folder), "failed to locate: `{}'.".format(src_folder)
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)

        video_names = [name for name in os.listdir(src_folder) \
                            if os.path.isfile(os.path.join(src_folder, name))]

        src_videos = [os.path.join(src_folder, vid_name.replace(";", "\;").replace("&", "\&")) for vid_name in video_names]
        dst_videos = [os.path.join(dst_folder, vid_name.replace(";", "\;").replace("&", "\&")) for vid_name in video_names]

        convert_video_wapper(src_videos=src_videos, 
                             dst_videos=dst_videos, 
                             cmd_format=cmd_format)
        
    logging.info("- Done.")
