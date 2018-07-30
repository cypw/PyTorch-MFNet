import os
import csv
import logging

import shutil

def move(src, dest):
    shutil.move(src, dest)
    logging.info("move '{}' -> '{}'".format(src, dest))

def rename_folder_name(csv_file, root_folder):
    assert os.path.exists(csv_file)

    f = open(csv_file, 'rt')
    reader = csv.reader(f, delimiter=',')
    for i, row in enumerate(reader):
        if i < 1:
            continue        
        category = row[0].replace(' ', '_')        
        dir_old = os.path.join(root_folder, category.replace('_', ' '))
        dir_new = os.path.join(root_folder, category)
        if dir_old != dir_new and os.path.exists(dir_old):
            move(dir_old, dir_new)
    f.close()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    root_folders = ['../raw/data/train',
                   '../raw/data/val']
    src_csv_list = ['../raw/list/kinetics_train.csv',
                    '../raw/list/kinetics_val.csv',]
                    
    # check
    for csv_file in src_csv_list:
        assert os.path.exists(csv_file), \
            "falied to locate '{}'".format(src_csv_list)

    # rename folder name
    for (csv_file, root_folder) in zip(src_csv_list, root_folders):
        rename_folder_name(csv_file=csv_file, root_folder=root_folder)
        
    # finished
    logging.info("- Done.")
