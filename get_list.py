import os
import random

if __name__ == '__main__':
    main_dir = '/home/ashelee/Original_OEM/xBD_trainset/images'
    fn_train = open('file_train.txt', 'w')
    fn_test = open('file_test.txt', 'w')
    for root, dirs, files in os.walk(main_dir):
        for file in files:
            if random.random() < 0.8:
                fn_train.write(file + '\n')
            else:
                fn_test.write(file + '\n')