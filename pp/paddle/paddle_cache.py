
from PIL import Image
import numpy as np 

import os
import glob
import pickle

import argparse
from functools import partial
import concurrent.futures as futures


def decode(path, root):
    im = Image.open(path)
    im = np.asarray(im)
    
    _path = os.path.join(root, os.path.basename(path) + '.pkl')

    with open(_path, 'wb') as f:
        pickle.dump(im, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_root', '-s', type=str)
    parser.add_argument('--cache_root', '-c', type=str)
    parser.add_argument('--max_workers', '-w', type=int, default=32, )
    parser.add_argument('--max_depth', '-d', type=int, default=1, )
    parser.add_argument('--ext', '-e', type=str, nargs='+', default='jpg')
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.source_root, '/'.join(['*', ] * args.max_depth) + '.jpg'))

    with futures.ThreadPoolExecutor(args.max_workers) as executor:
        image_infos = executor.map(decode, files)

    print(args)






from PIL import Image                                                                                               
import numpy as np                                                                                                                                                                                                               
import os                                                                                                           
import glob                                                                                                         
import pickle                                                                                                       
import concurrent.futures as futures   

                                                                             
def decode(path, root='./buffer/'):                                                                                 
    im = Image.open(path)                                                                                           
    im = np.array(im)                                                                                               
    # print(path, im.shape)                                                                                         
    assert len(im.shape) == 3, ''                                                                                   

    _path = os.path.join(root, os.path.basename(path) + '.pkl')                                                     
                                                                                                                    
    with open(_path, 'wb') as f:                                                                                    
        pickle.dump(im, f)                                                                                          
                                                                                                                    
                                                                                                                    
files = glob.glob('./dataset/coco/*/*.jpg')                                                                         
print(len(files))                                                                                                   
print(files[0])                                                                                                     
                                                                                                                    
with futures.ThreadPoolExecutor(32) as executor:                                                                    
    image_infos = executor.map(decode, files)                                                                       
                                                