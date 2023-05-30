import os
import requests
import concurrent.futures as futures

import json
from PIL import Image
from tqdm import tqdm


def download_and_save(url, root='./'):
    '''download and save
    '''
    im = Image.open(requests.get(url, stream=True).raw).convert('RGB')

    path = os.path.join(root, os.path.basename(url))
    if not os.path.exists(path):
        im.save(path)


def generate_urls():
    urls = ['https://static.flickr.com/2723/4385058960_b0f291553e.jpg', ] * 1000
    return urls


if __name__ == '__main__':

    urls = generate_urls()

    with futures.ThreadPoolExecutor(64) as executor:
        infos = list(tqdm(executor.map(download_and_save, urls), total=len(urls)))

