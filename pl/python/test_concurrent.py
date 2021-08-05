from concurrent import futures



def func(ids):
    return 0


with futures.ThreadPoolExecutor(32) as executor:
    image_infos = executor.map(func, [0, 1, 2])
