import io
from io import StringIO, BytesIO

# open



import threading
MUTEX = threading.Lock()
import fcntl

import pickle

@staticmethod
def dump(obj, path):
    # MUTEX.acquire()
    try:
        with open(path, 'wb') as f:
            fcntl.lockf(f.fileno(), fcntl.LOCK_EX)
            pickle.dump(obj, f)
            fcntl.lockf(f.fileno(), fcntl.LOCK_UN)
    except:
        pass
    finally:
        # MUTEX.release()
        pass

