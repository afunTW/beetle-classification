"""
Support function for python profilling
"""

import logging
import time
from datetime import datetime
from functools import wraps

LOGGER = logging.getLogger(__name__)

def func_profiling(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        time_spent = datetime.now() - start_time
        fullname = '{}.{}'.format(func.__module__, func.__name__)
        LOGGER.debug('{}[args={}, kwargs={}] completed in {}'.format(
            fullname, args, kwargs, str(time_spent)
        ))
        return result
    return wrapped

@func_profiling
def test_func_profiling():
    import random
    sleep_sec = random.randrange(1,3)
    LOGGER.debug('random sleep in {} sec'.format(sleep_sec))
    time.sleep(sleep_sec)
    LOGGER.debug('Wake up')

if __name__ == '__main__':
    """testing"""
    import sys
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)12s:L%(lineno)3s [%(levelname)8s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout
        )
    test_func_profiling()
