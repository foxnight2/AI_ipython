

from logging import log
from loguru import logger

# logfile = logger.add('log_{time:YYYY-MM-DD HH:mm:ss}.txt', format='{time}, {level}, {message}')
logfile = logger.add('log_{time}.txt', format='{time}, {level}, {message}')


logger.info('information-0')
logger.warning('warning')

logger.remove(logfile)
logger.info('information-1')

