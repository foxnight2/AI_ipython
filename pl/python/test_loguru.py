

from logging import log
from loguru import logger


logfile = logger.add('./log.txt', )
logger.info('information-0')
logger.warning('warning')

logger.remove(logfile)
logger.info('information-1')

