"""
日志工具方法
"""

import logging
import sys


def get_logger(logger_name: str, log_level=logging.INFO):
    """

    :param logger_name:
    :param log_level:
    :return:
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = False

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    return logger
