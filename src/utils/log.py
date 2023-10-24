import logging


def setup_logger(stdout_level=logging.INFO):
    # Create logger
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Create console handler
    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(stdout_level)
    stdout_handler.setFormatter(formatter)

    # Add stdout handler to logger
    logger.addHandler(stdout_handler)

    return logger
