import os
import logging

logger = logging.getLogger("main." + __name__.split(".")[-1])

def create_directories_for_file(file_path):
    # Extract the directory portion of the file path
    directory = os.path.dirname(file_path)

    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Directories created: {directory}")
    else:
        logger.debug("No directories to create; the necessary directories exist already.")