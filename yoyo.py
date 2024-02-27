import logging
import os
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

main_folder = 'experimenttest'

if not os.path.exists(main_folder):
    os.makedirs(main_folder)



log_file1 = os.path.join(main_folder, 'errors.log')
log_file2 = os.path.join(main_folder, 'errors2.log')


l1 = setup_logger('first_logger', log_file1)
l1.info('This is just info message')

l2 = setup_logger('second_logger', log_file2)
l2.info('this is second')
# logging.basicConfig(filename=log_file1, level=logging.INFO)
# logging.basicConfig(filename=log_file1, level=logging.INFO)


