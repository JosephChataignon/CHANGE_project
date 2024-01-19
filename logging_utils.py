# -*- coding: utf-8 -*-
"""
Helper file intended to help in setting up the logs
and in logging some informations
"""

import sys, logging, traceback
import torch



def setup_logging(filename, root_logger):
    root_logger.setLevel(logging.DEBUG)
    log_format = logging.Formatter('%(asctime)s %(levelname)s : %(message)s')
    # if interactive session
    if hasattr(sys, 'ps1'):
        print("Assuming this is a live session, logging only to console")
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(log_format)
        root_logger.addHandler(handler)
    else:
        # detailed logs go into a file
        detailed_log_handler = logging.FileHandler(filename)
        detailed_log_handler.setLevel(logging.DEBUG)
        detailed_log_handler.setFormatter(log_format)
        root_logger.addHandler(detailed_log_handler)
        # normal logs go to the output stream
        out_handler = logging.StreamHandler(sys.stdout)
        out_handler.setLevel(logging.INFO)
        out_handler.setFormatter(log_format)
        root_logger.addHandler(out_handler)
        logging.info(f"Detailed logs are written to: {filename}")
    # log uncaught exceptions
    sys.excepthook = log_exceptions

def log_exceptions(type, value, tb):
    # for line in traceback.TracebackException(type, value, tb).format(chain=True):
    #     logging.exception(line)
    exception_traceback = traceback.TracebackException(type, value, tb).format(chain=True)
    logging.exception(exception_traceback)
    logging.exception(value)
    # calls default excepthook
    sys.__excepthook__(type, value, tb)




def display_CUDA_info(device):
    logging.debug(f"CUDA available: {torch.cuda.is_available()}")
    logging.debug(f"Devices available: {[torch.cuda.device(i) for i in range(torch.cuda.device_count())]}")
    debug_str = f"Now using device: {device}"
    if device.type == 'cuda':
        debug_str += '\n\t'+torch.cuda.get_device_name(0)
        debug_str += '\n\tMemory Usage:'
        debug_str += f'\n\t\tAllocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB'
        debug_str += f'\n\t\tCached:    {round(torch.cuda.memory_reserved(0) /1024**3,1)} GB'

    logging.debug(debug_str)
    logging.debug(f'\tDefault location for tensors: {torch.rand(3).device}')










