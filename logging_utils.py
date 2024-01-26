# -*- coding: utf-8 -*-
"""
Helper file intended to help in setting up the logs
and in logging some informations
"""

import sys, logging, traceback
import torch, transformers



def setup_logging(filename, root_logger, transformers_logger):
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers = []
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
        # logs from transformers module
        transformers_logger.handlers = []
        transformers_handler = logging.FileHandler(filename)
        transformers_handler.setLevel(logging.DEBUG)
        transformers_handler.setFormatter(log_format)
        transformers_logger.addHandler(transformers_handler)
        # log basic info
        logging.info(f"Detailed logs are written to: {filename}")
        log_system_info()
    # log uncaught exceptions
    sys.excepthook = log_exceptions

def log_exceptions(exc_type, exc_value, exc_traceback, logging=logging):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    # calls default excepthook
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


def log_system_info():
    logging.debug('System info')
    logging.debug(f"\tOS detected: {sys.platform}")
    logging.debug(f"\tPython version: {sys.version}")
    logging.debug(f"\tCUDA drivers version: {torch.version.cuda}")




def display_CUDA_info(device):
    logging.debug(f"CUDA available: {torch.cuda.is_available()}")
    logging.debug(f"Devices available: {[torch.cuda.device(i) for i in range(torch.cuda.device_count())]}")
    debug_str = f"Now using device: {device}"
    if device.type == 'cuda':
        debug_str += '\n\t'+torch.cuda.get_device_name(0)
        debug_str += '\n\tMemory Usage:'
        debug_str += f'\n\t\tTotal available: {round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)} GB'
        debug_str += f'\n\t\tAllocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB'
        debug_str += f'\n\t\tCached:    {round(torch.cuda.memory_reserved(0) /1024**3,1)} GB'

    logging.debug(debug_str)
    logging.debug(f'The default location for tensors is: {torch.rand(3).device}')










