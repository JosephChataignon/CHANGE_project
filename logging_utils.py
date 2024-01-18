# -*- coding: utf-8 -*-
"""
Helper file intended to help in setting up the logs
and in logging some informations
"""

import logging
import torch



def setup_logging(filename):
    root_logger = logging.getLogger()
    log_format = logging.Formatter('%(asctime)s %(levelname)s : %(message)s')
    try:
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
    except:
        # if __file__ fails in the try block
        print("Assuming this is a live session, logging only to console")
        root_logger.setLevel(logging.DEBUG)


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










