#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module for functions related to identifying the computation devices.
"""

import logging

import theano

def get_default_device(requested):
    """Enumerates the devices that have been configured for Theano and returns
    the name of the device that will be used by default.

    Devices and contexts are rather complicated in Theano.
    1) If no GPU is configured, the only device that can be used is cpu and any
       other requests will raise an exception.
    2) If a single GPU is enabled using ``device=cuda0``, the GPU device is
       ``None``, and only ``None`` or cpu can be selected.
    3) If multiple GPUs are enabled using ``contexts=dev0->cuda0;dev1->cuda1``
       and the default GPU is defined using ``device=cuda0``, then the default
       device will be ``None`` and dev0 does not exist. Either ``None``, dev1,
       or cpu can be selected.
    4) If multiple GPUs are enabled and the default GPU is not defined, then the
       ``None`` device does not exist. If ``None`` is requested, the first GPU
       device is returned.

    :type requested: str
    :param requested: ``None`` to automatically selected a default device; any
                      other selection either returns the same value or raises an
                      exception

    :rtype: str
    :returns: 'cpu' if no GPUs are enabled, otherwise ``None`` or the name of
              the first GPU device
    """

    try:
        names = []
        for name in theano.gpuarray.type.list_contexts():
            context = theano.gpuarray.type.get_context(name)
            logging.info('Context {} device="{}" ID="{}"'.format(
                         name, context.devname, context.unique_id))
            names.append(name)
        if requested is None:
            if any(name is None for name in names):
                return None
            elif len(names) > 0:
                logging.info('The first GPU device ("{}") will be used as the '
                             'default device.'.format(names[0]))
                return names[0]
            else:
                logging.info('Theano is not using a GPU or an old version of '
                             'libgpuarray is installed.')
                return None
        elif requested in names:
            logging.info('"{}" selected as the default device.'
                         .format(requested))
            return requested
        else:
            raise ValueError('Theano is not configured to use device "{}".'
                             .format(requested))
    except AttributeError:
        # No GPU support or we couldn't list the contexts because an old version
        # of libgpuarray is installed.
        if requested is None:
            logging.info('Theano is not using a GPU or an old version of '
                         'libgpuarray is installed.')
            return None
        elif requested == 'cpu':
            logging.info('Using CPU for computation.')
            return requested
        else:
            raise ValueError('Theano is not configured to use device "{}". '
                             'Only cpu is supported.'.format(requested))

def log_free_mem():
    """Writes the available GPU memory to the debug log.
    """

    for name in theano.gpuarray.type.list_contexts():
        context = theano.gpuarray.type.get_context(name)
        free_mbytes = context.free_gmem / (1024 * 1024)
        logging.debug("Available memory on GPU %s: %.0f MB", name, free_mbytes)
