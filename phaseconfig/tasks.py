import os
import logging
from gi.repository import Ufo


LOG = logging.getLogger(__name__)
pm = Ufo.PluginManager()


def get_task(name, processing_node=None, **kwargs):
    task = pm.get_task(name)
    task.set_properties(**kwargs)
    if processing_node and task.uses_gpu():
        LOG.debug("Assigning task '%s' to node %d", name, processing_node.get_index())
        task.set_proc_node(processing_node)

    return task


def get_file_reader(params, processing_node=None):
    reader = get_task('read', processing_node=processing_node)
    set_node_props(reader, params)
    return reader


def get_projection_reader(params, processing_node=None):
    reader = get_file_reader(params, processing_node)
    setup_read_task(reader, params.projections, params)
    width, height = determine_shape(params, params.projections)
    return reader, width, height


def get_padding(params, pm, width, height, processing_node=None):
    padded_width = next_power_of_two(width + 32)
    padded_height = next_power_of_two(height + 32)
    pad = get_task('pad',
                    width=padded_width,
                    height=padded_height
                    x=(padded_width - width) / 2,
                    y=(padded_height - height) / 2,
                    addressing_mode=params.retrieval_padding_mode,
                    processing_node=processing_node)
    return pad
