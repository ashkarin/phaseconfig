import os
import logging
import numpy as np
from gi.repository import Ufo
from phaseconfig.tasks import (set_node_props, get_task, get_projection_reader,
                               get_padding, setup_read_task)
from phaseconfig.util import (get_filenames, get_coord_range, get_profile,
                              smooth_1d, swap, read_image, write_image)

import matplotlib
matplotlib.use('Agg')

LOG = logging.getLogger(__name__)

def build_power_spectrum_density_pipeline(params, graph, scheduler, processing_node=None):
    """
    Build a pipeline for computing a power spectrum density. All the settings are
    provided in *params*. *graph* is used for making the connections. Returns the
    power spectrum density.
    """
    if params.projections is None:
        raise RuntimeError("You must specify --projections")

    if params.output is None:
        raise RuntimeError("You must specify --output")

    # Get know how much projections to make a stack of proper size
    n_projections = None
    if params.number:
        n_projections = params.number
    else:
        n_projections = len(get_filenames(params.projections))

    LOG.debug("Number of projections: {}".format(n_projections))

    reader, width, height = get_projection_reader(params, processing_node=processing_node)
    pad = get_padding(params, width, height, processing_node=processing_node)
    fft = get_task('fft', dimensions=2, processing_node=processing_node)
    stack = get_task('stack', number=n_projections, processing_node=processing_node)
    print "STACK DIM: ", n_projections

    # Connect thigs together
    graph.connect_nodes(reader, pad)
    graph.connect_nodes(pad, fft)
    graph.connect_nodes(fft, stack)

    # TODO: Ideally we need introduce a new ufo filter to compute a
    # power spectrum density and return this filter as a result.
    # But we will create a write filter, write a temporary data to the file,
    # read it as numpy array, compute PSD and rewrite the file.

    writer = get_task('write',
                      filename=params.output,
                      processing_node=processing_node)
    graph.connect_nodes(stack, writer)
    scheduler.run(graph)

    # Read the temporary data
    proj_stack = read_image(params.output)
    print "proj_stack.shape: ", proj_stack.shape
    _d, _h, _w = proj_stack.shape
    proj_stack = np.array([proj[:, 0:_w:2] + 1j * proj[:, 1:_w:2] for proj in proj_stack])
    proj_stack = 10.0 * np.log10(np.abs(proj_stack) ** 2)
    psd_img = np.sum(proj_stack, axis=0)

    # UFO supplies a non-shifted fourier spaces
    psd_img = np.fft.fftshift(psd_img)

    # Rewrite the image
    write_image(psd_img, params.output)


def compute_filter(params, power_spectral_density_shape=None):
    energy = params.energy
    prop_distance = params.propagation_distance
    pixel_size = params.pixel_size
    reg_rate = params.regularization_rate
    threshold = params.thresholding_rate
    method = params.retrieval_method

    if energy is None:
        raise RuntimeError("You must specify --energy")

    if prop_distance is None:
        raise RuntimeError("You must specify --propagation-distance")

    if pixel_size is None:
        raise RuntimeError("You must specify --pixel-size")

    if reg_rate is None:
        raise RuntimeError("You must specify --regularization-rate")

    if threshold is None:
        raise RuntimeError("You must specify --threshold")

    psd_path = params.power_spectral_density
    if not power_spectral_density_shape:
        if (psd_path is None) or not os.path.isfile(psd_path):
            raise RuntimeError("You must specify --power-spectral-density")

    known_methods = ['tie', 'qp', 'qphalfsine', 'qp2']
    if method not in known_methods:
        raise RuntimeError("You must specify --retrieval-method=['tie', 'qp', 'qphalfsine', 'qp2']")

    # Read PSD image to grab it's dimensions
    if not power_spectral_density_shape:
        psd_img = read_image(psd_path)
        power_spectral_density_shape = psd_img.shape

    Lambda = 6.62606896e-34 * 299792458 / (energy * 1.60217733e-16)
    prefac = 2 * np.pi * Lambda * prop_distance / (pixel_size ** 2)

    # Make reversed coord. gird for the filter and normalize it
    height, width = power_spectral_density_shape
    X = get_coord_range(width, inverse=False)
    Y = get_coord_range(height, inverse=False)
    X = swap(X)
    Y = swap(Y)

    gx, gy = np.meshgrid(X, Y)
    ngy = gy / height
    ngx = gx / width

    # Compute sin_arg and sin values
    sin_arg = prefac * (ngx**2 + ngy**2) / 2.0
    sin_value = np.sin(sin_arg)

    # Compute the filter representation
    filter_data = None
    if method == 'tie':
        filter_data = 0.5 / (sin_arg + np.power(10, -reg_rate))

    elif method == 'qp':
        mask = np.logical_and(sin_arg > 0.5*np.pi, np.abs(sin_value) < threshold)
        filter_data = 0.5 * np.sign(sin_value) / (np.abs(sin_value) + np.power(10, -reg_rate))
        filter_data[mask] = 0

    elif method == 'qphalfsine':
        mask = np.logical_and(sin_arg > 0.5*np.pi, np.abs(sin_value) < threshold)
        mask = np.logical_or(mask, sin_arg >= np.pi)
        filter_data = 0.5 * np.sign(sin_value) / (np.abs(sin_value) + np.power(10, -reg_rate))
        filter_data[mask] = 0

    elif method == 'qp2':
        mask = np.logical_and(sin_arg > 0.5*np.pi, np.abs(sin_value) < threshold)
        filter_data = 0.5 * np.sign(sin_value) / (np.abs(sin_value) + np.power(10, -reg_rate))
        filter_data[mask] = np.sign(filter_data[mask]) / (2 * (threshold + np.power(10, -reg_rate)))

    filter_data[0, 0] = 0.5 * np.power(10, reg_rate);
    filter_data = np.fft.fftshift(filter_data)
    return filter_data


def get_power_spectrum_density(params):
    psd_path = params.power_spectral_density
    if psd_path is None or not os.path.isfile(psd_path):
        raise RuntimeError("You must specify --power-spectral-density")
    return read_image(psd_path)


def get_raw_profiles(psd_img, filter_img):
    """Get profiles of the power spectrum density and phase retrieval filter
    along with the radial position."""
    r, profile_psd = get_profile(psd_img)
    _, profile_filter = get_profile(filter_img)

    # Smooth the data profile
    profile_psd = smooth_1d(profile_psd)

    return (r, profile_psd, profile_filter)


def get_norm_profiles(psd_img, filter_img):
    """Get normalized profiles of the power spectrum density and phase
    retrieval filter along with the radial position."""
    r, profile_psd, profile_filter = get_raw_profiles(psd_img, filter_img)

    n_profile_psd = profile_psd / np.max(profile_psd)
    n_profile_filter = profile_filter / np.max(profile_filter)

    # Shift data profile so that the mean value of both profie would be
    # on the same level
    #n_profile_psd -= (np.mean(n_profile_psd) - np.mean(n_profile_filter))
    p0 = int(len(r)*0.1)
    p1 = len(r) - p0
    n_profile_psd -= np.min(n_profile_psd[p0:p1])

    return r, n_profile_psd, n_profile_filter


def save_profiles_image(params, r, profile_psd, profile_filter):
    if params.output is None:
        raise RuntimeError("You must specify --output")

    # Plot the image and save it
    import matplotlib.pyplot as plt
    filter_label = "Phase retrieval: {}".format(params.retrieval_method)

    fig, axes = plt.subplots(1, figsize=(20, 10))
    axes.plot(r, profile_psd, 'b', label="Power Spectral Density", linewidth=1)
    axes.plot(r, profile_filter, 'r--', label=filter_label, linewidth=1)
    axes.set_ylim(-0.3, 0.3)
    plt.legend(bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.5, prop={'size': 20})
    plt.savefig(params.output, bbox_inches='tight')


def analyze(args):
    filter_img = compute_filter(args)
    psd_img = get_power_spectrum_density(args)
    r, profile_psd, profile_filter = get_norm_profiles(psd_img, filter_img)
    save_profiles_image(args, r, profile_psd, profile_filter)


def compute_phase_spectrum_density(args):
    graph = Ufo.TaskGraph()
    scheduler = Ufo.Scheduler()

    build_power_spectrum_density_pipeline(args, graph, scheduler)
    # scheduler.run(graph) # runned inside for now
    duration = scheduler.props.time

    return duration

def generate_filter(args):
    if args.output is None:
        raise RuntimeError("You must specify --output")
    filter_img = compute_filter(args)
    import tifffile
    tifffile.imsave(args.output, filter_img.astype(np.float32))
