# coding: utf-8
import os
import ismrmrd
import ismrmrd.xsd
from ismrmrdtools import simulation, transform
import numpy as np
import argparse

def create(filename='testdata.h5', matrix_size=256, coils=8, oversampling=2, repetitions=1, acceleration=1, noise_level=0.05):

    # Generate the phantom and coil sensitivity maps
    phan = simulation.phantom(matrix_size)
    csm = simulation.generate_birdcage_sensitivities(matrix_size, coils)
    coil_images = np.tile(phan,(coils, 1, 1)) * csm

    # Oversample if needed
    if oversampling>1:
        padding = round((oversampling*phan.shape[1] - phan.shape[1])/2)
        phan = np.pad(phan,((0,0),(padding,padding)),mode='constant')
        csm = np.pad(csm,((0,0),(0,0),(padding,padding)),mode='constant')
        coil_images = np.pad(coil_images,((0,0),(0,0),(padding,padding)),mode='constant')

    # The number of points in x,y,kx,ky
    nx = matrix_size
    ny = matrix_size
    nkx = oversampling*nx
    nky = ny

    # Open the dataset
    dset = ismrmrd.Dataset(filename, "dataset", create_if_needed=True)

    # Create the XML header and write it to the file
    header = ismrmrd.xsd.ismrmrdHeader()

    # Experimental Conditions
    exp = ismrmrd.xsd.experimentalConditionsType()
    exp.H1resonanceFrequency_Hz = 128000000
    header.experimentalConditions = exp

    # Acquisition System Information
    sys = ismrmrd.xsd.acquisitionSystemInformationType()
    sys.receiverChannels = coils
    header.acquisitionSystemInformation = sys

    # Encoding
    encoding = ismrmrd.xsd.encodingType()
    encoding.trajectory = ismrmrd.xsd.trajectoryType('cartesian')

    # encoded and recon spaces
    efov = ismrmrd.xsd.fieldOfViewMmType()
    efov.x = oversampling*256
    efov.y = 256
    efov.z = 5
    rfov = ismrmrd.xsd.fieldOfViewMmType()
    rfov.x = 256
    rfov.y = 256
    rfov.z = 5
    
    ematrix = ismrmrd.xsd.matrixSizeType()
    ematrix.x = nkx
    ematrix.y = nky
    ematrix.z = 1
    rmatrix = ismrmrd.xsd.matrixSizeType()
    rmatrix.x = nx
    rmatrix.y = ny
    rmatrix.z = 1
    
    espace = ismrmrd.xsd.encodingSpaceType()
    espace.matrixSize = ematrix
    espace.fieldOfView_mm = efov
    rspace = ismrmrd.xsd.encodingSpaceType()
    rspace.matrixSize = rmatrix
    rspace.fieldOfView_mm = rfov
    
    # Set encoded and recon spaces
    encoding.encodedSpace = espace
    encoding.reconSpace = rspace
    
    # Encoding limits
    limits = ismrmrd.xsd.encodingLimitsType()
    
    limits1 = ismrmrd.xsd.limitType()
    limits1.minimum = 0
    limits1.center = round(ny/2)
    limits1.maximum = ny - 1
    limits.kspace_encoding_step_1 = limits1
    
    limits_rep = ismrmrd.xsd.limitType()
    limits_rep.minimum = 0
    limits_rep.center = round(repetitions / 2)
    limits_rep.maximum = repetitions - 1
    limits.repetition = limits_rep
    
    limits_rest = ismrmrd.xsd.limitType()
    limits_rest.minimum = 0
    limits_rest.center = 0
    limits_rest.maximum = 0
    limits.kspace_encoding_step_0 = limits_rest
    limits.slice = limits_rest    
    limits.average = limits_rest
    limits.contrast = limits_rest
    limits.kspaceEncodingStep2 = limits_rest
    limits.phase = limits_rest
    limits.segment = limits_rest
    limits.set = limits_rest
    
    encoding.encodingLimits = limits
    header.encoding.append(encoding)

    dset.write_xml_header(header.toXML('utf-8'))

    # Synthesize the k-space data
    Ktrue = transform.transform_image_to_kspace(coil_images,(1,2))

    # Create an acquistion and reuse it
    acq = ismrmrd.Acquisition()
    acq.resize(nkx, coils)
    acq.version = 1
    acq.available_channels = coils
    acq.center_sample = round(nkx/2)
    acq.read_dir[0] = 1.0
    acq.phase_dir[1] = 1.0
    acq.slice_dir[2] = 1.0

    # Initialize an acquisition counter
    counter = 0
    
    # Write out a few noise scans
    for n in range(32):
        noise = noise_level * (np.random.randn(coils, nkx) + 1j * np.random.randn(coils, nkx))
        # here's where we would make the noise correlated
        acq.scan_counter = counter
        acq.clearAllFlags()
        acq.setFlag(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)
        acq.data[:] = noise
        dset.append_acquisition(acq)
        counter += 1 # increment the scan counter
    
    # Loop over the repetitions, add noise and write to disk
    # simulating a T-SENSE type scan
    for rep in range(repetitions):
        noise = noise_level * (np.random.randn(coils, nky, nkx) + 1j * np.random.randn(coils, nky, nkx))
        # here's where we would make the noise correlated
        K = Ktrue + noise
        acq.idx.repetition = rep
        for acc in range(acceleration):
            for line in np.arange(acc,nky,acceleration):
                # set some fields in the header
                acq.scan_counter = counter
                acq.idx.kspace_encode_step_1 = line
                acq.clearAllFlags()
                if line == 0:
                     acq.setFlag(ismrmrd.ACQ_FIRST_IN_ENCODE_STEP1)
                     acq.setFlag(ismrmrd.ACQ_FIRST_IN_SLICE)
                     acq.setFlag(ismrmrd.ACQ_FIRST_IN_REPETITION)
                elif line == nky - 1:
                     acq.setFlag(ismrmrd.ACQ_LAST_IN_ENCODE_STEP1)
                     acq.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
                     acq.setFlag(ismrmrd.ACQ_LAST_IN_REPETITION)
                # set the data and append
                acq.data[:] = K[:,line,:]
                dset.append_acquisition(acq)
                counter += 1

    # Clean up
    dset.close()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', help='output filename')
    parser.add_argument('-m', '--matrix-size', type=int, dest='matrix_size', help='k-space matrix size')
    parser.add_argument('-c', '--coils', type=int, help='number of coils')
    parser.add_argument('-s', '--oversampling', type=int, help='oversampling')
    parser.add_argument('-r', '--repetitions', type=int, help='number of repetitions')
    parser.add_argument('-a', '--acceleration', type=int, help='acceleration')
    parser.add_argument('-n', '--noise-level', type=float, dest='noise_level', help='noise level')

    parser.set_defaults(output='testdata.h5', matrix_size=256, coils=8,
            oversampling=2, repetitions=1, acceleration=1, noise_level=0.05)

    args = parser.parse_args()

    create(args.output, args.matrix_size, args.coils, args.oversampling,
            args.repetitions, args.acceleration, args.noise_level)

if __name__ == "__main__":
    main()
