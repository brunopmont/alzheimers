#!/usr/bin/env python

"""
"""

import sys
import argparse
import os
import io

import pandas as pd

import numpy
import nibabel

import theano
import lasagne

import adnet

import utils

# File paths
PATH_MODEL = os.path.abspath(os.path.join(os.sep, 'model'))

PATH_ADNET = os.path.join(PATH_MODEL,
                          'adnet.npz')

PATH_MEAN_STD = os.path.join(PATH_MODEL,
                             'mean_std.npz')

# Brain slices
SLICE_NII_IDX0 = slice(24, 169)
SLICE_NII_IDX1 = slice(24, 206)
SLICE_NII_IDX2 = slice( 6, 161)

# Layer output
LAYER_NAME = 'prob'

# Normalization npz
INPUT_MEAN = 'mean'
INPUT_STD = 'std'

OUTPUT_HEADER = 'CN, MCI, AD'

def load_expected_values(csv_path):
    df = pd.read_csv(csv_path)
    return df


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_dir', type=str,
                        help='input brain directory file path')
    parser.add_argument('output_dir', type=str,
                        help='output directory text file path')

    args = parser.parse_args(args=argv)
    return args


def get_nii_data(path):
    data = nibabel.load(path).get_data()
    data = data[SLICE_NII_IDX0,
                SLICE_NII_IDX1,
                SLICE_NII_IDX2]
    return data


def get_mean_std(path):
    data = numpy.load(path)
    mean = data[INPUT_MEAN]
    std = data[INPUT_STD]
    return mean, std


def norm_data(data, mean, std):
    data = numpy.where(std != 0, (data - mean) / std, 0)
    return data


def get_input_data(image, mean_std, dtype):
    mean, std = get_mean_std(mean_std)
    data = get_nii_data(image)
    data = norm_data(data, mean, std)
    data = data.astype(dtype)
    return data


# Generate theano function
def get_func(network, input_var):
    output = lasagne.layers.get_output(network, deterministic=True)
    theano_fn = theano.function([input_var], output)
    return theano_fn


def load_cnn_model(path, cnn_model):
    with numpy.load(path) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(cnn_model, param_values)


def find_layer(network, layer):
    while hasattr(network, 'name') and network.name != layer:
        if hasattr(network, 'input_layer'):
            network = network.input_layer
        else:
            network = None

    if network is None:
        raise ValueError('Requested layer (%s) not found' % layer)

    return network


def cnn_process(image, model, mean_std, layer=LAYER_NAME):

    # Prepare theano variable
    input_var = theano.tensor.TensorType(theano.config.floatX,
                                         (False,)*5)('inputs')

    # Create and load cnn model
    network = adnet.build_model(input_var)
    load_cnn_model(model, network)

    # Get layer
    network = find_layer(network, layer)

    # Get function
    theano_fn = get_func(network, input_var)

    # Prepare input image
    input_data = get_input_data(image, mean_std, theano.config.floatX)

    # Run
    output = theano_fn([input_data])

    print("run")

    return output[0]


def main(argv):
    right, wrong = 0, 0

    #if sys.stdout.encoding != 'utf-8':
    #    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    # Parse arguments
    args = parse_args(argv)
    print("Args: %s" % str(args))

    # Prepare paths
    input_dir = utils.parse_path(args.input_dir, utils.REPO)
    model = utils.parse_path(PATH_ADNET, utils.REPO)
    mean_std = utils.parse_path(PATH_MEAN_STD, utils.REPO)
    output_dir = utils.parse_path(args.output_dir, utils.REPO)

    # Load expected values from CSV
    expected_values_df = load_expected_values(os.path.abspath('/repo/ADNI1_Screening_10_04_2024.csv'))

    for file in os.listdir(input_dir):
        # CNN processing
        out_data = cnn_process(os.path.join(input_dir, file),
                            model,
                            mean_std,
                            LAYER_NAME)
        

        # Output
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        numpy.savetxt(os.path.join(output_dir, file.replace('nii.gz', 'txt')), out_data,
                    header=OUTPUT_HEADER)

        # Compare with expected values
        subject_id = os.path.basename(file).replace('.nii.gz', '')  # Supondo que o nome do arquivo contém o ID do sujeito

        # Filtrar o DataFrame para o sujeito atual
        subject_data = expected_values_df[expected_values_df['Subject'] == subject_id]

        if not subject_data.empty:
            expected_value = subject_data['Group'].values[0]  # Pegue o grupo como valor esperado
            if ((max(out_data[0], out_data[1], out_data[2]) == out_data[0]) and (expected_value == 'CN')):
                right += 1
            elif ((max(out_data[0], out_data[1], out_data[2]) == out_data[1]) and (expected_value == 'MCI')):
                right += 1
            elif ((max(out_data[0], out_data[1], out_data[2]) == out_data[2]) and (expected_value == 'Alzheimer')):
                right += 1
            else:
                wrong += 1
            print(f"Comparando {subject_id}: Saída = {out_data}, Esperado = {expected_value}")
            # Aqui você pode adicionar lógica para verificar se o resultado do modelo está correto
        else:
            print(f"Nenhum dado esperado encontrado para {subject_id}.")

    print(f"Certos: {right}\nErrados: {wrong}\nAcurácia{right/wrong}")
    print("Done!")


if __name__ == "__main__":
    main(sys.argv[1:])
