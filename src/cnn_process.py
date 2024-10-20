#!/usr/bin/env python

"""
"""

import sys
import argparse
import os
import io

import pandas as pd
from sklearn.metrics import confusion_matrix


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

    print('')

    return output[0]


def main(argv):
    right, wrong = 0, 0
    predicted = ''

    # confusion matrix
    true_labels = []
    predicted_labels = []

    # Parse arguments
    args = parse_args(argv)
    print("Args: %s" % str(args))

    # Prepare paths
    input_dir = utils.parse_path(args.input_dir, utils.REPO)
    model = utils.parse_path(PATH_ADNET, utils.REPO)
    mean_std = utils.parse_path(PATH_MEAN_STD, utils.REPO)
    output_dir = utils.parse_path(args.output_dir, utils.REPO)

    # Load expected values from CSV
    expected_values_df = load_expected_values(os.path.abspath('/mnt/e/ADNI/ADNI1/ADNI1_Screening_10_04_2024.csv'))

    output_file = os.path.basename(output_dir) + '.txt'

    # Abrir arquivo para salvar logs
    with open(os.path.join(output_dir, output_file), 'w') as log_file:

        # Escrever os argumentos no arquivo de log
        log_file.write(f"Args: {str(args)}\n")

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

            # Busca dados do paciente no csv
            subject_id = os.path.basename(file).replace('.nii.gz', '')
            subject_data = expected_values_df[expected_values_df['Subject'] == subject_id]

            if not subject_data.empty:
                expected_value = subject_data['Group'].values[0]  # Busca o label

                if (max(out_data[0], out_data[1], out_data[2]) == out_data[0]):
                    predicted = 'CN'
                elif (max(out_data[0], out_data[1], out_data[2]) == out_data[1]):
                    predicted = 'MCI'
                elif max(out_data[0], out_data[1], out_data[2]) == out_data[2]:
                    predicted = 'AD'

                if predicted == expected_value:
                    right += 1
                else:
                    wrong += 1

                true_labels.append(expected_value)
                predicted_labels.append(predicted)

                # Escrever comparação no log
                print(f"Comparando {subject_id}: Saida = {predicted}, Esperado = {expected_value}")
                log_file.write(f"Comparando {subject_id}: Saida = {predicted}, Esperado = {expected_value}\n")
            else:
                log_file.write(f"Nenhum dado esperado encontrado para {subject_id}.\n")

        # Contagem dos valores reais para cada classe
        true_cn = true_labels.count('CN')
        true_mci = true_labels.count('MCI')
        true_ad = true_labels.count('AD')

        # Contagem dos valores preditos para cada classe
        pred_cn = predicted_labels.count('CN')
        pred_mci = predicted_labels.count('MCI')
        pred_ad = predicted_labels.count('AD')

        # Salvar resumo no arquivo de log
        log_file.write(f"\nRESUMO DOS RESULTADOS:\n")
        log_file.write(f"RIGHT: {right}\nWRONG: {wrong}\nACCURACY: {right/(right+wrong)}\n")
        log_file.write(f"TRUE: \n   CN: {true_cn}\n   MCI: {true_mci}\n   AD: {true_ad}\n")
        log_file.write(f"PREDICTED:  \n   CN: {pred_cn}\n   MCI: {pred_mci}\n   AD: {pred_ad}\n")

        # Gerar a matriz de confusão
        labels = ['CN', 'MCI', 'AD']
        conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)

        # Exibir a matriz de confusão no log
        conf_matrix_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)
        log_file.write(f"\nMatriz de Confusao:\n{conf_matrix_df}\n")




if __name__ == "__main__":
    main(sys.argv[1:])