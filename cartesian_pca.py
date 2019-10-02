
# ----------------------------------------
# USAGE:
# ----------------------------------------
# python3 cartesian_pca.py config_file.config IO_functions_file

# ----------------------------------------
# CODE OUTLINE
# ----------------------------------------
#   Step1: Use Step1 code from Allosteric_Paths_in_Proteins.
#   Step2: Essential Dynamics (this code)
#       1) Load in user defined parameters
#       2) Load in necessary functions from module files
#       3) PCA on cartesian coordinate covariance
#           a) PCA
#           b) Project timestep data onto PC eigenvectors

# ----------------------------------------
# PREAMBLE:
# ----------------------------------------

import sys
import os
import importlib
import numpy as np
import MDAnalysis

config_file = sys.argv[1]
IO_functions_file = sys.argv[2]

config_parser = importlib.import_module(IO_functions_file.split('.py')[0],package=None).cartesian_pca_config_parser
summary = importlib.import_module(IO_functions_file.split('.py')[0],package=None).cartesian_pca_summary

# ----------------------------------------
# FUNCTIONS:
# ----------------------------------------

def main():
    system_descriptor = parameters['output_directory'] + parameters['system_descriptor']

    data = np.loadtxt(parameters['cartesian_coordinates_file'])
    mean_vector = np.ndarray.flatten(np.loadtxt(parameters['average_structure_file']))
    covariance_matrix = np.loadtxt(parameters['covariance_matrix_file'])

    # ----------------------------------------
    # 3a) PCA
    # ----------------------------------------
    zero_padded_string_formatting = '%0'+'%d'%(int(np.log10(len(mean_vector)))+1)+'d'
    eigenvector_output_filename = parameters['output_directory'] + zero_padded_string_formatting +'.' + parameters['system_descriptor'] + '.pca_eigenvector.dat'
    eigenvector_matrix = pca_calc(covariance_matrix,system_descriptor,eigenvector_output_filename)
    print('Finished the principle component analysis on the covariance matrix. Onto projecting the raw data onto the eigenvectors.')

    # ----------------------------------------
    # 3b) Project timestep data onto PC eigenvectors
    # ----------------------------------------
    zero_padded_string_formatting = '%0'+'%d'%(int(np.log10(parameters['nProjections']))+1)+'d'
    projected_data_figure_names = parameters['output_directory'] + zero_padded_string_formatting +'.' + parameters['system_descriptor'] + '.projected_data.1d_hist.' + parameters['figure_format']
    data_projection(data,mean_vector,eigenvector_matrix,parameters['nProjections'],system_descriptor,plotting_bool = parameters['plotting_boolean'],eigenvec_projection_figure_names=projected_data_figure_names,nBins=250,test_eigenvec_projections=True)

    # ----------------------------------------
    # SUMMARY OUTPUT 
    # ----------------------------------------
    if parameters['write_summary']:
        summary_filename = system_descriptor + '.cartesian_pca.summary'
        summary(summary_filename,sys.argv,parameters)

# ----------------------------------------
# 1) LOAD IN USER DEFINED PARAMETERS
# ----------------------------------------
parameters = {}
config_parser(config_file,parameters)

# ----------------------------------------
# SETTING UP THE OUTPUT DIRECTORY
# ----------------------------------------
if parameters['output_directory'][-1] != os.sep:
    parameters['output_directory'] += os.sep

# ----------------------------------------
# 2) LOADING IN NECESSARY FUNCTIONS FROM MODULE FILES
# ----------------------------------------
pca_calc = importlib.import_module(parameters['pca_functions_file'].split('.py')[0],package=None).pca_calc

data_projection = importlib.import_module(parameters['pca_functions_file'].split('.py')[0],package=None).data_projection

# ----------------------------------------
# MAIN
# ----------------------------------------
if __name__ == '__main__':
    main()

