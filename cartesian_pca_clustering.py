#!/home/rbdavid/bin/python
# ----------------------------------------
# USAGE:

# ----------------------------------------
# PREAMBLE:

import sys
import os
import importlib
import numpy as np
import MDAnalysis
from IO import config_parser,summary

config_file = sys.argv[1]

# ----------------------------------------
# FUNCTIONS:
# ----------------------------------------

def main():
        
        system_descriptor = parameters['output_directory'] + parameters['system_descriptor']

        data = np.loadtxt(parameters['user_defined_data_file'])
        mean_vector = np.loadtxt(parameters['user_defined_mean_vector_file'])
        variance_vector = np.loadtxt(parameters['user_defined_variance_vector_file'])
        covariance_matrix = np.loadtxt(parameters['user_defined_covariance_matrix_file'])

        # ----------------------------------------
        # Plotting Mean, Variance, Covariance, and Correlation results
        if parameters['plotting_boolean']:
                # plotting covariance matrix as 2d heatmap 
                covar_matrix_heatmap_figure_name = system_descriptor + '.covar_matrix.heatmap.' + parameters['figure_format']
                plot_2dmatrix(covariance_matrix,covar_matrix_heatmap_figure_name,cbar_label='Covariance of Collective Variables ($\AA^{2}$)',plotting_cmap='bwr',v_range=[-np.max(covariance_matrix),np.max(covariance_matrix)])
            
        # ----------------------------------------
        # PCA analysis
        zero_padded_string_formatting = '%0'+'%d'%(int(np.log10(len(mean_vector)))+1)+'d'
        eigenvector_output_filename = parameters['output_directory'] + zero_padded_string_formatting +'.' + parameters['system_descriptor'] + '.pca_eigenvector.dat'

        # perform pca analysis on covariance matrix
        eigenvector_matrix = pca_calc(covariance_matrix,system_descriptor,eigenvector_output_filename)
        print 'Finished the principle component analysis on the covariance matrix. Onto projecting the raw data onto the eigenvectors.'
        
        # ----------------------------------------
        # projection analysis
        zero_padded_string_formatting = '%0'+'%d'%(int(np.log10(parameters['nProjections']))+1)+'d'
        projected_data_figure_names = parameters['output_directory'] + zero_padded_string_formatting +'.' + parameters['system_descriptor'] + '.projected_data.1d_hist.' + parameters['figure_format']
        
        projected_data = data_projection(data,mean_vector,variance_vector,eigenvector_matrix,parameters['nProjections'],system_descriptor,plotting_bool = parameters['plotting_boolean'],eigenvec_projection_figure_names=projected_data_figure_names,nBins=250,test_eigenvec_projections=True)

        # ----------------------------------------
        # clustering analysis and plotting
        zero_padded_string_formatting = '%0'+'%d'%(int(np.log10(np.max(parameters['nCluster_list']))+1))+'d'
        cluster_labels_output_string = parameters['output_directory'] + zero_padded_string_formatting + '.' + parameters['system_descriptor']
        cluster_figure_names = parameters['output_directory'] + zero_padded_string_formatting + '.' + parameters['system_descriptor'] + '.clustering.' + parameters['figure_format']
        
        kmeans_clustering(projected_data,parameters['nCluster_list'],system_descriptor,cluster_labels_output_string,cluster_figure_names)
        print 'Finished clustering the data. Done with the analyses encoded by this script. How does the data look?'

        # ----------------------------------------
        # SUMMARY OUTPUT 
        if parameters['write_summary']:
                summary_filename = system_descriptor + '.cartesian_pca_clustering.summary'
        	summary(summary_filename)

# ----------------------------------------
# CREATING PARAMETER DICTIONARY
parameters = {}
config_parser(config_file)

# ----------------------------------------
# LOADING IN NECESSARY FUNCTIONS FROM MODULE FILES

plot_2dmatrix = importlib.import_module(parameters['pca_clustering_functions_file'].split('.')[0],package=None).plot_2dmatrix

pca_calc = importlib.import_module(parameters['pca_clustering_functions_file'].split('.')[0],package=None).pca_calc

data_projection = importlib.import_module(parameters['pca_clustering_functions_file'].split('.')[0],package=None).data_projection

kmeans_clustering = importlib.import_module(parameters['pca_clustering_functions_file'].split('.')[0],package=None).kmeans_clustering

# ----------------------------------------
# CREATING OUTPUT DIRECTORY
if parameters['output_directory'][-1] != os.sep:
        parameters['output_directory'] += os.sep

if os.path.exists(parameters['output_directory']):
        print 'The output directory, ', parameters['output_directory'], ' already exists. Please delete this directory or select a different one for output before proceeding.'
        sys.exit()
else:
        os.mkdir(parameters['output_directory'])

# ----------------------------------------
# MAIN
if __name__ == '__main__':
	main()

