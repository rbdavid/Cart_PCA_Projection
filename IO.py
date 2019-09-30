
# ----------------------------------------
# PREAMBLE:
# ----------------------------------------

import importlib
import sys

# ----------------------------------------
# FUNCTIONS: 
# ----------------------------------------
def config_parser(config_file,parameters):	
    """ Function to take config file and create/fill the parameter dictionary (created before function call). 

    Usage: 
        parameters = {}     # initialize the dictionary to be filled with keys and values
        config_parser(config_file,parameters)
    
    Arguments:
        config_file: string object that corresponds to the local or global position of the config file to be used for this analysis.
        parameters: dictionary object that will contain the keys and values for parameters to be used in the script
    
    """
    necessary_parameters = ['output_directory','system_descriptor','pca_clustering_functions_file','nCluster_list','cartesian_coordinates_file','average_structure_file','covariance_matrix_file']
    all_parameters = ['output_directory','system_descriptor','pca_clustering_functions_file','nCluster_list','cartesian_coordinates_file','average_structure_file','covariance_matrix_file','plotting_boolean','nProjections','figure_format','write_summary']

    # NECESSARY PARAMETERS ARE INITIALIZED IN DICTIONARY WITH EMPTY STRINGS:
    for i in range(len(necessary_parameters)):
        parameters[necessary_parameters[i]] = ''
    
    # SETTING DEFAULT PARAMETERS FOR OPTIONAL PARAMETERS:
    parameters['plotting_boolean'] = False
    parameters['nProjections'] = 2
    parameters['figure_format'] = 'png'
    parameters['write_summary'] = True 
    
    # GRABBING PARAMETER VALUES FROM THE CONFIG FILE:
    with open(config_file) as f:
        exec(compile(f.read(),config_file,'exec'),parameters)
    
    # TESTING IF ANY PARAMETER HAS BEEN LEFT EMPTY:
    for key, value in list(parameters.items()):
        if value == '':
            print('%s has not been assigned a value. This variable is necessary for the script to run. Please declare this variable within the config file.' %(key))
            sys.exit()

def summary(summary_filename,arguments,parameters):
    """ Function to create a text file that holds important information about the analysis that was just performed. Outputs the version of MDAnalysis, how to rerun the analysis, and the parameters used in the analysis.
    
    Usage:
        summary(summary_filename,arguments,parameters)
    
    Arguments:
        summary_filename: string object of the file name to be written that holds the summary information.
        arguments: list object containing the terminal line arguments read into the script
        parameters: dictionary object containing the key and values associated with parameters used in the script
    
    """
    with open(summary_filename,'w') as f:
    	f.write('To recreate this analysis, run this line:\n')
    	for i in range(len(arguments)):
    	    f.write('%s ' %(arguments[i]))
    	f.write('\n\n')
    	f.write('Parameters used:\n')
        for key, value in list(parameters.items()):
            if key == '__builtins__':
                continue
            if type(value) == int or type(value) == float:
                f.write("%s = %s\n" %(key,value))
            else:
                f.write("%s = '%s'\n" %(key,value))
    	f.write('\n')

