
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
#       3) Analyze Trajectory and Project Data onto Eigenvectors
#           a) Create the atom selection to be subsequently used
#           b) Align trajectory to an alignment structure
#           c) Project aligned, mean-centered cartesian coordinates onto Eigenvectors

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

config_parser = importlib.import_module(IO_functions_file.split('.py')[0],package=None).align_and_project_config_parser
summary = importlib.import_module(IO_functions_file.split('.py')[0],package=None).align_and_project_summary

# ----------------------------------------
# FUNCTIONS:
# ----------------------------------------

def main():
    system_descriptor = parameters['output_directory'] + parameters['system_descriptor']
    alignment_coordinates = np.loadtxt(parameters['alignment_structure_data_file'])
    nNodes = alignment_coordinates.shape[0]
    nCartCoords = nNodes*3

    # ----------------------------------------
    # 3a) CREATE THE MDAnalysis.Universe OBJECT AND DESIRED ATOM SELECTIONS
    # ----------------------------------------
    u = MDAnalysis.Universe(parameters['pdb'])
    selection_list = make_selections(u,parameters['output_directory'] + 'node_selections.txt',parameters['substrate_node_definition'],parameters['substrate_selection_string'], nonstandard_substrates_selection = parameters['nonstandard_substrates_selection'], homemade_selections = parameters['homemade_selections'])
    print('Number of analysis nodes:', len(selection_list), '\nNode selections written out to', parameters['output_directory'] + 'node_selections.txt')
    
    # ----------------------------------------
    # 3b) ANALYZE TRAJECTORIES
    # ----------------------------------------
    # i) FILL A NUMPY ARRAY WITH NODE POSITIONS and ii) ALIGN THE NODE POSITIONS TO THE ALIGNMENT COORDINATES
    # ----------------------------------------
    Node_trajectory = traj_alignment(u,selection_list,parameters['substrate_selection_string'],alignment_coordinates,parameters['substrate_node_definition'],parameters['trajectory_list'],parameters['output_directory'])
    nSteps = Node_trajectory.shape[0]
    Node_trajectory = Node_trajectory.reshape((nSteps,nCartCoords))
    alignment_coordinates = alignment_coordinates.reshape((nCartCoords))

    # ----------------------------------------
    # 3c) PROJECT CART. COORDS ONTO EIGENVECTORS
    # ----------------------------------------
    projected_data_figure_names = parameters['output_directory'] + '%0' + '%d'%(int(np.log10(parameters['nProjections']))+1) + 'd.' + parameters['system_descriptor'] + '.projected_data.1d_hist.' + parameters['figure_format']
    eigenvector_data = np.zeros((nCartCoords,parameters['nProjections']))
    for i in list(range(parameters['nProjections'])):
        eigenvector_data[:,i] = np.loadtxt(parameters['eigenvector_file_naming']%(i))

    data_projection(Node_trajectory,alignment_coordinates,eigenvector_data,parameters['nProjections'],system_descriptor,test_eigenvec_projections=False,eigenvec_projection_figure_names=projected_data_figure_names)

    # ----------------------------------------
    # SUMMARY OUTPUT 
    # ----------------------------------------
    if parameters['write_summary']:
        summary_filename = system_descriptor + '.align_and_project.summary'
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

if os.path.exists(parameters['output_directory']):
    print('The output directory, ', parameters['output_directory'], 'already exists. Please select a different directory name for output.')
    sys.exit()
else:
    os.mkdir(parameters['output_directory'])

# ----------------------------------------
# 2) LOADING IN NECESSARY FUNCTIONS FROM MODULE FILES
# ----------------------------------------
make_selections = importlib.import_module(parameters['make_selections_functions_file'].split('.py')[0],package=None).make_selections

traj_alignment = importlib.import_module(parameters['align_and_project_functions_file'].split('.py')[0],package=None).traj_alignment

data_projection = importlib.import_module(parameters['align_and_project_functions_file'].split('.py')[0],package=None).data_projection

# ----------------------------------------
# MAIN
# ----------------------------------------
if __name__ == '__main__':
    main()

