
# USAGE:
# from pca_clustering_functions import *

# PREAMBLE:

import MDAnalysis
from MDAnalysis.analysis.align import rotation_matrix
from MDAnalysis.analysis.distances import distance_array
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator
from numpy.linalg import *

# SUBROUTINES/FUNCTIONS:

def traj_alignment(universe, selection_list, selection_string, alignment_coordinates, node_definition, trajectory_list, output_directory,step=1):
    """
    """

    print('Beginning trajectory analysis.')
    
    # ----------------------------------------
    # IO NAMING VARIABLES
    # ----------------------------------------
    trajectory_file_name = output_directory + 'node_positions_trajectory.dat' 
    
    # ----------------------------------------
    # CREATING ATOM SELECTIONS
    # ----------------------------------------
    u_selection = universe.select_atoms(selection_string)
    nNodes = len(selection_list)
    nNodes_range = list(range(nNodes))

    # ----------------------------------------
    # ANALYZE TRAJECTORIES
    # ----------------------------------------
    all_pos_Nodes = []
    nSteps = 0
    for traj in trajectory_list:
        print('Loading trajectory', traj)
        universe.load_new(traj)
        nSteps += len(universe.trajectory)//step
        for ts in universe.trajectory[::step]:
            u_selection.translate(-u_selection.center_of_mass())
            R, d = rotation_matrix(u_selection.positions,alignment_coordinates)
            u_selection.rotate(R)
            #if node_definition.upper() == 'COM':
            #    all_pos_Nodes.append([selection_list[i].center_of_mass() for i in nNodes_range])
            #elif node_definition.upper() == 'ATOMIC':
            #    all_pos_Nodes.append([selection_list[i].position for i in nNodes_range])
            all_pos_Nodes.append([selection_list[i].position for i in nNodes_range])
    
    print('Analyzed', nSteps, 'frames.')
    all_pos_Nodes = np.array(all_pos_Nodes)
    np.savetxt(trajectory_file_name,all_pos_Nodes.reshape((nSteps,nNodes*3)),header='Shape: nSteps x (nNodes x 3)',fmt='%f')

    return all_pos_Nodes

def data_projection(data,mean_vector,eigvec,nProjections,system_descriptor,plotting_bool=True,eigenvec_projection_figure_names='%d.projected_data.1d_hist.png',nBins=100,test_eigenvec_projections=True):
    """
    """

    nProjection_range = list(range(nProjections))
    data -= mean_vector
    projection_data = np.zeros((len(data),nProjections),dtype=np.float64)
    for i in nProjection_range:
        projection_data[:,i] = np.dot(data,eigvec[:,i])
        if plotting_bool:
            events,edges,patches = plt.hist(projection_data[:,i],bins=nBins,histtype='bar',density=True)
            plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
            plt.xlabel('Data projected onto Eigenvector %d'%(i))
            plt.ylabel('Probability Density')
            plt.savefig(eigenvec_projection_figure_names %(i),dpi=600,transparent=True)
            print('Projecting data onto eigenvector', i,'. The x-axis range is:', plt.xlim())
            plt.close()

    if test_eigenvec_projections:
        print('The data has been mean centered :. the average of the projected data should be zero. The dot product of the eigenvectors should be zero. The slope of projected data should be close to zero as well.')
        for i in nProjection_range[:-2]:
            print('Eigenvec', i, ' average = ', np.mean(projection_data[:,i]))
            for j in nProjection_range[i+1:]:
                print('Eigenvec', i, 'and eigenvec', j,': dot product = ', np.dot(eigvec[:,i],eigvec[:,j]), 'slope, intercept of linear least squares:', np.polyfit(projection_data[:,i],projection_data[:,j],deg=1))

    np.savetxt(system_descriptor+'.projected_data.dat',projection_data,fmt='%f')

