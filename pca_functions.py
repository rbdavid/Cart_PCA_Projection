
# USAGE:
# from pca_clustering_functions import *

# PREAMBLE:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator
from sklearn.cluster import KMeans

# SUBROUTINES/FUNCTIONS:

#def plot_vector_as_2dheatmap(vector,two_dimensional_boolean_matrix,figure_name,cbar_label='',plotting_cmap='bwr',v_range=None,minor_ticks=1,major_ticks=10):
#        """
#        """
#        vector = np.copy(vector)
#        two_dimensional_boolean_matrix = np.copy(two_dimensional_boolean_matrix)
#        nNodes = len(two_dimensional_boolean_matrix)
#	nNodes_range = range(nNodes)
#        two_dimensional_matrix_from_vector = np.zeros((nNodes,nNodes),dtype=np.float64)
#        counter = 0
#        for i in nNodes_range:
#                for j in nNodes_range:
#                        if two_dimensional_boolean_matrix[i][j]:
#                                two_dimensional_matrix_from_vector[i][j] = vector[counter]
#                                two_dimensional_matrix_from_vector[j][i] = vector[counter]
#                                counter += 1
#
#        node_range = range(nNodes+1)
#        fig, ax = plt.subplots()
#        ax.tick_params(which='major',length=6,width=2)
#        ax.tick_params(which='minor',length=3,width=1)
#        ax.xaxis.set_minor_locator(MultipleLocator(minor_ticks))
#        ax.xaxis.set_major_locator(MultipleLocator(major_ticks))
#        ax.yaxis.set_minor_locator(MultipleLocator(minor_ticks))
#        ax.yaxis.set_major_locator(MultipleLocator(major_ticks))
#
#        if v_range != None:
#                temp = plt.pcolormesh(node_range,node_range,two_dimensional_matrix_from_vector,cmap=plotting_cmap,vmin=v_range[0],vmax=v_range[1])
#        else:
#                temp = plt.pcolormesh(node_range,node_range,two_dimensional_matrix_from_vector,cmap=plotting_cmap)
#        cb1 = plt.colorbar()
#        cb1.set_label(r'%s'%(cbar_label))
#
#        xlabels = [str(int(x)) for x in temp.axes.get_xticks()[:]]
#        ylabels = [str(int(y)) for y in temp.axes.get_yticks()[:]]
#        temp.axes.set_xticks(temp.axes.get_xticks(minor=True)[:]+0.5,minor=True)
#        temp.axes.set_xticks(temp.axes.get_xticks()[:]+0.5)
#        temp.axes.set_yticks(temp.axes.get_yticks(minor=True)[:]+0.5,minor=True)
#        temp.axes.set_yticks(temp.axes.get_yticks()[:]+0.5)
#        temp.axes.set_xticklabels(xlabels)
#        temp.axes.set_yticklabels(ylabels)
#
#        plt.xlim((-0.5,nNodes+0.5))
#        plt.ylim((-0.5,nNodes+0.5))
#        plt.xlabel('Atom index in atom selection',size=14)
#        plt.ylabel('Atom index in atom selection',size=14)
#        ax.set_aspect('equal')
#        plt.tight_layout()
#        plt.savefig(figure_name,dpi=600,transparent=True)
#        plt.close()
#
#def plot_2dmatrix(square_matrix,figure_name,cbar_label='',plotting_cmap='bwr',v_range=None,minor_ticks=10,major_ticks=100):
#        """
#        """
#        nNodes = len(square_matrix)
#        node_range = range(nNodes+1)
#        fig, ax = plt.subplots()
#        ax.tick_params(which='major',length=6,width=2)
#        ax.tick_params(which='minor',length=3,width=1)
#        ax.xaxis.set_minor_locator(MultipleLocator(minor_ticks))
#        ax.xaxis.set_major_locator(MultipleLocator(major_ticks))
#        ax.yaxis.set_minor_locator(MultipleLocator(minor_ticks))
#        ax.yaxis.set_major_locator(MultipleLocator(major_ticks))
#        
#        if v_range != None:
#                temp = plt.pcolormesh(node_range,node_range,square_matrix,cmap=plotting_cmap,vmin=v_range[0],vmax=v_range[1])
#        else:
#                temp = plt.pcolormesh(node_range,node_range,square_matrix,cmap=plotting_cmap)
#        cb1 = plt.colorbar()
#        cb1.set_label(r'%s'%(cbar_label))
#
#        xlabels = [str(int(x)) for x in temp.axes.get_xticks()[:]]
#        ylabels = [str(int(y)) for y in temp.axes.get_yticks()[:]]
#        temp.axes.set_xticks(temp.axes.get_xticks(minor=True)[:]+0.5,minor=True)
#        temp.axes.set_xticks(temp.axes.get_xticks()[:]+0.5)
#        temp.axes.set_yticks(temp.axes.get_yticks(minor=True)[:]+0.5,minor=True)
#        temp.axes.set_yticks(temp.axes.get_yticks()[:]+0.5)
#        temp.axes.set_xticklabels(xlabels)
#        temp.axes.set_yticklabels(ylabels)
#
#        plt.xlim((-0.5,nNodes+0.5))
#        plt.ylim((-0.5,nNodes+0.5))
#        plt.xlabel('Collective Variable Index',size=14)
#        plt.ylabel('Collective Variable Index',size=14)
#        ax.set_aspect('equal')
#        plt.tight_layout()
#        plt.savefig(figure_name,dpi=600,transparent=True)
#        plt.close()

def pca_calc(square_matrix,system_descriptor,eigenvector_output_filename):
    """
    """
    # CALCULATE THE EIGENVALUES AND EIGENVECTORS OF THE SQUARE MATRIX
    eigval,eigvec = np.linalg.eig(square_matrix)    # NOTE: EIGENVEC IS ORGANIZED WHERE COMPONENTS OF EACH INDIVIDUAL EIGENVECTOR ARE STORED IN THE COLUMN (SECOND INDEX)
    # SORT THE EIGENVALUES AND EIGENVECTORS FROM LARGEST VALUED TO SMALLEST VALUED
    idx = eigval.argsort()[::-1]    # get arrangement of eigval indices that sort from largest value to smallest value
    eigval = eigval[idx]    # rearrange eigval
    eigvec = eigvec[:,idx]  # rearrange the columns of eigvec

    # ANALYZE THE EIGENVALUES AND CUMULATIVE EIGENVALUE TO HELP THE USER DECIDE THE NUMBER OF EIGENVECTORS THAT ADEQUATELY DESCRIBE THE DATASET
    nVec = len(eigvec)
    nVec_range = list(range(nVec))
    cumulative_eigval = np.zeros(nVec,dtype=np.float64)
    total_eigval = 0
    for i in nVec_range:
        total_eigval += eigval[i]
        cumulative_eigval[i] = total_eigval

    # OUTPUT EIGENVALUES AND EIGENVECTORS TO FILE
    with open(system_descriptor+'.pca_eigenvalues.dat','w') as f:
        f.write('# Eigenvalue   Frac_Total   Cumulative   Frac_Cumulative\n')
        for i in nVec_range:
            f.write('%f   %f   %f   %f\n' %(eigval[i],eigval[i]/total_eigval,cumulative_eigval[i],cumulative_eigval[i]/total_eigval))
            np.savetxt(eigenvector_output_filename%(i),eigvec[:,i],fmt='%f')

    # RETURN THE EIGENVEC ARRAY
    return eigvec

def data_projection(data,mean_vector,eigvec,nProjections,system_descriptor,plotting_bool=True,eigenvec_projection_figure_names='%d.projected_data.1d_hist.png',nBins=100,test_eigenvec_projections=True):
    """
    """

    nProjection_range = list(range(nProjections))
    data -= mean_vector
    projection_data = np.zeros((len(data),nProjections),dtype=np.float64)
    for i in nProjection_range:
        projection_data[:,i] = np.dot(data,eigvec[:,i])
        if plotting_bool:
            events,edges,patches = plt.hist(projection_data[:,i],bins=nBins,histtype='bar',normed=True)
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

