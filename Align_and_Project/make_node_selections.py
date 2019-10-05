
import MDAnalysis
import sys

# ----------------------------------------
# CLASSIFICATION OF NONSTANDARD RESIDUES:
# ----------------------------------------
nucleic = ['A5','A3','A','G5','G3','G','C5','C3','C','T5','T3','T','U5','U3','U']
triphosphate = ['atp','adp','PHX']
other = ['MG']

# ----------------------------------------
# HOMEMADE ATOM SELECTION STRINGS FOR THE NONSTANDARD RESIDUES
# ----------------------------------------
sugar = "name C5' C4' O4' C1' C3' C2' O2' " + " C5* C4* O4* C1* C3* O3* C2* O2* "		# NO HYDROGENS; DOES NOT INCLUDE THE O5' atom (which I will include in the phosphate atom selection string...; the atoms with * are found in triphosphates;
sugar_5= sugar + " O5'"		# NO HYDROGENS
sugar_3= sugar + " O3' "	# NO HYDROGENS
base = 'name N9 C8 N7 C5 C6 N6 N1 C2 N3 C4 O6 N4 C2 O2 O4'	# NO HYDROGENS; selection string that will select all appropriate atoms for any of the nucleic residues...
a_phos = 'name O5* O2A O1A PA O3A'
b_phos = 'name PB O1B O2B O3B'
g_phos = 'name PG O1G O2G O3G'
inorg_phos = 'name P O1 O2 O3 O4'	# NO HYDROGENS

# ----------------------------------------
# FUNCTION USED TO MAKE ANY OF THE HOMEMADE ATOM SELECTIONS FOR THE NONSTANDARD RESIDUES
# ----------------------------------------

def make_selections(analysis_universe,file_name,node_definition,selection_string,nonstandard_substrates_selection=None,homemade_selections=None):
    """
    """
    selection_list = []
    count = 0
    with open(file_name,'w') as f:
        # ----------------------------------------
        # SUBSTRATE SELECTION - CENTER OF MASS OF RESIDUES
        # ----------------------------------------
        if node_definition.upper() == 'COM':
            f.write('# Node description: node id, Resname, Resid\n')
            substrate_selection = analysis_universe.select_atoms(selection_string)
            nResidues_range = list(range(substrate_selection.n_residues))
            for i in nResidues_range:
                temp = substrate_selection.residues[i].atoms
                selection_list.append(temp)
                f.write("%5d   COM %5s %s\n" %(count,temp.resnames[0],temp.resids[0]))
                count += 1
        
        # ----------------------------------------
        # SUBSTRATE SELECTION - ATOMS
        # ----------------------------------------
        elif node_definition.upper() == 'ATOMIC':
            f.write('# Node description: node id, atom name, atom index, residresname\n')
            substrate_selection = analysis_universe.select_atoms(selection_string)
            nAtoms_range = list(range(substrate_selection.n_atoms))
            for i in nAtoms_range:
                temp = substrate_selection.atoms[i]
                selection_list.append(temp)
                f.write("%5d %5s %7s   %s%s\n" %(count,temp.name,temp.index+1,temp.resname,temp.resid))
                count += 1
        
        # ----------------------------------------
        # SUBSTRATE SELECTION - MORE COMPLEX STUFF... MULTI-SITE COARSE GRAINING HERE?
        # ----------------------------------------
        else:
            print('The substrate_node_definition parameter is not understood.')
            sys.exit()
        
        # ----------------------------------------
        # NONSTANDARD SUBSTRATES - USER DEVELOPED SELECTIONS
        # ----------------------------------------
        count = len(selection_list)
        if nonstandard_substrates_selection != None:
            nonstandard_substrates = analysis_universe.select_atoms(nonstandard_substrates_selection)
            nResidues_range = list(range(nonstandard_substrates.n_residues))
            for i in nResidues_range:
                temp_resname = nonstandard_substrates.residues[i].resname
                temp_resid = nonstandard_substrates.residues[i].resid
                if temp_resname in homemade_selections:
                    make_nonstandard_selections(analysis_universe,temp_resname,temp_resid,f,selection_list,count)
                    count = len(selection_list)
                else:
                    temp = analysis_universe.select_atoms('resname %s and resid %d' %(temp_resname,temp_resid))
                    selection_list.append(temp)
                    f.write("%5d   COM %5s %s\n" %(count,temp_resname,temp_resid))
                    count = len(selection_list)
    
    return selection_list

def make_nonstandard_selections(analysis_universe,resname,resid,output_file,selection_list,count):
    """A function that takes in a residue name and creates a non-standard MDAnalysis atom selection
    
    Usage: make_nonstandard_selection(........)
    
    Arguments:
    	analysis_universe: MDAnalysis Universe object to be used as the analysis universe.
    	resname: string of the residue name;
    	resid: int of the residue ID number;
    	output_file: file object that is to be written to;
        ...
    """
    
    # ----------------------------------------
    # CREATING THE NUCLEIC SELECTIONS
    # ----------------------------------------
    if resname in nucleic:
        # CREATING THE SLECTION FOR THE BASE OF NUCLEIC RESIDUES
        sel_string = 'resname %s and resid %d and %s' %(resname,resid,base)
        temp = analysis_universe.select_atoms(sel_string)
        selection_list.append(temp)
        output_file.write("%5d   COM %5s %s\n" %(count,resname,sel_string))
        count +=1
        # CREATING THE SLECTION FOR THE SUGAR OF NUCLEIC RESIDUES
        if resname in ['A5','G5','C5','T5','C5']:
            sel_string = 'resname %s and resid %d and %s' %(resname,resid,sugar_5)
            temp = analysis_universe.select_atoms(sel_string)
            selection_list.append(temp)
            output_file.write("%5d   COM %5s %s\n" %(count,resname,sel_string))
            count +=1
            return
        elif resname in ['A3','U3','C3','G3']:
            sel_string = 'resname %s and resid %d and %s' %(resname,resid,sugar_3)
            temp = analysis_universe.select_atoms(sel_string)
            selection_list.append(temp)
            output_file.write("%5d   COM %5s %s\n" %(count,resname,sel_string))
            count +=1
        else:
            sel_string = 'resname %s and resid %d and %s' %(resname,resid,sugar)
            temp = analysis_universe.select_atoms(sel_string)
            selection_list.append(temp)
            output_file.write("%5d   COM %5s %s\n" %(count,resname,sel_string))
            count +=1
        
        # CREATING THE SLECTION FOR THE PHOSPHATE OF NUCLEIC RESIDUES
        sel_string = "(resname %s and resid %s and name P OP1 OP2 O5') or (resid %s and name O3')" %(resname,resid,resid-1)
        temp = analysis_universe.select_atoms(sel_string)
        selection_list.append(temp)
        output_file.write("%5d   COM %5s %s\n" %(count,resname,sel_string))
        count += 1
        return
    
    # ----------------------------------------
    # CREATING THE TRIPHOSPHATE ATOM SELECTIONS
    elif resname in triphosphate:
        if resname in ['atp','adp']:
            # CREATING THE SLECTION FOR THE BASE OF TRIPHOSPHATES
            sel_string = 'resname %s and resid %d and %s' %(resname,resid,base)
            temp = analysis_universe.select_atoms(sel_string)
            selection_list.append(temp)
            output_file.write("%5d   COM %5s %s\n" %(count,resname,sel_string))
            count +=1
            # CREATING THE SLECTION FOR THE SUGAR OF TRIPHOSPHATES
            sel_string = 'resname %s and resid %d and %s' %(resname,resid,sugar)
            temp = analysis_universe.select_atoms(sel_string)
            selection_list.append(temp)
            output_file.write("%5d   COM %5s %s\n" %(count,resname,sel_string))
            count +=1
    
        # CREATING THE SLECTION FOR THE PHOSPHATES OF TRIPHOSPHATES
        if resname == 'atp':
            sel_string = 'resname %s and resid %d and %s' %(resname,resid,a_phos)
            temp = analysis_universe.select_atoms(sel_string)
            selection_list.append(temp)
            output_file.write("%5d   COM %5s %s\n" %(count,resname,sel_string))
            count +=1
            sel_string = 'resname %s and resid %d and %s' %(resname,resid,b_phos)
            temp = analysis_universe.select_atoms(sel_string)
            selection_list.append(temp)
            output_file.write("%5d   COM %5s %s\n" %(count,resname,sel_string))
            count +=1
            sel_string = 'resname %s and resid %d and %s' %(resname,resid,g_phos)
            temp = analysis_universe.select_atoms(sel_string)
            selection_list.append(temp)
            output_file.write("%5d   COM %5s %s\n" %(count,resname,sel_string))
            count +=1
            return

        elif resname == 'adp':
            sel_string = 'resname %s and resid %d and %s' %(resname,resid,a_phos)
            temp = analysis_universe.select_atoms(sel_string)
            selection_list.append(temp)
            output_file.write("%5d   COM %5s %s\n" %(count,resname,sel_string))
            count +=1
            sel_string = 'resname %s and resid %d and %s' %(resname,resid,b_phos)
            temp = analysis_universe.select_atoms(sel_string)
            selection_list.append(temp)
            output_file.write("%5d   COM %5s %s\n" %(count,resname,sel_string))
            count +=1
            return

        # CREATING THE SLECTION FOR INORGANIC PHOSPHATE MOLECULE
        elif resname == 'PHX':
            sel_string = 'resname %s and resid %d and %s' %(resname,resid,inorg_phos)
            temp = analysis_universe.select_atoms(sel_string)
            selection_list.append(temp)
            output_file.write("%5d   COM %5s %s\n" %(count,resname,sel_string))
            count +=1
            return
    
    # ----------------------------------------
    # CREATING ANY REMAINING ATOM SELECTIONS...
    elif resname in other:
        sel_string = 'resname %s and resid %d' %(resname,resid)
        temp = analysis_universe.select_atoms(sel_string)
        selection_list.append(temp)
        output_file.write("%5d   COM %5s %s\n" %(count,resname,sel_string))
        count +=1
        return

