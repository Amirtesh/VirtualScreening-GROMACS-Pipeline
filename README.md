# VirtualScreening-GROMACS-Pipeline

**Usage of Vina_linux.pl and Smina_linux.pl**

Keep a file name ligand.txt containing the names of all the ligand files for virtual screening purpose and conf.txt file containing grid box coordinates, receptor file name and other parameters required for docking with Vina or Smina (The parameters are similar)

## Virtual screening with Vina- 
Run the following command in the directory containing the Vina_linx.pl script file, all ligand files and the ligand.txt file and conf.txt. Enter ligand.txt after running this command to start the process of screening

```
perl Vina_linux.pl
```

## Virtual screening with Smina- 
Run the following command in the directory containing the Smina_linx.pl script file, all ligand files and the ligand.txt file and conf.txt. Enter ligand.txt after running this command to start the process of screening

```
perl Smina_linux.pl
```

## Virtual screening with QVina-
Run the following command in the directory containing the QVina_linx.pl script file, all ligand files and the ligand.txt file and conf.txt. Enter ligand.txt after running this command to start the process of screening

```
perl QVina_linux.pl
```


## Gromacs_all_codes_complex.txt

This file contains all the necessary codes to perform simulation of protein-ligand complex with gromacs, perform analysis such as RMSD, RMSF, SASA, Radius of Gyration, Number of H bonds, Minimum Distance, Free energy landscape and Principal component analysis. It also contains the commands required to perform MMPBSA analysis with gmx_MMPBSA software installed.

## Gromacs_all_codes_protein.txt

This file contains all the necessary codes to perform simulation of protein with gromacs, perform analysis such as RMSD, RMSF, SASA, Radius of Gyration, Number of H bonds, Free energy landscape and Principal component analysis.

## GromacsGUI.py

A PyQT5 based gui for analysis of gromacs trajectories after simulation. Allows centering of trajectory, running of rmsd, rmsf, gyration, sasa, HBond, Mindist, PCA and FEL analysis. The plot multiple files window can be used to open multiple xvg files and plot together, with options to save images. The 3d FEL window can be used to plot Gibbs Free Energy Landscape plots in both 2D and 3D. Inidividual scripts are also available without the use of a GUI.

Usage:

```python
python3 GromacsGUI.py
```

## scatter_plot_xvg.py

The scatter_plot_xvg.py program is used to plot the PCA_2d.xvg file obtained after performing PCA in scatter plot form, utilizing seaborn library of python

Usage:

```python
python3 scatter_plot_xvg.py PCA_2d.xvg
```

## plotXVG.py

Alternative to xmgrace to plot output xvg files from gromacs. Supports single file and mulitple files, with pressing of 's' to save image in 600dpi (ignore the save pop up after clicking 's'). Legends can be changed manually also after opening.

Usage:

```python
python3 plotXVG.py file1.xvg file2.xvg
```

## plot3DFEL.py

Alternative to eps file output from PCA analysis in gromacs. Works with the xpm file (PCA_sham.xpm) generated. Plots both the regular 2D plot and a 3D plot side by side and 3D plot can be rotated without affecting the 2D plot

Usage:

```python
python3 plot3DFEL.py PCA_sham.xpm
```
