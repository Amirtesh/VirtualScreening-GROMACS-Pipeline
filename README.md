# VirtualScreening-GROMACS-Pipeline

**Usage of Vina_linux.pl and Smina_linux.pl**

Keep a file name ligand.txt containing the names of all the ligand files for virtual screening purpose and conf.txt file containing grid box coordinates, receptor file name and other parameters required for docking with Vina or Smina (The parameters are similar)

Virtual screening with Vina- Run the following command in the directory containing the Vina_linx.pl script file, all ligand files and the ligand.txt file and conf.txt. Enter ligand.txt after running this command to start the process of screening

```
perl Vina_linux.pl
```

Virtual screening with Smina- Run the following command in the directory containing the Vina_linx.pl script file, all ligand files and the ligand.txt file and conf.txt. Enter ligand.txt after running this command to start the process of screening

```
perl Smina_linux.pl
```


**Gromacs_all_codes_complex.txt**

This file contains all the necessary codes to perform simulation of protein-ligand complex with gromacs, perform analysis such as RMSD, RMSF, SASA, Radius of Gyration, Number of H bonds, Free energy landscape and Principal component analysis. It also contains the commands required to perform MMPBSA analysis with gmx_MMPBSA software installed.

**Gromacs_all_codes_protein.txt**

This file contains all the necessary codes to perform simulation of protein with gromacs, perform analysis such as RMSD, RMSF, SASA, Radius of Gyration, Number of H bonds, Free energy landscape and Principal component analysis.

**scatter_plot_xvg.py**

The scatter_plot_xvg.py program is used to plot the PCA_2d.xvg file obtained after performing PCA in scatter plot form, utilizing seaborn library of python

Usage:

```python
python3 scatter_plot_xvg.py PCA_2d.xvg
```
