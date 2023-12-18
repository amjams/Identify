# Automated hyperspectral distinction of biological features in large-scale electron microscopy data
This is the code and data repository of the paper “Automated hyperspectral distinction of biological features in large-scale electron microscopy data”, which is submitted for review. Here, you can find links to view the full EM maps accompanying the paper, code to reproduce data analysis described in the paper, and downloadable EDX-EM data for reuse.

Index to the downloadable data
---------
[nanotomy.org](http://www.nanotomy.org/PW/temp03/Duinkerken2023SUB/index.html)

Description of the downloadable data
---------
1) Raw (.emd): The raw EMD files containing the HSI, which are produced by the microscope.
2) Pre-processes (.npz): The preprocessed (binned/denoised etc.) HSI cubes saved in numpy compressed arrays, which are flattened to 2D. These files contain the arrays that used in most processing in this repository (named spectrum_2D).
3) NPZ raw files (not downloadable): this is an intermediate step of processesing, which involved packaging the .emd file as npz. files, which contain the HSI summed up to a defined number of frames, the EM image, and metadata (xray_energies). These NPZ files are not downloadable, but can be generated from the EMD files using this [file](https://github.com/amjams/Identify/blob/main/scripts/data_preprocessing/emd_to_npz.py).

Links to viewable EM maps
---------
[HAADF](http://www.nanotomy.org/avivator/?image_url=http://www.nanotomy.org/PW/temp03/Duinkerken2023SUB/figures/fig2/HAADF.ome.tiff)

[Colored abundance map](http://www.nanotomy.org/avivator/?image_url=http://www.nanotomy.org/PW/temp03/Duinkerken2023SUB/figures/fig2/Multicolor.ome.tiff)

[Multi-channel abundance maps](http://www.nanotomy.org/avivator/?image_url=http://www.nanotomy.org/PW/temp03/Duinkerken2023SUB/figures/fig2/abundance_maps.ome.tiff)

Note to reviewers
---------
The results of the paper, per figure, can be reproduced using the scripts in this repository as follows:
#### Figure 1
1) The visualization of the HSI can be reproduced using this [file](https://github.com/amjams/Identify/blob/main/scripts/data_visualization/Figure_1A_HSI.ipynb).
2) The endmember extraction and abundance map calculation can reproduced using this [file](https://github.com/amjams/Identify/blob/main/scripts/data_visualization/Figure_1B%2BC_UMAP%2BAbundanceMaps.ipynb).
3) The elemental maps are saved [here](https://github.com/amjams/Identify/tree/main/secondary_data/Figure1_elementmaps) and can be reproduced from this [file](https://github.com/amjams/Identify/blob/main/scripts/data_visualization/Figure_1C_ElementMaps.ipynb). This file uses the HSI raw data, which are not stored in this directory, but can be downloaded from [here](http://www.nanotomy.org/PW/temp03/Duinkerken2023SUB/index.html).
#### Figure 2
1) The stack of UMAP embeddings in the figure can be reproduced using this [file](https://github.com/amjams/Identify/blob/main/scripts/data_visualization/Figure_2A_IterativeEmbeddings.ipynb). Note that this is just an illustration for the figure. To compute all endmember extraction iteration, which are used to for the clustering, use the instruction below.
2) To reproduce the scatter plot from the precomputed embeddings and clusters, use this [file](https://github.com/amjams/Identify/blob/main/scripts/data_visualization/Figure_2A_Clustering.ipynb).
3) To reproduce the heatmap of the endmembers, use this [file](https://github.com/amjams/Identify/blob/main/scripts/data_visualization/Figure_2C_HeatMap.ipynb).
#### Figure 3
- The scripts for reproducing the figure can be found [here](https://github.com/amjams/Identify/tree/main/scripts/data_analysis/segmentation)
#### Figure 4
- This figure, the qdot endmember extraction, can be reproduced using this [file](https://github.com/amjams/Identify/blob/main/scripts/data_visualization/Figure_S1_QDVCA.ipynb).
#### Figure 5
- The colored UMAP embeddings can be reproduced using this [file](https://github.com/amjams/Identify/blob/main/scripts/data_visualization/Figure_S2_ColoredUMAP.ipynb).

References
---------
The endmember extraction algorithm used in this study was adapted from the work of [Vermeulen et al. (2021)](https://www.sciencedirect.com/science/article/abs/pii/S1386142521001232), and using code from the corresponding [repository](https://github.com/NU-ACCESS/UMAP). The implementation of Vertex Component Analysis is from this [repository](https://github.com/Laadr/VCA).

Licensing
---------

Copyright (C) 2023 Ahmad Alsahaf and Peter Duinkerken

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
