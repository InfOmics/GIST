# GIST
Leveraging Graph Information for Spatially Informed Patient Data Analysis with GIST
![alt text](image-1.png)



Create an environment if necessary
````
conda create -n gist python==3.10.0 r-base==4.3.1 -y
conda activate gist
````
Verify R home is in the conda environment 
````
which R

````
/home/youruser/anaconda3/envs/gist/lib/R

install requirements
````
pip install -r requirements.txt

````
Install mclust packages
````
Rscript -e 'install.packages("mclust", repos="https://cran.r-project.org", type="source")'


````
install GIST packages
````
pip install git+https://github.com/gospelnnadi/GIST.git

````

run_GIST.py  contains the GIST pipeline. 
python run_GIST.py &> output.log

### Data Availability ###
The spatial transcriptomics datasets are available at:  https://doi.org/10.5281/zenodo.15277298

### Citation

G. O. Nnadi, V. Bonnici, S. Avesani, E. Viesi and R. Giugno, "Leveraging Graph Information for Spatially Informed Patient Data Analysis with GIST," 2025 IEEE Conference on Computational Intelligence in Bioinformatics and Computational Biology (CIBCB), Tainan, Taiwan, 2025, pp. 1-8, doi: 10.1109/CIBCB66090.2025.11177089. keywords: {Measurement;Computational modeling;Transcriptomics;Computer architecture;Contrastive learning;Brain modeling;Spatial databases;Graph neural networks;Indexes;Gene expression;Domain identification;Graph representation;Spatial transcriptomics}.

