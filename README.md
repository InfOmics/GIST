# GIST
Leveraging Graph Information for Spatially Informed Patient Data Analysis with GIST

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
Rscript -e 'install.packages("mclust",repos="http://cran.us.r-project.org")'

````
install GIST packages
````
pip install git+https://github.com/gospelnnadi/GIST.git

````

run_GIST.py  contains the GIST pipeline. 
python run_GIST.py &> output.log

### Data Availability ###
The spatial transcriptomics datasets are available at:  https://doi.org/10.5281/zenodo.15277298

