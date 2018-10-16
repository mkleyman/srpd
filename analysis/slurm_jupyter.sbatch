#!/bin/bash
#SBATCH -p zbj1
#SBATCH --mem=20G
#SBATCH --job-name jupyter_notebook
#SBATCH --output jupyter-log-%J.txt


## get tunneling info
XDG_RUNTIME_DIR=""
ipnport=$(shuf -i8000-9999 -n1)
ipnip=$(hostname -i)

## print tunneling instructions to jupyter-log-{jobid}.txt
echo -e "
    Copy/Paste this in your local terminal to ssh tunnel with remote
    -----------------------------------------------------------------
    ssh -N -L $ipnport:$ipnip:$ipnport eramamur@lanec1.compbio.cs.cmu.edu
    -----------------------------------------------------------------
	
    Then open a browser on your local machine to the following address
    ------------------------------------------------------------------
    localhost:$ipnport  (prefix w/ https:// if using password)
    ------------------------------------------------------------------
    "
	
## start an ipcluster instance and launch jupyter server
cd /projects/pfenninggroup/machineLearningForComputationalBiology/SingleCell/neuron_collab/analysis
jupyter-notebook --NotebookApp.iopub_data_rate_limit=100000000000000 --no-browser --port=$ipnport --ip=$ipnip
