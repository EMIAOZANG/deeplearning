#!/bin/bash

#PBS -l nodes=1:ppn=8
#PBS -l walltime=48:00:00
#PBS -l mem=64GB
#PBS -N lstm_gridsearch
#PBS -e mercer:/scratch/jl6583/${PBS_JOBNAME}.e${PBS_JOBID}

#PBS -o mercer:/scratch/jl6583/${PBS_JOBNAME}.o${PBS_JOBID}

module purge
module load torch-deps/7
module load torch/intel/20151009
cd /scratch/jl6583/deeplearning/A3/seq_generator/
th model_tuna.lua
