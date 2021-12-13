#!/bin/bash
#SBATCH --array=0-999
#SBATCH --time=48:00:00
#SBATCH --account=___
#SBATCH --mail-user=___
#SBATCH --mail-type=ALL
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=2048M

if [[ "$PATH" =~ (^|:)"___"(|/)(:|$) ]]; then
	echo "LAMMPS included"
else
	export PATH=$PATH:/home/dwong/projects/def-rottler/dwong/LAMMPS/src
fi

nTrajectories=10
equilNum=$((SLURM_ARRAY_TASK_ID/nTrajectories+1))
trajectoryNum=$((SLURM_ARRAY_TASK_ID-(equilNum-1)*nTrajectories+1))

dir=$(printf "Equil%03d/Quench/%03d-%02d" $equilNum $equilNum $trajectoryNum)
cd $dir

srun lmp_mpi -in in.quench.lmp

fileName=${PWD/*\//}
rm *.tar.gz
tar -czvf $fileName".tar.gz" quench_*
rm quench_*
