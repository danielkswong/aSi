import os
import shutil
import csv
import numpy as np

# Flags
preEquilFlag=0
postEquilFlag=0
coolFlag=0
quenchFlag=1
nebFlag=0
hessianFlag=0

# Parameters
nSamples=100
nTrajectories=100
nPreEquil=1000000
nPostEquil=nPreEquil
nCool=3500000
T_i=3500
T_f=0.001
T_search=600
cutoff=2.75
N1=100000
N2=100000
nPartitions=50
potential='tersoff'

# Writes LAMMPS scripts
def writePreEquilibrate(nPreEquil,T_i,seed):
    with open('in.preequilibrate.lmp','w') as f:
        f.write('# 3d amorphous silicon (pre-equilibrate)\n')
        f.write('# T_i = '+str(T_i)+'\n')
        f.write('# nPreEquil = '+str(nPreEquil)+'\n')
        f.write('# seed = '+str(seed)+'\n')
        f.write('\n')
        f.write('variable T_i equal '+str(T_i)+'\n')
        f.write('variable nPreEquil equal '+str(nPreEquil)+'\n')
        f.write('variable nRDF equal 1000\n')
        f.write('variable nMSD equal 1000\n')
        f.write('variable nThermo equal 1000\n')
        f.write('variable rdfBins equal 100\n')
        f.write('\n')
        f.write('log log.preequil.lmp\n')
        f.write('\n')
        f.write('dimension 3\n')
        f.write('units metal\n')
        f.write('boundary p p p\n')
        f.write('atom_style atomic\n')
        f.write('atom_modify map array sort 0 0.0\n')
        f.write('\n')
        f.write('lattice diamond 5.465517665\n')
        f.write('region box block 0 5 0 5 0 5 units lattice\n')
        f.write('create_box 1 box\n')
        f.write('create_atoms 1 box\n')
        f.write('mass 1 28.0855\n')
        f.write('\n')
        f.write('velocity all create ${T_i} '+str(seed)+'\n')
        f.write('\n')
        f.write('pair_style '+potential+'\n')
        f.write('pair_coeff * * Si.'+potential+' Si\n')
        f.write('\n')
        f.write('neighbor 2.0 bin\n')
        f.write('neigh_modify every 1 delay 0 check no\n')
        f.write('\n')
        f.write('dump data all custom 1000 preequil_pos.lammps id type x y z vx vy vz\n')
        f.write('\n')
        f.write('thermo ${nThermo}\n')
        f.write('thermo_style custom step temp epair emol pe ke etotal press vol\n')
        f.write('\n')
        f.write('comm_modify cutoff 20\n')
        f.write('compute RDF all rdf ${rdfBins} cutoff 10\n')
        f.write('compute MSD all msd\n')
        f.write('variable volume equal vol\n')
        f.write('fix 1 all ave/time 1 ${nThermo} ${nThermo} c_thermo_temp c_thermo_pe c_thermo_press v_volume file preequil_thermo.txt\n')
        f.write('fix 2 all ave/time 1 ${nRDF} ${nRDF} c_RDF[*] file preequil_rdf.txt mode vector\n')
        f.write('fix 3 all ave/time 1 ${nMSD} ${nMSD} c_MSD[*] file preequil_msd.txt\n')
        f.write('fix equil all nvt temp ${T_i} ${T_i} $(100.0*dt)\n')
        f.write('run ${nPreEquil}\n')
        f.write('\n')
        f.write('unfix 1\n')
        f.write('unfix 2\n')
        f.write('unfix 3\n')
        f.write('unfix equil\n')
        f.write('\n')
        f.write('write_restart restart.preequil')

def writeCool(nCool,T_i,T_f):
    with open('in.cool.lmp','w') as f:
        f.write('# 3d amorphous silicon (cool)\n')
        f.write('# T_i = '+str(T_i)+'\n')
        f.write('# T_f = '+str(T_f)+'\n')
        f.write('# n = '+str(nCool)+'\n')
        f.write('\n')
        f.write('variable T_i equal '+str(T_i)+'\n')
        f.write('variable T_f equal '+str(T_f)+'\n')
        f.write('variable nCool equal '+str(nCool)+'\n')
        f.write('variable nRDF equal 1000\n')
        f.write('variable nThermo equal 1000\n')
        f.write('variable rdfBins equal 100\n')
        f.write('\n')
        f.write('log log.cool.lmp\n')
        f.write('\n')
        f.write('dimension 3\n')
        f.write('units metal\n')
        f.write('boundary p p p\n')
        f.write('atom_style atomic\n')
        f.write('atom_modify map array sort 0 0.0\n')
        f.write('\n')
        f.write('lattice diamond 5.465517665\n')
        f.write('region box block 0 5 0 5 0 5 units lattice\n')
        f.write('\n')
        f.write('read_restart restart.preequil\n')
        f.write('\n')
        f.write('pair_style '+potential+'\n')
        f.write('pair_coeff * * Si.'+potential+' Si\n')
        f.write('\n')
        f.write('neighbor 2.0 bin\n')
        f.write('neigh_modify every 1 delay 0 check no\n')
        f.write('\n')
        f.write('dump data all custom 1000 cool_pos.lammps id type x y z vx vy vz\n')
        f.write('\n')
        f.write('thermo ${nThermo}\n')
        f.write('thermo_style custom step temp epair emol pe ke etotal press vol\n')
        f.write('\n')
        f.write('comm_modify cutoff 20\n')
        f.write('compute RDF all rdf ${rdfBins} cutoff 10\n')
        f.write('variable volume equal vol\n')
        f.write('fix 1 all ave/time 1 ${nThermo} ${nThermo} c_thermo_temp c_thermo_pe c_thermo_press v_volume file cool_thermo.txt\n')
        f.write('fix 2 all ave/time 1 ${nRDF} ${nRDF} c_RDF[*] file cool_rdf.txt mode vector\n')
        f.write('fix cool all nvt temp ${T_i} ${T_f} $(100.0*dt)\n')
        f.write('run ${nCool}\n')
        f.write('\n')
        f.write('unfix 1\n')
        f.write('unfix 2\n')
        f.write('unfix cool\n')
        f.write('\n')
        f.write('write_restart restart.cool')

def writePostEquilibrate(nPostEquil,nCool,T_i):
    with open('in.postequilibrate.lmp','w') as f:
        f.write('# 3d amorphous silicon (post-equilibrate)\n')
        f.write('# T_i = '+str(T_i)+'\n')
        f.write('# nPostEquil = '+str(nPostEquil)+'\n')
        f.write('\n')
        f.write('variable T_i equal '+str(T_i)+'\n')
        f.write('variable nPostEquil equal '+str(nPostEquil)+'\n')
        f.write('variable nCool equal '+str(nCool)+'\n')
        f.write('variable nRDF equal 1000\n')
        f.write('variable nMSD equal 1000\n')
        f.write('variable nThermo equal 1000\n')
        f.write('variable rdfBins equal 100\n')
        f.write('\n')
        f.write('log log.postequil.lmp\n')
        f.write('\n')
        f.write('dimension 3\n')
        f.write('units metal\n')
        f.write('boundary p p p\n')
        f.write('atom_style atomic\n')
        f.write('atom_modify map array sort 0 0.0\n')
        f.write('\n')
        f.write('lattice diamond 5.465517665\n')
        f.write('region box block 0 5 0 5 0 5 units lattice\n')
        f.write('\n')
        f.write('read_restart restart.cool\n')
        f.write('\n')
        f.write('pair_style '+potential+'\n')
        f.write('pair_coeff * * Si.'+potential+' Si\n')
        f.write('\n')
        f.write('neighbor 2.0 bin\n')
        f.write('neigh_modify every 1 delay 0 check no\n')
        f.write('\n')
        f.write('dump data all custom 1000 postequil_pos.lammps id type x y z vx vy vz\n')
        f.write('\n')
        f.write('thermo ${nThermo}\n')
        f.write('thermo_style custom step temp epair emol pe ke etotal press vol\n')
        f.write('\n')
        f.write('comm_modify cutoff 20\n')
        f.write('compute RDF all rdf ${rdfBins} cutoff 10\n')
        f.write('compute MSD all msd\n')
        f.write('variable volume equal vol\n')
        f.write('fix 1 all ave/time 1 ${nThermo} ${nThermo} c_thermo_temp c_thermo_pe c_thermo_press v_volume file postequil_thermo.txt\n')
        f.write('fix 2 all ave/time 1 ${nRDF} ${nRDF} c_RDF[*] file postequil_rdf.txt mode vector\n')
        f.write('fix 3 all ave/time 1 ${nMSD} ${nMSD} c_MSD[*] file postequil_msd.txt\n')
        f.write('fix equil all nvt temp ${T_i} ${T_i} $(100.0*dt)\n')
        f.write('run ${nPostEquil}\n')
        f.write('\n')
        f.write('unfix 1\n')
        f.write('unfix 2\n')
        f.write('unfix 3\n')
        f.write('unfix equil\n')
        f.write('\n')
        f.write('write_restart restart.postequil')

def writeQuench(nPreEquil,nCool,nPostEquil,T_search,seed,cutoff):
    with open('in.quench.lmp','w') as f:
        f.write('# 3d amorphous siicon (quench)\n')
        f.write('# nPreEquil = '+str(nPreEquil)+'\n')
        f.write('# nCool = '+str(nCool)+'\n')
        f.write('# nPostEquil = '+str(nPostEquil)+'\n')
        f.write('# T_search = '+str(T_search)+'\n')
        f.write('# seed = '+str(seed)+'\n')
        f.write('# cutoff = '+str(cutoff)+'\n')
        f.write('\n')
        f.write('variable nPreEquil equal '+str(nPreEquil)+'\n')
        f.write('variable nCool equal '+str(nCool)+'\n')
        f.write('variable nPostEquil equal '+str(nPostEquil)+'\n')
        f.write('variable nSearch equal 10\n')
        f.write('variable nQuench equal 10000\n')
        f.write('variable nThermo equal 50\n')
        f.write('variable T_search equal '+str(T_search)+'\n')
        f.write('variable E_tol equal 0.0\n')
        f.write('variable F_tol equal 1.0e-6\n')
        f.write('\n')
        f.write('log log.quench.lmp\n')
        f.write('\n')
        f.write('dimension 3\n')
        f.write('units metal\n')
        f.write('boundary p p p\n')
        f.write('atom_style atomic\n')
        f.write('atom_modify map array sort 0 0.0\n')
        f.write('\n')
        f.write('lattice diamond 5.465517665\n')
        f.write('region box block 0 5 0 5 0 5 units lattice\n')
        f.write('\n')
        f.write('read_restart restart.postequil\n')
        f.write('\n')
        f.write('velocity all create ${T_search} '+str(seed)+'\n')
        f.write('\n')
        f.write('pair_style '+potential+'\n')
        f.write('pair_coeff * * Si.'+potential+' Si\n')
        f.write('\n')
        f.write('neighbor 2.0 bin\n')
        f.write('neigh_modify every 1 delay 0 check no\n')
        f.write('\n')
        f.write('compute coord all coord/atom cutoff '+str(cutoff)+'\n')
        f.write('variable coord1 equal c_coord[1]\n')
        f.write('\n')
        f.write('thermo ${nThermo}\n')
        f.write('thermo_style custom step temp epair emol etotal press pe v_coord1\n')
        f.write('\n')
        f.write('write_dump all custom temp.pos id type xu yu zu vx vy vz\n')
        f.write('minimize ${E_tol} ${F_tol} ${nQuench} ${nQuench}\n')
        f.write('write_dump all custom quench_coord.lammps id type c_coord modify sort id\n')
        f.write('write_dump all custom quench_pos.lammps id type x y z xu yu zu modify sort id\n')
        f.write('print $(pe) file quench_energies.lammps\n')
        f.write('\n')
        f.write('fix search all nvt temp ${T_search} ${T_search} $(100.0*dt)\n')
        f.write('label loop_i\n')
        f.write('variable i loop 2000\n')
        f.write('\t'+'variable current_time equal ${nPreEquil}+${nCool}+${nPostEquil}+($i-1)*${nSearch}\n')
        f.write('\t'+'read_dump temp.pos ${current_time} x y z vx vy vz wrapped no\n')
        f.write('\t'+'run ${nSearch}\n')
        f.write('\t'+'write_dump all custom temp.pos id type xu yu zu vx vy vz\n')
        f.write('\n')
        f.write('\t'+'minimize ${E_tol} ${F_tol} ${nQuench} ${nQuench}\n')
        f.write('\t'+'write_dump all custom quench_coord.lammps id type c_coord modify sort id append yes\n')
        f.write('\t'+'write_dump all custom quench_pos.lammps id type x y z xu yu zu modify sort id append yes\n')
        f.write('\t'+'print $(pe) append quench_energies.lammps\n')
        f.write('next i\n')
        f.write('jump SELF loop_i\n')
        f.write('\n')
        f.write('unfix search')

def writeNEB(N1,N2,nPartitions):
    with open('in.neb.lmp','w') as f:
        f.write('# 3d amorphous silicon (NEB)\n')
        f.write('# N1 = '+str(N1)+'\n')
        f.write('# N2 = '+str(N2)+'\n')
        f.write('# nPartitions = '+str(nPartitions)+'\n')
        f.write('\n')
        f.write('log log.neb.lmp\n')
        f.write('\n')
        f.write('dimension 3\n')
        f.write('units metal\n')
        f.write('boundary p p p\n')
        f.write('atom_style atomic\n')
        f.write('atom_modify map array sort 0 0.0\n')
        f.write('\n')
        f.write('lattice diamond 5.465517665\n')
        f.write('region box block 0 5 0 5 0 5 units lattice\n')
        f.write('\n')
        f.write('create_box 1 box\n')
        f.write('create_atoms 1 box\n')
        f.write('mass 1 28.0855\n')
        f.write('\n')
        f.write('variable nThermo equal 1000\n')
        f.write('variable N1 equal '+str(N1)+'\n')
        f.write('variable N2 equal '+str(N2)+'\n')
        f.write('variable Nevery equal 50\n')
        f.write('variable nPartitions equal '+str(nPartitions)+'\n')
        f.write('variable E_tol equal 0.0\n')
        f.write('variable F_tol equal 1.0e-6\n')
        f.write('\n')
        f.write('velocity all create 600 1\n')
        f.write('\n')
        f.write('pair_style '+potential+'\n')
        f.write('pair_coeff * * Si.'+potential+' Si\n')
        f.write('\n')
        f.write('neighbor 2.0 bin\n')
        f.write('neigh_modify every 1 delay 0 check no\n')
        f.write('\n')
        f.write('thermo ${nThermo}\n')
        f.write('\n')
        f.write('fix 1 all neb 1.0\n')
        f.write('partition yes 1 fix 2 all setforce 0.0 0.0 0.0\n')#
        f.write('partition yes ${nPartitions} fix 2 all setforce 0.0 0.0 0.0\n')#
        f.write('min_style quickmin\n')
        f.write('\n')
        f.write('variable u uloop ${nPartitions}\n')
        f.write('\n')
        f.write('dump replicas all custom 1000 dump.neb.$u id type x y z xu yu zu\n')
        f.write('read_dump pre.lammps 0 x y z wrapped no\n')
        f.write('neb ${E_tol} ${F_tol} ${N1} ${N2} ${Nevery} final post.lammps')

def writeHessian():
    with open('in.hessian.lmp','w') as f:
        f.write('# 3d amorphous silicon (Hessian)\n')
        f.write('\n')
        f.write('variable gamma equal 0.000001\n')
        f.write('variable nThermo equal 1000\n')
        f.write('\n')
        f.write('log log.hessian.lmp\n')
        f.write('\n')
        f.write('dimension 3\n')
        f.write('units metal\n')
        f.write('boundary p p p\n')
        f.write('atom_style atomic\n')
        f.write('atom_modify map array sort 0 0.0\n')
        f.write('\n')
        f.write('lattice diamond 5.465517665\n')
        f.write('region box block 0 5 0 5 0 5 units lattice\n')
        f.write('\n')
        f.write('create_box 1 box\n')
        f.write('create_atoms 1 box\n')
        f.write('mass 1 28.0855\n')
        f.write('\n')
        f.write('pair_style '+potential+'\n')
        f.write('pair_coeff * * Si.'+potential+' Si\n')
        f.write('\n')
        f.write('neighbor 2.0 bin\n')
        f.write('neigh_modify every 1 delay 0 check no\n')
        f.write('\n')
        f.write('thermo ${nThermo}\n')
        f.write('thermo_style custom press pxx pxy pxz pyy pyz pzz vol\n')
        f.write('\n')
        f.write('read_dump ini.lammps 0 x y z wrapped no\n')
        f.write('dynamical_matrix all regular ${gamma} file ini.dat\n')
        f.write('run 0\n')
        f.write('print $(press) file ini_press.dat\n')
        f.write('print $(pxx) append ini_press.dat\n')
        f.write('print $(pxy) append ini_press.dat\n')
        f.write('print $(pxz) append ini_press.dat\n')
        f.write('print $(pyy) append ini_press.dat\n')
        f.write('print $(pyz) append ini_press.dat\n')
        f.write('print $(pzz) append ini_press.dat\n')
        f.write('print $(vol) append ini_press.dat\n')
        f.write('\n')
        f.write('read_dump ts.lammps 0 x y z wrapped no\n')
        f.write('dynamical_matrix all regular ${gamma} file ts.dat\n')
        f.write('run 0\n')
        f.write('print $(press) file ts_press.dat\n')
        f.write('print $(pxx) append ts_press.dat\n')
        f.write('print $(pxy) append ts_press.dat\n')
        f.write('print $(pxz) append ts_press.dat\n')
        f.write('print $(pyy) append ts_press.dat\n')
        f.write('print $(pyz) append ts_press.dat\n')
        f.write('print $(pzz) append ts_press.dat\n')
        f.write('print $(vol) append ts_press.dat\n')
        f.write('\n')
        f.write('read_dump fin.lammps 0 x y z wrapped no\n')
        f.write('dynamical_matrix all regular ${gamma} file fin.dat\n')
        f.write('run 0\n')
        f.write('print $(press) file fin_press.dat\n')
        f.write('print $(pxx) append fin_press.dat\n')
        f.write('print $(pxy) append fin_press.dat\n')
        f.write('print $(pxz) append fin_press.dat\n')
        f.write('print $(pyy) append fin_press.dat\n')
        f.write('print $(pyz) append fin_press.dat\n')
        f.write('print $(pzz) append fin_press.dat\n')
        f.write('print $(vol) append fin_press.dat\n')

if preEquilFlag:
    for i in range(1,nSamples+1):
        folder='Equil%03d' % (i)
        if not os.path.isdir(folder):
            os.mkdir(folder)
        os.chdir(folder)
        if not os.path.isdir('PreEquil'):
            os.mkdir('PreEquil')
        os.chdir('PreEquil')
        writePreEquilibrate(nPreEquil,T_i,i)
        os.chdir('../../')
        shutil.copy('Si.'+potential,folder+'/PreEquil/')

if coolFlag:
    for i in range(1,nSamples+1):
        folder='Equil%03d' % (i)
        os.chdir(folder)
        if not os.path.isdir('Cool'):
            os.mkdir('Cool')
        os.chdir('Cool')
        writeCool(nCool,T_i,T_f)
        os.chdir('../PreEquil/')
        shutil.copy('restart.preequil','../Cool/')
        shutil.copy('Si.'+potential,'../Cool/')
        os.chdir('../../')

if postEquilFlag:
    for i in range(1,nSamples+1):
        folder='Equil%03d' % (i)
        os.chdir(folder)
        if not os.path.isdir('PostEquil'):
            os.mkdir('PostEquil')
        os.chdir('PostEquil')
        writePostEquilibrate(nPostEquil,nCool,T_f)
        os.chdir('../Cool/')
        shutil.copy('restart.cool','../PostEquil/')
        shutil.copy('Si.'+potential,'../PostEquil')
        os.chdir('../../')

if quenchFlag:
    for i in range(1,nSamples+1):
        folder='Equil%03d' % (i)
        os.chdir(folder)
        if not os.path.isdir('Quench'):
            os.mkdir('Quench')
        os.chdir('Quench')
        for j in range(1,nTrajectories+1):
            trajectoryFolder='%03d-%03d' % (i,j)
            if not os.path.isdir(trajectoryFolder):
                os.mkdir(trajectoryFolder)
            os.chdir(trajectoryFolder)
            writeQuench(nPreEquil,nCool,nPostEquil,T_search,j,cutoff)
            os.chdir('..')
        os.chdir('../PostEquil/')
        for j in range(1,nTrajectories+1):
            trajectoryFolder='%03d-%03d' % (i,j)
            shutil.copy('restart.postequil','../Quench/'+trajectoryFolder)
            shutil.copy('Si.'+potential,'../Quench/'+trajectoryFolder)
        os.chdir('../../')

if nebFlag:
    transitionNumbersFile='transitionNumbers_T='+str(T_search)+'_filter1.csv'
    with open(transitionNumbersFile,'r') as csvfile:
        csvreader=csv.reader(csvfile,delimiter=',')
        transitionNumbers=np.asarray(list(csvreader),dtype=int)

    for i in range(len(transitionNumbers)):
        folder='Equil%03d' % (transitionNumbers[i,0])
        os.chdir(folder)
        if not os.path.isdir('NEB'):
            os.mkdir('NEB')
        os.chdir('NEB')
        trajectoryFolder='%03d-%03d' % (transitionNumbers[i,0],transitionNumbers[i,1])
        if not os.path.isdir(trajectoryFolder):
            os.mkdir(trajectoryFolder)
        os.chdir(trajectoryFolder)
        writeNEB(N1,N2,nPartitions)
        os.chdir('../../Cool/')
        shutil.copy('Si.'+potential,'../NEB/'+trajectoryFolder)
        os.chdir('../../')

if hessianFlag:
    transitionNumbersFile='transitionNumbers_T='+str(T_search)+'_no_minima.csv'
    with open(transitionNumbersFile,'r') as csvfile:
        csvreader=csv.reader(csvfile,delimiter=',')
        transitionNumbers=np.asarray(list(csvreader),dtype=int)

    for i in range(len(transitionNumbers)):
        folder='Equil%03d' % (transitionNumbers[i,0])
        os.chdir(folder)
        if not os.path.isdir('Hessian'):
            os.mkdir('Hessian')
        os.chdir('Hessian')
        trajectoryFolder='%03d-%03d' % (transitionNumbers[i,0],transitionNumbers[i,1])
        if not os.path.isdir(trajectoryFolder):
            os.mkdir(trajectoryFolder)
        os.chdir(trajectoryFolder)
        writeHessian()
        os.chdir('../../Cool/')
        shutil.copy('Si.'+potential,'../Hessian/'+trajectoryFolder)
        os.chdir('../../')