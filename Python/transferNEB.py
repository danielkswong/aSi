import csv
import numpy as np
import tarfile
import os
import io
import gzip

# Flag parameters
pushFlag=0
pullFlag=1-pushFlag

# Relevant parameters
T=600
nAtoms=1000
lineStart=9
dataDirectory='../Data_v1/'

# Read CSV transition number file
print('Reading CSV transition file...')

os.chdir(dataDirectory)
transitionNumbersFile='transitionNumbers_T='+str(T)+'_no_minima.csv'
with open(transitionNumbersFile,'r') as csvfile:
    csvreader=csv.reader(csvfile,delimiter=',')
    transitionNumbers=np.asarray(list(csvreader),dtype=int)

# Extract and push position files of NEB dumps to find dynamical matrix
if pushFlag:
    print('Extracting and transferring files...')
    # Pushing the position files
    for i in np.arange(len(transitionNumbers)):
        if os.path.exists('ini.lammps'):
            os.remove('ini.lammps')
        if os.path.exists('ts.lammps'):
            os.remove('ts.lammps')
        if os.path.exists('fin.lammps'):
            os.remove('fin.lammps')
        fileDirectory='dwong@cedar.computecanada.ca:/home/dwong/scratch/Statistics/Equil%03d/Hessian/Equil%03d-%02d' % (transitionNumbers[i,0],transitionNumbers[i,0],transitionNumbers[i,1])

        print('Opening Equil%03d-%02d_NEB.tar.gz' % (transitionNumbers[i,0],transitionNumbers[i,1]))
        NEBFileName='Equil%03d-%02d_NEB.tar.gz' % (transitionNumbers[i,0],transitionNumbers[i,1])
        with tarfile.open(NEBFileName,mode='r:gz') as archive:
            energyFile=archive.extractfile('log.lammps')
            energyFileText=energyFile.readlines()
            energyFileText=[line.decode('UTF-8') for line in energyFileText]
            stringArray=energyFileText[-1].strip().split()
            energyArray=np.asarray(stringArray[10::2],dtype=np.float64)
            ind_max=np.where(energyArray==np.amax(energyArray))[0][0]
            energyFile.close()

            iniFileName='dump.neb.1'
            iniFile=archive.extractfile(iniFileName)
            iniFileText=iniFile.readlines()
            iniFileText=[line.decode('UTF-8') for line in iniFileText]
            positions=iniFileText[-(nAtoms+lineStart):]
            positions[1]='0\n'
            stringArray=positions[8].strip().split()
            stringArray=stringArray[0]+' '+stringArray[1]+' '+stringArray[2]+' '+stringArray[3]+' '+stringArray[7]+' '+stringArray[8]+' '+stringArray[9]+'\n'
            positions[8]=stringArray
            for index in np.arange(lineStart,nAtoms+lineStart):
                stringArray=positions[index].strip().split()
                stringArray=stringArray[0]+' '+stringArray[1]+' '+stringArray[5]+' '+stringArray[6]+' '+stringArray[7]+'\n'
                positions[index]=stringArray
            iniFile.close()
            with open('ini.lammps','w') as f:
                f.write(''.join(positions))
            os.system('scp ini.lammps '+fileDirectory)
            
            tsFileName='dump.neb.'+str(int(ind_max+1))
            tsFile=archive.extractfile(tsFileName)
            tsFileText=tsFile.readlines()
            tsFileText=[line.decode('UTF-8') for line in tsFileText]s
            positions=tsFileText[-(nAtoms+lineStart):]
            positions[1]='0\n'
            stringArray=positions[8].strip().split()
            stringArray=stringArray[0]+' '+stringArray[1]+' '+stringArray[2]+' '+stringArray[3]+' '+stringArray[7]+' '+stringArray[8]+' '+stringArray[9]+'\n'
            positions[8]=stringArray
            for index in np.arange(lineStart,nAtoms+lineStart):
                stringArray=positions[index].strip().split()
                stringArray=stringArray[0]+' '+stringArray[1]+' '+stringArray[5]+' '+stringArray[6]+' '+stringArray[7]+'\n'
                positions[index]=stringArray
            tsFile.close()
            with open('ts.lammps','w') as f:
                f.write(''.join(positions))
            os.system('scp ts.lammps '+fileDirectory)
            
            finFileName='dump.neb.50'
            finFile=archive.extractfile(finFileName)
            finFileText=finFile.readlines()
            finFileText=[line.decode('UTF-8') for line in finFileText]
            positions=finFileText[-(nAtoms+lineStart):]
            positions[1]='0\n'
            stringArray=positions[8].strip().split()
            stringArray=stringArray[0]+' '+stringArray[1]+' '+stringArray[2]+' '+stringArray[3]+' '+stringArray[7]+' '+stringArray[8]+' '+stringArray[9]+'\n'
            positions[8]=stringArray
            for index in np.arange(lineStart,nAtoms+lineStart):
                stringArray=positions[index].strip().split()
                stringArray=stringArray[0]+' '+stringArray[1]+' '+stringArray[5]+' '+stringArray[6]+' '+stringArray[7]+'\n'
                positions[index]=stringArray
            finFile.close()
            with open('fin.lammps','w') as f:
                f.write(''.join(positions))
            os.system('scp in.lammps '+fileDirectory)
            
    # Pushing the CSV file containing the transition numbers
    os.system('scp -i ~/.ssh/id_rsa.pub transitionNumbers_T='+str(T)+'_no_minima.csv dwong@cedar.computecanada.ca:/home/dwong/scratch/Statistics')

# Pull NEB energy barrier files from cluster
if pullFlag:
    print('Pulling files...')
    for i in np.arange(len(transitionNumbers)):
        folderName='Equil%03d-%02d' % (transitionNumbers[i,0],transitionNumbers[i,1])
        fileName='Equil%03d-%02d_Hessian.tar.gz' % (transitionNumbers[i,0],transitionNumbers[i,1])
        print('Pulling '+fileName)
        directoryName='dwong@cedar.computecanada.ca:/home/dwong/scratch/Statistics/Equil%03d/Hessian/Equil%03d-%02d/' % (transitionNumbers[i,0],transitionNumbers[i,0],transitionNumbers[i,1])
        os.system('scp '+directoryName+fileName+' .')

