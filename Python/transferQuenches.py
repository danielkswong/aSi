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
dataDirectory='../Data_v1'

# Read CSV transition number file
print('Reading CSV transition file...')

os.chdir(dataDirectory)
transitionNumbersFile='transitionNumbers_T='+str(T)+'_filter1.csv'
with open(transitionNumbersFile,'r') as csvfile:
    csvreader=csv.reader(csvfile,delimiter=',')
    transitionNumbers=np.asarray(list(csvreader),dtype=int)

# Extract and push position files for NEB algorithm on cluster
if pushFlag:
    print('Extracting and transferring files...')
    
    # Push position files
    for i in np.arange(len(transitionNumbers)):
        if os.path.exists('pre.lammps'):
            os.remove('pre.lammps')
        if os.path.exists('post.lammps'):
            os.remove('post.lammps')
        
        print('Pushing position files from Equil%03d-%d.tar.gz' % (transitionNumbers[i,0],transitionNumbers[i,1]))
        with tarfile.open('Equil%03d-%d.tar.gz' % (transitionNumbers[i,0],transitionNumbers[i,1])) as archive:
            fileDirectory='dwong@cedar.computecanada.ca:/home/dwong/scratch/Statistics/Equil%03d/Quench/Equil%03d-%d' % (transitionNumbers[i,0],transitionNumbers[i,0],transitionNumbers[i,1])

            preFileName='quench_positions_T='+str(T)+'_i='+str(transitionNumbers[i,2])+'.lammps.gz'
            quenchFile1=archive.extractfile(preFileName)
            byteInfo=io.BytesIO(quenchFile1.read())
            positionFile=gzip.GzipFile(fileobj=byteInfo)
            fileText=positionFile.readlines()
            fileText=[line.decode('UTF-8') for line in fileText]
            fileText[1]='0\n'
            with open('pre.lammps','w') as f:
                f.write(''.join(fileText))
                
            positionFile.close()
            quenchFile1.close()
            os.system('scp -i ~/.ssh/id_rsa.pub pre.lammps '+fileDirectory)

            postFileName='quench_positions_T='+str(T)+'_i='+str(transitionNumbers[i,2]+1)+'.lammps.gz'
            quenchFile2=archive.extractfile(postFileName)
            byteInfo=io.BytesIO(quenchFile2.read())
            positionFile=gzip.GzipFile(fileobj=byteInfo)

            with open('post.lammps','w') as f:
                f.write(str(nAtoms)+'\n')
                for j in np.arange(lineStart):
                    positionFile.readline()
                for atomNumber in np.arange(nAtoms):
                    line=positionFile.readline().decode('UTF-8')
                    stringArray=line.strip().split(' ')
                    f.write(str(atomNumber+1)+' '+stringArray[2]+' '+stringArray[3]+' '+stringArray[4]+'\n')
            
            positionFile.close()
            quenchFile2.close()
            os.system('scp -i ~/.ssh/id_rsa.pub post.lammps '+fileDirectory)

    # Push CSV file containing the transition numbers of quenches
    os.system('scp -i ~/.ssh/id_rsa.pub transitionNumbers_T='+str(T)+'_filter1.csv dwong@cedar.computecanada.ca:/home/dwong/scratch/Statistics')

# Pull NEB energy barrier files from cluster
if pullFlag:
    print('Pulling files from Cedar...')

    for i in np.arange(len(transitionNumbers)):
        fileName='Equil%03d-%02d_NEB.tar.gz' % (transitionNumbers[i,0],transitionNumbers[i,1])
        print('Pulling '+fileName)
        directoryName='dwong@cedar.computecanada.ca:/home/dwong/scratch/Statistics/Equil%03d/Quench/Equil%03d-%02d/' % (transitionNumbers[i,0],transitionNumbers[i,0],transitionNumbers[i,1])
        os.system('scp -i ~/.ssh/id_rsa.pub '+directoryName+fileName+' ./Equil%03d-%02d_NEB.tar.gz' % (transitionNumbers[i,0],transitionNumbers[i,1]))

