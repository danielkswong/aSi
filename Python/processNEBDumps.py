import tarfile
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as manimation
import math
import csv

# Flag parameters
readFlag=1
writeFlag=1-readFlag
movieFlag=0
plotFlag=1

# Relevant parameters
T=600
nPartitions=50
nAtoms=1000
nColumns=6
screenX=13.66
screenY=7.68
minZScore=1
dataDirectory='../Data_v1/'
plotDirectory='../Plots_v1/'
cutoff=2.75

# Determines nearest neighbours
def findNearestNeighbours(pos,cutoff):
    nn=np.empty(len(pos),dtype=object)
    coord=np.zeros(len(pos),dtype=int)
    for i in range(len(pos)):
        d_i=pos[i,3:]
        nn_i=[]
        for j in range(len(pos)):
            if i==j:
                continue
            else:
                d_j=pos[j,3:]
                d=math.sqrt(sum((d_i-d_j)**2))
            if d <= cutoff:
                nn_i.append(j)
        nn[i]=nn_i
        coord[i]=len(nn_i)
    return nn,coord

# Reads in NEB files
if writeFlag:
    os.chdir(dataDirectory)
    
    with open('maxTLSNumbers.csv','r') as csvfile:
        csvreader=csv.reader(csvfile,delimiter=',')
        maxTLSNumbers=np.asarray(list(csvreader),dtype=np.float64)

    NEBFiles=[]
    for entry in maxTLSNumbers:
        fileName='Equil%03d-%02d_NEB.tar.gz' % (entry[0],entry[1])
        NEBFiles.append(fileName)
    nFiles=len(NEBFiles)
    NEBFiles.sort()

    atomicPositionArray=np.zeros((nFiles,nPartitions,nAtoms,nColumns),dtype=np.float64)
    atomicEnergyArray=np.zeros((nFiles,nPartitions),dtype=np.float64)
    atomicZScoreArray=np.zeros((nFiles,nAtoms),dtype=np.float64)
    
    for i in range(nFiles):
        print('Reading '+NEBFiles[i]+'...')
        with tarfile.open(NEBFiles[i],mode='r:gz') as archive:
            files=archive.getnames()
            if len(files)<(nPartitions+1):
                raise Exception('Missing dump and/or NEB log files in .tar.gz')

            for j in np.linspace(1,50,dtype=int):
                dumpFile=archive.extractfile('dump.neb.%d' %j)
                dumpFileText=dumpFile.readlines()
                dumpFileText=[line.decode('UTF-8') for line in dumpFileText[-(nAtoms):]]
                dumpFile.close()

                for lineNumber in range(nAtoms):
                    line=dumpFileText[lineNumber]
                    stringArray=np.asarray(line.strip().split(),dtype=np.float64)
                    atomicPositionArray[i,j-1,lineNumber,:]=stringArray[2:]

            energyFile=archive.extractfile('log.lammps')
            energyFileText=energyFile.readlines()
            energyFile.close()

            energyArray=np.asarray(energyFileText[-1].strip().split(),dtype=np.float64)[10::2]
            atomicEnergyArray[i]=energyArray-min(energyArray[0],energyArray[-1])         

    consDisp=atomicPositionArray[:,1:]-atomicPositionArray[:,0:-1]
    netConsDisp2=consDisp[:,:,:,3]**2+consDisp[:,:,:,4]**2+consDisp[:,:,:,5]**2
    totalConsDisp2=np.sum(netConsDisp2,axis=1)
    meanTotalConsDisp2=np.mean(totalConsDisp2,axis=1)
    stdTotalConsDisp2=np.std(totalConsDisp2,axis=1)

    for i in np.arange(nFiles):
        atomicZScoreArray[i]=(totalConsDisp2[i]-meanTotalConsDisp2[i])/stdTotalConsDisp2[i]
    
    # Writes energy barriers and transition numbers as CSV files
    print('Saving NEB data as CSV files...')
    
    np.savetxt('atomicPositionArray.csv',np.reshape(atomicPositionArray,(nFiles,-1)),delimiter=',')
    np.savetxt('atomicEnergyArray.csv',atomicEnergyArray,delimiter=',')
    np.savetxt('atomicZScoreArray.csv',atomicZScoreArray,delimiter=',')

# Reads data if CSV file is located
if readFlag:
    print('Reading CSV file...')

    os.chdir(dataDirectory)
    with open('maxTLSNumbers.csv','r') as csvfile:
        csvreader=csv.reader(csvfile,delimiter=',')
        maxTLSNumbers=np.asarray(list(csvreader),dtype=np.float64)
    with open('atomicPositionArray.csv','r') as csvfile:
        csvreader=csv.reader(csvfile,delimiter=',')
        atomicPositionArray=np.asarray(list(csvreader),dtype=np.float64)
    with open('atomicEnergyArray.csv','r') as csvfile:
        csvreader=csv.reader(csvfile,delimiter=',')
        atomicEnergyArray=np.asarray(list(csvreader),dtype=np.float64)
    with open('atomicZScoreArray.csv','r') as csvfile:
        csvreader=csv.reader(csvfile,delimiter=',')
        atomicZScoreArray=np.asarray(list(csvreader),dtype=np.float64)

    nFiles=len(maxTLSNumbers)
    atomicPositionArray=np.reshape(atomicPositionArray,(nFiles,nPartitions,nAtoms,nColumns))

# Generate plots
if plotFlag:
    print('Generating coordination plots...')

    os.chdir(plotDirectory)
    coordinationArray=np.zeros((nFiles,nPartitions,nAtoms),dtype=int)
    for i in range(nFiles):
        for j in range(nPartitions):
            _,coordinationArray[i,j]=findNearestNeighbours(atomicPositionArray[i,j],cutoff)
            print('%03d-%02d done' %(i,j))
    for i in range(nFiles):
        coordChange=coordinationArray[i,1:]-coordinationArray[i,0]
        netCoordChange=np.sum(abs(coordChange),axis=1)
        plt.plot(netCoordChange)
        plt.ylabel(r'Coordination change from initial position')
        plt.xlabel(r'NEB Image')
        plt.title(r'Net Coordination Change vs NEB Image')
        plt.savefig('coord_%03d-%02d.png' % (maxTLSNumbers[i,0],maxTLSNumbers[i,1]))
        plt.clf()

        print('coord_%03d-%02d.png saved' % (maxTLSNumbers[i,0],maxTLSNumbers[i,1]))

# Genrates .mp4 file of NEB energy profile as algorithm evolves
if movieFlag:
    print('Generating .mp4 files...')

    os.chdir(plotDirectory)
    atomicZScoreArray[atomicZScoreArray < minZScore]=0
    for i in range(nFiles):
        trajectory=maxTLSNumbers[i]
        
        # Generates plot of atomic Z score
        plt.plot(atomicZScoreArray[i])
        plt.xlabel(r'Atom Number')
        plt.ylabel(r'Z Score')
        plt.title(r'Z Score vs Atom Number')
        plt.savefig('atomic_z_%03d-%02d' % (trajectory[0],trajectory[1]) + '.png')
        plt.clf()

        # Generates movie of NEB evolution
        FFMpegWriter=manimation.writers['ffmpeg']
        metadata=dict(title='Atomic Positions',artist='Daniel Wong', comment='Tracks atomic positins as NEB algorithm evolves')
        writer=FFMpegWriter(fps=5,metadata=metadata)

        ind=np.nonzero(atomicZScoreArray[i])
        xMin=np.min(atomicPositionArray[i,:,ind,0])
        yMin=np.min(atomicPositionArray[i,:,ind,1])
        zMin=np.min(atomicPositionArray[i,:,ind,2])
        xMax=np.max(atomicPositionArray[i,:,ind,0])
        yMax=np.max(atomicPositionArray[i,:,ind,1])
        zMax=np.max(atomicPositionArray[i,:,ind,2])

        fig=plt.figure()
        ax=fig.add_gridspec(1,4)
        fig.subplots_adjust(left=0.1,right=0.9,hspace=0.2)
        with writer.saving(fig,'atomic_positions_%03d_%02d.mp4' % (trajectory[0],trajectory[1]),100):
            for j in range(nPartitions):
                ax1=fig.add_subplot(ax[0,0:3],projection='3d')
                ax1.scatter(atomicPositionArray[i,j,:,0],atomicPositionArray[i,j,:,1],atomicPositionArray[i,j,:,2],s=atomicZScoreArray[i])
                nn,_=findNearestNeighbours(atomicPositionArray[i,j],cutoff)
                for index in ind[0]:
                    for neighbour in nn[index]:
                        bondX=[atomicPositionArray[i,j,index,0],atomicPositionArray[i,j,neighbour,0]]
                        bondY=[atomicPositionArray[i,j,index,1],atomicPositionArray[i,j,neighbour,1]]
                        bondZ=[atomicPositionArray[i,j,index,2],atomicPositionArray[i,j,neighbour,2]]
                        ax1.plot(bondX,bondY,bondZ,'r-',linewidth=0.5)
                ax1.set_xlabel(r'X ($\AA$)')
                ax1.set_ylabel(r'Y ($\AA$)')
                ax1.set_zlabel(r'Z ($\AA$)')
                ax1.set_xlim((xMin,xMax))
                ax1.set_ylim((yMin,yMax))
                ax1.set_zlim((zMin,zMax))
                ax2=fig.add_subplot(ax[0,3:])
                ax2.plot(atomicEnergyArray[i,0:(j+1)])
                ax2.set_xlabel(r'NEB Image')
                ax2.set_ylabel(r'Energy (eV)')
                ax2.set_ylim((0,max(atomicEnergyArray[i])))
                fig.tight_layout()
                writer.grab_frame()
                plt.clf()

                print(str(j+1)+'/'+str(nPartitions)+' image saved')
        print('Trajectory %03d-%02d' % (trajectory[0],trajectory[1])+' saved')
