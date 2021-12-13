import tarfile
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as manimation
import math
import csv

# Flag parameters
readFlag=0
writeFlag=1-readFlag
plotFlag=1
movieFlag=0

# Relevant parameters
T=600
nPartitions=50
nAtoms=1000
lineStart=9
screenX=13.66
screenY=7.68
dataDirectory='../Data_v1/'
plotDirectory='../Plots_v1/'

# Determine if local minima exists
def minima_exists(array):
    location=len(array)-1
    for i in np.arange(len(array)-1):
        diff=array[i+1]-array[i]
        if diff<0:
            location=i
            break
    for i in np.arange(location,len(array)-1):
        diff=array[i+1]-array[i]
        if diff>0:
            return True
    return False

# Reads in NEB files
if writeFlag:
    os.chdir(dataDirectory)
    NEBFiles=[]
    for file in os.listdir(os.getcwd()):
        if file.startswith('Equil') and file.endswith('_NEB.tar.gz'):
            NEBFiles.append(file)
    nFiles=len(NEBFiles)
    NEBFiles.sort()

    NEBEnergyBarriers=np.zeros((nFiles,3),dtype=np.float64)
    NEBEnergyAsym=np.zeros((nFiles,3),dtype=np.float64)

    # Reads in energies and filters out pathological minimum energy paths
    deleteInd=[]
    initialRepDisp=np.zeros((nFiles,3),dtype=np.float64)
    finalRepDisp=np.zeros((nFiles,3),dtype=np.float64)
    print('Reading NEB files...')
    for i in np.arange(nFiles):
        equilNo=int(NEBFiles[i][5:8])
        velocityNo=int(NEBFiles[i].replace('_NEB.tar.gz','').split('-')[1])
        with tarfile.open(NEBFiles[i],mode='r:gz') as archive:
            files=archive.getnames()
            if len(files)<(nPartitions+1):
                raise Exception('Missing dump and/or NEB log files in .tar.gz')

            energyFile=archive.extractfile('log.lammps')
            energyFileText=energyFile.readlines()
            energyFileText=[line.decode('UTF-8') for line in energyFileText]
            energyFile.close()

            initialFile=archive.extractfile('dump.neb.1')
            initialFileText=initialFile.readlines()
            initialFileText=[line.decode('UTF-8') for line in initialFileText]
            initialFile.close()

            finalFile=archive.extractfile('dump.neb.50')
            finalFileText=finalFile.readlines()
            finalFileText=[line.decode('UTF-8') for line in finalFileText]
            finalFile.close()

        lenInitial=int(len(initialFileText)/(nAtoms+lineStart))
        lenFinal=int(len(finalFileText)/(nAtoms+lineStart))
        if lenInitial!=lenFinal:
            raise Exception('Dump files are not the same length')

        initialReplicaPosition=np.zeros((lenInitial,nAtoms,3),dtype=np.float64)
        finalReplicaPosition=np.zeros((lenFinal,nAtoms,3),dtype=np.float64)
        for j in np.arange(lenInitial):
            index=0
            for line in initialFileText[j*nAtoms+(j+1)*lineStart:(j+1)*nAtoms+(j+1)*lineStart]:
                stringArray=np.asarray(line.strip().split(),dtype=np.float64)
                initialReplicaPosition[j,index,:]=stringArray[5:]
                index+=1

        for j in np.arange(lenFinal):
            index=0
            for line in finalFileText[j*nAtoms+(j+1)*lineStart:(j+1)*nAtoms+(j+1)*lineStart]:
                stringArray=np.asarray(line.strip().split(),dtype=np.float64)
                finalReplicaPosition[j,index,:]=stringArray[5:]
                index+=1

        iniInitialNEBDisp=initialReplicaPosition[1:]-initialReplicaPosition[0]
        netInitialNEBDisp2=iniInitialNEBDisp[:,:,0]**2+iniInitialNEBDisp[:,:,1]**2+iniInitialNEBDisp[:,:,2]**2
        avgInitialNEBDisp=np.sqrt(np.sum(netInitialNEBDisp2,1))/nAtoms

        iniFinalNEBDisp=finalReplicaPosition[1:]-finalReplicaPosition[0]
        netFinalNEBDisp2=iniFinalNEBDisp[:,:,0]**2+iniFinalNEBDisp[:,:,1]**2+iniFinalNEBDisp[:,:,2]**2
        avgFinalNEBDisp=np.sqrt(np.sum(netFinalNEBDisp2,1))/nAtoms

        initialRepDisp[i,:]=np.asarray([equilNo,velocityNo,avgInitialNEBDisp[-1]])
        finalRepDisp[i,:]=np.asarray([equilNo,velocityNo,avgFinalNEBDisp[-1]])
        
        array=np.asarray(energyFileText[-1].strip().split(),dtype=np.float64)
        energyBarrier=(array[6]+array[7])/2
        energyAsym=array[6]-array[7]
        energyArray=array[10::2]
        if minima_exists(energyArray) or (np.max(energyArray)==energyArray[0] and np.max(energyArray)==energyArray[-1]):
            deleteInd.append(i)
        NEBEnergyBarriers[i,:]=np.asarray([equilNo,velocityNo,energyBarrier])
        NEBEnergyAsym[i,:]=np.asarray([equilNo,velocityNo,energyAsym])
        
        print(NEBFiles[i]+' completed')

    # Writes energy barriers and transition numbers as CSV files
    print('Saving NEB data as CSV files...')
    
    np.savetxt('barriers_T='+str(T)+'_NEB.csv',NEBEnergyBarriers,delimiter=',')
    np.savetxt('energyAsym_T='+str(T)+'_NEB.csv',NEBEnergyAsym,delimiter=',')
    np.savetxt('initialRepDisp_T='+str(T)+'_NEB.csv',initialRepDisp,delimiter=',')
    np.savetxt('finalRepDisp_T='+str(T)+'_NEB.csv',finalRepDisp,delimiter=',')
    NEBEnergyBarriers=np.delete(NEBEnergyBarriers,deleteInd,0)
    NEBEnergyAsym=np.delete(NEBEnergyAsym,deleteInd,0)
    initialRepDisp=np.delete(initialRepDisp,deleteInd,0)
    finalRepDisp=np.delete(finalRepDisp,deleteInd,0)
    np.savetxt('barriers_T='+str(T)+'_NEB_no_minima.csv',NEBEnergyBarriers,delimiter=',')
    np.savetxt('energyAsym_T='+str(T)+'_NEB_no_minima.csv',NEBEnergyAsym,delimiter=',')
    np.savetxt('initialRepDisp_T='+str(T)+'_NEB_no_minima.csv',initialRepDisp,delimiter=',')
    np.savetxt('finalRepDisp_T='+str(T)+'_NEB_no_minima.csv',finalRepDisp,delimiter=',')
    np.savetxt('transitionNumbers_T='+str(T)+'_no_minima.csv',NEBEnergyBarriers[:,:2],delimiter=',')

# Reads data if CSV file is located
if readFlag:
    print('Reading CSV file...')

    os.chdir(dataDirectory)
    with open('barriers_T='+str(T)+'_NEB_no_minima.csv','r') as csvfile:
        csvreader=csv.reader(csvfile,delimiter=',')
        NEBEnergyBarriers=np.asarray(list(csvreader),dtype=np.float64)
    with open('energyAsym_T='+str(T)+'_NEB_no_minima.csv','r') as csvfile:
        csvreader=csv.reader(csvfile,delimiter=',')
        NEBEnergyAsym=np.asarray(list(csvreader),dtype=np.float64)
    with open('initialRepDisp_T='+str(T)+'_NEB_no_minima.csv','r') as csvfile:
        csvreader=csv.reader(csvfile,delimiter=',')
        initialRepDisp=np.asarray(list(csvreader),dtype=np.float64)
    with open('finalRepDisp_T='+str(T)+'_NEB_no_minima.csv','r') as csvfile:
        csvreader=csv.reader(csvfile,delimiter=',')
        finalRepDisp=np.asarray(list(csvreader),dtype=np.float64)

# Plots energy barriers
if plotFlag:
    os.chdir(plotDirectory)
    
    print('Generating plots...')

    plt.hist(NEBEnergyBarriers[:,2],bins=50,density=False,ec='k')
    plt.title(r'Average Energy Barriers Histogram, $T$ = '+str(T)+' K',y=1.05)
    plt.xlabel(r'Average energy barrier (eV)')
    plt.ylabel(r'Frequency')
    plt.ticklabel_format(style='sci',axis='x',scilimits=(0,0),useMathText=True)
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0),useMathText=True)
    plt.savefig('barrier_hist_T='+str(T)+'_NEB_no_minima.png')
    plt.clf()

    print('barrier_hist_T='+str(T)+'_NEB_no_minima.png saved')
    
    plt.plot(initialRepDisp[:,2])
    plt.title(r'Net Displacement of First Replica')
    plt.xlabel(r'TLS Candidate')
    plt.ylabel(r'Average displacement $d$ (angstrom)')
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0),useMathText=True)    
    plt.savefig('iniRepDisp_NEB_no_minima.png')
    plt.clf()

    print('iniRepDisp_NEB_no_minima.png saved')

    plt.plot(finalRepDisp[:,2])
    plt.title(r'Net Displacement of Last Replica')
    plt.xlabel(r'TLS Candidate')
    plt.ylabel(r'Average displacement $d$ (angstrom)')
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0),useMathText=True)    
    plt.savefig('finRepDisp_NEB_no_minima.png')
    plt.clf()

    print('finRepDisp_NEB_no_minima.png saved')

    plt.plot(NEBEnergyBarriers[:,2],NEBEnergyAsym[:,2],'bo',mec='k')
    plt.title(r'Energy Asymmetry vs Average Energy Barrier, $T$ = '+str(T)+' K',y=1.05)
    plt.xlabel(r'Average energy barrier (eV)')
    plt.ylabel(r'Energy asymmetry (eV)')
    plt.ticklabel_format(style='sci',axis='x',scilimits=(0,0),useMathText=True)
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0),useMathText=True)
    plt.savefig('energyAsym_vs_barrier_T='+str(T)+'_NEB_no_minima.png')
    plt.clf()

    print('energyAsym_vs_barrier_T='+str(T)+'_NEB_no_minima.png')
    
# Genrates .mp4 file of NEB energy profile as algorithm evolves
if movieFlag:    
    print('Generating .mp4 files...')
    trajectories=[(50,2),(50,3),(50,6),(50,9)]
    for trajectory in trajectories:
        # Reads NEB images
        os.chdir(dataDirectory)
        
        barriers=[]
        iterations=[]
        replicas=[]
        climbingNEBStart=0
        fileName='Equil%03d-%02d_NEB.tar.gz' % (trajectory[0],trajectory[1])
        with tarfile.open(fileName,mode='r:gz') as archive:
            file=archive.extractfile('log.lammps')
            text=file.readlines()
            text=[line.decode('UTF-8') for line in text]
            for i in np.arange(len(text)):
                stringArray=text[i].strip().split()
                if stringArray[0].isdigit():
                    replicas.append(stringArray[10::2])
                    iterations.append(stringArray[0])
                    barriers.append((float(stringArray[6])+float(stringArray[7]))/2)
                elif stringArray[0]=='Climbing':
                    line=text[i+2].strip().split()
                    climbingNEBStart=int(line[0])
        replicas=np.asarray(replicas,dtype=np.float64)
        barriers=np.asarray(barriers,dtype=np.float64)
        iterations=np.asarray(iterations,dtype=np.float64)
        shift=min(replicas[0][0],replicas[0][-1])
        replicas=replicas-shift
        
        # Generates movie of NEB evolution
        os.chdir(plotDirectory)
        
        FFMpegWriter=manimation.writers['ffmpeg']
        metadata=dict(title='NEB Image Tracker',artist='Daniel Wong', comment='Tracks energy vs NEB image as NEB algorithm evolves')
        writer=FFMpegWriter(fps=10,metadata=metadata)

        fig=plt.gcf()
        fig.set_size_inches(screenX,screenY)
        with writer.saving(fig,'NEB_%03d_%02d.mp4' % (trajectory[0],trajectory[1]),100):
            for j in np.arange(1,len(barriers)):
                barrier_max=np.max(barriers[:(j+1)])
                barrier_min=np.min(barriers[:(j+1)])
                plt.subplot(2,1,1)
                plt.plot(iterations[:(j+1)],barriers[:(j+1)])
                plt.xlabel('Iteration')
                plt.ylabel('Average Energy Barrier (eV)')
                plt.ylim((barrier_min,barrier_max))
                plt.xlim((0,iterations[j]))
                if (iterations[j]>=climbingNEBStart):
                    plt.vlines(climbingNEBStart,barrier_min,barrier_max,linestyles='dashed',colors='red',label='Barrier-climbing NEB begins')
                plt.subplot(2,1,2)
                plt.plot(replicas[j])
                plt.xlabel('NEB Image')
                plt.ylabel('Energy (eV)')
                plt.ylim((np.min(replicas[j]),np.max(replicas[j])))
                writer.grab_frame()
                plt.clf()

                print(str(j)+'/'+str(len(barriers)-1)+' image saved')
        print('Trajectory '+str(trajectory[0])+'-'+str(trajectory[1])+' saved')
