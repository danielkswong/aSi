import tarfile
import os
import io
import gzip
import numpy as np
from scipy.signal import argrelextrema
from matplotlib import pyplot as plt
import math
import csv

# Flags
readFlag=1
writeFlag=1-readFlag
plotFlag=1

# Relevant parameters
T=600
nEquils=100
nVelocities=10
nAtoms=1000
nColumns=6
nDumps=2001
lineStart=9
crit_Damart=1
dataDirectory='../Data_v1/'
plotDirectory='../Plots_v1/'

# Read in minimized position files and energies
if writeFlag:
    os.chdir(dataDirectory)
    quenchFiles=[]
    for file in os.listdir(os.getcwd()):
        if file.startswith('Equil') and not file.endswith('Hessian.tar.gz') and not file.endswith('NEB.tar.gz'):
            quenchFiles.append(file)
    nFiles=len(quenchFiles)
    if nFiles!=nEquils*nVelocities:
        raise Exception('Folder contains missing trajectory files')
    quenchFiles.sort()

    quenchAvgDisp=np.zeros((nEquils*nVelocities*(nDumps-1),3),dtype=np.float64)
    quenchAvgConsDisp=np.zeros((nEquils*nVelocities*(nDumps-1),3),dtype=np.float64)
    quenchAvgDisp2=np.zeros((nEquils*nVelocities*(nDumps-1),3),dtype=np.float64)
    quenchAvgConsDisp2=np.zeros((nEquils*nVelocities*(nDumps-1),3),dtype=np.float64)
    quenchMaxConsDisp=np.zeros((nEquils*nVelocities*(nDumps-1),3),dtype=np.float64)
    quenchPR=np.zeros((nEquils*nVelocities*(nDumps-1),3),dtype=np.float64)
    quenchEnergyAsym=np.zeros((nEquils*nVelocities*(nDumps-1),3),dtype=np.float64)

    # Reads in position and energy data
    print('Reading data...')
    for i in np.arange(nFiles):
        equilNo=int(quenchFiles[i][5:8])
        velocityNo=int(quenchFiles[i].replace('.tar.gz','').split('-')[1])
        positionArray=np.zeros((nDumps,nAtoms,nColumns),dtype=np.float64)
        energyArray=np.zeros(nDumps,dtype=np.float64)
        with tarfile.open(quenchFiles[i],mode='r:gz') as archive:
            files=archive.getnames()
            if len(files)<(nDumps+1):
                raise Exception('Missing position and/or energy files in *.tar.gz')
            for file in files:
                fileName=file.replace('i=','')
                fileName=fileName.replace('.lammps.gz','')
                fileName=fileName.split('_')
                f=archive.extractfile(file)
                if fileName[1]=='energies':
                    for index in np.arange(nDumps):
                        energyArray[index]=float(f.readline().decode('UTF-8').strip())
                else:
                    index=int(fileName[-1])
                    byteInfo=io.BytesIO(f.read())
                    positionFile=gzip.GzipFile(fileobj=byteInfo)
                    for j in np.arange(lineStart):
                        positionFile.readline()
                    for atomNumber in np.arange(nAtoms):
                        line=positionFile.readline().decode('UTF-8')
                        stringArray=np.asarray(line.strip().split(' '),dtype=np.float64)
                        positionArray[index,atomNumber,:]=stringArray[2:]
                    positionFile.close()
                f.close()

        # Process data
        iniDisp=positionArray[1:]-positionArray[0]
        consDisp=positionArray[1:]-positionArray[0:-1]

        # Net displacement squared from initial position/previous position for each atom in a given dump
        netDisp2=iniDisp[:,:,3]**2+iniDisp[:,:,4]**2+iniDisp[:,:,5]**2
        netConsDisp2=consDisp[:,:,3]**2+consDisp[:,:,4]**2+consDisp[:,:,5]**2
        # Average displacement summed over all atoms
        avgDisp=np.sqrt(np.sum(netDisp2,1))/nAtoms
        avgConsDisp=np.sqrt(np.sum(netConsDisp2,1))/nAtoms
        # Average squared displacement summed over all atoms
        avgDisp2=np.sum(netDisp2,1)/nAtoms
        avgConsDisp2=np.sum(netConsDisp2,1)/nAtoms
        # Maximum net displacement squared
        maxDisp=np.sqrt(np.amax(netDisp2,1))
        maxConsDisp=np.sqrt(np.amax(netConsDisp2,1))
        # Participation ratio
        PR=np.sum(consDisp[:,:,3]**2+consDisp[:,:,4]**2+consDisp[:,:,5]**2,1)**2/np.sum(consDisp[:,:,3]**4+consDisp[:,:,4]**4+consDisp[:,:,5]**4,1)
        # Energy asymmetry
        energyAsym=energyArray[1:]-energyArray[0:-1]

        # Place processed values into array to be saved later into text file
        saveIndex=(nDumps-1)*nVelocities*(equilNo-1)+(nDumps-1)*(velocityNo-1)
        
        quenchAvgDisp[saveIndex:saveIndex+nDumps-1,0]=equilNo
        quenchAvgConsDisp[saveIndex:saveIndex+nDumps-1,0]=equilNo
        quenchAvgDisp2[saveIndex:saveIndex+nDumps-1,0]=equilNo
        quenchAvgConsDisp2[saveIndex:saveIndex+nDumps-1,0]=equilNo
        quenchMaxConsDisp[saveIndex:saveIndex+nDumps-1,0]=equilNo
        quenchPR[saveIndex:saveIndex+nDumps-1,0]=equilNo
        quenchEnergyAsym[saveIndex:saveIndex+nDumps-1,0]=equilNo

        quenchAvgDisp[saveIndex:saveIndex+nDumps-1,1]=velocityNo
        quenchAvgConsDisp[saveIndex:saveIndex+nDumps-1,1]=velocityNo
        quenchAvgDisp2[saveIndex:saveIndex+nDumps-1,1]=velocityNo
        quenchAvgConsDisp2[saveIndex:saveIndex+nDumps-1,1]=velocityNo
        quenchMaxConsDisp[saveIndex:saveIndex+nDumps-1,1]=velocityNo
        quenchPR[saveIndex:saveIndex+nDumps-1,1]=velocityNo
        quenchEnergyAsym[saveIndex:saveIndex+nDumps-1,1]=velocityNo

        quenchAvgDisp[saveIndex:saveIndex+nDumps-1,2]=avgDisp
        quenchAvgConsDisp[saveIndex:saveIndex+nDumps-1,2]=avgConsDisp
        quenchAvgDisp2[saveIndex:saveIndex+nDumps-1,2]=avgDisp2
        quenchAvgConsDisp2[saveIndex:saveIndex+nDumps-1,2]=avgConsDisp2
        quenchMaxConsDisp[saveIndex:saveIndex+nDumps-1,2]=maxConsDisp
        quenchPR[saveIndex:saveIndex+nDumps-1,2]=PR
        quenchEnergyAsym[saveIndex:saveIndex+nDumps-1,2]=energyAsym

        print(quenchFiles[i]+' completed')

    # Writes processed quench data as CSV files
    print('Saving data as CSV files...')
    
    np.savetxt('avgDisp_T='+str(T)+'_quench.csv',quenchAvgDisp,delimiter=',')
    np.savetxt('avgConsDisp_T='+str(T)+'_quench.csv',quenchAvgConsDisp,delimiter=',')
    np.savetxt('avgDisp2_T='+str(T)+'_quench.csv',quenchAvgDisp2,delimiter=',')
    np.savetxt('avgConsDisp2_T='+str(T)+'_quench.csv',quenchAvgConsDisp2,delimiter=',')
    np.savetxt('maxConsDisp_T='+str(T)+'_quench.csv',quenchMaxConsDisp,delimiter=',')
    np.savetxt('PR_T='+str(T)+'_quench.csv',quenchPR,delimiter=',')
    np.savetxt('energyAsym_T='+str(T)+'_quench.csv',quenchEnergyAsym,delimiter=',') 

# Reads data if CSV file is located
if readFlag:
    print('Reading CSV files...')

    os.chdir(dataDirectory)
    with open('avgDisp_T='+str(T)+'_quench.csv','r') as csvfile:
        csvreader=csv.reader(csvfile,delimiter=',')
        quenchAvgDisp=np.asarray(list(csvreader),dtype=np.float64)
    with open('avgConsDisp_T='+str(T)+'_quench.csv','r') as csvfile:
        csvreader=csv.reader(csvfile,delimiter=',')
        quenchAvgConsDisp=np.asarray(list(csvreader),dtype=np.float64)
    with open('avgDisp2_T='+str(T)+'_quench.csv','r') as csvfile:
        csvreader=csv.reader(csvfile,delimiter=',')
        quenchAvgDisp2=np.asarray(list(csvreader),dtype=np.float64)
    with open('avgConsDisp2_T='+str(T)+'_quench.csv','r') as csvfile:
        csvreader=csv.reader(csvfile,delimiter=',')
        quenchAvgConsDisp2=np.asarray(list(csvreader),dtype=np.float64)
    with open('maxConsDisp_T='+str(T)+'_quench.csv','r') as csvfile:
        csvreader=csv.reader(csvfile,delimiter=',')
        quenchMaxConsDisp=np.asarray(list(csvreader),dtype=np.float64)
    with open('PR_T='+str(T)+'_quench.csv','r') as csvfile:
        csvreader=csv.reader(csvfile,delimiter=',')
        quenchPR=np.asarray(list(csvreader),dtype=np.float64)
    with open('energyAsym_T='+str(T)+'_quench.csv','r') as csvfile:
        csvreader=csv.reader(csvfile,delimiter=',')
        quenchEnergyAsym=np.asarray(list(csvreader),dtype=np.float64)

# Generate plots
if plotFlag:

    # Obtain distance criterion
    quenchAvgConsDispHist=np.histogram(quenchAvgConsDisp[:,2],bins='auto',density=False)
    quenchAvgConsDisp2Hist=np.histogram(quenchAvgConsDisp2[:,2],bins='auto',density=False)
    quenchMaxConsDispHist=np.histogram(quenchMaxConsDisp[:,2],bins='auto',density=False)
    min_dCritInd=np.argmax((quenchAvgConsDispHist[0][1:]-quenchAvgConsDispHist[0][0:-1])>=0)
    max_dCritInd=np.argmax(quenchAvgConsDispHist[0][min_dCritInd:])+min_dCritInd
    dCritInd=np.argmin(quenchAvgConsDispHist[0][min_dCritInd:max_dCritInd])+min_dCritInd
    dCrit=(quenchAvgConsDispHist[1][dCritInd]+quenchAvgConsDispHist[1][dCritInd+1])/2
    min_d2CritInd=np.argmax((quenchAvgConsDisp2Hist[0][1:]-quenchAvgConsDisp2Hist[0][0:-1])>=0)
    max_d2CritInd=np.argmax(quenchAvgConsDisp2Hist[0][min_d2CritInd:])+min_d2CritInd
    d2CritInd=np.argmin(quenchAvgConsDisp2Hist[0][min_d2CritInd:max_d2CritInd])+min_d2CritInd
    d2Crit=(quenchAvgConsDisp2Hist[1][d2CritInd]+quenchAvgConsDisp2Hist[1][d2CritInd+1])/2        
    min_dConsCritInd=np.argmax((quenchMaxConsDispHist[0][1:]-quenchMaxConsDispHist[0][0:-1])>=0)
    max_dConsCritInd=np.argmax(quenchMaxConsDispHist[0][min_dConsCritInd:])+min_dConsCritInd
    dConsCritInd=np.argmin(quenchMaxConsDispHist[0][min_dConsCritInd:max_dConsCritInd])+min_dConsCritInd
    dConsCrit=(quenchMaxConsDispHist[1][dConsCritInd]+quenchMaxConsDispHist[1][dConsCritInd+1])/2

    # Change to plot directory
    os.chdir(plotDirectory)

    print('Generating plots...')

    # Average consecutive displacement plots
    plt.hist(quenchAvgConsDisp[:,2],bins='auto',density=False,ec='k')
    plt.title(r'Average Consecutive Displacement Histogram, $T$ = '+str(T) + ' K',y=1.05)
    xlims=(0.0,max(quenchAvgConsDisp[:,2]))
    ylims=plt.ylim()
    plt.vlines(dCrit,0,ylims[1],linestyles='dashed',colors='red',label=r'$d_{C}$ = '+'{:.3e} '.format(dCrit,s=3)+r'$\AA$')
    plt.vlines(float(crit_Damart)/nAtoms,0,ylims[1],linestyles='dashed',colors='green',label='Damart and Rodney = '+'{:.3e} '.format(float(crit_Damart)/nAtoms,s=3)+r'$\AA$')
    plt.xlabel(r'Average consecutive displacement ($\AA$)')
    plt.ylabel(r'Frequency')
    plt.ticklabel_format(style='sci',axis='x',scilimits=(0,0),useMathText=True)
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0),useMathText=True)
    plt.legend(bbox_to_anchor=(1,1),loc='upper left')
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.savefig('avg_cons_disp_hist_T='+str(T)+'_quench.png',bbox_inches='tight')

    print('avg_cons_disp_hist_T='+str(T)+'_quench.png saved')

    maxFreq=quenchAvgConsDispHist[0][max_dCritInd]
    zoomY=math.ceil(maxFreq/10**(math.floor(math.log10(maxFreq))))*10**(math.floor(math.log10(maxFreq)))
    plt.xlim((xlims[0],xlims[1]/3))
    plt.ylim((ylims[0],zoomY))
    plt.savefig('avg_cons_disp_hist_T='+str(T)+'_quench_zoom.png',bbox_inches='tight')
    plt.clf()

    print('avg_cons_disp_hist_T='+str(T)+'_quench_zoom.png saved')

    # Average consecutive displacement vs energy asymmetry plots
    plt.plot(quenchAvgConsDisp[:,2],quenchEnergyAsym[:,2],'bo',mec='k')
    plt.title(r'Average Consecutive Displacement vs Energy Asymmetry, $T$ = ' + str(T) + ' K',y=1.05)
    xlims=(0.0,max(quenchAvgConsDisp[:,2]))
    ylims=plt.ylim()
    plt.vlines(dCrit,ylims[0],ylims[1],linestyles='dashed',colors='red',label='$d_{C}$ = '+'{:.3e} '.format(dCrit,s=3)+r'$\AA$')
    plt.vlines(float(crit_Damart)/nAtoms,ylims[0],ylims[1],linestyles='dashed',colors='green',label='Damart = '+'{:.3e} '.format(float(crit_Damart)/nAtoms,s=3)+r'$\AA$')
    plt.xlabel(r'Average consecutive displacement ($\AA$)')
    plt.ylabel(r'Energy asymmetry (eV)')
    plt.ticklabel_format(style='sci',axis='x',scilimits=(0,0),useMathText=True)
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0),useMathText=True)
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.savefig('avg_cons_disp_vs_energy_T='+str(T)+'_quench.png')

    print('avg_cons_disp_vs_energy_T='+str(T)+'_quench.png saved')

    plt.xlim((xlims[0],xlims[1]/3))
    plt.savefig('avg_cons_disp_vs_energy_T='+str(T)+'_quench_zoom.png')
    plt.clf()

    print('avg_cons_disp_vs_energy_T='+str(T)+'_quench_zoom.png saved')

    # Average consecutive displacement squared plots
    plt.hist(quenchAvgConsDisp2[:,2],bins='auto',density=False,ec='k')
    plt.title(r'Average Consecutive Displacement Squared Histogram, $T$ = '+str(T) + ' K',y=1.05)
    xlims=(0.0,max(quenchAvgConsDisp2[:,2]))
    ylims=plt.ylim()
    plt.vlines(d2Crit,0,ylims[1],linestyles='dashed',colors='red',label=r'$d^{2}_{C}$ = '+'{:.3e} '.format(d2Crit,s=3)+r'$\AA$')
    plt.xlabel(r'Average consecutive displacement squared ($\AA^{2}$)')
    plt.ylabel(r'Frequency')
    plt.ticklabel_format(style='sci',axis='x',scilimits=(0,0),useMathText=True)
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0),useMathText=True)
    plt.legend(bbox_to_anchor=(1,1),loc='upper left')
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.savefig('avg_cons_disp2_hist_T='+str(T)+'_quench.png',bbox_inches='tight')

    print('avg_cons_disp2_hist_T='+str(T)+'_quench.png saved')
    
    maxFreq=quenchAvgConsDisp2Hist[0][max_d2CritInd]
    zoomY=math.ceil(maxFreq/10**(math.floor(math.log10(maxFreq))))*10**(math.floor(math.log10(maxFreq)))
    plt.xlim((xlims[0],xlims[1]/50))
    plt.ylim((ylims[0],zoomY))
    plt.savefig('avg_cons_disp2_hist_T='+str(T)+'_quench_zoom.png',bbox_inches='tight')
    plt.clf()

    print('avg_cons_disp2_hist_T='+str(T)+'_quench_zoom.png saved')


if plotFlag:
    # Change to plot directory
    os.chdir(plotDirectory)
    
    # Average consecutive displacement squared vs energy asymmetry plots
    plt.plot(quenchAvgConsDisp2[:,2],quenchEnergyAsym[:,2],'bo',mec='k')
    plt.title(r'Average Consecutive Displacement Squared vs Energy Asymmetry, $T$ = ' + str(T) + ' K',y=1.05)
    xlims=(0.0,max(quenchAvgConsDisp2[:,2]))
    ylims=plt.ylim()
    plt.vlines(d2Crit,ylims[0],ylims[1],linestyles='dashed',colors='red',label=r'$d^{2}_{C}$ = '+'{:.3e} '.format(d2Crit,s=3)+r'$\AA$')
    plt.xlabel(r'Average consecutive displacement squared ($\AA^{2}$)')
    plt.ylabel(r'Energy asymmetry (eV)')
    plt.ticklabel_format(style='sci',axis='x',scilimits=(0,0),useMathText=True)
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0),useMathText=True)
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.savefig('avg_cons_disp2_vs_energy_T='+str(T)+'_quench.png')

    print('avg_cons_disp2_vs_energy_T='+str(T)+'_quench.png saved')

    plt.xlim((xlims[0],xlims[1]/50))
    plt.savefig('avg_cons_disp2_vs_energy_T='+str(T)+'_quench_zoom.png')
    plt.clf()

    print('avg_cons_disp2_vs_energy_T='+str(T)+'_quench_zoom.png saved')

if plotFlag:
    # Change to plot directory
    os.chdir(plotDirectory)
    
    # Maximal consecutive atomic displacement plots
    plt.hist(quenchMaxConsDisp[:,2],bins='auto',density=False,ec='k')
    plt.title(r'Maximum Consecutive Atomic Displacement Histogram, $T$ = '+str(T) + ' K',y=1.05)
    xlims=(0.0,max(quenchMaxConsDisp[:,2]))
    ylims=plt.ylim()
    plt.vlines(dConsCrit,0,ylims[1],linestyles='dashed',colors='red',label=r'$d_{C}$ = '+'{:.3e} '.format(dConsCrit,s=3)+r'$\AA$')
    plt.xlabel(r'Maximum consecutive atomic displacement ($\AA$)')
    plt.ylabel(r'Frequency')
    plt.ticklabel_format(style='sci',axis='x',scilimits=(0,0),useMathText=True)
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0),useMathText=True)
    plt.legend(bbox_to_anchor=(1,1),loc='upper left')
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.savefig('max_cons_disp_hist_T='+str(T)+'_quench.png',bbox_inches='tight')

    print('max_cons_disp_hist_T='+str(T)+'_quench.png saved')

    maxFreq=quenchMaxConsDispHist[0][max_dConsCritInd]
    zoomY=math.ceil(maxFreq/10**(math.floor(math.log10(maxFreq))))*10**(math.floor(math.log10(maxFreq)))
    plt.xlim((xlims[0],xlims[1]/2))
    plt.ylim((ylims[0],zoomY))
    plt.savefig('max_cons_disp_hist_T='+str(T)+'_quench_zoom.png',bbox_inches='tight')
    plt.clf()

    print('max_cons_disp_hist_T='+str(T)+'_quench_zoom.png saved')

if plotFlag:
    # Change to plot directory
    os.chdir(plotDirectory)
    
    # Maximal consecutive atomic displacement vs energy asymmetry plots
    plt.plot(quenchMaxConsDisp[:,2],quenchEnergyAsym[:,2],'bo',mec='k')
    plt.title(r'Maximum Consecutive Atomic Displacement vs Energy Asymmetry, $T$ = ' + str(T) + ' K',y=1.05)
    xlims=(0.0,max(quenchMaxConsDisp[:,2]))
    ylims=plt.ylim()
    plt.vlines(dConsCrit,ylims[0],ylims[1],linestyles='dashed',colors='red',label=r'$d_{C}$ = '+'{:.3e} '.format(dConsCrit,s=3)+r'$\AA$')
    plt.xlabel(r'Maximum consecutive atomic displacement ($\AA$)')
    plt.ylabel(r'Energy asymmetry (eV)')
    plt.ticklabel_format(style='sci',axis='x',scilimits=(0,0),useMathText=True)
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0),useMathText=True)
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.savefig('max_cons_disp_vs_energy_T='+str(T)+'_quench.png')

    print('max_cons_disp_vs_energy_T='+str(T)+'_quench.png saved')

    plt.xlim((xlims[0],xlims[1]/2))
    plt.savefig('max_cons_disp_vs_energy_T='+str(T)+'_quench_zoom.png')
    plt.clf()

    print('max_cons_disp_vs_energy_T='+str(T)+'_quench_zoom.png saved')

if plotFlag:
    # Change to plot directory
    os.chdir(plotDirectory)
    
    # Maximal consecutive atomic displacement vs average consecutive displacement plots
    plt.plot(quenchMaxConsDisp[:,2],quenchAvgConsDisp[:,2],'bo',mec='k')
    plt.title(r'Maximum Consecutive Atomic Displacement vs Average Consecutive Displacement, $T$ = ' + str(T) + ' K',y=1.05)
    plt.xlabel(r'Maximum consecutive atomic displacement ($\AA$)')
    plt.ylabel(r'Average consecutive displacement ($\AA$)')
    plt.ticklabel_format(style='sci',axis='x',scilimits=(0,0),useMathText=True)
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0),useMathText=True)
    plt.savefig('max_cons_disp_vs_avg_cons_disp_T='+str(T)+'_quench.png')
    plt.clf()

    print('max_cons_disp_vs_avg_cons_disp_T='+str(T)+'_quench.png saved')
