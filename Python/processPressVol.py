import numpy as np
import tarfile
import os
import math
import csv
import matplotlib.pyplot as plt

# Flags
readFlag=1
writeFlag=1-readFlag
plotFlag=1

# Relevant parameters
T=600
dataDirectory='../Data_v1/'
plotDirectory='../Plots_v1/'
fileLength=8

# Read Hessian files
if writeFlag:
    hessianFiles=[]
    os.chdir(dataDirectory)
    for file in os.listdir(os.getcwd()):
        if file.startswith('Equil') and file.endswith('_Hessian.tar.gz'):
            hessianFiles.append(file)
    nFiles=len(hessianFiles)
    hessianFiles.sort()

    iniPressVol=np.zeros((nFiles,10),dtype=np.float64)
    finPressVol=np.zeros((nFiles,10),dtype=np.float64)
    
    for i in np.arange(nFiles):
        print('Reading '+hessianFiles[i]+'...')

        equilNo=int(hessianFiles[i][5:8])
        velocityNo=int(hessianFiles[i].replace('_Hessian.tar.gz','').split('-')[1])

        with tarfile.open(hessianFiles[i],mode='r:gz') as archive:
            iniFile=archive.extractfile('ini_press.dat')
            iniText=iniFile.readlines()
            iniText=[line.decode('UTF-8') for line in iniText]
            iniFile.close()

            if len(iniText)!=fileLength:
                raise Exception('Missing lines from ini_press.dat file')

            for j in np.arange(len(iniText)):
                line=iniText[j].strip().split()
                iniPressVol[i,0]=equilNo
                iniPressVol[i,1]=velocityNo
                iniPressVol[i,j+2]=float(line[0])

            finFile=archive.extractfile('fin_press.dat')
            finText=finFile.readlines()
            finText=[line.decode('UTF-8') for line in finText]
            finFile.close()

            if len(finText)!=fileLength:
                raise Exception('Missing lines from fin_press.dat file')

            for j in np.arange(len(finText)):
                line=finText[j].strip().split()
                finPressVol[i,0]=equilNo
                finPressVol[i,1]=velocityNo
                finPressVol[i,j+2]=float(line[0])

    np.savetxt('ini_press_vol.csv',iniPressVol,delimiter=',')
    np.savetxt('fin_press_vol.csv',finPressVol,delimiter=',')

if readFlag:
    os.chdir(dataDirectory)
    
    print('Reading CSV files...')

    with open('ini_press_vol.csv','r') as csvfile:
        csvreader=csv.reader(csvfile,delimiter=',')
        iniArray=np.asarray(list(csvreader),dtype=np.float64)
    with open('fin_press_vol.csv','r') as csvfile:
        csvreader=csv.reader(csvfile,delimiter=',')
        finArray=np.asarray(list(csvreader),dtype=np.float64)
    with open('barriers_T='+str(T)+'_NEB_no_minima.csv','r') as csvfile:
        csvreader=csv.reader(csvfile,delimiter=',')
        NEBEnergyBarriers=np.asarray(list(csvreader),dtype=np.float64)
    with open('freq_hessian.csv','r') as csvfile:
        csvreader=csv.reader(csvfile,delimiter=',')
        freqArray=np.asarray(list(csvreader),dtype=np.float64)

    barriers=np.zeros(len(freqArray),dtype=np.float64)
    iniPressVol=np.zeros((len(freqArray),fileLength),dtype=np.float64)
    finPressVol=np.zeros((len(freqArray),fileLength),dtype=np.float64)

    for i in np.arange(len(freqArray)):
        equilNo=int(freqArray[i,0])
        velocityNo=int(freqArray[i,1])

        iniIndex=np.argwhere(np.all(iniArray[:,:2]==(equilNo,velocityNo),axis=1))[0][0]
        iniPressVol[i]=iniArray[iniIndex,2:]
        finIndex=np.argwhere(np.all(finArray[:,:2]==(equilNo,velocityNo),axis=1))[0][0]
        finPressVol[i]=finArray[finIndex,2:]
        barrierIndex=np.argwhere(np.all(NEBEnergyBarriers[:,:2]==(equilNo,velocityNo),axis=1))[0][0]
        barriers[i]=NEBEnergyBarriers[barrierIndex,2]

    vol=finPressVol[0,-1]
    conversion_factor=(1e5)*(1e-30)/1.602e-19
    gamma=vol*(finPressVol[:,1:7]-iniPressVol[:,1:7])*conversion_factor
    gammaL2=1./5*(gamma[:,0]**2+gamma[:,3]**2+gamma[:,5]**2)+2./15*(gamma[:,0]*gamma[:,3]+gamma[:,0]*gamma[:,5]+gamma[:,3]*gamma[:,5])+4./15*(gamma[:,1]**2+gamma[:,2]**2+gamma[:,4]**2)
    gammaT2=1./15*(gamma[:,0]**2+gamma[:,3]**2+gamma[:,5]**2)-1./15*(gamma[:,0]*gamma[:,3]+gamma[:,0]*gamma[:,5]+gamma[:,3]*gamma[:,5])+3./15*(gamma[:,1]**2+gamma[:,2]**2+gamma[:,4]**2)
    L2avg=np.mean(gammaL2)
    T2avg=np.mean(gammaT2)
    ratioAvg=np.mean(gammaL2/gammaT2)

# Generates gamma^2 plots and ratio plot
if plotFlag:
    print('Printing gamma^2 plots and ratio plots...')
    os.chdir(plotDirectory)
    
    plt.plot(barriers,gammaL2,'bo',mec='k')
    xlims=plt.xlim()
    plt.plot(xlims,[L2avg,L2avg],'r--')
    plt.yscale('log')
    plt.ylabel(r'$\gamma_{L}^{2}$ (eV$^{2}$)')
    plt.xlabel(r'$V$ (eV)')
    plt.title('Squared Longitudinal Deformation Potential vs Average Energy Barrier')
    plt.savefig('gammaL2.png')
    plt.clf()

    plt.plot(barriers,gammaT2,'bo',mec='k')
    xlims=plt.xlim()
    plt.plot(xlims,[T2avg,T2avg],'r--')
    plt.yscale('log')
    plt.ylabel(r'$\gamma_{T}^{2}$ (eV$^{2}$)')
    plt.xlabel(r'$V$ (eV)')
    plt.title('Squared Transverse Deformation Potential vs Average Energy Barrier')
    plt.savefig('gammaT2.png')
    plt.clf()

    plt.plot(barriers,gammaL2/gammaT2,'bo',mec='k')
    xlims=plt.xlim()
    plt.plot(xlims,[ratioAvg,ratioAvg],'r--')
    plt.yscale('log')
    plt.ylabel(r'$\gamma_{T}^{2}/\gamma_{L}^{2}$')
    plt.xlabel(r'$V$ (eV)')
    plt.title('Squared Deformation Potential Ratio vs Average Energy Barrier')
    plt.savefig('ratio.png')
    plt.clf()
