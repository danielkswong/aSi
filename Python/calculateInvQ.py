import numpy as np
import tarfile
import os
import math
import csv
import matplotlib.pyplot as plt


# Flags
plotFlag=1

# Relevant parameters
T=600
kB=8.617333262e-5
fileLength=8
screenX=13.66
screenY=7.68
dataDirectory='../Data_v1/'
plotDirectory='../Plots_v1/'

# Modulus in Pa
K=90.9*1e9
E=125*1e9
G=3*K*E/(9*K-E)
M=3*K*(3*K+E)/(9*K-E)

# Reads in TLS parameters
print('Reading CSV files...')

os.chdir(dataDirectory)
with open('ini_press_vol.csv','r') as csvfile:
    csvreader=csv.reader(csvfile,delimiter=',')
    iniArray=np.asarray(list(csvreader),dtype=np.float64)
with open('fin_press_vol.csv','r') as csvfile:
    csvreader=csv.reader(csvfile,delimiter=',')
    finArray=np.asarray(list(csvreader),dtype=np.float64)
with open('barriers_T='+str(T)+'_NEB_no_minima.csv','r') as csvfile:
    csvreader=csv.reader(csvfile,delimiter=',')
    NEBEnergyBarriers=np.asarray(list(csvreader),dtype=np.float64)
with open('energyAsym_T='+str(T)+'_NEB_no_minima.csv','r') as csvfile:
    csvreader=csv.reader(csvfile,delimiter=',')
    NEBEnergyAsym=np.asarray(list(csvreader),dtype=np.float64)
with open('freq_hessian.csv','r') as csvfile:
    csvreader=csv.reader(csvfile,delimiter=',')
    freqArray=np.asarray(list(csvreader),dtype=np.float64)

barriers=np.zeros(len(freqArray),dtype=np.float64)
energyAsym=np.zeros(len(freqArray),dtype=np.float64)
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
    energyAsymIndex=np.argwhere(np.all(NEBEnergyAsym[:,:2]==(equilNo,velocityNo),axis=1))[0][0]
    energyAsym[i]=NEBEnergyAsym[energyAsymIndex,2]
omega1=freqArray[:,2]
omega2=freqArray[:,3]

# Calculating deformation potential tensor
vol=finPressVol[0,-1]
conversion_factor=(1e5)*(1e-30)/1.602e-19
conversion_factor_2=(1e-30)/1.602e-19
gamma=vol*(finPressVol[:,1:7]-iniPressVol[:,1:7])*conversion_factor
gammaL2=1./5*(gamma[:,0]**2+gamma[:,3]**2+gamma[:,5]**2)+2./15*(gamma[:,0]*gamma[:,3]+gamma[:,0]*gamma[:,5]+gamma[:,3]*gamma[:,5])+4./15*(gamma[:,1]**2+gamma[:,2]**2+gamma[:,4]**2)
gammaT2=1./15*(gamma[:,0]**2+gamma[:,3]**2+gamma[:,5]**2)-1./15*(gamma[:,0]*gamma[:,3]+gamma[:,0]*gamma[:,5]+gamma[:,3]*gamma[:,5])+3./15*(gamma[:,1]**2+gamma[:,2]**2+gamma[:,4]**2)
L2avg=np.mean(gammaL2)
T2avg=np.mean(gammaT2)

# Calculating summands (longitudinal and transverse) for different TLS at various T and omega
TArray=np.linspace(10,210,20001)
inputFreq=np.asarray([1e-10,1e-9,1e-8])
summandLArray=np.zeros((len(inputFreq),len(TArray),len(freqArray)),dtype=np.float64)
summandTArray=np.zeros((len(inputFreq),len(TArray),len(freqArray)),dtype=np.float64)
for i in np.arange(len(inputFreq)):
    for j in np.arange(len(TArray)):
        omega=inputFreq[i]
        T=TArray[j]
        A=1./(4.*kB*T)/(np.cosh(energyAsym/(2.*kB*T)+0.5*np.log(omega2/omega1))**2)
        tau=math.pi/np.sqrt(omega1*omega2)*np.exp(barriers/(kB*T))/np.cosh(energyAsym/(2.*kB*T)+0.5*np.log(omega2/omega1))
        summandL=A*omega*tau*gammaL2/(1+omega**2*tau**2)
        summandLArray[i,j]=summandL
        summandT=A*omega*tau*gammaT2/(1+omega**2*tau**2)
        summandTArray[i,j]=summandT
invQArrayL=1./(vol*M*conversion_factor_2)*np.sum(summandLArray,axis=2)
invQArrayT=1./(vol*G*conversion_factor_2)*np.sum(summandTArray,axis=2)

intSummandLArray=np.sum(summandLArray,axis=1)
intSummandTArray=np.sum(summandTArray,axis=1)

maxTLSindL=np.argpartition(-intSummandLArray[0],13)[:13]
maxTLSindL.sort()
maxTLSindT=np.argpartition(-intSummandTArray[0],14)[:14]
maxTLSindT.sort()
maxTLSind=np.union1d(maxTLSindL,maxTLSindT)
maxTLSNumbers=freqArray[maxTLSind,:2]
np.savetxt('maxTLSNumbers.csv',maxTLSNumbers,fmt='%i',delimiter=',')

if plotFlag:
    # Generates plots
    os.chdir(plotDirectory)
    
    print('Plotting mechanical loss (longitudinal and transverse)...')
    
    plt.plot(TArray,invQArrayL[0],label=r'$\omega$ = 100 Hz')
    plt.plot(TArray,invQArrayL[1],label=r'$\omega$ = 1000 Hz')
    plt.plot(TArray,invQArrayL[2],label=r'$\omega$ = 10000 Hz')
    plt.yscale('log')
    plt.title(r'Longitudinal $Q^{-1}$ vs $T$')
    plt.ylabel(r'$Q^{-1}$')
    plt.xlabel(r'$T$ (K)')
    plt.legend(bbox_to_anchor=(1,1),loc='upper left')
    fig=plt.gcf()
    fig.set_size_inches(screenX,screenY)
    plt.savefig('invQ_L.png',bbox_inches='tight')
    plt.clf()

    plt.plot(TArray,invQArrayT[0],label=r'$\omega$ = 100 Hz')
    plt.plot(TArray,invQArrayT[1],label=r'$\omega$ = 1000 Hz')
    plt.plot(TArray,invQArrayT[2],label=r'$\omega$ = 10000 Hz')
    plt.yscale('log')
    plt.title(r'Transverse $Q^{-1}$ vs $T$')
    plt.ylabel(r'$Q^{-1}$')
    plt.xlabel(r'$T$ (K)')
    plt.legend(bbox_to_anchor=(1,1),loc='upper left')
    fig=plt.gcf()
    fig.set_size_inches(screenX,screenY)
    plt.savefig('invQ_T.png',bbox_inches='tight')
    plt.clf()

    index=11500
    T=int(TArray[index])
    
    for i in np.arange(len(inputFreq)):
        plt.plot(summandLArray[i,index,:],'b-')
        plt.title(r'Summand Magnitude for Longitudinal $Q^{-1}$')
        plt.xlabel(r'TLS Candidate')
        plt.ylabel(r'Summand')
        plt.yscale('log')
        plt.savefig('summandL_T='+str(T)+'_omega='+str(inputFreq[i])+'.png')
        plt.clf()

        plt.plot(summandTArray[i,index,:],'b-')
        plt.title(r'Summand Magnitude for Transverse $Q^{-1}$')
        plt.xlabel(r'TLS Candidate')
        plt.ylabel(r'Summand')
        plt.yscale('log')
        plt.savefig('summandT_T='+str(T)+'_omega='+str(inputFreq[i])+'.png')
        plt.clf()

        plt.plot(intSummandLArray[i],'b-')
        plt.title(r'Integrated Summand Magnitude for Longitudinal $Q^{-1}$')
        plt.xlabel(r'TLS Candidate')
        plt.ylabel(r'Integrated Summand')
        plt.savefig('int_summandL_omega='+str(inputFreq[i])+'.png')
        plt.clf()

        plt.plot(intSummandTArray[i],'b-')
        plt.title(r'Integrated Summand Magnitude for Transverse $Q^{-1}$')
        plt.xlabel(r'TLS Candidate')
        plt.ylabel(r'Integrated Summand')
        plt.savefig('int_summandT_omega='+str(inputFreq[i])+'.png')
        plt.clf()

    colour=plt.cm.rainbow(np.linspace(0,1,len(maxTLSindL)))
    for i in np.arange(len(inputFreq)):
        plt.plot(TArray,invQArrayL[i],'k-')
        plt.yscale('log')
        plt.title(r'Longitudinal $Q^{-1}$ vs $T$')
        plt.ylabel(r'$Q^{-1}$')
        plt.xlabel(r'$T$ (K)')
        for j in np.arange(len(maxTLSindL)):
            ind=maxTLSindL[j]
            plt.plot(TArray,1./(vol*M*conversion_factor_2)*summandLArray[i,:,ind],label='TLS %03d-%02d' % (freqArray[ind,0],freqArray[ind,1]),c=colour[j],alpha=0.8)
        plt.legend(bbox_to_anchor=(1,1),loc='upper left')
        plt.ylim(1e-8,1e-1)
        fig=plt.gcf()
        fig.set_size_inches(screenX,screenY)
        plt.savefig('sig_summandL_omega='+str(inputFreq[i])+'.png',bbox_inches='tight')
        plt.clf()

    colour=plt.cm.rainbow(np.linspace(0,1,len(maxTLSindT)))
    for i in np.arange(len(inputFreq)):
        plt.plot(TArray,invQArrayT[i],'k-')
        plt.yscale('log')
        plt.title(r'Transverse $Q^{-1}$ vs $T$')
        plt.ylabel(r'$Q^{-1}$')
        plt.xlabel(r'$T$ (K)')
        for j in np.arange(len(maxTLSindT)):
            ind=maxTLSindT[j]
            plt.plot(TArray,1./(vol*G*conversion_factor_2)*summandTArray[i,:,ind],label='TLS %03d-%02d' % (freqArray[ind,0],freqArray[ind,1]),c=colour[j],alpha=0.8)
        plt.legend(bbox_to_anchor=(1,1),loc='upper left')
        plt.ylim(1e-8,1e-1)
        fig=plt.gcf()
        fig.set_size_inches(screenX,screenY)
        plt.savefig('sig_summandT_omega='+str(inputFreq[i])+'.png',bbox_inches='tight')
        plt.clf()
