import tarfile
import numpy as np
import math
import os
import csv
import matplotlib.pyplot as plt

# Flags
readFlag=1
writeFlag=1-readFlag
plotFlag=1

# Relevant parameters
nAtoms=1000
eigTol=1e-6
T=600
dataDirectory='../Data_v1/'
plotDirectory='../Plots_v1/'
    
if writeFlag:
    # Find all Hessian files
    os.chdir(dataDirectory)
    hessianFiles=[]
    for file in os.listdir(os.getcwd()):
        if file.startswith('Equil') and file.endswith('_Hessian.tar.gz'):
            hessianFiles.append(file)
    nFiles=len(hessianFiles)
    hessianFiles.sort()

    freqArray=[]

    # Read Hessian files
    for i in np.arange(nFiles):
        
        print('Reading '+hessianFiles[i]+'...')

        equilNo=int(hessianFiles[i][5:8])
        velocityNo=int(hessianFiles[i].replace('_Hessian.tar.gz','').split('-')[1])
        
        iniHessian=np.zeros((3*nAtoms**2,3),dtype=np.float64)
        tsHessian=np.zeros((3*nAtoms**2,3),dtype=np.float64)
        finHessian=np.zeros((3*nAtoms**2,3),dtype=np.float64)

        with tarfile.open(hessianFiles[i],mode='r:gz') as archive:
            tsFile=archive.extractfile('ts.dat')
            tsDyn=tsFile.readlines()
            tsDyn=[line.decode('UTF-8') for line in tsDyn]
            tsFile.close()
            
            if len(tsDyn)!=3*nAtoms**2:
                raise Exception('Missing lines in ts.dat file')

            for j in np.arange(len(tsDyn)):
                line=tsDyn[j].strip().split()
                tsHessian[j]=np.asarray(line,dtype=np.float64)
            tsHessian=np.reshape(tsHessian,(3*nAtoms,3*nAtoms))
            tsEigVal=np.linalg.eigvalsh(tsHessian)

            if not (tsEigVal[0]<0):
                print('Saddle point does not have one negative eigenvalue')
                print(tsEigVal[0])
                continue
            if not np.all(tsEigVal[1:4]<eigTol):
                print('Saddle point does not have three eigenvalues of zero')
                print(tsEigVal[1:4])
                continue
            if not np.all(-eigTol<tsEigVal[1:4]):
                print('Saddle point does not have three eigenvalues of zero')
                print(tsEigVal[1:4])
                continue
            print('Eigenvalues of transition state diagonalized')

            iniFile=archive.extractfile('ini.dat')
            iniDyn=iniFile.readlines()
            iniDyn=[line.decode('UTF-8') for line in iniDyn]
            iniFile.close()

            if len(iniDyn)!=3*nAtoms**2:
                raise Exception('Missing lines in ini.dat file')

            for j in np.arange(len(iniDyn)):
                line=iniDyn[j].strip().split()
                iniHessian[j]=np.asarray(line,dtype=np.float64)
            iniHessian=np.reshape(iniHessian,(3*nAtoms,3*nAtoms))
            iniEigVal=np.linalg.eigvalsh(iniHessian)
            
            if not np.all(iniEigVal[0:3]<eigTol):
                print('Initial state does not have three eigenvalues of zero')
                print(iniEigVal[0:3])
                continue
            if not np.all(-eigTol<iniEigVal[0:3]):
                print('Initial state does not have three eigenvalues of zero')
                print(iniEigVal[0:3])
                continue
            print('Eigenvalues of initial state diagonalized')

            finFile=archive.extractfile('fin.dat')
            finDyn=finFile.readlines()
            finDyn=[line.decode('UTF-8') for line in finDyn]
            finFile.close()
        
            if len(finDyn)!=3*nAtoms**2:
                raise Exception('Missing lines in fin.dat file')

            for j in np.arange(len(finDyn)):
                line=finDyn[j].strip().split()
                finHessian[j]=np.asarray(line,dtype=np.float64)
            finHessian=np.reshape(finHessian,(3*nAtoms,3*nAtoms))
            finEigVal=np.linalg.eigvalsh(finHessian)

            if not np.all(finEigVal[0:3]<eigTol):
                print('Final state does not have three eigenvalues of zero')
                print(finEigVal[0:3])
                continue
            if not np.all(-eigTol<finEigVal[0:3]):
                print('Final state does not have three eigenvalues of zero')
                print(finEigVal[0:3])
                continue
            print('Eigenvalues of final state diagonalized')

        omega1=np.sqrt(np.prod(iniEigVal[4:]/tsEigVal[4:])*iniEigVal[3])/2/math.pi
        omega2=np.sqrt(np.prod(finEigVal[4:]/tsEigVal[4:])*finEigVal[3])/2/math.pi

        freqs=[equilNo,velocityNo,omega1,omega2]
        freqArray.append(freqs)

    # Save attempt frequencies in CSV file
    freqArray=np.asarray(freqArray,dtype=np.float64)
    np.savetxt('freq_hessian.csv',freqArray,delimiter=',')

if readFlag:
    # Read CSV files
    print('Reading CSV file...')

    os.chdir(dataDirectory)
    with open('freq_hessian.csv','r') as csvfile:
        csvreader=csv.reader(csvfile,delimiter=',')
        freqArray=np.asarray(list(csvreader),dtype=np.float64)
    with open('barriers_T='+str(T)+'_NEB_no_minima.csv','r') as csvfile:
        csvreader=csv.reader(csvfile,delimiter=',')
        NEBEnergyBarriers=np.asarray(list(csvreader),dtype=np.float64)

    barrierIndices=[]
    for indices in freqArray[:,:2]:
        VList=np.all(NEBEnergyBarriers[:,:2]==indices,1)
        VIndex=np.where(VList==True)[0][0]
        barrierIndices.append(VIndex)

    omega1=freqArray[:,2]
    omega2=freqArray[:,3]
    barrier=NEBEnergyBarriers[barrierIndices,2]
    avg=np.mean(np.sqrt(omega1*omega2))
    
# Generate plots of attempt frequencies
if plotFlag:
    os.chdir(plotDirectory)
    
    plt.plot(barrier,np.sqrt(omega1*omega2),'bo',mec='k')
    xlims=plt.xlim()
    plt.plot(xlims,[avg,avg],'r--')
    plt.yscale('log')
    plt.ylabel(r'$\sqrt{\omega_{1}\omega_{2}}$ (THz)')
    plt.xlabel(r'$V$ (eV)')
    plt.title('Effective Attempt Frequency vs Average Energy Barrier')
    plt.savefig('eff_freq.png')
    plt.clf()
    
