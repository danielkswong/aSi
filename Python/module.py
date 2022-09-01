import numpy as np
import tarfile
import os
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from math import sqrt,pi
import csv

def readCSV(file):
    """Reads CSV files quickly into arrays

    Parameters
    ----------
    file : str
        File name of CSV file

    Returns
    -------
    data : array
        Array of data contained in CSV file
    """
    
    with open(file,'r') as csvfile:
        csvreader=csv.reader(file)
        data=np.asarray(list(csvreader))
    return data

def readLAMMPS_quench(file,nQuenches,nAtoms,lineSkip=9):
    """Reads LAMMPS *.tar.gz quench files into arrays

    Parameters
    ----------
    file : str
        File name of LAMMPS *.tar.gz file
    nQuenches : int
        Number of quenches
    nAtoms : int
        Number of atoms
    lineSkip : int, optional
        Number of lines to skip in *.lammps file before reading data

    Returns
    -------
    pos : numpy float array (nQuenches x nAtoms x 6)
        Position array of wrapped and unwrapped x, y, and z coordinates for each atom for each quench
    coord : numpy int array (nQuenches x nAtoms)
        Coordination number array of each atom for each quench
    energy : numpy float array (nQuenches x 1)
        Potential energy array of system for each quench
    
    """

    pos=np.zeros((nQuenches,nAtoms,6),dtype=np.float64)
    coord=np.zeros((nQuenches,nAtoms),dtype=int)
    energy=np.zeros(nQuenches,dtype=np.float64)
    with tarfile.open(file,mode='r:gz') as archive:
        posFile=archive.extractfile('quench_pos.lammps')
        for quench in range(nQuenches):
            for j in range(lineSkip):
                posFile.readline()
            for atom in range(nAtoms):
                line=posFile.readline().decode('UTF-8')
                tempData=np.asarray(line.strip().split(' '),dtype=np.float64)
                pos[quench,atom,:]=tempData[2:]
        posFile.close()

        coordFile=archive.extractfile('quench_coord.lammps')
        for quench in range(nQuenches):
            for j in range(lineSkip):
                coordFile.readline()
            for atom in range(nAtoms):
                line=coordFile.readline().decode('UTF-8')
                tempData=np.asarray(line.strip().split(' '),dtype=int)
                coord[quench,atom]=tempData[2]
        coordFile.close()

        energyFile=archive.extractfile('quench_energies.lammps')
        for quench in range(nQuenches):
            tempData=energyFile.readline().decode('UTF-8').strip()
            energy[quench]=float(tempData)
        energyFile.close()

    return pos,coord,energy

def readLAMMPS_neb(file,nPartitions,nAtoms,lineSkip=9):
    """Reads LAMMPS *.tar.gz NEB files into arrays

    Parameters
    ----------
    file : str
        File name of LAMMPS *.tar.gz file
    nPartitions : int
        Number of partitions used in NEB algorithm
    nAtoms : int
        Number of atoms
    lineSkip : int, optional
        Number of lines to skip in *.lammps file before reading data

    Raises
    ------
    Exception
        Raise exception if number of files in *.tar.gz file is incorrect

    Returns
    -------
    pos : numpy float array (nPartitions x nAtoms x 6)
        Position array of wrapped and unwrapped x, y, and z coordinates for each atom for each NEB image (aka partition)
    energy : numpy float array (nPartitions x 1)
        Potential energy array of system for each partition
    V : float
        Average energy barrier
    Delta : float
        Energy asymmetry
        
    """
    
    pos=np.zeros((nPartitions,nAtoms,6),dtype=np.float64)
    with tarfile.open(file,mode='r:gz') as archive:
        files=archive.getnames()
        if len(files)!=(nPartitions+1):
            raise Exception('Incorrect number of output files in *.tar.gz file')

        for partition in np.linspace(1,50,dtype=int):
            dumpFile=archive.extractfile('dump.neb.%d' % partition)
            dumpFileText=dumpFile.readlines()
            dumpFileText=[line.decode('UTF-8') for line in dumpFileText[-(nAtoms):]]
            dumpFile.close()

            for atom in range(nAtoms):
                line=dumpFileText[atom]
                posData=np.asarray(line.strip().split(),dtype=np.float64)
                pos[partition-1,atom,:]=posData[2:]

        logFile=archive.extractfile('log.lammps')
        logText=logFile.readlines()
        logText=[line.decode('UTF-8') for line in logText]
        logFile.close()
                
    logData=np.asarray(logText[-1].strip().split(),dtype=np.float64)
    energy=logData[10::2]
    V=(logData[6]+logData[7])/2
    Delta=logData[6]-logData[7]

    return pos,energy,V,Delta

def readLAMMPS_hessian(file,nAtoms,eigTol=1e-6):
    """Calculates attempt frequencies from LAMMPS *.tar.gz Hessian files

    Parameters
    ----------
    file : str
        File name of LAMMPS *.tar.gz file
    nAtoms : int
        Number of atoms
    eigTol : float, optional
        Eigenvalue tolerance required to be considered equal to zero

    Raises
    ------
    Exception
        Raises exception if *.dat files do not have the correct number of lines
    ValueError
        Raises ValueError if dynamical matrix does not have the correct values for its eigenvalues

    Returns
    -------
    omega1 : float
        Attempt frequency of initial state of TLS
    omega2 : float
        Attempt frequency of final state of TLS
        
    """

    with tarfile.open(file,mode='r:gz') as archie:
        tsFile=archive.extractfile('ts.dat')
        tsDyn=tsFile.readlines()
        tsDyn=[line.decode('UTF-8') for line in tsDyn]
        tsFile.close()

        if len(tsDyn)!=3*nAtoms**2:
                raise Exception('Missing lines in ts.dat file')

        for j in range(len(tsDyn)):
            line=tsDyn[j].strip().split()
            tsHessian[j]=np.asarray(line,dtype=np.float64)
        tsHessian=np.reshape(tsHessian,(3*nAtoms,3*nAtoms))
        tsEigVal=np.linalg.eigvalsh(tsHessian)

        if not (tsEigVal[0]<0):
            print(tsEigVal[0])
            raise ValueError('Saddle point does not have one negative eigenvalue')
        if not np.all(abs(tsEigVal[1:4])<eigTol):
            print(tsEigVal[1:4])
            raise ValueError('Saddle point does not have three eigenvalues of zero')
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
        
        if not np.all(abs(iniEigVal[0:3])<eigTol):
            print(iniEigVal[0:3])
            raise ValueError('Initial state does not have three eigenvalues of zero')
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

        if not np.all(abs(finEigVal[0:3])<eigTol):
            print(finEigVal[0:3])
            raise ValueError('Final state does not have three eigenvalues of zero')
        print('Eigenvalues of final state diagonalized')

    omega1=sqrt(np.prod(iniEigVal[4:]/tsEigVal[4:])*iniEigVal[3])/2./pi
    omega2=sqrt(np.prod(finEigVal[4:]/tsEigVal[4:])*finEigVal[3])/2./pi

    return omega1,omega2
    
def minima_exists(data):
    """Determines if a minimum exists in data

    Parameters
    ----------
    data : numpy float array
        Numpy array of data

    Returns
    -------
    exist : boolean
        Returns true or false depending if a minimum exists or not in the data
        
    """

    exist=False
    location=len(data)-1
    for i in range(len(data)-1):
        diff=data[i+1]-data[i]
        if diff<0:
            location=i
            break
    for i in range(location,len(data)-1):
        diff=data[i+1]-data[i]
        if diff>0:
            exist=True
    return exist

def findNearestNeighbours(pos,cutoff,boxLength):
    """Finds all nearest neighbours within a shell for every atom in a position array (assuming periodic boundary conditions)

    Parameters
    ----------
    pos : numpy float array (n x 3)
        Position array of x, y, and z coordinates of n atoms
    cutoff : float
        Only atoms within the cutoff radius of the shell are considered to be nearest neighbours
        Cutoff should have the same units as pos
    boxLength : float
        Simulation box length
        Box length should have the same units as pos

    Returns
    -------
    nn : numpy list array (n x 1)
        Numpy array of lists containing indices of each atom's nearest neighbour within the cutoff
    
    """

    nAtoms=len(pos)
    nn=np.empty(nAtoms,dtype=object)
    x=pos[:,0]
    y=pos[:,1]
    z=pos[:,2]
    xs=[x-boxLength for k in range(9)]+[x for k in range(9)]+[x+boxLength for k in range(9)]
    xs=np.ndarray.flatten(np.array(xs))
    ys=[[[y-boxLength for k in range(3)],[y for k in range(3)],[y+boxLength for k in range(3)]] for l in range (3)]
    ys=np.ndarray.flatten(np.array(ys))
    zs=[[z-boxLength,z,z+boxLength] for k in range(9)]
    zs=np.ndarray.flatten(np.array(zs))
    extendedPos=np.transpose(np.vstack((xs,ys,zs)))
    for i in range(nAtoms):
        atomPos=pos[i,:3]
        disp=np.linalg.norm(extendedPos-atomPos,axis=1)
        nnInd=np.mod(np.argwhere(disp<cutoff),nAtoms)
        nnInd=nnInd[nnInd!=i]
        nn[i]=nnInd
    return nn

def findCoordination(pos,cutoff,boxLength):
    """Finds the coordination number of each atom within a shell with a radius given by the cutoff

    Parameters
    ----------
    pos : numpy float array (n x 3)
        Position array of x, y, and z coordinates of n atoms
    cutoff : float
        Only atoms within the cutoff radius of the shell are considered to be nearest neighbours
        Cutoff should have the same units as pos
    boxLength : float
        Simulation box length
        Box length should have the same units as pos

    Returns
    -------
    coordination : numpy int array (n x 1)
        Numpy array of containing coordination number of each atom
    
    """

    nn=findNearestNeighbours(pos,cutoff,boxLength)
    coordination=np.array([len(nn[i]) for i in range(len(nn))],dtype=int)
    return coordination

def findDivAndCurl(pos,boxLength,n):
    """Interpolates the displacement between atomic positions and calculates local divergence and each component of local curl

    Parameters
    ----------
    pos : numpy float array (nPartitions x n x 3)
        Position array of x, y, and z coordinates of n atoms for each partition
    boxLength : float
        Simulation box length
        Box length should have the same units as pos
    n : int
        Number of grid points used along one axis during meshing

    Returns
    -------
    divArray : numpy float array ((nPartitions-1) x (n+1) x (n+1) x (n+1))
        Local divergence on interpolated grid points
    curlXArray : numpy float array ((nPartitions-1) x (n+1) x (n+1) x (n+1))
        Local x-component of curl on interpolated grid points
    curlYArray : numpy float array ((nPartitions-1) x (n+1) x (n+1) x (n+1))
        Local y-component of curl on interpolated grid points
    curlZArray : numpy float array ((nPartitions-1) x (n+1) x (n+1) x (n+1))
        Local z-component of curl on interpolated grid points
    """
    
    nPartitions=len(pos)

    xx_=np.linspace(0,boxLength,n+1)
    yy_=np.linspace(0,boxLength,n+1)
    zz_=np.linspace(0,boxLength,n+1)

    xx,yy,zz=np.meshgrid(xx_,yy_,zz_,indexing='ij')
    xx=xx.reshape((np.prod(xx.shape),))
    yy=yy.reshape((np.prod(yy.shape),))
    zz=zz.reshape((np.prod(zz.shape),))
    gridPoints=np.column_stack((xx,yy,zz))

    consDisp=pos[1:]-pos[0:-1]
    divArray=np.zeros((nPartitions-1,n+1,n+1,n+1),dtype=np.float64)
    curlXArray=np.zeros((nPartitions-1,n+1,n+1,n+1),dtype=np.float64)
    curlYArray=np.zeros((nPartitions-1,n+1,n+1,n+1),dtype=np.float64)
    curlZArray=np.zeros((nPartitions-1,n+1,n+1,n+1),dtype=np.float64)

    for i in range(nPartitions-1):
        x=pos[i,:,0]
        y=pos[i,:,1]
        z=pos[i,:,2]

        xs=[x-boxLength for k in range(9)]+[x for k in range(9)]+[x+boxLength for k in range(9)]
        xs=np.ndarray.flatten(np.array(xs))
        ys=[[[y-boxLength for k in range(3)],[y for k in range(3)],[y+boxLength for k in range(3)]] for l in range (3)]
        ys=np.ndarray.flatten(np.array(ys))
        zs=[[z-boxLength,z,z+boxLength] for k in range(9)]
        zs=np.ndarray.flatten(np.array(zs))
        dataPoints=np.transpose(np.vstack((xs,ys,zs)))

        u=consDisp[i,:,3]
        v=consDisp[i,:,4]
        w=consDisp[i,:,5]
        us=np.concatenate((u,)*27)
        vs=np.concatenate((v,)*27)
        ws=np.concatenate((w,)*27)
        
        u_interp=interpolate.griddata(dataPoints,us,gridPoints,method='linear')
        u_interp=np.reshape(u_interp,(n+1,n+1,n+1))
        v_interp=interpolate.griddata(dataPoints,vs,gridPoints,method='linear')
        v_interp=np.reshape(v_interp,(n+1,n+1,n+1))
        w_interp=interpolate.griddata(dataPoints,ws,gridPoints,method='linear')
        w_interp=np.reshape(w_interp,(n+1,n+1,n+1))
        grad_u=np.gradient(u_interp,boxLength/n)
        grad_v=np.gradient(v_interp,boxLength/n)
        grad_w=np.gradient(w_interp,boxLength/n)

        div=grad_u[0]+grad_v[1]+grad_w[2]
        curl_x=grad_w[1]-grad_v[2]
        curl_y=grad_u[2]-grad_w[0]
        curl_z=grad_v[0]-grad_u[1]

        divArray[i]=div
        curlXArray[i]=curl_x
        curlYArray[i]=curl_y
        curlZArray[i]=curl_z

    return divArray,curlXArray,curlYArray,curlZArray
    
def findBonds(nn):
    """Finds all unique bond pairs given a list of each atom's nearest neighbour

    Parameters
    ----------
    nn : numpy list array (n x 1)
        Numpy array of lists containing indices of each atom's nearest neighbour within the cutoff

    Returns
    -------
    bonds :  numpy int array (n_bonds x 2)
        Numpy array containing the index of the atoms making up each unique bond (n_bonds in total)

    """

    firstInd=np.hstack([np.full(len(nn[i]),i) for i in range(len(nn))])
    secondInd=np.hstack(nn)
    bonds=np.c_[firstInd,secondInd]
    bonds=np.sort(bonds,axis=1)
    bonds=np.unique(bonds,axis=0)
    return bonds

def unwrapDisplacements(disp,cutoff,boxLength):
    """Unwraps the displacement values from periodic boundary conditions if a component of the displacement lies outside the cutoff range

    Parameters
    ----------
    disp : numpy float array (nDisp x 3)
        Displacement array in x, y, and z coordinates of nDisp atoms
    cutoff : float
        Only atoms within the cutoff radius of the shell are considered to be nearest neighbours
        Cutoff should have the same units as disp
    boxLength : float
        Simulation box length
        Box length should have the same units as disp

    Returns
    -------
    dx : numpy float array (nDisp x 1)
        Unwrapped displacement array of x coordinates
    dy : numpy float array (nDisp x 1)
        Unwrapped displacement array of y coordinates
    dz : numpy float array (nDisp x 1)
        Unwrapped displacement array of z coordinates

    """

    dx=disp[0]
    dy=disp[1]
    dz=disp[2]

    dx=dx-(dx>cutoff)*boxLength+(dx<-cutoff)*boxLength
    dy=dy-(dy>cutoff)*boxLength+(dy<-cutoff)*boxLength
    dz=dz-(dz>cutoff)*boxLength+(dz<-cutoff)*boxLength
    
    return dx,dy,dz

def calcQInv(omega,T,V,Delta,omega1,omega2,gammaSquared,K,E,mode='longitudinal'):
    """Calculates mechanical loss curve from array of TLS parameters (doi: 10.1103/PhysRevB.97.014201)

    Parameters
    ----------
    omega : float
        Frequency at which the mechanical loss curve is calculated in the same units as omega1 and omega2
    T : numpy float array (n x 1)
        Numpy array of floats containing temperatures of interest
    V : numpy float array (nTLS x 1)
        Numpy array of floats containing the average energy barriers for the TLS in eV
    Delta : numpy float array (nTLS x 1)
        Numpy array of floats containing the energy asymmetry between potential minima for the TLS in eV
    omega1 : numpy float array (nTLS x 1)
        Numpy array of floats containing the attempt frequency of the first potential minima
    omega2 : numpy float array (nTLS x 1)
        Numpy array of floats containing the attempt frequency of the second potential minima
    gammaSquared : numpy float array (nTLS x 1)
        Numpy array of floats containing either the longitudinal or transverse squared deformation potential
    K : float
        Bulk modulus in Pa
    E : float
        Young's modulus in Pa
    mode : str, optional
        String indicating which mode of mechanical loss if calculated
        'longitudinal' : Longitudinal mechanical loss
        'transverse' : Transverse mechanical loss

    Returns
    -------
    QInv : numpy float array (n x 1)
        Mechanical loss calculated for TLS
    """

    kB=8.617333262e-5
    G=3*K*E/(9*K-E)
    M=3*K*(3*K+E)/(9*K-E)

    A=1./(4.*kB*T)/(np.cosh(energyAsym/(2.*kB*T)+0.5*np.log(omega2/omega1))**2)
    tau=pi/np.sqrt(omega1*omega2)*np.exp(V/(kB*T))/np.cosh(Delta/(2.*kB*T)+0.5*np.log(omega2/omega1))
    summand=A*omega*tau*gammaSquared/(1+omega**2*tau**2)
    invQArray=1./(vol*M*conversion_factor_2)*np.sum(summandLArray,axis=2)
    print('INCOMPLETE!!!')
    return

def calc_q4(pos,nn,boxLength):
    """Calculates tetrahedrality parameter q4 (doi: 10.1038/35053024) from bond angles of
    nearest neighbours for every atom in a position array (assuming periodic boundary conditions)

    Parameters
    ----------
    pos : numpy float array (n x 3)
        Position array of x, y, and z coordinates of n atoms
    nn : numpy list array (n x 1)
        Numpy array of lists containing indices of each atom's nearest neighbours
    boxLength : float
        Simulation box length
        Box length should have the same units of pos

    Returns
    -------
    q4 : numpy float array (n x 1)
        Array of q4 order parameter for each atom
    
    """

    nAtoms=len(pos)
    q4=np.zeros(nAtoms,dtype=np.float64)
    for i in range(nAtoms):
        x=pos[nn[i],0]
        y=pos[nn[i],1]
        z=pos[nn[i],2]
        xs=[x-boxLength for k in range(9)]+[x for k in range(9)]+[x+boxLength for k in range(9)]
        xs=np.ndarray.flatten(np.array(xs))
        ys=[[[y-boxLength for k in range(3)],[y for k in range(3)],[y+boxLength for k in range(3)]] for l in range (3)]
        ys=np.ndarray.flatten(np.array(ys))
        zs=[[z-boxLength,z,z+boxLength] for k in range(9)]
        zs=np.ndarray.flatten(np.array(zs))
        extendedPos=np.transpose(np.vstack((xs,ys,zs)))

        disp=pos[i,:3]-extendedPos
        netDisp=np.sqrt(disp[:,0]**2+disp[:,1]**2+disp[:,2]**2)
        disp=disp[netDisp<cutoff]
        dotProd=np.dot(disp,disp.T)
        mag=np.sqrt(np.outer(np.diag(dotProd),np.diag(dotProd)))
        cosMatrix=dotProd/mag
        cosArray=cosMatrix[np.triu_indices_from(cosMatrix,k=1)]
        q4[i]=1.-3./8.*np.sum((cosArray+1./3.)**2.)
    return q4
    
def calc_theta(pos1,pos2,bonds,cutoff,boxLength):
    """Calculates angular displacement theta in degrees for each bond (assuming periodic boundary conditions)

    Parameters
    ----------
    pos1 : numpy float array (n x 3)
        Initial position array of x, y, and z coordinates of n atoms
    pos2 : numpy float array (n x 3)
        Final position array of x, y, and z coordinates of n atoms
    bonds : numpy int array (nBonds x 2)
        Numpy array containing the index of the atoms making up each unique bond (nBonds in total)
    cutoff : float
        Only atoms within the cutoff radius of the shell are considered to be nearest neighbours
        Cutoff should have the same units as pos1 and pos2
    boxLength : float
        Simulation box length
        Box length should have the same units of pos1 and pos2

    Returns
    -------
    theta : numpy float array (nBonds x 1)
        Numpy array of floats containing angular displacement of each atom's initial bond within the cutoff
        
    """

    nBonds=len(bonds)
    theta=np.empty(nBonds,dtype=np.float64)
    
    disp1=pos1[bonds[:,0]]-pos1[bonds[:,1]]
    dx1,dy1,dz1=unwrapDisplacements(disp1,cutoff,boxLength)

    disp2=pos2[bonds[:,0]]-pos2[bonds[:,1]]
    dx2,dy2,dz2=unwrapDisplacements(disp2,cutoff,boxLength)

    cosArray=(dx1*dx2+dy1*dy2+dz1*dz2)/np.sqrt(dx1**2+dy1**2+dz1**2)/np.sqrt(dx2**2+dy2**2+dz2**2)
    theta=np.arccos(np.clip(cosArray,-1.0,1.0))/pi*180
    return theta

def checkSlurmFiles(jobID,jobType='cool',lenThresh=100):
    """Checks for unsuccessful Slurm jobs during batch run

    Parameters
    ----------
    jobID : int
        Slurm job ID
    jobType : str, optional
        String indicating type of job
        'cool' : Outputs from cooling
        'quench' : Outputs from minima search through quenches
        'neb' : Outputs from NEB algorithm
    lenThresh : int, optional
        File must be greater than length threshold to be considered 

    Raises
    ------
    Exception
        Raise exception if job type is not 'cool', 'quench', or 'neb'

    Returns
    -------
    fails : numpy str array
        Numpy array containing failed job names
    
    """

    jobTypes=['cool','quench','neb']
    if (jobType not in jobTypes):
        raise Exception('Job type is not allowed')
    
    files=[]
    for file in os.listdir(os.getcwd()):
        if file.startswith('slurm-'+str(jobID)):
            files.append(file)

    fails=[]
    if jobType=='cool':
        for file in files:
            with open(file,'r') as f:
                text=f.readlines()
            if text[-2]!='System init for write_restart ...\n':
                fails.append(file)
            elif len(text)<lenThresh:
                fails.append(file)
            else:
                continue
            print(file+' failed')
    elif jobType=='quench':
        for file in files:
            with open(file,'r') as f:
                text=f.readlines()
            if text[-1]!='quench_pos.lammps\n':
                fails.append(file)
            elif len(text)<lenThresh:
                fails.append(file)
            else:
                continue
            print(file+' failed')
    elif jobType=='neb':
        for file in files:
            with open(file,'r') as f:
                text=f.readlines()
            if text[-2]!='log.lammps\n':
                fails.append(file)
            elif len(text)<lenThresh:
                fails.append(file)
            else:
                continue
            print(file+' failed')
    return fails
    
