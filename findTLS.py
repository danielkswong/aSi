# Determine TLS parameters

# Import libraries
import os
import numpy as np
from scipy.signal import argrelextrema
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import math

# Flags
plotFlag=0
writeFlag=0
checkFlag=0

# Relevant parameters
T=700
nAtoms=1000
nDump=10
nPartitions=50
maxIter=20000
conversion_factor=(1./nAtoms)*(1./6.022e23)*(4184.)*(1/1.602e-19)
minStyle="quickmin"

# Read in quench position files
quenchFiles=[]
dir_path=os.path.dirname(os.path.realpath(__file__))
for file in os.listdir(dir_path):
    if file.startswith("quench_positions_T="+str(T)+"_"):
            quenchFiles.append(file)
nFiles=len(quenchFiles)

# Sort by time
##quenchFiles.sort(key=lambda x: os.stat(os.path.join(dir_path, x)).st_ctime)

# Sort by name
files=np.empty(nFiles,dtype=object)
for i in range(nFiles):
    file=quenchFiles[i].replace("i=","")
    file=file.replace(".lammps","")
    fileName=file.split("_")
    index=int(fileName[3])
    files[index]=quenchFiles[i]
quenchFiles=files


positionArray=[]
for i in range(nFiles):
    tempArray=[]
    file=open(quenchFiles[i],'r')
    for i in range(9):
        file.readline()
    for line in file:
        stringArray=line.strip()
        stringArray=stringArray.split(' ')
        tempArray.append(stringArray)
    positionArray.append(tempArray)
    file.close()    

positionArray=np.asarray(positionArray,dtype=np.float64)

# Read in quench energies
energyFile=open("quench_energies_T="+str(T)+".txt")
energyArray=[]
for line in energyFile:
    energyArray.append(float(line))
energyArray=np.array(energyArray)
energyArray=energyArray*conversion_factor
energyFile.close()

# Process data

iniDisp=positionArray[1:]-positionArray[0]
consDisp=positionArray[1:]-positionArray[0:-1]

# Net displacement squared from initial position/previous position for each atom in a given dump
netDisp2=iniDisp[:,:,5]**2+iniDisp[:,:,6]**2+iniDisp[:,:,7]**2
netConsDisp2=consDisp[:,:,5]**2+consDisp[:,:,6]**2+consDisp[:,:,7]**2
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
PR=np.sum(consDisp[:,:,5]**2+consDisp[:,:,6]**2+consDisp[:,:,7]**2,1)**2/np.sum(consDisp[:,:,5]**4+consDisp[:,:,6]**4+consDisp[:,:,7]**4,1)
# Energy asymmetry
energyAsym=energyArray[1:]-energyArray[0:-1]


# Distance criteria
# Damart and Rodney
critD=1
dispD=avgConsDisp*nAtoms
percentD=float(len(dispD[dispD>critD]))/len(energyAsym)
indD=np.argwhere(dispD>critD).flatten()
# Cheng
critC=0.01
dispC=np.sum(np.sqrt(netConsDisp2),axis=1)/nAtoms
percentC=float(len(dispC[dispC>critC]))/len(energyAsym)
indC=np.argwhere(dispC>critC).flatten()

indCrit=sorted(set(indC).intersection(indD))

# Read in barriers and stopping iteration
barriers=[]
iterations=[]
replicas=np.zeros((len(indCrit),nPartitions))
file=open('barrier_'+minStyle+'_'+str(nPartitions)+'x1.lammps','r')
lineNumber=0
for i in range(4):
    lineNumber+=1
    file.readline()
previousTwoLine=[]
previousLine=[]
index=0
j=0
for currentLine in file:
    lineNumber+=1
    stringArray=currentLine.strip()
    stringArray=stringArray.split()
    if stringArray[0]=='0':
        barriers.append(float(previousTwoLine[6]))
        if checkFlag and barriers[-1]*conversion_factor>0.05:
            print(lineNumber-403)
            print(barriers[-1])
        iterations.append(float(previousTwoLine[0]))
        if any(i==index for i in indCrit):
            replicas[j]=np.asarray(previousTwoLine[10::2],dtype=np.float64)
            j+=1
        index+=1
    previousTwoLine=previousLine
    previousLine=stringArray
file.close()
barriers.append(float(previousLine[6]))
barriers=np.asarray(barriers)
barriers=barriers*conversion_factor
iterations.append(float(previousLine[0]))
iterations=np.asarray(iterations)

# Threshold for top ten most displaced positions
thresh=np.sort(avgConsDisp)[-10]
# Array containing dump number for TLS
consTLSArray=np.where(avgConsDisp>=thresh)[0]
# Indices of events past the distance criterion
hist_fit=np.histogram(avgConsDisp2,bins="auto",normed=False)
d0_2=np.min(argrelextrema(hist_fit[0],np.less_equal))
d0_2=(hist_fit[1][d0_2]+hist_fit[1][d0_2+1])/2
ind_c=np.where(avgConsDisp2>d0_2)[0]
ind_nc=np.where(avgConsDisp2<=d0_2)[0]
# Indices of non-zero energy barriers
ind_nzeb=np.where(barriers>1.0e-3)[0]
ind_zeb=np.where(barriers<=1.0e-3)[0]

# Generate plots and PDFs
if plotFlag:
    plt.plot(avgDisp)
    plt.title("Average Displacement from Initial Position over Time, T = " + str(T))
    plt.xlabel("Time (arbitrary)")
    plt.ylabel("Average displacement from initial position d (angstrom)")
    plt.savefig("avg_disp_plot_T="+str(T)+".png")
    plt.clf()

    plt.hist(avgDisp,bins="auto",normed=False)
    plt.title("Average Displacement from Initial Position PDF, T = " + str(T))
    plt.xlabel("Average displacement from initial position d (angstrom)")
    plt.ylabel("P(d)")
    plt.savefig("avg_disp_PDF_T="+str(T)+".png")
    plt.clf()

if plotFlag:
    plt.plot(avgDisp2)
    plt.title("Average Displacement Squared from Initial Position over Time, T = " + str(T))
    plt.xlabel("Time (arbitrary)")
    plt.ylabel("Average displacement squared from initial position d^2 (angstrom)")
    plt.savefig("avg_disp2_plot_T="+str(T)+".png")
    plt.clf()
    
    plt.hist(avgDisp2,bins="auto",normed=False)
    plt.title("Average Displacement Squared from Initial Position PDF, T = " + str(T))
    plt.xlabel("Average displacement squared from initial position d^2 (angstrom)")
    plt.ylabel("P(d^2)")
    plt.savefig("avg_disp2_PDF_T="+str(T)+".png")
    plt.clf()

if plotFlag:
    plt.plot(avgConsDisp)
    plt.plot(consTLSArray,avgConsDisp[consTLSArray],'ro')
    plt.title("Average Displacement from Previous Position over Time, T = " + str(T))
    plt.xlabel("Time (arbitrary)")
    plt.ylabel("Average displacement from previous position d (angstrom)")
    plt.savefig("avg_cons_disp_plot_T="+str(T)+".png")
    plt.clf()
    
    plt.hist(avgConsDisp,bins="auto",normed=False)
    plt.title("Average Displacement from Previous Position PDF, T = " + str(T))
    plt.xlabel("Average displacement from previous position d (angstrom)")
    plt.ylabel("P(d)")
    plt.savefig("avg_cons_disp_PDF_T="+str(T)+".png")
    plt.clf()

if plotFlag:
    plt.plot(avgConsDisp2)
    plt.title("Average Displacement Squared from Previous Position over Time, T = " + str(T))
    plt.xlabel("Time (arbitrary)")
    plt.ylabel("Average displacement squared from previous position d^2 (angstrom)")
    plt.savefig("avg_cons_disp2_plot_T="+str(T)+".png")
    plt.clf()

    plt.hist(avgConsDisp2,bins="auto",normed=False)
    plt.vlines(d0_2,0,np.max(hist_fit[0]),linestyles='dashed',colors='red',label="x = "+"{:.3e}".format(d0_2,s=3))
    plt.legend(loc="upper right")
    plt.title("Average Displacement Squared from Previous Position PDF, T = " + str(T))
    plt.xlabel("Average displacement squared from previous position d^2 (angstrom)")
    plt.ylabel("P(d^2)")
    plt.savefig("avg_cons_disp2_PDF_T="+str(T)+".png")
    plt.clf()

if plotFlag:
    plt.hist(maxConsDisp[ind_c],bins="auto",normed=False)
    plt.title("Max Displacement From Previous Position PDF, T = " + str(T))
    plt.xlabel("Max displacement from previous position d_max (angstrom)")
    plt.ylabel("P(d_max)")
    plt.savefig("max_cons_disp_PDF_T="+str(T)+".png")
    plt.clf()

if plotFlag:
    plt.plot(PR)
    plt.title("Participation Ratio (All) over Time, T = " + str(T))
    plt.xlabel("Time (arbitrary)")
    plt.ylabel("Participation ratio PR")
    plt.savefig("PR_plot_T="+str(T)+".png")
    plt.clf()
    
    plt.hist(PR,bins="auto",normed=False)
    plt.title("Participation Ratio (All) PDF, T = " + str(T))
    plt.xlabel("Participation ratio PR")
    plt.ylabel("P(PR)")
    plt.savefig("PR_PDF_T="+str(T)+".png")
    plt.clf()

    plt.plot(PR[ind_nc])
    plt.title("Participation Ratio (Non-Cage Breaking) over Time, T = " + str(T))
    plt.xlabel("Time (arbitrary)")
    plt.ylabel("Participation ratio PR")
    plt.savefig("PR_nc_plot_T="+str(T)+".png")
    plt.clf()
    
    plt.hist(PR[ind_nc],bins="auto",normed=False,log=False)
    plt.title("Participation Ratio (Non-Cage Breaking) PDF, T = " + str(T))
    plt.xlabel("Participation ratio PR")
    plt.ylabel("P(PR)")
    plt.savefig("PR_nc_PDF_T="+str(T)+".png")
    plt.clf()

    plt.hist(PR[ind_nc],bins="auto",normed=False,log=False)
    plt.title("Participation Ratio (Non-Cage Breaking, Zoomed) PDF, T = " + str(T))
    plt.xlabel("Participation ratio PR")
    plt.ylabel("P(PR)")
    plt.xlim((0,100))
    plt.savefig("PR_nc_zoom_PDF_T="+str(T)+".png")
    plt.clf()

if plotFlag:
    plt.plot(abs(energyAsym))
    plt.xlabel("Time (arbitrary)")
    plt.ylabel("Energy Asymmetry Delta V (eV)")
    plt.xlim((0,len(energyAsym)))
    plt.title("Energy Asymmetry over Time, T = " +str(T))
    plt.savefig("energy_asym_plot_T="+str(T)+".png")
    plt.clf()

    plt.hist(abs(energyAsym),bins=40,normed=False,log=False)
    plt.title("Energy Asymmetry PDF, T = " + str(T))
    plt.xlabel("Energy Asymmetry Delta V (eV)")
    plt.ylabel("P(Delta V)")
    plt.savefig("energy_asym_PDF_T="+str(T)+".png")
    plt.clf()

    plt.hist(abs(energyAsym)[ind_nc],bins=40,normed=False,log=False)
    plt.title("Energy Asymmetry (Non-Cage Breaking) PDF, T = " + str(T))
    plt.xlabel("Energy Asymmetry Delta V (eV)")
    plt.ylabel("P(Delta V)")
    plt.savefig("energy_asym_nc_PDF_T="+str(T)+".png")
    plt.clf()

if plotFlag:
    plt.plot(barriers)
    plt.xlabel("Time (arbitrary)")
    plt.ylabel("Energy Barrier V (eV)")
    plt.xlim((0,len(barriers)))
    plt.ylim((0,max(barriers)))
    plt.title("Energy Barrier over Time, T = " +str(T))
    plt.savefig("energy_barrier_plot_T="+str(T)+"_"+minStyle+"_"+str(nPartitions)+"x1.png")
    plt.clf()

    plt.hist(barriers,bins=40,normed=False,log=False)
    plt.title("Energy Barrier PDF, T = " + str(T))
    plt.xlabel("Energy Barrier V (eV)")
    plt.ylabel("P(V)")
    plt.xlim((0,max(barriers)))
    plt.savefig("energy_barrier_PDF_T="+str(T)+"_"+minStyle+"_"+str(nPartitions)+"x1.png")
    plt.clf()

if plotFlag:
    plt.hist(abs(energyAsym)[ind_nzeb],bins=40,normed=False,log=False)
    plt.title("Energy Asymmetry (Non-Zero Energy Barrier) PDF, T = " + str(T))
    plt.xlabel("Energy Asymmetry Delta V (eV)")
    plt.ylabel("P(Delta V)")
    plt.savefig("energy_asym_nzeb_PDF_T="+str(T)+".png")
    plt.clf()

    plt.hist(PR[ind_nzeb],bins="auto",normed=False,log=False)
    plt.title("Participation Ratio (Non-Zero Energy Barrier) PDF, T = " + str(T))
    plt.xlabel("Participation ratio PR")
    plt.ylabel("P(PR)")
    plt.savefig("PR_nzeb_PDF_T="+str(T)+".png")
    plt.clf()

    plt.hist(barriers[ind_nzeb],bins="auto",normed=False,log=False)
    plt.title("Energy Barriers (Non-Zero Energy Barrier) PDF, T = " + str(T))
    plt.xlabel("Energy Barriers V (eV)")
    plt.ylabel("P(V)")
    plt.savefig("energy_barriers_nzeb_PDF_T="+str(T)+"_"+minStyle+"_"+str(nPartitions)+"x1.png")
    plt.clf()

if plotFlag:
    plt.plot(barriers,abs(energyAsym),'o')
    plt.title("Energy Barrier vs Energy Asymmetry, T = " + str(T))
    plt.xlabel("Energy Barrier V (eV)")
    plt.ylabel("Energy Asymmetry delta V (eV)")
    plt.savefig("energy_barrier_asym_plot_T="+str(T)+"_"+minStyle+"_"+str(nPartitions)+"x1.png")
    plt.clf()

# Check if barriers are satisfying force criterion
if checkFlag:
    plt.plot(iterations,barriers,'o')
    plt.vlines(maxIter,0,np.max(barriers),linestyles='dashed',colors='red',label="Max iteration = "+str(maxIter))
    plt.xlim((0,1.1*maxIter))
    plt.ylim((0,1.1*max(barriers)))
    plt.title("Force Criterion Check for Energy Barriers")
    plt.xlabel("Stopping Iteration Number")
    plt.ylabel("Energy Barrier V (eV)")
    plt.savefig("criterion_check_"+minStyle+"_"+str(nPartitions)+"x1.png")
    plt.clf()

# Plot TLS displacements in real space
if plotFlag:
    for i in consTLSArray:
        maxDisp=max(netConsDisp2[i])
        fig=plt.gcf()
        ax=fig.add_subplot(projection="3d")
        ax=Axes3D(fig)
        weight=100*netConsDisp2[i]/maxDisp
        ax.scatter(positionArray[i,:,2],positionArray[i,:,3],positionArray[i,:,4],s=weight)
        ax.set_xlim(0,27.3275)
        ax.set_ylim(0,27.3275)
        ax.set_zlim(0,27.3275)
        plt.title("i = "+str(i)+", PR = "+str(PR[i]))
        fig.set_size_inches(13.66,7.68)
        plt.savefig('tls_'+str(i)+'.png',dpi=100)
        plt.clf()

# Writes data
if writeFlag:
    np.savetxt('avgConsDisp2_Dump='+str(nDump)+'.csv',avgConsDisp2,delimiter=',')
    np.savetxt('PR_Dump='+str(nDump)+'.csv',PR,delimiter=',')
    np.savetxt('energy_Dump='+str(nDump)+'.csv',energyAsym,delimiter=',')
    np.savetxt('energyBarrier_'+minStyle+'_'+str(nPartitions)+'x1_Dump='+str(nDump)+'.csv',barriers,delimiter=',')
