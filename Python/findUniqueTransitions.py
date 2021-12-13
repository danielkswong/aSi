import numpy as np
import tarfile
import io
import gzip
import csv
import os

# Relevant parameters
T=600
nEquils=100
nVelocities=10
nAtoms=1000
nDumps=2001
nColumns=6
lineStart=9
dataDirectory='../Data_v1/'

# Calculates average consecutive displacement between two atomic positions
def calculateDisplacement(pos1,pos2,n):
    consDisp=pos2-pos1
    netConsDisp2=consDisp[:,3]**2+consDisp[:,4]**2+consDisp[:,5]**2
    avgConsDisp=np.sqrt(np.sum(netConsDisp2,0))/n
    return avgConsDisp

# Read CSV files
print('Reading CSV file...')

os.chdir(dataDirectory)

with open('avgConsDisp_T='+str(T)+'_quench.csv','r') as csvfile:
    csvreader=csv.reader(csvfile,delimiter=',')
    quenchAvgConsDisp=np.asarray(list(csvreader),dtype=np.float64)

# Distance criteria
quenchAvgConsDispHist=np.histogram(quenchAvgConsDisp[:,2],bins='auto',density=False)
min_dCritInd=np.argmax((quenchAvgConsDispHist[0][1:]-quenchAvgConsDispHist[0][0:-1])>=0)
max_dCritInd=np.argmax(quenchAvgConsDispHist[0][min_dCritInd:])+min_dCritInd
dCritInd=np.argmin(quenchAvgConsDispHist[0][min_dCritInd:max_dCritInd])+min_dCritInd
dCrit=(quenchAvgConsDispHist[1][dCritInd]+quenchAvgConsDispHist[1][dCritInd+1])/2

# Arrays to store transition numbers and minima positions
transitionNumbers=np.zeros((nEquils*nVelocities,3),dtype=int)
minimaPositions=[]

# Finds first element greater than distance criterion and extracts them
print('Finding transition numbers...')

for equilNo in np.arange(nEquils):
    for velocityNo in np.arange(nVelocities):
        saveIndex=(nDumps-1)*nVelocities*equilNo+(nDumps-1)*velocityNo
        avgConsDisp=quenchAvgConsDisp[saveIndex:saveIndex+nDumps-1,2]
        transitions=np.argwhere(avgConsDisp>dCrit)
        if len(transitions)>0:
            transitionNumber=transitions[0][0]
            transitionNumbers[nVelocities*equilNo+velocityNo,:]=np.asarray([equilNo+1,velocityNo+1,transitionNumber],dtype=int)

# Tracks all unique transitions and generates indices of repeated transitions to delete
print('Finding unique transitions...')

transitionIndices=[]
deleteInd=[]
for i in np.arange(len(transitionNumbers)):
    print('Analyzing trajectory '+str(transitionNumbers[i,0])+'-'+str(transitionNumbers[i,1])+' (i = '+str(i+1)+') ...')
    with tarfile.open('Equil%03d-%d.tar.gz' % (transitionNumbers[i,0],transitionNumbers[i,1])) as archive:
        positionArray=np.zeros((nAtoms,nColumns),dtype=np.float64)
        quenchFile1=archive.extractfile('quench_positions_T='+str(T)+'_i='+str(transitionNumbers[i,2])+'.lammps.gz')
        byteInfo=io.BytesIO(quenchFile1.read())
        positionFile=gzip.GzipFile(fileobj=byteInfo)
        for j in np.arange(lineStart):
            positionFile.readline()
        for atomNumber in np.arange(nAtoms):
            line=positionFile.readline().decode('UTF-8')
            stringArray=np.asarray(line.strip().split(' '),dtype=np.float64)
            positionArray[atomNumber,:]=stringArray[2:]
        positionFile.close()
        quenchFile1.close()
        index1=0
        for j in np.arange(len(minimaPositions)):
            avgConsDisp=calculateDisplacement(positionArray,minimaPositions[j],nAtoms)
            if avgConsDisp<dCrit:
                index1=j
                break
        else:
            minimaPositions.append(positionArray)
            index1=len(minimaPositions)-1

        positionArray=np.zeros((nAtoms,nColumns),dtype=np.float64)
        quenchFile2=archive.extractfile('quench_positions_T='+str(T)+'_i='+str(transitionNumbers[i,2]+1)+'.lammps.gz')
        byteInfo=io.BytesIO(quenchFile2.read())
        positionFile=gzip.GzipFile(fileobj=byteInfo)
        for j in np.arange(lineStart):
            positionFile.readline()
        for atomNumber in np.arange(nAtoms):
            line=positionFile.readline().decode('UTF-8')
            stringArray=np.asarray(line.strip().split(' '),dtype=np.float64)
            positionArray[atomNumber,:]=stringArray[2:]
        positionFile.close()
        quenchFile2.close()
        index2=0
        for j in np.arange(len(minimaPositions)):
            avgConsDisp=calculateDisplacement(positionArray,minimaPositions[j],nAtoms)
            if avgConsDisp<dCrit:
                index2=j
                break
        else:
            minimaPositions.append(positionArray)
            index2=len(minimaPositions)-1

        for j in np.arange(len(transitionIndices)):
            if (index1,index2)==transitionIndices[j] or (index1,index2)==transitionIndices[j][::-1]:
                deleteInd.append(i)
                break
        else:
            transitionIndices.append((index1,index2))

# Delete repeated transitions
transitionNumbers=np.delete(transitionNumbers,deleteInd,0)

# Save transition numbers
print('Saving TLS candidates...')

np.savetxt('transitionNumbers_T='+str(T)+'_filter1.csv',transitionNumbers,fmt='%i',delimiter=',')
