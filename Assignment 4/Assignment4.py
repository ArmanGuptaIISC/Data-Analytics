import numpy as np
import scipy as sp # may be useful to compute probabilities
import time # may be useful to check the execution time of some function
import re
from math import *
from tqdm import tqdm

"""
Please refer to lecture slides.
Please refer to README file.
All the functions that you define must be able to handle corner cases/exceptions
"""

"""
Starting and ending locations (indices) of red and green exons in the reference sequence - Begins

1. Red Exon Locations
"""
RedExonPos = np.array([
    [149249757, 149249868], # R1
    [149256127, 149256423], # R2
    [149258412, 149258580], # R3
    [149260048, 149260213], # R4
    [149261768, 149262007], # R5
    [149264290, 149264400]  # R6
    ])
"""
2. Green Exon Locations
"""
GreenExonPos = np.array([
    [149288166, 149288277], # G1
    [149293258, 149293554], # G2
    [149295542, 149295710], # G3
    [149297178, 149297343], # G4
    [149298898, 149299137], # G5
    [149301420, 149301530]  # G6
    ])
"""
Starting and ending locations (indices) of red and green exons in the reference sequence - Ends
"""    

def loadLastCol(filename):
    """
    Input: Path of the file corresponding the last column (BWT).
    Output: The last column (BWT) in string format.
    """
    # function body - Begins
    LastCol = ''
    with open(filename) as f:
        lines = f.readlines()
        LastCol = ''.join(lines)
        
    # function body - Ends
    LastCol = re.sub('\n', '', LastCol)
    return LastCol #string data type

def loadRefSeq(filename):
    """
    Input: Path of the file containing the reference sequence.
    Output: The reference sequence in string format.
    """
    # function body - Begins
    RefSeq = ''
    with open(filename) as f:
        lines = f.readlines()
        RefSeq = ''.join(lines[1:])
    # function body - Ends
    RefSeq = re.sub('\n','',RefSeq)
    return RefSeq # string data type

def replaceAWithN(s):
    '''
     Helper Function : Replace N with A in read.
    '''
    return re.sub('N','A',s)
def loadReads(filename):
    """
    Input: Path of the file containing all the reads.
    Output: A list containing all the reads.
    """
    # function body - Begins
    with open(filename) as f:
        Reads = f.readlines()
    # function body - Ends
    Reads = [replaceAWithN(read.strip()) for read in Reads]
    return Reads # list of strings

def loadMapToRefSeq(filename):
    """
    Input: Path of the file containing mapping of the first column to the reference sequence.
    Output: numpy integer array containing the mapping.
    """
    # function body - Begins
    with open('../data/chrX_map.txt') as f:
        count = 0
        for i,e in tqdm(enumerate(f)):
            count +=1
    map = np.zeros(count,dtype= int)
    with open('../data/chrX_map.txt') as f:
        for i,e in tqdm(enumerate(f)):
            map[i] = int(e.strip())
    return map # numpy integer array

def rankDataStructure(LastCol, delta):
    '''
     Input:
        LastCol : Input from the chrX_last_col.txt 
        delta : Interval at which we will store the rank of each Base
    Output:
        numpy array containing rank of each base at fixed interval 'delta'
    '''
    size = int( ceil(len(LastCol)/delta))
    ds = np.zeros((size , 4),dtype = int)
    cA, cC, cT, cG  = 0, 0 ,0 ,0
    k = 0
    for i,ch in enumerate(LastCol):
        if ch == 'A':
            cA+=1
        elif ch == 'C':
            cC+=1
        elif ch == 'T':
            cT+=1
        elif ch == 'G':
            cG+=1
        if i%delta == 0:
            ds[k]= np.array([cA,cC,cG,cT],dtype= int)
            k+=1
    return ds

def findrank(rank_ds, delta ,LastCol ,base , index , first = False):
    '''
        Input : 
            rank_ds : 2D numpy array containing rank at fixed interval delta.
            delta : Integer denoting fixed interval at which rank is stored in rank_ds
            LastCol : String Input from the chrX_last_col.txt 
            base : Integer which denotes the base for which we have to compute rank
            index : Integer which contains the current index which will be taken as reference for computing the rank of base
            first : Boolean which will be true if we are computing rank of first occurance of base in the band otherwise false 
                    if we are computing rank of the last occurance of base.

        Output : Find the rank of the base considering index as reference point.
    '''
    if base == '$':
        return 1

    # Index of Rank Data Structure which we need to refer
    ds_above_index = index // delta
    # Index of the LastCol which corresponds to row at ds_above_index in Rank_DS
    actual_above_index = ds_above_index * delta

    cur_index = index
    count = 0
    while cur_index > actual_above_index:
        if LastCol[cur_index] == base:
            count +=1
        cur_index-=1
    if first:
        # Count will be incremented because the next base will lie below the current index thus we compute 
        # number of a particular base above current index and once we get the rank, we will check for the base
        # in first index of the current band and if it is not there we will increment the rank for the obvious reason.
        if LastCol[index] != base:
            count +=1 
    if base == 'A':
        return rank_ds[ds_above_index][0] + count
    elif base == 'C':
        return rank_ds[ds_above_index][1] + count
    elif base == 'G':
        return rank_ds[ds_above_index][2] + count
    elif base == 'T':
        return rank_ds[ds_above_index][3] + count
    else:
        raise Exception("Invalid Base : "+base) 


def getFirstCol(LastCol):
    '''
     helper function : Return First Column by Sorting LastCol and computing the first and last occurance of each base in
                       first column.
    '''
    FirstCol = sorted(LastCol)
    if FirstCol[0] =='$':
        FirstCol.remove('$')
        FirstCol.append('$')
    FirstCol =''.join(FirstCol)
    firstIndex = np.zeros((4,2),dtype = int)
    firstIndex[0,:] = FirstCol.find('A'),FirstCol.rfind('A')
    firstIndex[1,:] = FirstCol.find('C'),FirstCol.rfind('C')
    firstIndex[2,:] = FirstCol.find('G'),FirstCol.rfind('G')
    firstIndex[3,:] = FirstCol.find('T'),FirstCol.rfind('T')
    
    return  FirstCol, firstIndex

def stringMismatch(s1, s2):
    '''
     helper function : Return the number of position where string mismatches.
    '''
    c = [i for i in range(len(s1)) if s1[i] != s2[i]]
    return len(c)


def searchForSubstring2(LastCol, FirstColSIndex , rank_ds , delta, read):
    '''
        Input : 
            rank_ds : 2D numpy array containing rank at fixed interval delta.
            delta : Integer denoting fixed interval at which rank is stored in rank_ds
            LastCol : String Input from the chrX_last_col.txt 
            read : single read from the Reads input.
            FirstColSIndex : 2D numpy array containing starting and ending position of each base in first column.

        Output:
            Range of band ie. start and end pos where the matching string lie.
    '''
    start_band = None
    if  read[-1] == 'A':
        start_band = FirstColSIndex[0]
    elif read[-1] =='C':
        start_band = FirstColSIndex[1]
    elif read[-1] =='G':
        start_band = FirstColSIndex[2]
    elif read[-1] == 'T':
        start_band = FirstColSIndex[3]
    
    cur_band = start_band
    for c in read[::-1][1:]:    
        first_rank = findrank(rank_ds , delta, LastCol ,c ,cur_band[0], True)
        last_rank = findrank(rank_ds , delta, LastCol ,c , cur_band[1], False)
        if(first_rank > last_rank):
            return []
        if c == 'A':
            start = FirstColSIndex[0][0] + first_rank -1
            end = FirstColSIndex[0][0] + last_rank - 1
            cur_band = np.array([start , end])
        elif c == 'C':
            start = FirstColSIndex[1][0] + first_rank -1
            end = FirstColSIndex[1][0] + last_rank - 1
            cur_band = np.array([start , end])
        elif c == 'G':
            start = FirstColSIndex[2][0] + first_rank -1
            end = FirstColSIndex[2][0] + last_rank - 1
            cur_band = np.array([start , end])
        elif c == 'T':
            start = FirstColSIndex[3][0] + first_rank -1
            end = FirstColSIndex[3][0] + last_rank - 1
            cur_band = np.array([start , end])  
    return cur_band

def MatchReadToLoc(read):
    """
    Input: a read (string)
    Output: list of potential locations at which the read may match the reference sequence. 
    Refer to example in the README file.
    IMPORTANT: This function must also handle the following:
        1. cases where the read does not match with the reference sequence
        2. any other special case that you may encounter
    """
    # function body - Begins

    ## indices contains band where the matching string lies. If no matching string found , it get empty array.
    indices = searchForSubstring2(LastCol , FirstColSIndex , rank_ds , delta , read)
    positions = []

    ## If Matching String Not Found
    if len(indices) == 0:
        optlen = len(read) // 3 ## Since Two Miss matches are allowed.
        sub_reads = [read[0:optlen] , read[optlen:2*optlen] , read[optlen*2:]]
        positionsToCheck = []
        for i,sub_read in enumerate(sub_reads):
            sub_indices = searchForSubstring2(LastCol , FirstColSIndex , rank_ds , delta , sub_read)
            if len(sub_indices) == 0:
                continue
            for ind in range(sub_indices[0] , sub_indices[1]+1):
                temp = Map[ind] - i * optlen 
                if temp >= 0:
                    positionsToCheck.append(temp)
        for p in positionsToCheck:
            # checking if mismatch tolerance <= 2 condition satified.
            if stringMismatch(RefSeq[p : p + len(read)] ,read) <= 2:
                positions.append(p)

    ## If Matching String Found
    else:
        for ind in range(indices[0] , indices[1]+1):
            positions.append(Map[ind])

    positions = np.unique(positions)  
    # function body - Ends
    return [positions , len(indices) == 0]  # list of potential locations at which the read may match the reference sequence.



def WhichExon(positions):
    """
    Input: list of potential locations at which the read may match the reference sequence.
    Output: Update(increment) to the counts of the 12 exons
    IMPORTANT: This function must also handle the following:
        1. cases where the read does not match with the reference sequence
        2. cases where there are more than one matches (match at two exons)
        3. any other special case that you may encounter
    """
    which_exon = np.zeros(12)
    # r1,r2,r3,r4,r5,r6,g1,g2,g3,g4,g5,g6 = 0,0,0,0,0,0,0,0,0,0,0,0
    # function body - Begins
    positions ,isMismatch  = positions
    if len(positions) == 0:
        return which_exon
    
    for position in positions:
        start_position = position
        end_position = position + 100
        for i , exon in enumerate(RedExonPos):
            if exon[0]<=start_position and start_position <= exon[1] or exon[0] <=end_position and end_position <=exon[1]:
                which_exon[i] +=1 

        for i , exon in enumerate(GreenExonPos):
            if exon[0]<=start_position and start_position <= exon[1] or exon[0] <=end_position and end_position <=exon[1]:
                which_exon[i+6] +=1 
    # function body - Ends    
    if np.sum(which_exon) > 0:
        return which_exon / np.sum(which_exon)
    else:
        return which_exon


def softmax(x):
    return np.exp(x) / np.exp(x).sum()
def ComputeProb(ExonMatchCounts):
    """
    Input: The counts for each exon
    Output: Probabilities of each of the four configurations (a list of four real numbers)
    """
    # function body - Begins
    v = [ExonMatchCounts[1] / ExonMatchCounts[7],ExonMatchCounts[2] / ExonMatchCounts[8],ExonMatchCounts[3] / ExonMatchCounts[9],ExonMatchCounts[4] / ExonMatchCounts[10]]
    c = np.zeros((4,4))
    c[0] =  np.array([0.5,0.5,0.5,0.5])
    c[1] =  np.array([1,1,0,0])
    c[2] =  np.array([0.33,0.33,1,1])
    c[3] =  np.array([0.33,0.33,0.33,1])
    x = np.zeros(4)
    for i in range(4):
        x[i] = np.dot(v,c[i]) / (np.linalg.norm(v) * np.linalg.norm(c[i]))
    # function body - ends
    return softmax(x)


def BestMatch(ListProb):
    """
    Input: Probabilities of each of the four configurations (a list of four real numbers)
    Output: Most likely configuration (an integer). Refer to lecture slides
    """
    # function body - Begins
    MostLikelyConfiguration = np.argmax(ListProb)
    # function body - ends
    return MostLikelyConfiguration # it holds 0, 1, 2, or 3

if __name__ == "__main__":
    # load all the data files
    begin = time.time()
    LastCol = loadLastCol("../data/chrX_last_col.txt") # loads the last column
    print('Done Reading chrX_last_col.txt. Time taken =',time.time()-begin)

    begin = time.time()
    RefSeq = loadRefSeq("../data/chrX.fa") # loads the reference sequence
    print('Done Reading chrX.fa. Time taken =',time.time()-begin)

    begin = time.time()
    Reads = loadReads("../data/reads") # loads the reads
    print('Done Reading reads file. Time taken =',time.time()-begin)

    begin = time.time()
    Map = loadMapToRefSeq("../data/chrX_map.txt") # loads the mapping to the reference sequence
    print('Done Reading chrX_map.txt. Time taken =',time.time()-begin)

    delta = 20
    begin = time.time()
    FirstCol , FirstColSIndex = getFirstCol(LastCol)
    print('Computed first col data from LastCol. Time taken =',time.time()-begin)

    begin = time.time()
    rank_ds = rankDataStructure(LastCol,delta)
    print('Computed Rank Data Structure. Time Taken = ',time.time() - begin)

    # run the functions
    ExonMatchCounts = np.zeros(12) # initialize the counts for exons
    for read in tqdm(Reads): # update the counts for exons
        positions = MatchReadToLoc(read) # get the list of potential match locations
        ExonMatchCounts += WhichExon(positions) # update the counts of exons, if applicable
    
    print('Exon match counts : ', ExonMatchCounts)
    ListProb = ComputeProb(ExonMatchCounts) # compute probabilities of each of the four configurations
    print('Probabilities : ', ListProb)
    MostLikely = BestMatch(ListProb) # find the most likely configuration
    print("Configuration %d is the best match"%MostLikely)
