from tqdm import tqdm
import pickle
import argparse
import re
import numpy as np

def findRuns(lst):
    """ yield (start, end) tuples for all runs of ident. numbers in lst 
    >>> list(findRuns([1,1,1,0,0,1,0,1,1,1]))
    [(0, 3), (5, 6), (7, 10)]
    """
    start,end=False,False

    for i,x in enumerate(lst):
        if x and start is False:
            start=i
        if x==0 and start is not False and end is False:
            end=i-1
        if start is not False and end is not False:
            yield start,end+1       #and len is (end-start)
            start,end=False,False
    
    if start is not False:
        yield start,i+1       #and len is (end-start)
        
def calcCropitScore(guideSeq, otSeq):
    """
    see http://www.ncbi.nlm.nih.gov/pmc/articles/PMC4605288/ PMID 26032770

    >>> int(calcCropitScore("GGGGGGGGGGGGGGGGGGGG","GGGGGGGGGGGGGGGGGGGG"))
    650

    # mismatch in 3' part
    >>> int(calcCropitScore("GGGGGGGGGGGGGGGGGGGA","GGGGGGGGGGGGGGGGGGGG"))
    575

    # mismatch in 5' part
    >>> int(calcCropitScore("AGGGGGGGGGGGGGGGGGGG","GGGGGGGGGGGGGGGGGGGG"))
    642

    # only mismatches -> least likely offtarget
    >>> int(calcCropitScore("AAAAAAAAAAAAAAAAAAAA","GGGGGGGGGGGGGGGGGGGG"))
    -27
    """
    if len(otSeq)==23:
        guideSeq = guideSeq[:20]
        otSeq = otSeq[:20]
    assert len(guideSeq)==len(otSeq)==20 ,(len(guideSeq),len(otSeq))

    penalties = [5,5,5,5,5,5,5,5,5,5,70,70,70,70,70,50,50,50,50,50]
    score = 0.0

    # do the score only for the non-mism positions
    misList = []
    score = 0.0
    for i in range(0, 20):
        if guideSeq[i]!=otSeq[i]:
            misList.append(1)
        else:
            misList.append(0)
            score += penalties[i]
    
    # get the runs of mismatches and update score for these positions
    consecPos = set()
    singlePos = set()
    for start, end in findRuns(misList):
        if end-start==1:
            score += -penalties[start]/2.0
        else:
            # mean if they happen to fall into different segments
            startScore = penalties[start]
            endScore = penalties[end-1]
            score += -((startScore+endScore)/2.0)

    return score
    
def calcCcTopScore(guideSeq, otSeq):
    """
    calculate the CC top score
    see http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0124633#sec002
    # no mismatch -> most likely off-target
    >>> int(calcCcTopScore("GGGGGGGGGGGGGGGGGGGG","GGGGGGGGGGGGGGGGGGGG"))
    224

    # mismatch in 5' partF
    >>> int(calcCcTopScore("AGGGGGGGGGGGGGGGGGGG","GGGGGGGGGGGGGGGGGGGG"))
    222

    # mismatch in 3' part
    >>> int(calcCcTopScore("GGGGGGGGGGGGGGGGGGGA","GGGGGGGGGGGGGGGGGGGG"))
    185F

    # only mismatches -> least likely offtarget
    >>> int(calcCcTopScore("AAAAAAAAAAAAAAAAAAAA","GGGGGGGGGGGGGGGGGGGG"))
    0
    """
    if len(otSeq)==23:
        guideSeq = guideSeq[:20]
        otSeq = otSeq[:20]

    if not (len(guideSeq)==len(otSeq)==20):
        raise Exception("Not 20bp long: %s %dbp<-> %s %dbp" % (guideSeq, len(guideSeq), otSeq, len(otSeq)))
    score = 0.0
    for i in range(0, 20):
        if guideSeq[i]!=otSeq[i]:
            score += 1.2**(i+1)
    return 224.0-score

def get_parser():
    parser = argparse.ArgumentParser(description='Calculates CFD score')
    parser.add_argument('--wt',
        type=str,
        help='WT 23mer sgRNA sequence')
    parser.add_argument('--off',
        type=str,
        help='Off-target 23mer sgRNA sequence')
    return parser

#Reverse complements a given string
def revcom(s):
    basecomp = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A','U':'A'}
    letters = list(s[::-1])
    letters = [basecomp[base] for base in letters]
    return ''.join(letters)

#Unpickle mismatch scores and PAM scores
def get_mm_pam_scores():
    try:
        mm_scores = pickle.load(open('mismatch_score.pkl','rb'))
        pam_scores = pickle.load(open('pam_scores.pkl','rb'))
        return (mm_scores,pam_scores)
    except: 
        raise Exception("Could not find file with mismatch scores or PAM scores")

#Calculates CFD score
def calc_cfd(wt,sg,pam):
    mm_scores,pam_scores = get_mm_pam_scores()
    score = 1
    sg = sg.replace('T','U')
    wt = wt.replace('T','U')
    s_list = list(sg)
    wt_list = list(wt)
    for i,sl in enumerate(s_list):
        if wt_list[i] == sl:
            score*=1
        else:
            key = 'r'+wt_list[i]+':d'+revcom(sl)+','+str(i+1)
            score*= mm_scores[key]
    score*=pam_scores[pam]
    return score



def calcCfdScore(guideSeq, otSeq):
    """ based on source code provided by John Doench
    >>> calcCfdScore("GGGGGGGGGGGGGGGGGGGGGGG", "GGGGGGGGGGGGGGGGGAAAGGG")
    0.4635989007074176
    >>> calcCfdScore("GGGGGGGGGGGGGGGGGGGGGGG", "GGGGGGGGGGGGGGGGGGGGGGG")
    1.0
    >>> calcCfdScore("GGGGGGGGGGGGGGGGGGGGGGG", "aaaaGaGaGGGGGGGGGGGGGGG")
    0.5140384614450001
    """
    wt = guideSeq.upper()
    off = otSeq.upper()
    m_wt = re.search('[^ATCG]',wt)
    m_off = re.search('[^ATCG]',off)
    if (m_wt is None) and (m_off is None):
        pam = off[-2:]
        sg = off[:-3]
        cfd_score = calc_cfd(wt,sg,pam)
        return cfd_score
        
hitScoreM = [0,0,0.014,0,0,0.395,0.317,0,0.389,0.079,0.445,0.508,0.613,0.851,0.732,0.828,0.615,0.804,0.685,0.583] 
def calcMitScore(string1,string2, startPos=0):
    """
    The MIT off-target score
    see 'Scores of single hits' on http://crispr.mit.edu/about
    startPos can be used to feed sequences longer than 20bp into this function

    the most likely off-targets have a score of 100
    >>> int(calcMitScore("GGGGGGGGGGGGGGGGGGGG","GGGGGGGGGGGGGGGGGGGG"))
    100

    # mismatches in the first three positions have no effect
    >>> int(calcMitScore("AGGGGGGGGGGGGGGGGGGG","GGGGGGGGGGGGGGGGGGGG"))
    100

    # less likely off-targets have lower scores
    >>> int(calcMitScore("GGGGGGGGGGGGGGGGGGGG","GGGGGGGGGGGGGGGGGGGA"))
    41
    """
    
    string1 = string1[:20]
    string2 = string2[:20]

    assert(len(string1)==len(string2)==20)

    dists = [] # distances between mismatches, for part 2
    mmCount = 0 # number of mismatches, for part 3
    lastMmPos = None # position of last mismatch, used to calculate distance

    score1 = 1.0
    for pos in range(0, len(string1)):
        if string1[pos]!=string2[pos]:
            mmCount+=1
            if lastMmPos!=None:
                dists.append(pos-lastMmPos)
            score1 *= 1-hitScoreM[pos]
            lastMmPos = pos

    # 2nd part of the score - distribution of mismatches
    if mmCount<2: # special case, not shown in the paper
        score2 = 1.0
    else:
        avgDist = sum(dists)/len(dists)
        score2 = 1.0 / (((19-avgDist)/19.0) * 4 + 1)

    # 3rd part of the score - mismatch penalty
    if mmCount==0: # special case, not shown in the paper
        score3 = 1.0
    else:
        score3 = 1.0 / (mmCount**2)

    score = score1 * score2 * score3 * 100
    return score 
 

# Because the output file of single execution is too large and GitHub does not allow uploading files over 100MB, so we execute it three times.    
fw = open("ot_score1.txt","w") 
fr = open("cas-offinder_output1.txt","r")
lineAll = fr.readlines()
for line in tqdm(lineAll):
	linelist = line.strip("\n").split("\t")
	if linelist[5] == "0":
		continue
	sgRNA = linelist[0][:-3]
	otSeq = linelist[3].upper()
	fw.write(sgRNA + "\t" + otSeq + "\t" + str(calcCropitScore(sgRNA,otSeq)) + "\t" + str(calcCcTopScore(sgRNA,otSeq)) + "\t" + str(calcMitScore(sgRNA,otSeq)) + "\t" + str(calcCfdScore(sgRNA,otSeq)) + "\n")	 	
fw.close()
fr.close()


fw = open("ot_score2.txt","w") 
fr = open("cas-offinder_output2.txt","r")
lineAll = fr.readlines()
for line in tqdm(lineAll):
	linelist = line.strip("\n").split("\t")
	if linelist[5] == "0":
		continue
	sgRNA = linelist[0][:-3]
	otSeq = linelist[3].upper()
	fw.write(sgRNA + "\t" + otSeq + "\t" + str(calcCropitScore(sgRNA,otSeq)) + "\t" + str(calcCcTopScore(sgRNA,otSeq)) + "\t" + str(calcMitScore(sgRNA,otSeq)) + "\t" + str(calcCfdScore(sgRNA,otSeq)) + "\n")	 	
fw.close()
fr.close()


	
fw = open("ot_score3.txt","w") 
fr = open("cas-offinder_output3.txt","r")
lineAll = fr.readlines()
for line in tqdm(lineAll):
	linelist = line.strip("\n").split("\t")
	if linelist[5] == "0":
		continue
	sgRNA = linelist[0][:-3]
	otSeq = linelist[3].upper()
	fw.write(sgRNA + "\t" + otSeq + "\t" + str(calcCropitScore(sgRNA,otSeq)) + "\t" + str(calcCcTopScore(sgRNA,otSeq)) + "\t" + str(calcMitScore(sgRNA,otSeq)) + "\t" + str(calcCfdScore(sgRNA,otSeq)) + "\n")
fw.close()
fr.close()


Bottom = ["end" + "\t" + "end" + "\t" + "0" + "\t" + "0"+ "\t" + "0"+ "\t" + "0"]	
  
fr = open("ot_score1.txt","r")
lineAll = fr.readlines()
fr.close()

fr = open("ot_score2.txt","r")
lineAll = fr.readlines() + lineAll
fr.close()

fr = open("ot_score3.txt","r")
lineAll = fr.readlines() + lineAll
fr.close()

lineAll = lineAll + Bottom
    
fw = open("../mediate_processing/POS.txt","w")
pre_sgRNA = lineAll[0].split("\t")[0]
CropitScore = 0
CcTopScore = 0
MitScore = 0
CfdScore = 0
for line in tqdm(lineAll):
	linelist = line.strip("\n").split("\t")
	sgRNA = linelist[0]
	if pre_sgRNA == sgRNA:
		CropitScore = CropitScore + float(linelist[2])
		CcTopScore = CcTopScore + float(linelist[3])
		MitScore = MitScore + float(linelist[4])
		CfdScore = CfdScore + float(linelist[5])
	else :
		fw.write(pre_sgRNA + "\t" + str(CropitScore) + "\t" + str(CcTopScore) + "\t" + str(MitScore) + "\t" + str(CfdScore) +"\n")
		pre_sgRNA = sgRNA
		CropitScore = float(linelist[2])
		CcTopScore = float(linelist[3])
		MitScore = float(linelist[4])
		CfdScore = float(linelist[5])
		
fw.close()
fr.close()	

		
