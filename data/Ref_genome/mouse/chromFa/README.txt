This directory contains the Jul. 2007 assembly of the mouse genome
(mm9, NCBI Build 37) in one gzip-compressed FASTA file per chromosome.

This assembly was produced by the Mouse Genome Sequencing Consortium,
and the National Center for Biotechnology Information (NCBI).
See also: http://www.ncbi.nlm.nih.gov/mapview/map_search.cgi?taxid=10090

Files included in this directory:

  - chr*.fa.gz: compressed FASTA sequence of each chromosome.
    Each chromosome is in a separate file in a gzip Fasta format.
    Repeats -- which are shown in lower case -- are annotated by 
    RepeatMasker run at the sensitive setting and Tandem Repeats Finder
    (repeats of period 12 or less).

md5sum.txt - MD5 checksum of these files to verify correct transmission

The main assembly is contained in the chrN.fa.gz files, where N is the name 
of the chromosome.  The chrN_random.fa.gz files contain clones that are not 
yet finished or cannot be placed with certainty at a specific place on 
the chromosome.  The chrUn_random.fa.gz sequence are unfinished clones,
or clones that can not be tenatively placed on any chromosome.

------------------------------------------------------------------
If you plan to download a large file or multiple files from this 
directory, we recommend that you use ftp rather than downloading the 
files via our website. To do so, ftp to hgdownload.cse.ucsc.edu, then 
go to the directory goldenPath/mm9/chromosomes. To download multiple 
files, use the "mget" command:

    mget <filename1> <filename2> ...
    - or -
    mget -a (to download all the files in the directory)

Alternate methods to ftp access.

Using an rsync command to download the entire directory:
    rsync -avzP rsync://hgdownload.cse.ucsc.edu/goldenPath/mm9/chromosomes/ .
For a single file, e.g. chrM.fa.gz
    rsync -avzP \
        rsync://hgdownload.cse.ucsc.edu/goldenPath/mm9/chromosomes/chrM.fa.gz .

Or with wget, all files:
    wget --timestamping \
        'ftp://hgdownload.cse.ucsc.edu/goldenPath/mm9/chromosomes/*'
With wget, a single file:
    wget --timestamping \
        'ftp://hgdownload.cse.ucsc.edu/goldenPath/mm9/chromosomes/chrM.fa.gz' \
        -O chrM.fa.gz

To uncompress the fa.gz files:
    gunzip <file>.fa.gz

All the files in this directory are freely available for public use.

This file last updated: 2007-07-26 - 26 July 2007
