Corpus of definition and non-definition sentences
Prepared by VP (vid.podpecan@ijs.si)
------------------------------------

The corpus folder contains a collection of files in the line-doc format (one sentence per line).
Each subfolder of the language subfolder (currently SL and EN) in SL and EN is one domain.
The files are named according to the label of its sentences.

There are different label formats but the general notation is as follows:

0 - not a definition
1 - not really a definition (weak)
2 - a definition

N - not a definition
N*** - not a definition (ask Senja about the meaning of addition characters)
Y - a definition
Y*** - maybe a definition (ask Senja about the meaning of addition characters)

*y - weak definition
?y - maybe a definition      1
n - not a definition
y - a definition


Some statistics:

               0   1   2
rsdo5bimdis  101  10  29
rsdo5kemcla   12   2   0
rsdo5bimucb   50  11  14
rsdo5jezcla    7   0   1
rsdo5kemdis  199  61  25
rsdo5vetdis  130  19   3
rsdo5bimcla   15   2   0
rsdo5jezucb   37  15   8
rsdo5kemucb   16   4   9
rsdo5vetucb  133  49  35
rsdo5jezdis  226  65  57
rsdo5vetcla   15   6   1


EvalKorpusTotale (only those with >10 sentences)
      count
N     12953
N?      175
Na       23
Na?      11
Nd       11
Nf       11
Ng       18
Np       32
Np?      30
Ns       26
Ns?      32
Nt       11
Nt?      18
Nw       36
Y        54
Y?       89
Yd?      11
Yl       11
Yl?      21
Ylt?     17
Yp?      25
Ys?      35
Yt?      22
Yv?      14
Yw?      18


Termframe
    count
*y    166
?y      1
n     792
y     259


Korpus_cel_eval_slo_finish.ttl
      count
N     16839
N?      378
Na       29
Nb       68
Nd       43
Ne?      20
Ng       11
Nl?      21
Nlp?     13
Np       66
Np?      86
Ns       24
Ns?      31
Nt       37
Nt?      47
Nw      111
Nw?      11
Nz?      12
Y       112
Y?      189
Ye?      26
Yl       21
Yl?      49
Ylp?     11
Ylt?     13
Ym?      14
Yn?      13
Yp?      34
Ys?      32
Yt?      31
Yv       14
Yv?      44
Yw       44
Yw?      33


DF_NDF_wiki_slo
count
Y: 3251
N: 14703
N1: 20684  (including non-definition sentences with term in the beggining of the sentence)
