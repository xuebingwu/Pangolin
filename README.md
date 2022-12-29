# Splice site prediction with Pangolin

A Google Colab notebook/web server for predicting splice site strength/usage from a single sequence using Pangolin

Analyze your own sequence here: https://colab.research.google.com/drive/1sxIP4vatYbTPMM0JXtxH4DRHHgPhq4MH

## Input
A single sequence, DNA or RNA

## Output

Predicted splice sites

Position  Sequence  Type  Heart	Liver	Brain	Testis
-----------------------------------------------------------------------
73     CCCGCCGCCAGgtaagcccg	5'SS	0.57	0.503	0.59	0.415
791 	 TGCCTTTTATGgtaataacg	5'SS	0.013	0.032	0.028	0.14
934 	 attctcgcagCTCACCATGG	3'SS	0.72	0.66	0.711	0.64
1062 	 CCAGGCACCAGgtaggggag	5'SS	0.906	0.897	0.926	0.909
1197 	 tccttcccagGGCGTGATGG	3'SS	0.88	0.845	0.898	0.892
1436 	 AGATGACCCAGgtgagtggc	5'SS	0.927	0.888	0.93	0.947
1878 	 tgccttacagATCATGTTTG	3'SS	0.924	0.871	0.912	0.921
2316 	 TTCCTTCCTGGgtgagtgga	5'SS	0.968	0.929	0.961	0.954
2412 	 tccctctcagGCATGGAGTC	3'SS	0.962	0.944	0.957	0.95
2593 	 TGAAGATCAAGgtgggtgtc	5'SS	0.957	0.92	0.969	0.956
2706 	 ccctcctcagATCATTGCTC	3'SS	0.955	0.943	0.965	0.949

Columns:
Position:	Site position in the sequence
Sequence:	20-nt junction sequence. Intron-lower case; Exon-upper case
Type    :	5' splice site (5'SS), 3' splice site (3'SS), or Undetermined, or Noncanonical
Scores  :	Site usage or P(splice) in each tissue

![Alt text](./sample-output.png?raw=true "Title")


