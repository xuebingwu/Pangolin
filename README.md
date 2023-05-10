# Splicing^Lab

A Google Colab notebook/web server for predicting tissue-specific splice site usage from a single sequence using Pangolin

Analyze your own sequence here: https://colab.research.google.com/github/xuebingwu/SplicingLab/blob/main/splicingScan.ipynb

## User interface
![interface](./colab-gui2.png?raw=true "Colab interface")

## Input
A single sequence, DNA or RNA

## Output
A figure and a table for predicted sites on both strands. Output files will be zipped and downloaded. See below the output when using ACTB pre-mRNA as input.

![ACTB-output](./sample-output2.png?raw=true "Sample output (ACTB pre-mRNA)")

# About <a name="Instructions"></a>

**Applications**
* Identify potential splicing artifacts in plasmid reporters.


**Limitations**
* A gmail account is required to run Google Colab notebooks.
* This notebook was designed for analyzing a single sequence. 
* Only sequences of length 1-150,000 bases have been tested. Longer sequences may fail due to a lack of memory.
* The first run is slow due to the need to install the `Pangolin` package.  
* GPU may not be available and running the prediction on CPU will be significantly slower. 
* Your browser can block the pop-up for downloading the result file. You can choose the `save_to_google_drive` option to upload to Google Drive instead or manually download the result file: Click on the little folder icon to the left, navigate to file: `res.zip`, right-click and select \"Download\".


**Bugs**
- If you encounter any bugs, please report the issue by emailing Xuebing Wu (xw2629 at cumc dot columbia dot edu)

**License**

* The source code of this notebook is licensed under [MIT](https://raw.githubusercontent.com/sokrypton/ColabFold/main/LICENSE). See details of the license for Pangolin [here](https://github.com/tkzeng/Pangolin/blob/main/LICENSE).

**Acknowledgments**
- We thank the [Pangolin](https://doi.org/10.1186/s13059-022-02664-4) team for developing an excellent model and open sourcing the software. 

- This notebook is modeld after the [ColabFold notebook](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb).


