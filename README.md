# Board mining

## Board event logs dataset

We have collecte a [dataset of 616 board event logs](https://drive.google.com/file/d/1D5rybwE4dx1vMQyrOHki1GuQXZdrHbvw/view?usp=sharing) downloaded from public Trello boards. You can use the previous link to download the full dataset or replicate the download process using:
1. ```index.js``` with node.js to download all JSON files
2. ```notebook.ipnyb```, which is a Jupyter Notebook that is used to transform the downloaded JSON file into a CSV file.

## Empirical analysis of board event log metrics

From the 616 board event logs, we filtered them to keep the logs of the boards that (i) represent their entire life (i.e. the log starts with a board creation event); and (ii) have over 2,000 events and over 12 weeks of use, to ensure an extensive/real use. This results in 63 logs to be analyzed. Then, we computed several metrics for these event logs and analyzed them together with a manual identification of the design patterns used in each of them. The procedure followed is detailed in:
1. ```analysis.ipynb```, which is a Jupyter Notebook that details the filtering process and the metric generation.
2. ```details.xlsx```, which is an Excel file that contains the metrics computed in the Jupyter notebook and the manual categorization of each board.

Several insights are derived from the analysis. First, there seems to be a lack of discipline using the boards since the real values of the metrics lie on average under 50%, showing in most cases a deficient use of the boards. Second, mechanisms to ensure, monitor, and recommend the right use of boards are missing. In this regard, it would be useful to have support for automatically detecting cards that should have been moved and have not been moved, or cards unassigned or not being used according to specific patterns. This may aid users to be more disciplined in detecting misuse or lack of attention to parts of the problem being managed. 


## Use cases 

We have analyzed in detail three boards to show how the board mining techniques can be used to gain insights into its use and evolution:
1. ```analysis_Oeagag_Trello.ipynb```
2. ```analysis_Wooting_roadmap.ipynb```
3. ```analysis_Zwar57_Sheets.ipynb```

## Board mining library

All operations that are used in the notebook and described in the paper have been implemented in a Python library for board mining (```bomi.py```).

