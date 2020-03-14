# Predictions On Task Performance

Repository for the Applied Cognitive Science Project at TU Darmstadt WS2019/2020

What must be done to run our project scripts:
* Make sure to have the requirements listed in ./requirements.txt installed on your system
* E.g. run "pip install requirements.txt"

In case you want to add a subject:
* Add a folder in ./assets/subjects/ called 'subject_XXXX/' (replace XXXX by the subject's vpn code
* Add the scans or photos of the filled questionnaire to a subfolder of this folder labeled ../subject_XXXX/raw/ 
* Add a .csv file containing the subject's answers, labeled ../subject_XXXX/raw/subject_XXXX_answers.csv
* Add the subject to assets/demographics.csv
* Run ./src/main.py to let our scripts create the results in ./assets/subjects/subject_XXXX/pdfs/,
 ./assets/subjects/subject_XXXX/analysis/ and ./assets/plots/
 
 For evaluation you can choose of three different scripts
 * ./src/average_and_individual_evaluation.py
 * ./src/confidence_evaluation.py
 * ./src/brier_vs_rating_evaluation.py
 * Results are saved in ./assets/plots/