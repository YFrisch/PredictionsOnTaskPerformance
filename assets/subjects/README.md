# Subjects

### About:
- This folder contains a folder for each subject with the data from our 
experiment.


- Project: Predictions on task performance
- Organization: Technische Universität Darmstadt
- Path: /assets/subjects/

### Subfolders:
- A subfolder has to be added for each subject.
- It has to be created by hand with the name 'subject_' + SUBJECTCODE. 
- Additionally a folder with the name 'raw' has to be created for each
subject, which should contain the data from the experiment.

### Raw folders:
- The raw folders for every subject should contain a PDF file and a CSV file.
- PDF: the scanned questionnaire (4 pages) named 'subject_' + SUBJECTCODE + '.pdf'
- CSV: the subject answers in the questionnaire named 'subject_' + 
SUBJECTCODE + '_answers.csv'
- Instructions for creating these files can be found in the top level README.md
- Additionally JPEGs are created for the PDF by our program. These are needed 
for processing the Questionnaire by the means of Computer Vision.

### pdf folders
- When processing the data, our program creates a 'pdf' folder for each subject
- These folders contain all the probability density functions that our 
program detected and extracted from the PDFs. These images are numbered in 
the order of occurrence from first page to last and top to bottom.

### analysis folders
- Our program creates 'analysis folders for each subject
- These folders contain three files
- SUBJECTCODE_brier_scores.csv: the brier score for each task
- SUBJECTCODE_probabilities.csv: the probabilities that we read from the 
probability density functions for each task
- SUBJECTCODE_task_scores: the score for each task
