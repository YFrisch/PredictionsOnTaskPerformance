# Main Python File
import os

from src.discrete_distribution_reader import DiscreteDistributionReader as DDR
from src.read_pdfs import extract_pdfs

# --------------- Read PDFs --------------- #

subject_code = f'AAAA'
folder_path = f'assets/subjects/subject_{subject_code}'

image_array = [f'{folder_path}/raw/subject_AAAA_p1.jpg',
               f'{folder_path}/raw/subject_AAAA_p2.jpg',
               f'{folder_path}/raw/subject_AAAA_p3.jpg']

extract_pdfs(image_path_array=image_array, dst_folder=f'{folder_path}/pdfs/')

print(os.getcwd())

# --------------- Simulate Distribution --------------- #

# dr = DiscreteDistributionReader(path="curve.png", points=5)
dr1 = DDR(path=f'{folder_path}/pdfs/pdf_task_1.jpg', points=5)
dr2 = DDR(path=f'{folder_path}/pdfs/pdf_task_2.jpg', points=5)
dr3 = DDR(path=f'{folder_path}/pdfs/pdf_task_3.jpg', points=5)

dr1.plot()
dr2.plot()
dr3.plot()

# --------------- Calculate Brier Score --------------- #

scores1 = [round(dr1.brier_score(i), 2) for i in range(0, 6)]
scores2 = [round(dr2.brier_score(i), 2) for i in range(0, 6)]
scores3 = [round(dr3.brier_score(i), 2) for i in range(0, 6)]

print("Brier Scores for Task 1 are: ", scores1)
print("Brier Scores for Task 2 are: ", scores2)
print("Brier Scores for Task 3 are: ", scores3)
