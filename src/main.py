# Main Python File
import os

from src.discrete_distribution_reader import DiscreteDistributionReader as DDR
from src.read_pdfs import extract_pdfs

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))

# --------------- Read PDFs --------------- #

subject_code = f'RNZK'
folder_path = f'assets/subjects/subject_{subject_code}'

image_array = [f'{folder_path}/raw/subject_{subject_code}_p1.jpg',
               f'{folder_path}/raw/subject_{subject_code}_p2.jpg',
               f'{folder_path}/raw/subject_{subject_code}_p3.jpg']

extract_pdfs(image_path_array=image_array, dst_folder=BASE_DIR + "/" + f'{folder_path}/pdfs/')

print(os.getcwd())

# --------------- Simulate Distribution --------------- #

# dr = DiscreteDistributionReader(path="curve.png", points=5)
dr1 = DDR(path=f'{folder_path}/pdfs/pdf_task_1.jpg', points=5)
dr2 = DDR(path=f'{folder_path}/pdfs/pdf_task_2.jpg', points=5)
dr3 = DDR(path=f'{folder_path}/pdfs/pdf_task_3.jpg', points=5)
dr4 = DDR(path=f'{folder_path}/pdfs/pdf_task_4.jpg', points=5)
dr5 = DDR(path=f'{folder_path}/pdfs/pdf_task_5.jpg', points=5)
dr6 = DDR(path=f'{folder_path}/pdfs/pdf_task_6.jpg', points=5)
dr7 = DDR(path=f'{folder_path}/pdfs/pdf_task_7.jpg', points=5)
dr8 = DDR(path=f'{folder_path}/pdfs/pdf_task_8.jpg', points=5)
# drx = DDR(path=f'{folder_path}/pdfs/pdf_task_x.jpg', points=5)

dr1.plot()

# --------------- Calculate Brier Score --------------- #

print("Brier Scores for Task 1 are: ", [round(dr1.brier_score(i), 2) for i in range(0, 6)])
print("Brier Scores for Task 2 are: ", [round(dr2.brier_score(i), 2) for i in range(0, 6)])
print("Brier Scores for Task 3 are: ", [round(dr3.brier_score(i), 2) for i in range(0, 6)])
print("Brier Scores for Task 4 are: ", [round(dr4.brier_score(i), 2) for i in range(0, 6)])
print("Brier Scores for Task 5 are: ", [round(dr5.brier_score(i), 2) for i in range(0, 6)])
print("Brier Scores for Task 6 are: ", [round(dr6.brier_score(i), 2) for i in range(0, 6)])
print("Brier Scores for Task 7 are: ", [round(dr7.brier_score(i), 2) for i in range(0, 6)])
print("Brier Scores for Task 8 are: ", [round(dr8.brier_score(i), 2) for i in range(0, 6)])
# print("Brier Scores for general performance are: ", [round(drx.brier_score(i), 2) for i in range(0, 6)])
