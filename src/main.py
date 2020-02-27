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

# --------------- Simulate Distribution --------------- #

dr = DDR(subject_code)
dr.plot(task_id=1)
dr.brier_score(task_id=1, vpn_points_for_task=4)
