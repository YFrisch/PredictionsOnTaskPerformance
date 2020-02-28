# Main Python File
import os
from src.discrete_distribution_reader import DiscreteDistributionReader as DDR
from src.read_pdfs import extract_pdfs

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))

# --------------- Read PDFs --------------- #

# subject_code = r'RNZK'
# subject_code = r'RSHA'
subject_code = r'ETLA'

folder_path = r'/assets/subjects/subject_{}'.format(str(subject_code))

image_array = [r'{}/raw/subject_{}_p1.jpg'.format(folder_path, subject_code),
               r'{}/raw/subject_{}_p2.jpg'.format(folder_path, subject_code),
               r'{}/raw/subject_{}_p3.jpg'.format(folder_path, subject_code)]

dst_path = BASE_DIR + folder_path + r"/pdfs/"
extract_pdfs(image_path_array=image_array, dst_folder=dst_path)

# --------------- Simulate Distribution --------------- #

dr = DDR(subject_code)
dr.plot(task_id=1)
dr.plot(task_id=2)
dr.plot(task_id=3)
dr.plot(task_id=4)
dr.plot(task_id=5)
dr.plot(task_id=6)
dr.plot(task_id=7)
dr.plot(task_id=8)



