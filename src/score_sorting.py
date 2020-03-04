import numpy as np


def score_sorting_task(answers, points_per_task):
    """ This function returns a discrete score for an input numpy array of actual answer positions for a sorting task.
        E.g. if a subjects answer for a task is a sorting of [A, C, E, B, D], we pass the correct positions
        [1, 3, 5, 2, 4] to the function.
        The method then calculates the 2-norm between the right answer [1, 2, 3, 4, 5]
        and the given answer [1, 3, 5, 2, 4].
        The interval [0, max-2-norm] is split into an array of a fixed amount of equidistant numbers (e.g. 5 or 10).
        The discrete rating for this task is the 'id' of the closes value of that array compared to the norm
        of the answer.
        E.g. if the subjects answer got a norm of 3.16 and we have a max-2-norm of 6.32 (worst possible answer) and we
        split the task into 5 possible ratings (1-5 points), we get the intervals [0, 1.27, 2.53, 3.79, 5.06, 6.32].
        The function would return a rating of 5 - 3 = 2.
    """
    def find_nearest(array, value):
        """ Returns entry of array that is closest to value.
        """
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    task_norm = np.linalg.norm(x=(np.array([1, 2, 3, 4, 5]) - answers), ord=2)
    max_norm = np.linalg.norm(x=(np.array([1, 2, 3, 4, 5]) - np.array([5, 4, 3, 2, 1])), ord=2)
    # We use a min 0 points and a max of points_per_task
    intervals = np.linspace(0, max_norm, points_per_task+1)
    rating = points_per_task - np.where(intervals == find_nearest(intervals, task_norm))[0][0]
    return rating
