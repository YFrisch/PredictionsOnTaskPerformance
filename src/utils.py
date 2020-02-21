import random


def draw_random():
    nums = list(range(1, 6))
    random.shuffle(nums)
    print(nums)

draw_random()
