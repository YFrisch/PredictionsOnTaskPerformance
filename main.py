# Main Python File
from DistributionReadIn import DistributionReader

dr = DistributionReader()
dr.read_dist(path="curve.png")
print(dr.return_dist()[0])
