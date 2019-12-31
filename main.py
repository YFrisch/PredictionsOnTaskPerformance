# Main Python File
from DistributionReadIn import DistributionReader

dr = DistributionReader()
dr.read_in_dist(path="curve.png")
dr.plot_dist(0)
