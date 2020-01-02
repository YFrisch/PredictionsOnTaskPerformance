# Main Python File
from DistributionReadIn import DistributionReader

dr = DistributionReader()
dr.read_in_dist(path="curve.png")
dr.plot_dist("curve.png")
dr.set_poly_degree(5)
dr.read_in_dist(path="curve2.jpg")
dr.plot_dist("curve2.jpg")
dr.set_poly_degree(5)
dr.read_in_dist(path="curve3.png")
dr.plot_dist("curve3.png")
dr.set_poly_degree(5)
dr.read_in_dist(path="curve4.png")
dr.plot_dist("curve4.png")