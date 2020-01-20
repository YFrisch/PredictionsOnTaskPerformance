# Main Python File
from DiscreteDistributionReader import DiscreteDistributionReader

dr = DiscreteDistributionReader(path="curve.png", points=5)
dr.plot()
dr2 = DiscreteDistributionReader(path="curve2.jpg", points=5)
dr2.plot()
dr3 = DiscreteDistributionReader(path="curve3.png", points=5)
dr3.plot()
dr4 = DiscreteDistributionReader(path="curve4.png", points=5)
dr4.plot()
