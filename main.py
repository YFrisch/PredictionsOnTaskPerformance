# Main Python File
from DiscreteDistributionReader import DiscreteDistributionReader

dr = DiscreteDistributionReader(path="curve.png", points=5)
dr2 = DiscreteDistributionReader(path="curve2.jpg", points=5)
dr3 = DiscreteDistributionReader(path="curve3.png", points=5)
dr4 = DiscreteDistributionReader(path="curve4.png", points=5)
dr5 = DiscreteDistributionReader(path="pdf_3.jpg", points=5)

dr.plot()
dr.brier_score(3)
