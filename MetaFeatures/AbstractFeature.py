class AbstractFeature:

    def __init__(self, X, y, *args):
        self.X = X
        self.y = y
        self.value = self.calculate()

    def calculate(self):
        pass
