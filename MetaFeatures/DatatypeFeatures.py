from MetaFeatures.AbstractFeature import AbstractFeature


class NumberOfCategoricalFeatures(AbstractFeature):

    def __init__(self, X, y, *args):
        super().__init__(X, y)

    def calculate(self):
        temp = [col for col in self.X.columns if self.X[col].dtype == "int" or self.X[col].dtype == "object"]
        return len(temp)


class NumberOfNumericalFeatures(AbstractFeature):

    def __init__(self, X, y, *args):
        if isinstance(args[0], int):
            self.number_of_categorical_features = args[0]
        super().__init__(X, y)

    def calculate(self):
        return len(self.X.columns) - self.number_of_categorical_features


class RatioCategoricalToNumerical(AbstractFeature):

    def __init__(self, X, y, *args):
        if isinstance(args[0], int) and isinstance(args[1], int):
            self.number_of_categorical_features= args[0]
            self.number_of_numerical_features = args[1]
        super().__init__(X, y)

    def calculate(self):
        if self.number_of_numerical_features == 0:
            temp = self.number_of_categorical_features
        else:
            temp = self.number_of_categorical_features / self.number_of_numerical_features
        return temp


class RatioNumericalToCategorical(AbstractFeature):

    def __init__(self, X, y, *args):
        if isinstance(args[0], int) and isinstance(args[1], int):
            self.number_of_categorical_features = args[0]
            self.number_of_numerical_features = args[1]
        super().__init__(X, y)

    def calculate(self):
        if self.number_of_categorical_features == 0:
            temp = self.number_of_numerical_features
        else:
            temp = self.number_of_numerical_features / self.number_of_categorical_features
        return temp


class CategoricalFeatures(AbstractFeature):

    def __init__(self, X, y, *args):
        self.number_of_categorical_features = None
        self.ratio_categorical_to_numerical = None
        self.ratio_numerical_to_categorical = None
        self.number_of_numerical_features = None
        super().__init__(X, y)

    def calculate(self):
        self.number_of_categorical_features = NumberOfCategoricalFeatures(self.X, self.y).value
        self.number_of_numerical_features = NumberOfNumericalFeatures(self.X, self.y, self.number_of_categorical_features).value
        self.ratio_categorical_to_numerical = RatioCategoricalToNumerical(self.X, self.y, self.number_of_categorical_features, self.number_of_numerical_features).value
        self.ratio_numerical_to_categorical = RatioNumericalToCategorical(self.X, self.y, self.number_of_categorical_features, self.number_of_numerical_features).value
        return [self.number_of_categorical_features,
                     self.ratio_categorical_to_numerical,
                     self.ratio_numerical_to_categorical]
