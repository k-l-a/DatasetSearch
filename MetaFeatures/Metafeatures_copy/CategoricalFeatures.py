class CategoricalFeatures:

    def __init__(self, basic_features):
        self.basic_features = basic_features
        self.X = basic_features.X
        self.y = basic_features.y
        self.number_of_categorical_features = None
        self.ratio_categorical_to_numerical = None
        self.ratio_numerical_to_categorical = None
        self.number_of_numerical_features = None

    # number-of-categorical-features
    def get_number_of_categorical_features(self):
        temp = [col for col in self.X.columns if self.X[col].dtype == "int" or self.X[col].dtype == "object"]
        self.number_of_categorical_features = len(temp)
        return self.number_of_categorical_features

    # number-of-numeric-features
    def get_number_of_numerical_features(self):
        if self.number_of_categorical_features is None:
            self.get_number_of_categorical_features()
        self.number_of_numerical_features = self.basic_features.number_of_features - self.number_of_categorical_features
        return self.number_of_numerical_features

    # ratio-categorical-to-numerical
    def get_ratio_categorical_to_numerical(self):
        if self.number_of_numerical_features is None:
            self.get_number_of_numerical_features()
        if self.number_of_numerical_features == 0:
            temp = float('inf')
        else:
            temp = self.number_of_categorical_features / self.number_of_numerical_features
        self.ratio_categorical_to_numerical = temp
        return self.ratio_categorical_to_numerical

    # ratio-numerical-to-categorical
    def get_ratio_numerical_to_categorical(self):
        if self.number_of_numerical_features is None:
            self.get_number_of_numerical_features()
        if self.number_of_categorical_features == 0:
            temp = float('inf')
        else:
            temp = 1 / self.ratio_categorical_to_numerical
        self.ratio_numerical_to_categorical = temp
        return self.ratio_numerical_to_categorical

    def calculate(self):
        self.get_number_of_categorical_features()
        self.get_number_of_numerical_features()
        self.get_ratio_categorical_to_numerical()
        self.get_ratio_numerical_to_categorical()
        return list([self.number_of_categorical_features,
                     self.ratio_categorical_to_numerical,
                     self.ratio_numerical_to_categorical])
