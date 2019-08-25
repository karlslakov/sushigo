class agent:
    def __init__(self, model, input_size, name="unnamed agent"):
        self.input_size = input_size
        self.model = model
        self.name = name
    
    def predict(self, features):
        return self.model.predict(features.reshape((1, self.input_size)))[0]
