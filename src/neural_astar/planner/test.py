import torch
import torch.nn as nn

class t:
    def __init__(self, text = "NAresh"):
        self.text = text
        # super(t, self).__init__()
        self.fc1 = nn.Linear(1, 1)

    def forward(self):
        print("Something {}".format(self.text))
        return 7
    
    def something(self):
        print("this is something function")
    
class t1(t):
    def __init__(self):
        super().__init__()

model = t1()
print(model.forward())