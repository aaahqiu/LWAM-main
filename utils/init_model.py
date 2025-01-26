from models import *

class ModelBase:
    def __init__(self, args):
        self.args = args

    def init_UNetNested(self):
        model = UNetNested(in_channels=3, n_classes=self.args.class_num)
        return model

    def init_LWRNetF(self):
        model = LWRNetF(n_classes=self.args.class_num)
        return model

def init_model(args,model_name):
    model_base = ModelBase(args)
    model = getattr(model_base, 'init_' + model_name)()
    return model


if __name__ == '__main__':
    model = init_model()
    print(model)