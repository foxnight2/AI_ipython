import ppdet
from ppdet.data.transform import BaseOperator, register_op, registered_ops



@register_op
class RandomCopyPaste(BaseOperator):
    def __init__(self, a=0):
        super().__init__()

        self.a = a 
    
    def __call__(self, sample, context=None):
        return super().__call__(sample, context)




if __name__ == '__main__':
    print( '', registered_ops)
    print('RandomCopyPaste' in registered_ops)