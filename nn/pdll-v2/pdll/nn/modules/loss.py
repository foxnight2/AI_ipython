
from pdll.autograd import Variable

from ..functional import softmax 
from ..functional import cross_entropy

from .module import Module


class Softmax(Module):
    '''softmax
    ''' 
    def __init__(self, axis: int=-1):
        super().__init__()
        self.axis = axis

    def forward(self, data: Variable) -> Variable: 
        return softmax(data, self.axis)
    
    def ext_repr(self, ) -> str:
        return ''



class CrossEntropyLoss(Module):
    '''crossentropyloss
    ''' 
    def __init__(self, axis: int=-1, reduction: str='mean'):
        super().__init__()
        self.axis = axis
        self.reduction = reduction

    def forward(self, data: Variable, target: Variable) -> Variable: 
        return cross_entropy(data, target, axis=self.axis, reduction=self.reduction)

    def ext_repr(self, ) -> str:
        return ''


class MSELoss(Module):
    def __init__(self, reduction: str='mean'):
        super().__init__()
        self.reduction = reduction.lower()

    def forward(self, data: Variable, target: Variable):
        '''
        '''
        if self.reduction == 'mean':
            return ((data - target) ** 2).sum() / data.shape[0]

        elif self.reduction == 'sum':
            return ((data - target) ** 2).sum()
    
        else:
            raise NotImplementedError
        
    def ext_repr(self, ) -> str:
        return ''
