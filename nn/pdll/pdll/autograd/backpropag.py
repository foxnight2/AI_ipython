
from pdll.backend import executor


class ExecuteEngine(object):
    
    def __init__(self, debug=False):
        self.debug = debug

    def build_graph(self, ):
        '''
        '''
        raise NotImplementedError

    def destroy_graph(self, ):
        '''
        '''
        raise NotImplementedError

    def backward_var(self, var, grad) -> None:
        ''' 
        '''
        var.grad += grad
        grads_input = var.creator._do_backward(grad)
        for _i, _grad in enumerate(grads_input):
            if _grad is not None:
                self.backward_var(var.creator.inputs[_i], _grad)

    def backward_fn(self, creator, grad) -> None:
        ''' 
        '''
        grads_input = creator._do_backward(grad)
        for _i, _grad in enumerate(grads_input):
            if _grad is not None:
                self.backward_fn(creator.previous_functions[_i][0], _grad)
        
        # self.destroy_graph()


class Leaf(object):
    '''leaf
    '''
    def __init__(self, tensor, requires_grad):
        self.tensor = tensor
        self.output_ids = {id(tensor): 0}
        self.previous_functions = []
        self.requires_grad = requires_grad

    def _do_backward(self, *grad_output):
        assert len(grad_output) == 1, ''
        if self.requires_grad:
            if self.tensor.grad is None:
                self.tensor.grad = type(self.tensor)(executor.np.zeros(self.tensor.shape))
            self.tensor.grad.storage += grad_output[0].storage
        return tuple()
