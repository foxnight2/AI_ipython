from collections import OrderedDict
from typing import Union, Tuple, List, Any, Iterable


from .tensor import Tensor


class Function(object):
    '''function
    '''
    def __init__(self, ):
        self.previous_functions = None
        self.output_ids = None
        self.input_ids = None
        self.needs_input_grad = None

    def _do_forward(self, *inputs: Any) -> Iterable[Tensor]:
        '''Tuple[Varible, ...]
        '''
        unpacked_input = []
        needs_input_grad = []
        previous_functions = []
        for var in inputs:
            if isinstance(var, Tensor):
                unpacked_input.append(var.storage)
                needs_input_grad.append(var.creator.requires_grad)
                previous_functions.append((var.creator, id(var)))
            else:
                unpacked_input.append(var)
                needs_input_grad.append(False)
                previous_functions.append((None, -1))
        
        self.needs_input_grad = needs_input_grad
        self.requires_grad = any(needs_input_grad)
        self.previous_functions = previous_functions

        raw_output = self.forward(*unpacked_input)
        if not isinstance(raw_output, tuple):
            raw_output = (raw_output, )

        output = tuple(Tensor(data, creator=self) for data in raw_output)

        self.input_ids = {id(var): i for i, var in enumerate(inputs)}
        self.output_ids = {id(var): i for i, var in enumerate(output)}

        return output

    __call__ = _do_forward

    def _do_backward(self, output_grad: Tensor) -> Tuple[Tensor, ...]:
        '''
        '''        
        _grad_inputs = self.backward(output_grad.storage) 
        if not isinstance(_grad_inputs, tuple):
            _grad_inputs = (_grad_inputs, )
        
        assert len(_grad_inputs) == len(self.previous_functions), f'{self.__class__.__name__} _do_backward'

        grad_inputs = []
        for _grad in _grad_inputs:
            if _grad is not None:
                grad_inputs.append(Tensor(_grad))
            else:
                grad_inputs.append(_grad)

        return tuple(grad_inputs)


    def forward(self, *inputs):
        '''backen.types -> backen.types
        '''
        raise NotImplementedError


    def backward(self, grad_output: Tensor):
        '''backen.types -> backen.types
        '''
        raise NotImplementedError



