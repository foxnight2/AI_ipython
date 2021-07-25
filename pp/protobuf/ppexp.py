

import pp_pb2 as pp
from google.protobuf import text_format


class _EXP(object):
    def __init__(self, ):
        pass
    
    @property
    def model(self, ):
        return build_model()

    def build_model(self, ):
        raise NotImplementedError('')
        
    def build_dataset(self, ):
        raise NotImplementedError('')
    
    def build_dataloader(self, ):
        raise NotImplementedError('')
    
    def build_optimizer(self, ):
        raise NotImplementedError('')

    def build_lr_scheduler(self, ):
        raise NotImplementedError('')



class PBEXP(_EXP):
    def __init__(self, pb_solver):
        solver = pp.SolverParameter() 
        text_format.Merge(open(pb_solver, 'rb').read(), solver)
    
        print(text_format)
        
    
class PYEXP(_EXP):
    def __init__(self, pb_solver):
        pass

    

