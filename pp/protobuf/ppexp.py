



class EXP(object):
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



class PBEXP(EXP):
    def __init__(self, pb_solver):
        pass
    
    
    
class PYEXP(EXP):
    def __init__(self, pb_solver):
        pass
