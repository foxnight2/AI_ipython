

class Node:
    pass


class TraceableFunction(Node):
    pass


class GradFunction(TraceableFunction):
    pass



class Edge:
    def __init__(self) -> None:
        
        self.grad_fn: GradFunction
        self.input_nr: int


class Variable:
    pass


class AutogradMeta:
    def __init__(self) -> None:

        # std::string name_;
        # Variable grad_;
        # std::shared_ptr<Node> grad_fn_;
        # std::weak_ptr<Node> grad_accumulator_;
        # // other fields and methods
        pass


class Tensor:
    pass




if __name__ == '__main__':

    edge = Edge()

