import abc

class BaseController(abc.ABC):
    '''
    An abstract base class for low level controllers.
    '''
    def __init__(self, sim):
        self.sim = sim
        self.action_space = None

    @abc.abstractmethod
    def set_action(self, action):
        '''
        Stores an action that will later affect the torques.
        '''
        pass

    @abc.abstractmethod
    def get_torque(self):
        '''
        Computes the raw motor torques/
        '''
        pass
