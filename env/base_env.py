from abc import ABC, abstractmethod
import time
import numpy as np

class BaseEnv(ABC):
    """
    Base env class for all deploy env (e.g. mujoco, real world)
    
    This abstract base class defines the common interface and functionality
    that all env implementations should follow.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize base env
        """
        self.num_actions = None
        # Required attributes that must be set by subclasses

        # Validate that subclasses properly initialize required attributes
        self._validate_required_attributes()
        
    def _validate_required_attributes(self):
        """Validate that all required attributes are properly set by subclasses"""
        required_attrs = []
        for attr in required_attrs:
            if getattr(self, attr) is None:
                raise NotImplementedError(f"Subclass must set '{attr}' attribute")
        
    @abstractmethod
    def reset(self):
        NotImplementedError("reset Not Implemented")
        pass
    
    @abstractmethod
    def step(self, mujoco_data):
        NotImplementedError("step Not Implemented")
        pass
    
    @abstractmethod
    def get_env_data(self) -> dict:
        NotImplementedError("get_env_data Not Implemented")
        pass
    
    @abstractmethod
    def get_joystick_val(self) -> dict:
        NotImplementedError("get_joystick_val Not Implemented")
        pass
    
    def get_tick_time(self) -> int:
        return int(time.time() * 1000)
    
    def __str__(self):
        # TODO
        """String representation of env"""
        pass    
    
    def __repr__(self):
        # TODO
        """Detailed representation of env"""
        pass