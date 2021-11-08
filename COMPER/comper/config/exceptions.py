import comper.config.transitions
class ExceptionFrameMode(Exception):
    
    """ Exception raised for errors in the framesmode parameter.

        Attributes:
        wrong_value -- input salary which caused the error  
        message -- custo mensage      
    """
    def __init__(self, wrong_value, message="Wrong value to 'framesmode'. Must be 'single' or 'staked'."):
        self.wrong_value = wrong_value
        self.message = message
        super().__init__(self.message)

class ExceptionDisplayScreen(Exception):
    
    """ Exception raised for errors in the framesmode parameter.

        Attributes:
        wrong_value -- input salary which caused the error  
        message -- custo mensage      
    """
    def __init__(self, wrong_value, message="Wrong value to 'display_screen'. Must be 0 or 1."):
        self.wrong_value = wrong_value
        self.message = message
        super().__init__(self.message)

class ExceptionPersistMemory(Exception):
    
    """ Exception raised for errors in the persist_memory parameter.

        Attributes:
        wrong_value -- input salary which caused the error  
        message -- custo mensage      
    """
    def __init__(self, wrong_value, message="Wrong value to 'persist_memories'. Must be 0 or 1."):
        self.wrong_value = wrong_value
        self.message = message
        super().__init__(self.message)

class ExceptionRunType(Exception):
    
    """ Exception raised for errors in the run type attribuite.

        Attributes:
        wrong_value -- input salary which caused the error  
        message -- custo mensage      
    """
    def __init__(self, wrong_value, message="Wrong value to 'run type'. Must be 1 or 2."):
        self.wrong_value = wrong_value
        self.message = message
        super().__init__(self.message)