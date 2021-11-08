import tensorflow as tf
from keras import backend as K
import os
class TensorFlowSettings(object):
    def apply(self,num_parallel_exec_units,inter_op_parallelism_threads=2,allow_soft_placement=True):
        
        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=num_parallel_exec_units,
                                 inter_op_parallelism_threads=inter_op_parallelism_threads,
                                 allow_soft_placement=allow_soft_placement,
                                 device_count = {'CPU': allow_soft_placement})
        session = tf.compat.v1.Session(config=config)
        #os.environ["OMP_NUM_THREADS"] = str(num_parallel_exec_units)
        tf.config.threading.set_inter_op_parallelism_threads(num_parallel_exec_units)

        os.environ["KMP_BLOCKTIME"] = "30"

        os.environ["KMP_SETTINGS"] = "1"

        os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

        return session

class KerasSettings(object):

    def apply(self,num_parallel_exec_units, inter_op_parallelism_threads=2, allow_soft_placement=True):
        
        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=num_parallel_exec_units,
                                inter_op_parallelism_threads=inter_op_parallelism_threads,
                                 allow_soft_placement=allow_soft_placement, 
                                 device_count = {'CPU': num_parallel_exec_units})
        session = tf.Session(config=config)

        K.set_session(session)

        os.environ["OMP_NUM_THREADS"] = str(num_parallel_exec_units)

        os.environ["KMP_BLOCKTIME"] = "30"

        os.environ["KMP_SETTINGS"] = "1"

        os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"


