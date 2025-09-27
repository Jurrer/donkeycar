"""

keras.py

Methods to create, use, save and load pilots. Pilots contain the highlevel
logic used to determine the angle and throttle of a vehicle. Pilots can
include one or more models to help direct the vehicles motion.

"""
import datetime
from abc import ABC, abstractmethod
from collections import deque

import numpy as np
from typing import Dict, Tuple, Optional, Union, List, Sequence, Callable, Any
from logging import getLogger

from tensorflow.python.data.ops.dataset_ops import DatasetV1, DatasetV2

import donkeycar as dk
from donkeycar.utils import normalize_image, linear_bin
from donkeycar.pipeline.types import TubRecord
from donkeycar.parts.interpreter import Interpreter, KerasInterpreter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Dense, Input,Convolution2D,
    MaxPooling2D, Activation, Dropout, Flatten, LSTM, BatchNormalization,
    Conv3D, MaxPooling3D, Conv2DTranspose, Add, GlobalAveragePooling2D,
    Reshape, Multiply, Lambda)

from tensorflow.keras.layers import TimeDistributed as TD
from tensorflow.keras.backend import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

ONE_BYTE_SCALE = 1.0 / 255.0

# type of x
XY = Union[float, np.ndarray, Tuple[Union[float, np.ndarray], ...]]


logger = getLogger(__name__)


class KerasPilot(ABC):
    """
    Base class for Keras models that will provide steering and throttle to
    guide a car.
    """
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3)) -> None:
        # self.model: Optional[Model] = None
        self.input_shape = input_shape
        self.optimizer = "adam"
        self.interpreter = interpreter
        self.interpreter.set_model(self)
        logger.info(f'Created {self} with interpreter: {interpreter}')

    def load(self, model_path: str) -> None:
        logger.info(f'Loading model {model_path}')
        self.interpreter.load(model_path)

    def load_weights(self, model_path: str, by_name: bool = True) -> None:
        self.interpreter.load_weights(model_path, by_name=by_name)

    def shutdown(self) -> None:
        pass

    def compile(self) -> None:
        pass

    @abstractmethod
    def create_model(self):
        pass

    def set_optimizer(self, optimizer_type: str,
                      rate: float, decay: float) -> None:
        if optimizer_type == "adam":
            optimizer = keras.optimizers.Adam(lr=rate, decay=decay)
        elif optimizer_type == "sgd":
            optimizer = keras.optimizers.SGD(lr=rate, decay=decay)
        elif optimizer_type == "rmsprop":
            optimizer = keras.optimizers.RMSprop(lr=rate, decay=decay)
        else:
            raise Exception(f"Unknown optimizer type: {optimizer_type}")
        self.interpreter.set_optimizer(optimizer)

    def get_input_shape(self, input_name) -> tf.TensorShape:
        return self.interpreter.get_input_shape(input_name)

    def seq_size(self) -> int:
        return 0

    def run(self, img_arr: np.ndarray, *other_arr: List[float]) \
            -> Tuple[Union[float, np.ndarray], ...]:
        """
        Donkeycar parts interface to run the part in the loop.

        :param img_arr:     uint8 [0,255] numpy array with image data
        :param other_arr:   numpy array of additional data to be used in the
                            pilot, like IMU array for the IMU model or a
                            state vector in the Behavioural model
        :return:            tuple of (angle, throttle)
        """
        norm_img_arr = normalize_image(img_arr)
        np_other_array = tuple(np.array(arr) for arr in other_arr)
        # create dictionary on the fly, we expect the order of the arguments:
        # img_arr, *other_arr to exactly match the order of the
        # self.output_shape() first dictionary keys, because that's how we
        # set up the model
        values = (norm_img_arr, ) + np_other_array
        # note output_shapes() returns a 2-tuple of dicts for input shapes
        # and output shapes(), so we need the first tuple here
        input_dict = dict(zip(self.output_shapes()[0].keys(), values))
        return self.inference_from_dict(input_dict)

    def inference_from_dict(self, input_dict: Dict[str, np.ndarray]) \
            -> Tuple[Union[float, np.ndarray], ...]:
        """ Inferencing using the interpreter
            :param input_dict:  input dictionary of str and np.ndarray
            :return:            typically tuple of (angle, throttle)
        """
        output = self.interpreter.predict_from_dict(input_dict)
        return self.interpreter_to_output(output)

    @abstractmethod
    def interpreter_to_output(
            self,
            interpreter_out: Sequence[Union[float, np.ndarray]]) \
            -> Tuple[Union[float, np.ndarray], ...]:
        """ Virtual method to be implemented by child classes for conversion
            :param interpreter_out:  input data
            :return:                 output values, possibly tuple of np.ndarray
        """
        pass

    def train(self,
              model_path: str,
              train_data: Union[DatasetV1, DatasetV2],
              train_steps: int,
              batch_size: int,
              validation_data: Union[DatasetV1, DatasetV2],
              validation_steps: int,
              epochs: int,
              verbose: int = 1,
              min_delta: float = .0005,
              patience: int = 5,
              show_plot: bool = False) -> tf.keras.callbacks.History:
        """
        trains the model
        """
        assert isinstance(self.interpreter, KerasInterpreter)
        model = self.interpreter.model
        self.compile()

        callbacks = [
            EarlyStopping(monitor='val_loss',
                          patience=patience,
                          min_delta=min_delta),
            ModelCheckpoint(monitor='val_loss',
                            filepath=model_path,
                            save_best_only=True,
                            verbose=verbose)]

        tic = datetime.datetime.now()
        logger.info('////////// Starting training //////////')
        history: tf.keras.callbacks.History = model.fit(
            x=train_data,
            steps_per_epoch=train_steps,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_data=validation_data,
            validation_steps=validation_steps,
            epochs=epochs,
            verbose=verbose,
            workers=1,
            use_multiprocessing=False)
        toc = datetime.datetime.now()
        logger.info(f'////////// Finished training in: {toc - tic} //////////')

        if show_plot:
            try:
                import matplotlib.pyplot as plt
                from pathlib import Path

                plt.figure(1)
                # Only do accuracy if we have that data
                # (e.g. categorical outputs)
                if 'angle_out_acc' in history.history:
                    plt.subplot(121)

                # summarize history for loss
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'validate'], loc='upper right')

                # summarize history for acc
                if 'angle_out_acc' in history.history:
                    plt.subplot(122)
                    plt.plot(history.history['angle_out_acc'])
                    plt.plot(history.history['val_angle_out_acc'])
                    plt.title('model angle accuracy')
                    plt.ylabel('acc')
                    plt.xlabel('epoch')

                plt.savefig(Path(model_path).with_suffix('.png'))
                # plt.show()

            except Exception as ex:
                print(f"problems with loss graph: {ex}")
            
        return history.history

    def x_transform(
            self,
            record: Union[TubRecord, List[TubRecord]],
            img_processor: Callable[[np.ndarray], np.ndarray]) \
            -> Dict[str, Union[float, np.ndarray]]:
        """ Transforms the record into dictionary for x for training the
        model to x,y, and applies an image augmentation. Here we assume the
        model only takes the image as input. All model input layer's names
        must be matched by dictionary keys."""
        assert isinstance(record, TubRecord), "TubRecord required"
        img_arr = record.image(processor=img_processor)
        return {'img_in': img_arr}

    def y_transform(self, record: Union[TubRecord, List[TubRecord]]) \
            -> Dict[str, Union[float, List[float]]]:
        """ Transforms the record into dictionary for y for training the
        model to x,y. All model ouputs layer's names must be matched by
        dictionary keys. """
        raise NotImplementedError(f'{self} not ready yet for new training '
                                  f'pipeline')

    def output_types(self) -> Tuple[Dict[str, np.typename], ...]:
        """ Used in tf.data, assume all types are doubles"""
        shapes = self.output_shapes()
        types = tuple({k: tf.float64 for k in d} for d in shapes)
        return types

    def output_shapes(self) -> Dict[str, tf.TensorShape]:
        return {}

    def __str__(self) -> str:
        """ For printing model initialisation """
        return type(self).__name__


class KerasCategorical(KerasPilot):
    """
    The KerasCategorical pilot breaks the steering and throttle decisions
    into discreet angles and then uses categorical cross entropy to train the
    network to activate a single neuron for each steering and throttle
    choice. This can be interesting because we get the confidence value as a
    distribution over all choices. This uses the dk.utils.linear_bin and
    dk.utils.linear_unbin to transform continuous real numbers into a range
    of discreet values for training and runtime. The input and output are
    therefore bounded and must be chosen wisely to match the data. The
    default ranges work for the default setup. But cars which go faster may
    want to enable a higher throttle range. And cars with larger steering
    throw may want more bins.
    """
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 throttle_range: float = 0.5):
        self.throttle_range = throttle_range
        super().__init__(interpreter, input_shape)

    def create_model(self):
        return default_categorical(self.input_shape)

    def compile(self):
        self.interpreter.compile(
            optimizer=self.optimizer,
            metrics=['accuracy'],
            loss={'angle_out': 'categorical_crossentropy',
                  'throttle_out': 'categorical_crossentropy'},
            loss_weights={'angle_out': 0.5, 'throttle_out': 0.5})

    def interpreter_to_output(self, interpreter_out):
        angle_binned, throttle_binned = interpreter_out
        N = len(throttle_binned)
        throttle = dk.utils.linear_unbin(throttle_binned, N=N,
                                         offset=0.0, R=self.throttle_range)
        angle = dk.utils.linear_unbin(angle_binned)
        return angle, throttle

    def y_transform(self, record: Union[TubRecord, List[TubRecord]]) \
            -> Dict[str, Union[float, List[float]]]:
        assert isinstance(record, TubRecord), "TubRecord expected"
        angle: float = record.underlying['user/angle']
        throttle: float = record.underlying['user/throttle']
        angle = linear_bin(angle, N=15, offset=1, R=2.0)
        throttle = linear_bin(throttle, N=20, offset=0.0, R=self.throttle_range)
        return {'angle_out': angle, 'throttle_out': throttle}

    def output_shapes(self):
        # need to cut off None from [None, 120, 160, 3] tensor shape
        img_shape = self.get_input_shape('img_in')[1:]
        shapes = ({'img_in': tf.TensorShape(img_shape)},
                  {'angle_out': tf.TensorShape([15]),
                   'throttle_out': tf.TensorShape([20])})
        return shapes

    def __str__(self) -> str:
        """ For printing model initialisation """
        return super().__str__() + f'-R:{self.throttle_range}'


class KerasLinear(KerasPilot):
    """
    The KerasLinear pilot uses one neuron to output a continuous value via
    the Keras Dense layer with linear activation. One each for steering and
    throttle. The output is not bounded.
    """
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 num_outputs: int = 2):
        self.num_outputs = num_outputs
        super().__init__(interpreter, input_shape)

    def create_model(self):
        return default_n_linear(self.num_outputs, self.input_shape)

    def compile(self):
        self.interpreter.compile(optimizer=self.optimizer, loss='mse')

    def interpreter_to_output(self, interpreter_out):
        steering = interpreter_out[0]
        throttle = interpreter_out[1]
        return steering[0], throttle[0]

    def y_transform(self, record: Union[TubRecord, List[TubRecord]]) \
            -> Dict[str, Union[float, List[float]]]:
        assert isinstance(record, TubRecord), 'TubRecord expected'
        angle: float = record.underlying['user/angle']
        throttle: float = record.underlying['user/throttle']
        return {'n_outputs0': angle, 'n_outputs1': throttle}

    def output_shapes(self):
        # need to cut off None from [None, 120, 160, 3] tensor shape
        img_shape = self.get_input_shape('img_in')[1:]
        shapes = ({'img_in': tf.TensorShape(img_shape)},
                  {'n_outputs0': tf.TensorShape([]),
                   'n_outputs1': tf.TensorShape([])})
        return shapes


class KerasMemory(KerasLinear):
    """
    The KerasLinearWithMemory is based on KerasLinear but uses the last n
    steering and throttle commands as input in order to produce smoother
    steering outputs
    """
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 mem_length: int = 3,
                 mem_depth: int = 0,
                 mem_start_speed: float = 0.0,
                 **kwargs):
        self.mem_length = mem_length
        self.mem_start_speed = mem_start_speed
        # create memory of [anlge=0, throttle=mem_start_speed] * mem_length
        self.mem_seq = deque([[0.0, mem_start_speed]] * mem_length)
        self.mem_depth = mem_depth
        super().__init__(interpreter, input_shape, **kwargs)

    def seq_size(self) -> int:
        return self.mem_length + 1

    def create_model(self):
        return default_memory(self.input_shape,
                              self.mem_length, self.mem_depth)

    def load(self, model_path: str) -> None:
        super().load(model_path)
        mem_shape = self.interpreter.get_input_shape('mem_in')
        # take the mem_shape (index 1), the length (index 1) and divide by 2.
        self.mem_length = mem_shape[1] // 2
        # create memory of [anlge=0, throttle=mem_start_speed] * mem_length
        self.mem_seq = deque([[0.0, self.mem_start_speed]] * self.mem_length)
        logger.info(f'Loaded {type(self).__name__} model with mem length'
                    f' {self.mem_length}')

    def run(self, img_arr: np.ndarray, *other_arr: List[float]) -> \
            Tuple[Union[float, np.ndarray], ...]:
        # Only called at start to fill the previous values
        np_mem_arr = np.array(self.mem_seq).reshape((2 * self.mem_length,))
        norm_img_arr = normalize_image(img_arr)
        # create dictionary on the fly, we expect the order of the arguments:
        # img_arr, *other_arr to exactly match the order of the
        # self.output_shape() first dictionary keys, because that's how we
        # set up the model
        values = (norm_img_arr, np_mem_arr)
        # note output_shapes() returns a 2-tuple of dicts for input shapes
        # and output shapes(), so we need the first tuple here
        input_dict = dict(zip(self.output_shapes()[0].keys(), values))
        angle, throttle = self.inference_from_dict(input_dict)
        # fill new values into back of history list for next call
        self.mem_seq.popleft()
        self.mem_seq.append([angle, throttle])
        return angle, throttle

    def x_transform(
            self,
            record: Union[TubRecord, List[TubRecord]],
            img_processor: Callable[[np.ndarray], np.ndarray]) \
            -> Dict[str, Union[float, np.ndarray]]:
        assert isinstance(record, list), 'List[TubRecord] expected'
        assert len(record) == self.mem_length + 1, \
            f"Record list of length {self.mem_length} required but " \
            f"{len(record)} was passed"
        img_arr = record[-1].image(processor=img_processor)
        mem = [[r.underlying['user/angle'], r.underlying['user/throttle']]
               for r in record[:-1]]
        np_mem = np.array(mem).reshape((2 * self.mem_length,))
        return {'img_in': img_arr, 'mem_in': np_mem}

    def y_transform(self, records: Union[TubRecord, List[TubRecord]]) \
            -> Dict[str, Union[float, List[float]]]:
        assert isinstance(records, list), 'List[TubRecord] expected'
        angle = records[-1].underlying['user/angle']
        throttle = records[-1].underlying['user/throttle']
        return {'n_outputs0': angle, 'n_outputs1': throttle}

    def output_shapes(self):
        # need to cut off None from [None, 120, 160, 3] tensor shape
        img_shape = self.get_input_shape('img_in')[1:]
        shapes = ({'img_in': tf.TensorShape(img_shape),
                   'mem_in': tf.TensorShape(2 * self.mem_length)},
                  {'n_outputs0': tf.TensorShape([]),
                   'n_outputs1': tf.TensorShape([])})
        return shapes

    def __str__(self) -> str:
        """ For printing model initialisation """
        return super().__str__() \
            + f'-L:{self.mem_length}-D:{self.mem_depth}'


class KerasInferred(KerasPilot):
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3)):
        super().__init__(interpreter, input_shape)

    def create_model(self):
        return default_n_linear(1, self.input_shape)

    def compile(self):
        self.interpreter.compile(optimizer=self.optimizer, loss='mse')

    def interpreter_to_output(self, interpreter_out):
        steering = interpreter_out[0]
        return steering, dk.utils.throttle(steering)

    def y_transform(self, record: Union[TubRecord, List[TubRecord]]) \
            -> Dict[str, Union[float, List[float]]]:
        assert isinstance(record, TubRecord), "TubRecord expected"
        angle: float = record.underlying['user/angle']
        return {'n_outputs0': angle}

    def output_shapes(self):
        # need to cut off None from [None, 120, 160, 3] tensor shape
        img_shape = self.get_input_shape('img_in')[1:]
        shapes = ({'img_in': tf.TensorShape(img_shape)},
                  {'n_outputs0': tf.TensorShape([])})
        return shapes


class KerasIMU(KerasPilot):
    """
    A Keras part that take an image and IMU vector as input,
    outputs steering and throttle
    """
    # keys for imu data in TubRecord
    imu_vec = [f'imu/{f}_{x}' for f in ('acl', 'gyr') for x in 'xyz']

    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 num_outputs: int = 2, num_imu_inputs: int = 6):
        self.num_outputs = num_outputs
        self.num_imu_inputs = num_imu_inputs
        super().__init__(interpreter, input_shape)

    def create_model(self):
        return default_imu(num_outputs=self.num_outputs,
                           num_imu_inputs=self.num_imu_inputs,
                           input_shape=self.input_shape)

    def compile(self):
        self.interpreter.compile(optimizer=self.optimizer, loss='mse')

    def interpreter_to_output(self, interpreter_out) \
            -> Tuple[Union[float, np.ndarray], ...]:
        steering = interpreter_out[0]
        throttle = interpreter_out[1]
        return steering[0], throttle[0]

    def x_transform(
            self,
            record: Union[TubRecord, List[TubRecord]],
            img_processor: Callable[[np.ndarray], np.ndarray]) \
            -> Dict[str, Union[float, np.ndarray]]:
        # this transforms the record into x for training the model to x,y
        assert isinstance(record, TubRecord), 'TubRecord expected'
        img_arr = record.image(processor=img_processor)
        imu_arr = np.array([record.underlying[k] for k in self.imu_vec])
        return {'img_in': img_arr, 'imu_in': imu_arr}

    def y_transform(self, record: Union[TubRecord, List[TubRecord]]) \
            -> Dict[str, Union[float, List[float]]]:
        assert isinstance(record, TubRecord), "TubRecord expected"
        angle: float = record.underlying['user/angle']
        throttle: float = record.underlying['user/throttle']
        return {'out_0': angle, 'out_1': throttle}

    def output_shapes(self):
        # need to cut off None from [None, 120, 160, 3] tensor shape
        img_shape = self.get_input_shape('img_in')[1:]
        # the keys need to match the models input/output layers
        shapes = ({'img_in': tf.TensorShape(img_shape),
                   'imu_in': tf.TensorShape([self.num_imu_inputs])},
                  {'out_0': tf.TensorShape([]),
                   'out_1': tf.TensorShape([])})
        return shapes


class KerasBehavioral(KerasCategorical):
    """
    A Keras part that take an image and Behavior vector as input,
    outputs steering and throttle
    """
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 throttle_range: float = 0.5,
                 num_behavior_inputs: int = 2):
        self.num_behavior_inputs = num_behavior_inputs
        super().__init__(interpreter, input_shape, throttle_range)

    def create_model(self):
        return default_bhv(num_bvh_inputs=self.num_behavior_inputs,
                           input_shape=self.input_shape)

    def x_transform(
            self,
            record: Union[TubRecord, List[TubRecord]],
            img_processor: Callable[[np.ndarray], np.ndarray]) \
            -> Dict[str, Union[float, np.ndarray]]:
        assert isinstance(record, TubRecord), 'TubRecord expected'
        # this transforms the record into x for training the model to x,y
        img_arr = record.image(processor=img_processor)
        bhv_arr = np.array(record.underlying['behavior/one_hot_state_array'])
        return {'img_in': img_arr, 'xbehavior_in': bhv_arr}

    def output_shapes(self):
        # need to cut off None from [None, 120, 160, 3] tensor shape
        img_shape = self.get_input_shape('img_in')[1:]
        # the keys need to match the models input/output layers
        shapes = ({'img_in': tf.TensorShape(img_shape),
                   'xbehavior_in': tf.TensorShape([self.num_behavior_inputs])},
                  {'angle_out': tf.TensorShape([15]),
                   'throttle_out': tf.TensorShape([20])})
        return shapes


class KerasLocalizer(KerasPilot):
    """
    A Keras part that take an image as input,
    outputs steering and throttle, and localisation category
    """
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 num_locations: int = 8):
        self.num_locations = num_locations
        super().__init__(interpreter, input_shape)

    def create_model(self):
        return default_loc(num_locations=self.num_locations,
                           input_shape=self.input_shape)

    def compile(self):
        self.interpreter.compile(optimizer=self.optimizer, metrics=['acc'],
                                 loss='mse')

    def interpreter_to_output(self, interpreter_out) \
            -> Tuple[Union[float, np.ndarray], ...]:
        angle, throttle, track_loc = interpreter_out
        loc = np.argmax(track_loc)
        return angle[0], throttle[0], loc

    def y_transform(self, record: Union[TubRecord, List[TubRecord]]) \
            -> Dict[str, Union[float, List[float]]]:
        assert isinstance(record, TubRecord), "TubRecord expected"
        angle: float = record.underlying['user/angle']
        throttle: float = record.underlying['user/throttle']
        loc = record.underlying['localizer/location']
        loc_one_hot = np.zeros(self.num_locations)
        loc_one_hot[loc] = 1
        return {'angle': angle, 'throttle': throttle, 'zloc': loc_one_hot}

    def output_shapes(self):
        # need to cut off None from [None, 120, 160, 3] tensor shape
        img_shape = self.get_input_shape('img_in')[1:]
        # the keys need to match the models input/output layers
        shapes = ({'img_in': tf.TensorShape(img_shape)},
                  {'angle': tf.TensorShape([]),
                   'throttle': tf.TensorShape([]),
                   'zloc': tf.TensorShape([self.num_locations])})
        return shapes


class KerasLSTM(KerasPilot):
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 seq_length=3,
                 num_outputs=2):
        self.num_outputs = num_outputs
        self.seq_length = seq_length
        super().__init__(interpreter, input_shape)
        self.img_seq = deque()
        self.optimizer = "rmsprop"

    def seq_size(self) -> int:
        return self.seq_length

    def create_model(self):
        return rnn_lstm(seq_length=self.seq_length,
                        num_outputs=self.num_outputs,
                        input_shape=self.input_shape)

    def compile(self):
        self.interpreter.compile(optimizer=self.optimizer, loss='mse')

    def x_transform(
            self,
            records: Union[TubRecord, List[TubRecord]],
            img_processor: Callable[[np.ndarray], np.ndarray]) \
        -> Dict[str, Union[float, np.ndarray]]:
        """ Transforms the record sequence into x for training the model to
            x, y. """
        assert isinstance(records, list), 'List[TubRecord] expected'
        assert len(records) == self.seq_length, \
            f"Record list of length {self.seq_length} required but " \
            f"{len(records)} was passed"
        img_arrays = [rec.image(processor=img_processor) for rec in records]
        return {'img_in': np.array(img_arrays)}

    def y_transform(self, records: Union[TubRecord, List[TubRecord]]) \
            -> Dict[str, Union[float, List[float]]]:
        """ Only return the last entry of angle/throttle"""
        assert isinstance(records, list), 'List[TubRecord] expected'
        angle = records[-1].underlying['user/angle']
        throttle = records[-1].underlying['user/throttle']
        return {'model_outputs': [angle, throttle]}

    def run(self, img_arr, *other_arr):
        if img_arr.shape[2] == 3 and self.input_shape[2] == 1:
            img_arr = dk.utils.rgb2gray(img_arr)

        while len(self.img_seq) < self.seq_length:
            self.img_seq.append(img_arr)

        self.img_seq.popleft()
        self.img_seq.append(img_arr)
        new_shape = (self.seq_length, *self.input_shape)
        img_arr = np.array(self.img_seq).reshape(new_shape)
        img_arr_norm = normalize_image(img_arr)
        input_dict = {'img_in': img_arr_norm}
        return self.inference_from_dict(input_dict)

    def interpreter_to_output(self, interpreter_out) \
            -> Tuple[Union[float, np.ndarray], ...]:
        steering = interpreter_out[0]
        throttle = interpreter_out[1]
        return steering, throttle

    def output_shapes(self):
        # need to cut off None from [None, 120, 160, 3] tensor shape
        img_shape = self.get_input_shape('img_in')[1:]
        # the keys need to match the models input/output layers
        shapes = ({'img_in': tf.TensorShape(img_shape)},
                  {'model_outputs': tf.TensorShape([self.num_outputs])})
        return shapes

    def __str__(self) -> str:
        """ For printing model initialisation """
        return f'{super().__str__()}-L:{self.seq_length}'


class Keras3D_CNN(KerasPilot):
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 seq_length=20,
                 num_outputs=2):
        self.num_outputs = num_outputs
        self.seq_length = seq_length
        super().__init__(interpreter, input_shape)
        self.img_seq = deque()

    def seq_size(self) -> int:
        return self.seq_length

    def create_model(self):
        return build_3d_cnn(self.input_shape, s=self.seq_length,
                            num_outputs=self.num_outputs)

    def compile(self):
        self.interpreter.compile(loss='mse', optimizer=self.optimizer)

    def x_transform(
            self,
            records: Union[TubRecord, List[TubRecord]],
            img_processor: Callable[[np.ndarray], np.ndarray]) \
            -> Dict[str, Union[float, np.ndarray]]:
        """ Transforms the record sequence into x for training the model to
            x, y. """
        assert isinstance(records, list), 'List[TubRecord] expected'
        assert len(records) == self.seq_length, \
            f"Record list of length {self.seq_length} required but " \
            f"{len(records)} was passed"
        img_seq = [rec.image(processor=img_processor) for rec in records]
        return {'img_in': np.array(img_seq)}

    def y_transform(self, records: Union[TubRecord, List[TubRecord]]) \
            -> Dict[str, Union[float, List[float]]]:
        """ Only return the last entry of angle/throttle"""
        assert isinstance(records, list), 'List[TubRecord] expected'
        angle = records[-1].underlying['user/angle']
        throttle = records[-1].underlying['user/throttle']
        return {'outputs': [angle, throttle]}

    def run(self, img_arr, *other_arr):
        if img_arr.shape[2] == 3 and self.input_shape[2] == 1:
            img_arr = dk.utils.rgb2gray(img_arr)

        while len(self.img_seq) < self.seq_length:
            self.img_seq.append(img_arr)

        self.img_seq.popleft()
        self.img_seq.append(img_arr)
        new_shape = (self.seq_length, *self.input_shape)
        img_arr = np.array(self.img_seq).reshape(new_shape)
        img_arr_norm = normalize_image(img_arr)
        input_dict = {'img_in': img_arr_norm}
        return self.inference_from_dict(input_dict)

    def interpreter_to_output(self, interpreter_out) \
            -> Tuple[Union[float, np.ndarray], ...]:
        steering = interpreter_out[0]
        throttle = interpreter_out[1]
        return steering, throttle

    def output_shapes(self):
        # need to cut off None from [None, 120, 160, 3] tensor shape
        img_shape = self.get_input_shape('img_in')[1:]
        # the keys need to match the models input/output layers
        shapes = ({'img_in': tf.TensorShape(img_shape)},
                  {'outputs': tf.TensorShape([self.num_outputs])})
        return shapes


class KerasLatent(KerasPilot):
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 num_outputs: int = 2):
        self.num_outputs = num_outputs
        super().__init__(interpreter, input_shape)

    def create_model(self):
        return default_latent(self.num_outputs, self.input_shape)

    def compile(self):
        loss = {"img_out": "mse", "n_outputs0": "mse", "n_outputs1": "mse"}
        weights = {"img_out": 100.0, "n_outputs0": 2.0, "n_outputs1": 1.0}
        self.interpreter.compile(optimizer=self.optimizer,
                                 loss=loss, loss_weights=weights)

    def interpreter_to_output(self, interpreter_out) \
            -> Tuple[Union[float, np.ndarray], ...]:
        steering = interpreter_out[1]
        throttle = interpreter_out[2]
        return steering[0][0], throttle[0][0]


def conv2d(filters, kernel, strides, layer_num, activation='relu'):
    """
    Helper function to create a standard valid-padded convolutional layer
    with square kernel and strides and unified naming convention

    :param filters:     channel dimension of the layer
    :param kernel:      creates (kernel, kernel) kernel matrix dimension
    :param strides:     creates (strides, strides) stride
    :param layer_num:   used in labelling the layer
    :param activation:  activation, defaults to relu
    :return:            tf.keras Convolution2D layer
    """
    return Convolution2D(filters=filters,
                         kernel_size=(kernel, kernel),
                         strides=(strides, strides),
                         activation=activation,
                         name='conv2d_' + str(layer_num))


def core_cnn_layers(img_in, drop, l4_stride=1):
    """
    Returns the core CNN layers that are shared among the different models,
    like linear, imu, behavioural

    :param img_in:          input layer of network
    :param drop:            dropout rate
    :param l4_stride:       4-th layer stride, default 1
    :return:                stack of CNN layers
    """
    x = img_in
    x = conv2d(24, 5, 2, 1)(x)
    x = Dropout(drop)(x)
    x = conv2d(32, 5, 2, 2)(x)
    x = Dropout(drop)(x)
    x = conv2d(64, 5, 2, 3)(x)
    x = Dropout(drop)(x)
    x = conv2d(64, 3, l4_stride, 4)(x)
    x = Dropout(drop)(x)
    x = conv2d(64, 3, 1, 5)(x)
    x = Dropout(drop)(x)
    x = Flatten(name='flattened')(x)
    return x


def default_n_linear(num_outputs, input_shape=(120, 160, 3)):
    drop = 0.2
    img_in = Input(shape=input_shape, name='img_in')
    x = core_cnn_layers(img_in, drop)
    x = Dense(100, activation='relu', name='dense_1')(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu', name='dense_2')(x)
    x = Dropout(drop)(x)

    outputs = []
    for i in range(num_outputs):
        outputs.append(
            Dense(1, activation='linear', name='n_outputs' + str(i))(x))

    model = Model(inputs=[img_in], outputs=outputs, name='linear')
    return model


def default_memory(input_shape=(120, 160, 3), mem_length=3, mem_depth=0):
    drop = 0.2
    drop2 = 0.1
    logger.info(f'Creating memory model with length {mem_length}, depth '
                f'{mem_depth}')
    img_in = Input(shape=input_shape, name='img_in')
    x = core_cnn_layers(img_in, drop)
    mem_in = Input(shape=(2 * mem_length,), name='mem_in')
    y = mem_in
    for i in range(mem_depth):
        y = Dense(4 * mem_length, activation='relu', name=f'mem_{i}')(y)
        y = Dropout(drop2)(y)
    for i in range(1, mem_length):
        y = Dense(2 * (mem_length - i), activation='relu', name=f'mem_c_{i}')(y)
        y = Dropout(drop2)(y)
    x = concatenate([x, y])
    x = Dense(100, activation='relu', name='dense_1')(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu', name='dense_2')(x)
    x = Dropout(drop)(x)
    activation = ['tanh', 'sigmoid']
    outputs = [Dense(1, activation=activation[i], name='n_outputs' + str(i))(x)
               for i in range(2)]
    model = Model(inputs=[img_in, mem_in], outputs=outputs, name='memory')
    return model


def default_categorical(input_shape=(120, 160, 3)):
    drop = 0.2
    img_in = Input(shape=input_shape, name='img_in')
    x = core_cnn_layers(img_in, drop, l4_stride=2)
    x = Dense(100, activation='relu', name="dense_1")(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu', name="dense_2")(x)
    x = Dropout(drop)(x)
    # Categorical output of the angle into 15 bins
    angle_out = Dense(15, activation='softmax', name='angle_out')(x)
    # categorical output of throttle into 20 bins
    throttle_out = Dense(20, activation='softmax', name='throttle_out')(x)

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out],
                  name='categorical')
    return model


def default_imu(num_outputs, num_imu_inputs, input_shape):
    drop = 0.2
    img_in = Input(shape=input_shape, name='img_in')
    imu_in = Input(shape=(num_imu_inputs,), name="imu_in")

    x = core_cnn_layers(img_in, drop)
    x = Dense(100, activation='relu')(x)
    x = Dropout(.1)(x)
    
    y = imu_in
    y = Dense(14, activation='relu')(y)
    y = Dense(14, activation='relu')(y)
    y = Dense(14, activation='relu')(y)
    
    z = concatenate([x, y])
    z = Dense(50, activation='relu')(z)
    z = Dropout(.1)(z)
    z = Dense(50, activation='relu')(z)
    z = Dropout(.1)(z)

    outputs = []
    for i in range(num_outputs):
        outputs.append(Dense(1, activation='linear', name='out_' + str(i))(z))
        
    model = Model(inputs=[img_in, imu_in], outputs=outputs, name='imu')
    return model


def default_bhv(num_bvh_inputs, input_shape):
    drop = 0.2
    img_in = Input(shape=input_shape, name='img_in')
    # tensorflow is ordering the model inputs alphabetically in tensorrt,
    # so behavior must come after image, hence we put an x here in front.
    bvh_in = Input(shape=(num_bvh_inputs,), name="xbehavior_in")

    x = core_cnn_layers(img_in, drop)
    x = Dense(100, activation='relu')(x)
    x = Dropout(.1)(x)
    
    y = bvh_in
    y = Dense(num_bvh_inputs * 2, activation='relu')(y)
    y = Dense(num_bvh_inputs * 2, activation='relu')(y)
    y = Dense(num_bvh_inputs * 2, activation='relu')(y)
    
    z = concatenate([x, y])
    z = Dense(100, activation='relu')(z)
    z = Dropout(.1)(z)
    z = Dense(50, activation='relu')(z)
    z = Dropout(.1)(z)
    
    # Categorical output of the angle into 15 bins
    angle_out = Dense(15, activation='softmax', name='angle_out')(z)
    # Categorical output of throttle into 20 bins
    throttle_out = Dense(20, activation='softmax', name='throttle_out')(z)

    model = Model(inputs=[img_in, bvh_in], outputs=[angle_out, throttle_out],
                  name='behavioral')
    return model


def default_loc(num_locations, input_shape):
    drop = 0.2
    img_in = Input(shape=input_shape, name='img_in')

    x = core_cnn_layers(img_in, drop)
    x = Dense(100, activation='relu')(x)
    x = Dropout(drop)(x)
    
    z = Dense(50, activation='relu')(x)
    z = Dropout(drop)(z)

    # linear output of the angle
    angle_out = Dense(1, activation='linear', name='angle')(z)
    # linear output of throttle
    throttle_out = Dense(1, activation='linear', name='throttle')(z)
    # Categorical output of location
    # Here is a crazy detail b/c TF Lite has a bug and returns the outputs
    # in the alphabetical order of the name of the layers, so make sure
    # this output comes last
    loc_out = Dense(num_locations, activation='softmax', name='zloc')(z)

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out, loc_out],
                  name='localizer')
    return model


def rnn_lstm(seq_length=3, num_outputs=2, input_shape=(120, 160, 3)):
    # add sequence length dimensions as keras time-distributed expects shape
    # of (num_samples, seq_length, input_shape)
    img_seq_shape = (seq_length,) + input_shape
    img_in = Input(shape=img_seq_shape, name='img_in')
    drop_out = 0.3

    x = img_in
    x = TD(Convolution2D(24, (5, 5), strides=(2, 2), activation='relu'))(x)
    x = TD(Dropout(drop_out))(x)
    x = TD(Convolution2D(32, (5, 5), strides=(2, 2), activation='relu'))(x)
    x = TD(Dropout(drop_out))(x)
    x = TD(Convolution2D(32, (3, 3), strides=(2, 2), activation='relu'))(x)
    x = TD(Dropout(drop_out))(x)
    x = TD(Convolution2D(32, (3, 3), strides=(1, 1), activation='relu'))(x)
    x = TD(Dropout(drop_out))(x)
    x = TD(MaxPooling2D(pool_size=(2, 2)))(x)
    x = TD(Flatten(name='flattened'))(x)
    x = TD(Dense(100, activation='relu'))(x)
    x = TD(Dropout(drop_out))(x)

    x = LSTM(128, return_sequences=True, name="LSTM_seq")(x)
    x = Dropout(.1)(x)
    x = LSTM(128, return_sequences=False, name="LSTM_fin")(x)
    x = Dropout(.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(.1)(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    out = Dense(num_outputs, activation='linear', name='model_outputs')(x)
    model = Model(inputs=[img_in], outputs=[out], name='lstm')
    return model


def build_3d_cnn(input_shape, s, num_outputs):
    """
    Credit: https://github.com/jessecha/DNRacing/blob/master/3D_CNN_Model/model.py

    :param input_shape:     image input shape
    :param s:               sequence length
    :param num_outputs:     output dimension
    :return:                keras model
    """
    drop = 0.5
    input_shape = (s, ) + input_shape
    img_in = Input(shape=input_shape, name='img_in')
    x = img_in
    # Second layer
    x = Conv3D(
            filters=16, kernel_size=(3, 3, 3), strides=(1, 3, 3),
            data_format='channels_last', padding='same', activation='relu')(x)
    x = MaxPooling3D(
            pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid',
            data_format=None)(x)
    # Third layer
    x = Conv3D(
            filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1),
            data_format='channels_last', padding='same', activation='relu')(x)
    x = MaxPooling3D(
        pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid',
        data_format=None)(x)
    # Fourth layer
    x = Conv3D(
            filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1),
            data_format='channels_last', padding='same', activation='relu')(x)
    x = MaxPooling3D(
            pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid',
            data_format=None)(x)
    # Fifth layer
    x = Conv3D(
            filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1),
            data_format='channels_last', padding='same', activation='relu')(x)
    x = MaxPooling3D(
            pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid',
            data_format=None)(x)
    # Fully connected layer
    x = Flatten()(x)

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(drop)(x)

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(drop)(x)

    out = Dense(num_outputs, name='outputs')(x)
    model = Model(inputs=[img_in], outputs=out, name='3dcnn')
    return model


def default_latent(num_outputs, input_shape):
    # TODO: this auto-encoder should run the standard cnn in encoding and
    #  have corresponding decoder. Also outputs should be reversed with
    #  images at end.
    drop = 0.2
    img_in = Input(shape=input_shape, name='img_in')
    x = img_in
    x = Convolution2D(24, 5, strides=2, activation='relu', name="conv2d_1")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(32, 5, strides=2, activation='relu', name="conv2d_2")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(32, 5, strides=2, activation='relu', name="conv2d_3")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(32, 3, strides=1, activation='relu', name="conv2d_4")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(32, 3, strides=1, activation='relu', name="conv2d_5")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, 3, strides=2, activation='relu', name="conv2d_6")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, 3, strides=2, activation='relu', name="conv2d_7")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, 1, strides=2, activation='relu', name="latent")(x)

    y = Conv2DTranspose(filters=64, kernel_size=3, strides=2,
                        name="deconv2d_1")(x)
    y = Conv2DTranspose(filters=64, kernel_size=3, strides=2,
                        name="deconv2d_2")(y)
    y = Conv2DTranspose(filters=32, kernel_size=3, strides=2,
                        name="deconv2d_3")(y)
    y = Conv2DTranspose(filters=32, kernel_size=3, strides=2,
                        name="deconv2d_4")(y)
    y = Conv2DTranspose(filters=32, kernel_size=3, strides=2,
                        name="deconv2d_5")(y)
    y = Conv2DTranspose(filters=1, kernel_size=3, strides=2, name="img_out")(y)
    
    x = Flatten(name='flattened')(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(drop)(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(drop)(x)

    outputs = [y]
    for i in range(num_outputs):
        outputs.append(Dense(1, activation='linear', name='n_outputs' + str(i))(x))
        
    model = Model(inputs=[img_in], outputs=outputs, name='latent')
    return model


class KerasAdvanced(KerasPilot):
    """
    Advanced Keras pilot with ResNet backbone, attention mechanisms,
    uncertainty quantification, and advanced loss functions.
    Memory-optimized for GPU efficiency.
    """
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 num_outputs: int = 2,
                 num_residual_blocks: int = 4,  # Reduced for memory efficiency
                 base_filters: int = 32,        # Reduced from 64
                 use_attention: bool = True,
                 mc_samples: int = 10,
                 uncertainty_weight: float = 0.1):
        self.num_outputs = num_outputs
        self.num_residual_blocks = num_residual_blocks
        self.base_filters = base_filters
        self.use_attention = use_attention
        self.mc_samples = mc_samples
        self.uncertainty_weight = uncertainty_weight
        super().__init__(interpreter, input_shape)

    def create_model(self):
        return advanced_resnet_with_uncertainty(
            input_shape=self.input_shape,
            num_outputs=self.num_outputs,
            num_residual_blocks=self.num_residual_blocks,
            base_filters=self.base_filters,
            use_attention=self.use_attention
        )

    def compile(self):
        self.interpreter.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
            loss={
                'steering_mean': advanced_huber_loss,
                'steering_var': 'mse',
                'throttle_mean': advanced_huber_loss,
                'throttle_var': 'mse'
            },
            loss_weights={
                'steering_mean': 1.0,
                'steering_var': self.uncertainty_weight,
                'throttle_mean': 1.0,
                'throttle_var': self.uncertainty_weight
            },
            metrics=['mae']
        )

    def interpreter_to_output(self, interpreter_out):
        steering_mean, steering_var, throttle_mean, throttle_var = interpreter_out
        return steering_mean[0], throttle_mean[0], steering_var[0], throttle_var[0]

    def run_with_uncertainty(self, img_arr: np.ndarray, *other_arr: List[float]):
        """Run inference with uncertainty quantification using Monte Carlo dropout."""
        norm_img_arr = normalize_image(img_arr)
        np_other_array = tuple(np.array(arr) for arr in other_arr)
        values = (norm_img_arr, ) + np_other_array
        input_dict = dict(zip(self.output_shapes()[0].keys(), values))

        # Enable training mode for dropout during inference
        predictions = []
        for _ in range(self.mc_samples):
            pred = self.interpreter.predict_from_dict(input_dict, training=True)
            predictions.append(pred)

        # Calculate mean and uncertainty across samples
        predictions = np.array(predictions)
        mean_steering = np.mean(predictions[:, 0])
        mean_throttle = np.mean(predictions[:, 2])
        uncertainty_steering = np.std(predictions[:, 0])
        uncertainty_throttle = np.std(predictions[:, 2])

        return mean_steering, mean_throttle, uncertainty_steering, uncertainty_throttle

    def y_transform(self, record: Union[TubRecord, List[TubRecord]]) \
            -> Dict[str, Union[float, List[float]]]:
        assert isinstance(record, TubRecord), 'TubRecord expected'
        angle: float = record.underlying['user/angle']
        throttle: float = record.underlying['user/throttle']
        # Initialize variance targets (will be learned during training)
        return {
            'steering_mean': angle,
            'steering_var': 0.1,  # Initial uncertainty estimate
            'throttle_mean': throttle,
            'throttle_var': 0.1
        }

    def output_shapes(self):
        img_shape = self.get_input_shape('img_in')[1:]
        shapes = (
            {'img_in': tf.TensorShape(img_shape)},
            {
                'steering_mean': tf.TensorShape([]),
                'steering_var': tf.TensorShape([]),
                'throttle_mean': tf.TensorShape([]),
                'throttle_var': tf.TensorShape([])
            }
        )
        return shapes


def advanced_huber_loss(y_true, y_pred, delta=1.0):
    """Advanced Huber loss with adaptive delta."""
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= delta
    squared_loss = tf.square(error) / 2
    linear_loss = delta * tf.abs(error) - tf.square(delta) / 2
    return tf.where(is_small_error, squared_loss, linear_loss)


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """Focal loss for handling class imbalance in steering angles."""
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    alpha_factor = tf.ones_like(y_true) * alpha
    alpha_t = tf.where(tf.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
    cross_entropy = -tf.math.log(p_t)
    weight = alpha_t * tf.pow((1 - p_t), gamma)
    focal_loss = weight * cross_entropy
    return tf.reduce_mean(focal_loss)


def uncertainty_loss(y_true, y_pred_var):
    """Loss function for uncertainty estimation."""
    return tf.reduce_mean(tf.square(y_pred_var))


def residual_block(x, filters, kernel_size=3, stride=1, dropout_rate=0.1):
    """ResNet-style residual block with dropout for uncertainty."""
    shortcut = x

    # First conv layer
    x = Convolution2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x, training=True)  # Always active for MC dropout

    # Second conv layer
    x = Convolution2D(filters, kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    # Adjust shortcut if needed
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Convolution2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # Add shortcut
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x, training=True)

    return x


def channel_attention(x, ratio=8):
    """Channel attention mechanism (SE-Net style)."""
    channels = x.shape[-1]

    # Global average pooling
    gap = GlobalAveragePooling2D()(x)

    # MLP
    mlp = Dense(channels // ratio, activation='relu')(gap)
    mlp = Dense(channels, activation='sigmoid')(mlp)

    # Reshape and multiply
    mlp = Reshape((1, 1, channels))(mlp)
    return Multiply()([x, mlp])


def spatial_attention(x):
    """Spatial attention mechanism."""
    # Average and max pooling across channels
    avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(x)
    max_pool = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(x)

    # Concatenate and convolve
    concat = concatenate([avg_pool, max_pool])
    attention = Convolution2D(1, 7, padding='same', activation='sigmoid')(concat)

    return Multiply()([x, attention])


def advanced_resnet_with_uncertainty(input_shape=(120, 160, 3),
                                   num_outputs=2,
                                   num_residual_blocks=4,
                                   base_filters=32,
                                   use_attention=True):
    """
    Memory-optimized ResNet architecture with attention mechanisms and uncertainty quantification.
    Balanced between performance and GPU memory usage.
    """
    img_in = Input(shape=input_shape, name='img_in')

    # Initial convolution - smaller kernel to save memory
    x = Convolution2D(base_filters, 5, strides=2, padding='same')(img_in)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)

    # Progressive residual blocks with controlled channel growth
    filters = base_filters
    for i in range(num_residual_blocks):
        stride = 2 if i > 0 and i % 2 == 0 else 1
        if i > 0 and i % 2 == 0:
            filters = min(filters * 2, 128)  # Cap at 128 to control memory

        x = residual_block(x, filters, stride=stride, dropout_rate=0.1)

        # Add attention every 2 blocks
        if use_attention and (i + 1) % 2 == 0:
            x = channel_attention(x, ratio=16)  # Higher ratio to reduce params
            x = spatial_attention(x)

    # One additional block for capacity without massive memory increase
    x = residual_block(x, filters, dropout_rate=0.15)
    if use_attention:
        x = channel_attention(x, ratio=16)

    # Global average pooling instead of flatten for better generalization
    x = GlobalAveragePooling2D()(x)

    # Smaller dense layers to reduce memory usage
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.25)(x, training=True)  # MC dropout
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x, training=True)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.15)(x, training=True)

    # Uncertainty quantification outputs
    # For each output, predict both mean and variance
    steering_mean = Dense(1, activation='tanh', name='steering_mean')(x)
    steering_var = Dense(1, activation='softplus', name='steering_var')(x)  # Ensure positive

    throttle_mean = Dense(1, activation='sigmoid', name='throttle_mean')(x)
    throttle_var = Dense(1, activation='softplus', name='throttle_var')(x)

    model = Model(
        inputs=[img_in],
        outputs=[steering_mean, steering_var, throttle_mean, throttle_var],
        name='advanced_resnet_uncertainty'
    )

    return model


class KerasAdvancedLite(KerasPilot):
    """
    Lightweight version of KerasAdvanced for smaller GPUs.
    Still includes modern techniques but with reduced memory footprint.
    """
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 num_outputs: int = 2,
                 base_filters: int = 24,
                 use_attention: bool = True,
                 mc_samples: int = 5,
                 uncertainty_weight: float = 0.1):
        self.num_outputs = num_outputs
        self.base_filters = base_filters
        self.use_attention = use_attention
        self.mc_samples = mc_samples
        self.uncertainty_weight = uncertainty_weight
        super().__init__(interpreter, input_shape)

    def create_model(self):
        return lightweight_resnet_with_uncertainty(
            input_shape=self.input_shape,
            base_filters=self.base_filters,
            use_attention=self.use_attention
        )

    def compile(self):
        self.interpreter.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
            loss={
                'steering_mean': advanced_huber_loss,
                'steering_var': 'mse',
                'throttle_mean': advanced_huber_loss,
                'throttle_var': 'mse'
            },
            loss_weights={
                'steering_mean': 1.0,
                'steering_var': self.uncertainty_weight,
                'throttle_mean': 1.0,
                'throttle_var': self.uncertainty_weight
            },
            metrics=['mae']
        )

    def interpreter_to_output(self, interpreter_out):
        steering_mean, steering_var, throttle_mean, throttle_var = interpreter_out
        return steering_mean[0], throttle_mean[0], steering_var[0], throttle_var[0]

    def y_transform(self, record: Union[TubRecord, List[TubRecord]]) \
            -> Dict[str, Union[float, List[float]]]:
        assert isinstance(record, TubRecord), 'TubRecord expected'
        angle: float = record.underlying['user/angle']
        throttle: float = record.underlying['user/throttle']
        return {
            'steering_mean': angle,
            'steering_var': 0.1,
            'throttle_mean': throttle,
            'throttle_var': 0.1
        }

    def output_shapes(self):
        img_shape = self.get_input_shape('img_in')[1:]
        shapes = (
            {'img_in': tf.TensorShape(img_shape)},
            {
                'steering_mean': tf.TensorShape([]),
                'steering_var': tf.TensorShape([]),
                'throttle_mean': tf.TensorShape([]),
                'throttle_var': tf.TensorShape([])
            }
        )
        return shapes


def lightweight_resnet_with_uncertainty(input_shape=(120, 160, 3),
                                       base_filters=24,
                                       use_attention=True):
    """
    Lightweight ResNet for GPUs with limited memory (4-6GB).
    Maintains advanced features while being memory efficient.
    """
    img_in = Input(shape=input_shape, name='img_in')

    # Initial convolution with aggressive downsampling
    x = Convolution2D(base_filters, 5, strides=2, padding='same')(img_in)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(2, strides=2, padding='same')(x)

    # Lightweight residual blocks
    x = residual_block(x, base_filters, stride=1, dropout_rate=0.1)
    if use_attention:
        x = channel_attention(x, ratio=8)

    x = residual_block(x, base_filters * 2, stride=2, dropout_rate=0.1)
    if use_attention:
        x = spatial_attention(x)

    x = residual_block(x, base_filters * 3, stride=2, dropout_rate=0.15)
    if use_attention:
        x = channel_attention(x, ratio=8)

    # Global average pooling
    x = GlobalAveragePooling2D()(x)

    # Compact dense layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x, training=True)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.15)(x, training=True)

    # Uncertainty outputs
    steering_mean = Dense(1, activation='tanh', name='steering_mean')(x)
    steering_var = Dense(1, activation='softplus', name='steering_var')(x)
    throttle_mean = Dense(1, activation='sigmoid', name='throttle_mean')(x)
    throttle_var = Dense(1, activation='softplus', name='throttle_var')(x)

    model = Model(
        inputs=[img_in],
        outputs=[steering_mean, steering_var, throttle_mean, throttle_var],
        name='lightweight_resnet_uncertainty'
    )

    return model
