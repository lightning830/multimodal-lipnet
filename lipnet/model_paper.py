from tensorflow.keras.layers import Conv3D, ZeroPadding3D, Conv1D, Conv2D, ZeroPadding2D, concatenate
from tensorflow.keras.layers import MaxPool3D, MaxPool2D, MaxPool1D, GlobalAveragePooling3D, GlobalAveragePooling2D, AveragePooling1D
from tensorflow.keras.layers import Dense, Activation, SpatialDropout3D, Flatten
from tensorflow.keras.layers import Bidirectional, TimeDistributed
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Reshape
from tensorflow.keras.models import Model
from lipnet.core.layers import CTC
from tensorflow.keras import backend as K
from tensorflow.nn import conv1d_transpose
import tensorflow as tf

import sys
sys.path.append('../../lipnet')
from resnet.residual_block import make_basic_block_layer, make_basic_block_layer3D, make_basic_block_layer1D


class LipNet(object):
    def __init__(self, img_c=3, img_w=100, img_h=50, frames_n=75, wave_n = 131328, fs = 44100, absolute_max_string_len=32, output_size=28):
        self.img_c = img_c
        self.img_w = img_w
        self.img_h = img_h
        self.frames_n = frames_n
        self.wave_n = wave_n
        self.fs = fs
        self.absolute_max_string_len = absolute_max_string_len
        self.output_size = output_size
        self.build()

    def build(self):
        if K.image_data_format() == 'channels_first':
            input_shape = (self.img_c, self.frames_n, self.img_w, self.img_h)
        else:
            input_shape = (self.frames_n, self.img_w, self.img_h, self.img_c)

        self.input_data = Input(name='the_input', shape=input_shape, dtype='float32')
        self.input_data2 = Input(name='the_input2', shape=(self.wave_n, 1), dtype='float32')

        # print(self.input_data2)

        self.zero1 = ZeroPadding3D(padding=(1, 2, 2), name='zero1')(self.input_data)
        self.conv1 = Conv3D(32, (5, 7, 7), strides=(1, 2, 2), kernel_initializer='he_normal', name='conv1')(self.zero1)
        self.batc1 = BatchNormalization(name='batc1')(self.conv1)#正規化
        self.actv1 = Activation('relu', name='actv1')(self.batc1)
        self.drop1 = SpatialDropout3D(0.5)(self.actv1)
        self.zero2 = ZeroPadding3D(padding=(1, 2, 2), name='zero2')(self.drop1)
        # self.maxp1 = MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max1')(self.drop1)
        self.resnet = Conv3D(filters=64, kernel_size=(1, 7, 7), strides=(1, 2, 2), padding="same")(self.zero2)
        self.resnet = BatchNormalization()(self.resnet)
        self.resnet = Activation('relu')(self.resnet)
        self.resnet = MaxPool3D(pool_size=(1, 3, 3), strides=(1,2,2), padding="same")(self.resnet)
        self.resnet = make_basic_block_layer3D(filter_num=64, blocks=3)(self.resnet)
        self.resnet = make_basic_block_layer3D(filter_num=128, blocks=4, stride=(1,2,2))(self.resnet)
        self.resnet = make_basic_block_layer3D(filter_num=256, blocks=6, stride=(1,2,2))(self.resnet)
        self.resnet = make_basic_block_layer3D(filter_num=512, blocks=3, stride=(1,2,2))(self.resnet)
        # self.resnet = GlobalAveragePooling3D()(self.resnet)
        self.resnet = TimeDistributed(Flatten())(self.resnet)

        self.gru_1 = Bidirectional(GRU(1024,  return_sequences=True, kernel_initializer='Orthogonal', name='gru1'), merge_mode='concat')(self.resnet)
        self.gru_2 = Bidirectional(GRU(1024,  return_sequences=True, kernel_initializer='Orthogonal', name='gru2'), merge_mode='concat')(self.gru_1)

        self.resnet2 = Conv1D(filters=64, kernel_size=int(5*10**-3*self.fs), strides=int(0.25*10**-3*self.fs), padding="same")(self.input_data2)
        # print(self.resnet2)
        self.resnet2 = BatchNormalization()(self.resnet2)
        self.resnet2 = Activation('relu')(self.resnet2)
        # self.resnet2 = Reshape((75, 64, 1))(self.resnet2)
        self.resnet2 = make_basic_block_layer1D(filter_num=64, blocks=2,)(self.resnet2)
        self.resnet2 = make_basic_block_layer1D(filter_num=128, blocks=2, stride=2)(self.resnet2)
        self.resnet2 = make_basic_block_layer1D(filter_num=256, blocks=2, stride=2)(self.resnet2)
        self.resnet2 = make_basic_block_layer1D(filter_num=512, blocks=2, stride=2)(self.resnet2)
        self.resnet2 = AveragePooling1D(pool_size=21, strides=20, padding="same")(self.resnet2)

        self.resnet2 = TimeDistributed(Flatten())(self.resnet2)
        self.gru_3 = Bidirectional(GRU(1024,  return_sequences=True, kernel_initializer='Orthogonal', name='gru3'), merge_mode='concat')(self.resnet2)
        self.gru_4 = Bidirectional(GRU(1024,  return_sequences=True, kernel_initializer='Orthogonal', name='gru4'), merge_mode='concat')(self.gru_3)

        self.concat = concatenate([self.gru_2, self.gru_4])

        self.gru_5 = Bidirectional(GRU(1024,  return_sequences=True, kernel_initializer='Orthogonal', name='gru5'), merge_mode='concat')(self.concat)
        self.gru_6 = Bidirectional(GRU(1024,  return_sequences=True, kernel_initializer='Orthogonal', name='gru6'), merge_mode='concat')(self.gru_5)

        # transforms RNN output to character activations:
        self.dense1 = Dense(self.output_size, kernel_initializer='he_normal', name='dense1')(self.gru_6)

        self.y_pred = Activation('softmax', name='softmax')(self.dense1)

        self.labels = Input(name='the_labels', shape=[self.absolute_max_string_len], dtype='float32')
        self.input_length = Input(name='input_length', shape=[1], dtype='int64')
        self.label_length = Input(name='label_length', shape=[1], dtype='int64')

        self.loss_out = CTC('ctc', [self.y_pred, self.labels, self.input_length, self.label_length])

        self.model = Model(inputs=[self.input_data, self.input_data2, self.labels, self.input_length, self.label_length], outputs=self.loss_out)

    def summary(self):
        # Model(inputs= self.input_data2, outputs=self.y_pred).summary()
        Model(inputs=[self.input_data, self.input_data2], outputs=self.y_pred).summary()

    def predict(self, input_batch):
        # print('aaaaaaaaaaaaaaaaaa', input_batch.shape)
        return self.test_function(input_batch)[0]  # the first 0 indicates test

    @property
    def test_function(self):
        # captures output of softmax so we can decode the output during visualization
        # print('bbbbbbbbbbbbbbb', [self.input_data, K.symbolic_learning_phase()])
        # return K.function(self.input_data2, [self.y_pred])
        return K.function([self.input_data, self.input_data2], [self.y_pred])
lipnet = LipNet()
lipnet.summary()