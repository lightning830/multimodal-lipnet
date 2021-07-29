from tensorflow.keras.layers import Conv3D, ZeroPadding3D, Conv2D, ZeroPadding2D, concatenate
from tensorflow.keras.layers import MaxPool3D
from tensorflow.keras.layers import Dense, Activation, SpatialDropout3D, Flatten
from tensorflow.keras.layers import Bidirectional, TimeDistributed
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from lipnet.core.layers import CTC
from tensorflow.keras import backend as K


class LipNet(object):
    def __init__(self, stft_w=48, stft_h=480, stft_c=3, stft_n=75, absolute_max_string_len=32, output_size=28):
        self.stft_w = stft_w
        self.stft_h = stft_h
        self.stft_c = stft_c
        self.stft_n = stft_n
        self.absolute_max_string_len = absolute_max_string_len
        self.output_size = output_size
        self.build()

    def build(self):
        input_shape = (self.stft_n, self.stft_h, self.stft_w, self.stft_c)

        self.input_data = Input(name='the_input', shape=input_shape, dtype='float32')

        self.zero4 = ZeroPadding3D(padding=(1, 2, 2), name='zero')(self.input_data)
        self.conv4 = Conv3D(32, (3, 5, 5), strides=(1, 2, 2), kernel_initializer='he_normal', name='conv')(self.zero4)
        self.batc4 = BatchNormalization(name='batc')(self.conv4)
        self.actv4 = Activation('relu', name='actv')(self.batc4)
        self.drop4 = SpatialDropout3D(0.5)(self.actv4)
        # self.maxp4 = MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max4')(self.drop4)

        self.zero5 = ZeroPadding3D(padding=(1, 2, 2), name='zero1')(self.drop4)
        self.conv5 = Conv3D(32, (3, 4, 4), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv1')(self.zero5)
        self.batc5 = BatchNormalization(name='batc1')(self.conv5)
        self.actv5 = Activation('relu', name='actv1')(self.batc5)
        self.drop5 = SpatialDropout3D(0.5)(self.actv5)
        # self.maxp5 = MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max5')(self.drop5)

        self.zero6 = ZeroPadding3D(padding=(1, 1, 1), name='zero2')(self.drop5)
        self.conv6 = Conv3D(32, (3, 4, 4), strides=(1, 2, 2), kernel_initializer='he_normal', name='conv2')(self.zero6)
        self.batc6 = BatchNormalization(name='batc2')(self.conv6)
        self.actv6 = Activation('relu', name='actv2')(self.batc6)
        self.drop6 = SpatialDropout3D(0.5)(self.actv6)
        # self.maxp6 = MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max6')(self.drop6)

        self.zero7 = ZeroPadding3D(padding=(1, 1, 1), name='zero3')(self.drop6)
        self.conv7 = Conv3D(64, (3, 2, 2), strides=(1, 2, 1), kernel_initializer='he_normal', name='conv3')(self.zero7)
        self.batc7 = BatchNormalization(name='batc3')(self.conv7)
        self.actv7 = Activation('relu', name='actv3')(self.batc7)
        self.drop7 = SpatialDropout3D(0.5)(self.actv7)
        # self.maxp7 = MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max7')(self.drop7)

        self.zero8 = ZeroPadding3D(padding=(1, 1, 1), name='zero4')(self.drop7)
        self.conv8 = Conv3D(64, (3, 2, 2), strides=(1, 2, 1), name='conv4')(self.zero8)
        self.batc8 = BatchNormalization(name='batc4')(self.conv8)
        self.actv8 = Activation('relu', name='actv4')(self.batc8)
        self.drop8 = SpatialDropout3D(0.5)(self.actv8)
        # self.maxp8 = MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max8')(self.drop8)
        print(self.drop8)
        self.resh2 = TimeDistributed(Flatten())(self.drop8)
        print(self.resh2)

        self.gru_3 = Bidirectional(GRU(32,  return_sequences=True, kernel_initializer='Orthogonal', name='gru1'), merge_mode='concat')(self.resh2)
        self.gru_4 = Bidirectional(GRU(32,  return_sequences=True, kernel_initializer='Orthogonal', name='gru2'), merge_mode='concat')(self.gru_3)

        # transforms RNN output to character activations:
        self.dense1 = Dense(self.output_size, kernel_initializer='he_normal', name='dense1')(self.gru_4)

        self.y_pred = Activation('softmax', name='softmax')(self.dense1)

        self.labels = Input(name='the_labels', shape=[self.absolute_max_string_len], dtype='float32')
        self.input_length = Input(name='input_length', shape=[1], dtype='int64')
        self.label_length = Input(name='label_length', shape=[1], dtype='int64')

        self.loss_out = CTC('ctc', [self.y_pred, self.labels, self.input_length, self.label_length])

        self.model = Model(inputs=[self.input_data, self.labels, self.input_length, self.label_length], outputs=self.loss_out)

    def summary(self):
        Model(inputs=self.input_data, outputs=self.y_pred).summary()

    def predict(self, input_batch):
        # print('aaaaaaaaaaaaaaaaaa', input_batch.shape)
        return self.test_function(input_batch)[0]  # the first 0 indicates test

    @property
    def test_function(self):
        # captures output of softmax so we can decode the output during visualization
        # print('bbbbbbbbbbbbbbb', [self.input_data, K.symbolic_learning_phase()])
        return K.function([self.input_data], [self.y_pred])

lipnet = LipNet()
lipnet.summary()