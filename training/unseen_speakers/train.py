from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
from lipnet.lipreading.generators import BasicGenerator
from lipnet.lipreading.callbacks import Statistics, Visualize
from lipnet.lipreading.curriculums import Curriculum
from lipnet.core.decoders import Decoder
from lipnet.lipreading.helpers import labels_to_text
from lipnet.utils.spell import Spell
from lipnet.model4 import LipNet
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os

np.random.seed(55)

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join('datasets')
OUTPUT_DIR   = os.path.join('results')
LOG_DIR      = os.path.join('logs')

PREDICT_GREEDY      = False
PREDICT_BEAM_WIDTH  = 200
PREDICT_DICTIONARY  = os.path.join(CURRENT_PATH,'..','..','common','dictionaries','grid.txt')

def curriculum_rules(epoch):
    return { 'sentence_length': -1, 'flip_probability': 0.5, 'jitter_probability': 0.05 }


def train(run_name, start_epoch, stop_epoch, img_c, img_w, img_h, frames_n, absolute_max_string_len, minibatch_size):
    curriculum = Curriculum(curriculum_rules)
    lip_gen = BasicGenerator(dataset_path=DATASET_DIR,
                                minibatch_size=minibatch_size,
                                img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                                absolute_max_string_len=absolute_max_string_len,
                                curriculum=curriculum, start_epoch=start_epoch).build()

    lipnet = LipNet(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                            absolute_max_string_len=absolute_max_string_len, output_size=lip_gen.get_output_size())
    lipnet.summary()
    lipnet.model.summary()

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)

    # load weight if necessary
    if start_epoch > 0:
        weight_file = os.path.join(OUTPUT_DIR, os.path.join(run_name, 'weights%02d.h5' % (start_epoch)))
        lipnet.model.load_weights(weight_file)

    spell = Spell(path=PREDICT_DICTIONARY)
    decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
                      postprocessors=[labels_to_text, spell.sentence])
    print('here')

    # define callbacks
    statistics  = Statistics(lipnet, lip_gen.next_val(), decoder, 256, output_dir=os.path.join(OUTPUT_DIR, run_name))
    visualize   = Visualize(os.path.join(OUTPUT_DIR, run_name), lipnet, lip_gen.next_val(), decoder, num_display_sentences=minibatch_size)
    tensorboard = TensorBoard(log_dir=os.path.join(LOG_DIR, run_name))
    csv_logger  = CSVLogger(os.path.join(LOG_DIR, "{}-{}.csv".format('training',run_name)), separator=',', append=True)
    checkpoint  = ModelCheckpoint(os.path.join(OUTPUT_DIR, run_name, "weights{epoch:02d}.h5"), monitor='val_loss', save_weights_only=True, mode='auto', period=1)

    history = lipnet.model.fit_generator(lip_gen.next_train(),
                        steps_per_epoch=lip_gen.default_training_steps, epochs=stop_epoch,
                        validation_data=lip_gen.next_val(), validation_steps=lip_gen.default_validation_steps,
                        callbacks=[checkpoint, statistics, visualize, lip_gen, tensorboard, csv_logger], 
                        initial_epoch=start_epoch, 
                        verbose=1,
                        max_queue_size=5,
                        workers=2,
            # use_multiprocessing=True
                        )
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    run_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # run_name = '2020-11-06-17-50-22'
    # run_name = '2020-11-08-00-32-55'
    # run_name = '2020-11-15-16-37-16'
    # run_name = '2020-12-21-19-36-11'
    # train(run_name, 33, 50, 3, 100, 50, 75, 32, 10)# もともと50


    # run_name = '2020-12-26-03-10-14'#stft0
    # run_name = '2021-01-02-02-45-52'#stft0 maxp
    # run_name = '2021-04-29-00-29-24'#mxpなし　マルチモーダル音声認識　結合層bgru128x2
    run_name = '2021-05-08-03-10-37'
    train(run_name, 1, 35, 3, 100, 50, 75, 32, 3)# もともと50