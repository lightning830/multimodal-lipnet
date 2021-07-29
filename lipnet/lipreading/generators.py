from lipnet.lipreading.helpers import text_to_labels
from lipnet.lipreading.videos import Video, Audio
from lipnet.lipreading.aligns import Align
from lipnet.helpers.threadsafe import threadsafe_generator
from lipnet.helpers.list import get_list_safe
from keras import backend as K
import numpy as np
import keras
import pickle
import os
import glob
import multiprocessing


# datasets/[train|val]/<sid>/<id>/<image>.png
# or datasets/[train|val]/<sid>/<id>.mpg
# datasets/align/<id>.align
class BasicGenerator(keras.callbacks.Callback):
    def __init__(self, dataset_path, minibatch_size, img_c, img_w, img_h, frames_n, absolute_max_string_len=30, **kwargs):
        self.dataset_path   = dataset_path
        self.minibatch_size = minibatch_size
        self.blank_label    = self.get_output_size() - 1
        self.img_c          = img_c
        self.img_w          = img_w
        self.img_h          = img_h
        self.frames_n       = frames_n
        self.absolute_max_string_len = absolute_max_string_len
        self.cur_train_index = multiprocessing.Value('i', 0)#int: i 初期値0
        self.cur_val_index   = multiprocessing.Value('i', 0)
        self.curriculum      = kwargs.get('curriculum', None)
        self.random_seed     = kwargs.get('random_seed', 13)
        self.vtype               = kwargs.get('vtype', 'mouth')
        self.face_predictor_path = kwargs.get('face_predictor_path', None)
        self.steps_per_epoch     = kwargs.get('steps_per_epoch', None)
        self.validation_steps    = kwargs.get('validation_steps', None)
        # Process epoch is used by non-training generator (e.g: validation)
        # because each epoch uses different validation data enqueuer
        # Will be updated on epoch_begin
        self.process_epoch       = -1
        # Maintain separate process train epoch because fit_generator only use
        # one enqueuer for the entire training, training enqueuer can contain
        # max_q_size batch data ahead than the current batch data which might be
        # in the epoch different with current actual epoch
        # Will be updated on next_train()
        self.shared_train_epoch  = multiprocessing.Value('i', -1)
        self.process_train_epoch = -1
        self.process_train_index = -1
        self.process_val_index   = -1

    def build(self, **kwargs):
        self.train_path     = os.path.join(self.dataset_path, 'train')
        self.val_path       = os.path.join(self.dataset_path, 'val')
        self.align_path     = os.path.join(self.dataset_path, 'align')
        self.build_dataset()
        # Set steps to dataset size if not set
        self.steps_per_epoch  = self.default_training_steps if self.steps_per_epoch is None else self.steps_per_epoch
        self.validation_steps = self.default_validation_steps if self.validation_steps is None else self.validation_steps
        return self

    @property
    def training_size(self):
        return len(self.train_list)

    @property
    def default_training_steps(self):
        return self.training_size // self.minibatch_size

    @property
    def validation_size(self):
        return len(self.val_list)

    @property
    def default_validation_steps(self):
        return self.validation_size // self.minibatch_size

    def get_output_size(self):
        return 28

    def get_cache_path(self):
        return self.dataset_path.rstrip(os.sep) + '.cache' #os.sep = "\\", rstrip(s) = 右側の文字sを消す、引数なしなら空白削除

    def enumerate_videos(self, path):
        video_list = []
        # print('video_path', video_path)# "datasets/train/s1/lbwo9p datasets/..."
        for video_path in glob.glob(path):
            try:
                if os.path.isfile(video_path):
                    video = Video(self.vtype, self.face_predictor_path).from_video(video_path)
                else:
                    video = Video(self.vtype, self.face_predictor_path).from_frames(video_path)
            except AttributeError as err:
                raise err
            except:
                print("Error loading video: ", video_path)
                continue
            if K.image_data_format() == 'channels_first' and video.data.shape != (self.img_c,self.frames_n,self.img_w,self.img_h):
                print("Video ",video_path," has incorrect shape ",str(video.data.shape),", must be ",str((self.img_c,self.frames_n,self.img_w,self.img_h)),"")
                continue
            if K.image_data_format() != 'channels_first' and video.data.shape != (self.frames_n,self.img_w,self.img_h,self.img_c):
                print("Video ",video_path," has incorrect shape ",str(video.data.shape),", must be ",str((self.frames_n,self.img_w,self.img_h,self.img_c))+"")
                continue
            stft = Video().from_frames_stft(video_path)
            if stft.data_stft.shape != (75, 480, 48, 3):
                print("Video ",video_path," has incorrect stft shape ",str(stft.data_stft.shape))
                continue
            video_list.append(video_path)
            # print(video_list)
        # for v in video_list:
        #     v = v.replace('\\', '/')
        # video_list
        return video_list

    def enumerate_align_hash(self, video_list):
        align_hash = {}
        for video_path in video_list:
            video_id = os.path.splitext(video_path)[0].split(os.sep)[-1]
            # print('video_id = ', video_id)
            align_path = os.path.join(self.align_path, video_id)+".align"
            # print('align_path = ', align_path)
            align_hash[video_id] = Align(self.absolute_max_string_len, text_to_labels).from_file(align_path)
        return align_hash

    def build_dataset(self):
        print(self.get_cache_path())
        print(os.path.isfile(self.get_cache_path()))
        if os.path.isfile(self.get_cache_path()):
            print("\nLoading dataset list from cache...")
            with open (self.get_cache_path(), 'rb') as fp:
                self.train_list, self.val_list, self.align_hash = pickle.load(fp)
                # print('trainpath', os.path.join(self.train_path, '*', '*'))
        else:
            print("\nEnumerating dataset list from disk...")
            self.train_list = self.enumerate_videos(os.path.join(self.train_path, '*', '*'))
            # print('train_list', self.train_list)
            # for i in self.train_list:
            #     i = i.replace('\\', '/')
            # print('train_list...', self.train_list)
            self.val_list   = self.enumerate_videos(os.path.join(self.val_path, '*', '*'))
            # print(self.train_list + self.val_list) #['datasets\\train\\test\\bbaf2n.mpg', 'datasets\\train\\test\\bbaf3s.mpg', 'datasets\\val\\test\\sbwo1s.mpg', 'datasets\\val\\test\\sbwo2p.mpg']
            self.align_hash = self.enumerate_align_hash(self.train_list + self.val_list)
            print('align_hash = ', self.align_hash)
            # {'bbaf2n': <lipnet.lipreading.aligns.Align object at 0x000002CE53B1BE10>..}
            with open(self.get_cache_path(), 'wb') as fp:
                pickle.dump((self.train_list, self.val_list, self.align_hash), fp)

        print("Found {} videos for training.".format(self.training_size))
        print("Found {} videos for validation.".format(self.validation_size))
        print("")

        np.random.shuffle(self.train_list)

    def get_align(self, _id):
        return self.align_hash[_id]

    def get_batch(self, index, size, train):
        if train:
            video_list = self.train_list
        else:
            video_list = self.val_list
        # print(video_list, index, size) 0, 50
        X_data_path = get_list_safe(video_list, index, size)#size分もってくる
        X_data = []
        X_data2 = []
        Y_data = []
        label_length = []
        input_length = []
        source_str = []
        # print('X_data_path', X_data_path)
        for path in X_data_path:
            video = Video().from_frames(path)
            stft = Audio().from_frames(path)
            # print(path.split(os.sep)[-1]) 'bbaf3s.mpg'
            # print('PATH,', path)
            align = self.get_align(os.path.splitext(path)[0].split(os.sep)[-1])
            video_unpadded_length = video.length
            # if self.curriculum is not None:
            #     video, align, video_unpadded_length = self.curriculum.apply(video, align)
            X_data.append(video.data)
            X_data2.append(stft.data)
            Y_data.append(align.padded_label)#[3, 2, 5, 26, ,,-1,-1,-1,] length=32
            label_length.append(align.label_length) # CHANGED [A] -> A, CHECK!
            # input_length.append([video_unpadded_length - 2]) # 2 first frame discarded
            input_length.append(video.length) # Just use the video padded length to avoid CTC No path found error (v_len < a_len)
            source_str.append(align.sentence) # CHANGED [A] -> A, CHECK!

        source_str = np.array(source_str)
        label_length = np.array(label_length)
        input_length = np.array(input_length)
        Y_data = np.array(Y_data)
        X_data = np.array(X_data).astype(np.float32) / 255 # Normalize image data to [0,1], TODO: mean normalization over training data
        X_data2 = np.array(X_data2).astype(np.float32) / 255
        # print('Y_data = ', Y_data)

        inputs = {'the_input': X_data, 
                    'the_input2': X_data2,
                  'the_labels': Y_data,
                  'input_length': input_length,
                  'label_length': label_length,
                
                  'source_str': source_str  # used for visualization only
                  }
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function

        return (inputs, outputs)

    dataset_path1 = os.path.join('pickle_data')
    @threadsafe_generator
    def next_train(self):
        #pickleデータ使用時、ジェネレータがうまくいってないので不使用なう
        """
        for p in glob.glob(os.path.join('pickle_data', 'train', '*')):
            with open (p, 'rb') as fp:
                z = pickle.load(fp)
            X_data = [r[0] for r in z]
            X_data2 = [r[1] for r in z]
            Y_data = [r[2] for r in z]
            input_length = [r[3] for r in z]
            label_length = [r[4] for r in z]
            source_str = [r[5] for r in z]

            X_data = np.array(X_data).astype(np.float32) / 255 # Normalize image data to [0,1], TODO: mean normalization over training data
            X_data2 = np.array(X_data2).astype(np.float32) / 255
            Y_data = np.array(Y_data)
            input_length = np.array(input_length)
            label_length = np.array(label_length)
            source_str = np.array(source_str)

            inputs = {'the_input': X_data, 
                    'the_input2': X_data2,
                    'the_labels': Y_data,
                    'input_length': input_length,
                    'label_length': label_length,
                
                    'source_str': source_str  # used for visualization only
                    }
            outputs = {'ctc': np.zeros([5])}  # dummy data for dummy loss function

            yield (inputs, outputs)

        """
        r = np.random.RandomState(self.random_seed)
        while 1:
            # print ("SI: {}, SE: {}".format(self.cur_train_index.value, self.shared_train_epoch.value))
            with self.cur_train_index.get_lock(), self.shared_train_epoch.get_lock():
                cur_train_index = self.cur_train_index.value
                self.cur_train_index.value += self.minibatch_size
                # Shared epoch increment on start or index >= training in epoch
                if cur_train_index >= self.steps_per_epoch * self.minibatch_size:
                    cur_train_index = 0
                    self.shared_train_epoch.value += 1
                    self.cur_train_index.value = self.minibatch_size
                if self.shared_train_epoch.value < 0:
                    self.shared_train_epoch.value += 1
                # Shared index overflow
                if self.cur_train_index.value >= self.training_size:
                    self.cur_train_index.value = self.cur_train_index.value % self.minibatch_size
                # Calculate differences between process and shared epoch
                epoch_differences = self.shared_train_epoch.value - self.process_train_epoch
            if epoch_differences > 0:
                self.process_train_epoch += epoch_differences
                # for i in range(epoch_differences):
                #     r.shuffle(self.train_list) # Catch up
                # print ("GENERATOR EPOCH {}".format(self.process_train_epoch))
                # print (self.train_list[0])
            # print ("PI: {}, SI: {}, SE: {}".format(cur_train_index, self.cur_train_index.value, self.shared_train_epoch.value))
            if self.curriculum is not None and self.curriculum.epoch != self.process_train_epoch:
                self.update_curriculum(self.process_train_epoch, train=True)
            # print ("Train [{},{}] {}:{}".format(self.process_train_epoch, epoch_differences, cur_train_index,cur_train_index+self.minibatch_size))
            ret = self.get_batch(cur_train_index, self.minibatch_size, train=True)
            # if epoch_differences > 0:
            #     print ("GENERATOR EPOCH {} - {}:{}".format(self.process_train_epoch, cur_train_index, cur_train_index + self.minibatch_size))
            #     print (ret[0]['source_str'])
            #     print ("-------------------")
            yield ret
        

    @threadsafe_generator
    def next_val(self):
        """
        for p in glob.glob(os.path.join('pickle_data', 'val', '*')):
            with open (p, 'rb') as fp:
                z = pickle.load(fp)
            X_data = [r[0] for r in z]
            X_data2 = [r[1] for r in z]
            Y_data = [r[2] for r in z]
            input_length = [r[3] for r in z]
            label_length = [r[4] for r in z]
            source_str = [r[5] for r in z]

            X_data = np.array(X_data).astype(np.float32) / 255 # Normalize image data to [0,1], TODO: mean normalization over training data
            X_data2 = np.array(X_data2).astype(np.float32) / 255
            Y_data = np.array(Y_data)
            input_length = np.array(input_length)
            label_length = np.array(label_length)
            source_str = np.array(source_str)

            inputs = {'the_input': X_data, 
                    'the_input2': X_data2,
                    'the_labels': Y_data,
                    'input_length': input_length,
                    'label_length': label_length,
                
                    'source_str': source_str  # used for visualization only
                    }
            outputs = {'ctc': np.zeros([5])}  # dummy data for dummy loss function

            yield (inputs, outputs)
        """
        while 1:
            with self.cur_val_index.get_lock():
                cur_val_index = self.cur_val_index.value
                self.cur_val_index.value += self.minibatch_size
                if self.cur_val_index.value >= self.validation_size:
                    self.cur_val_index.value = self.cur_val_index.value % self.minibatch_size
            if self.curriculum is not None and self.curriculum.epoch != self.process_epoch:
                self.update_curriculum(self.process_epoch, train=False)
            print ("Val [{}] {}:{}".format(self.process_epoch, cur_val_index,cur_val_index+self.minibatch_size))
            ret = self.get_batch(cur_val_index, self.minibatch_size, train=False)
            yield ret
        

    def on_train_begin(self, logs={}):
        with self.cur_train_index.get_lock():
            self.cur_train_index.value = 0
        with self.cur_val_index.get_lock():
            self.cur_val_index.value = 0

    def on_epoch_begin(self, epoch, logs={}):
        self.process_epoch = epoch

    def update_curriculum(self, epoch, train=True):
        self.curriculum.update(epoch, train=train)
        print("Epoch {}: {}".format(epoch, self.curriculum))


# datasets/video/<sid>/<id>/<image>.png
# or datasets/[train|val]/<sid>/<id>.mpg
# datasets/align/<id>.align
class RandomSplitGenerator(BasicGenerator):
    def build(self, **kwargs):
        self.video_path = os.path.join(self.dataset_path, 'video')
        self.align_path = os.path.join(self.dataset_path, 'align')
        self.val_split = kwargs.get('val_split', 0.2)
        self.build_dataset()
        # Set steps to dataset size if not set
        self.steps_per_epoch  = self.default_training_steps if self.steps_per_epoch is None else self.steps_per_epoch
        self.validation_steps = self.default_validation_steps if self.validation_steps is None else self.validation_steps
        return self
        
    def build_dataset(self):
        if os.path.isfile(self.get_cache_path()):
            print("\nLoading dataset list from cache...")
            with open (self.get_cache_path(), 'rb') as fp:
                self.train_list, self.val_list, self.align_hash = pickle.load(fp)
        else:
            print("\nEnumerating dataset list from disk...")
            video_list = self.enumerate_videos(os.path.join(self.video_path, '*', '*'))
            np.random.shuffle(video_list) # Random the video list before splitting
            if(self.val_split > 1): # If val_split is not a probability
                training_size = len(video_list) - self.val_split
            else: # If val_split is a probability
                training_size = len(video_list) - int(self.val_split * len(video_list))
            self.train_list = video_list[0:training_size]
            self.val_list   = video_list[training_size:]
            self.align_hash = self.enumerate_align_hash(self.train_list + self.val_list)
            with open(self.get_cache_path(), 'wb') as fp:
                pickle.dump((self.train_list, self.val_list, self.align_hash), fp)

        print("Found {} videos for training.".format(self.training_size))
        print("Found {} videos for validation.".format(self.validation_size))
        print("")




class BasicGenerator1(keras.callbacks.Callback):
    def __init__(self, dataset_path, minibatch_size, absolute_max_string_len=30, **kwargs):
        self.dataset_path   = dataset_path
        self.minibatch_size = minibatch_size
        self.blank_label    = self.get_output_size() - 1
        self.absolute_max_string_len = absolute_max_string_len
        self.cur_train_index = multiprocessing.Value('i', 0)#int: i 初期値0
        self.cur_val_index   = multiprocessing.Value('i', 0)
        self.curriculum      = kwargs.get('curriculum', None)
        self.random_seed     = kwargs.get('random_seed', 13)
        self.vtype               = kwargs.get('vtype', 'mouth')
        self.face_predictor_path = kwargs.get('face_predictor_path', None)
        self.steps_per_epoch     = kwargs.get('steps_per_epoch', None)
        self.validation_steps    = kwargs.get('validation_steps', None)
        # Process epoch is used by non-training generator (e.g: validation)
        # because each epoch uses different validation data enqueuer
        # Will be updated on epoch_begin
        self.process_epoch       = -1
        # Maintain separate process train epoch because fit_generator only use
        # one enqueuer for the entire training, training enqueuer can contain
        # max_q_size batch data ahead than the current batch data which might be
        # in the epoch different with current actual epoch
        # Will be updated on next_train()
        self.shared_train_epoch  = multiprocessing.Value('i', -1)
        self.process_train_epoch = -1
        self.process_train_index = -1
        self.process_val_index   = -1

    def build(self, **kwargs):
        self.train_path     = os.path.join(self.dataset_path, 'train')
        self.val_path       = os.path.join(self.dataset_path, 'val')
        self.align_path     = os.path.join(self.dataset_path, 'align')
        self.build_dataset()
        # Set steps to dataset size if not set
        self.steps_per_epoch  = self.default_training_steps if self.steps_per_epoch is None else self.steps_per_epoch
        self.validation_steps = self.default_validation_steps if self.validation_steps is None else self.validation_steps
        return self

    @property
    def training_size(self):
        return len(self.train_list)

    @property
    def default_training_steps(self):
        return self.training_size // self.minibatch_size

    @property
    def validation_size(self):
        return len(self.val_list)

    @property
    def default_validation_steps(self):
        return self.validation_size // self.minibatch_size

    def get_output_size(self):
        return 28

    def get_cache_path(self):
        return self.dataset_path.rstrip(os.sep) + '.cache'

    def enumerate_videos(self, path):
        #基本実行しないので気にしない
        video_list = []
        for video_path in glob.glob(path):
            try:
                if os.path.isfile(video_path):
                    video = Video(self.vtype, self.face_predictor_path).from_video(video_path)
                else:
                    video = Video(self.vtype, self.face_predictor_path).from_frames(video_path)
            except AttributeError as err:
                raise err
            except:
                print("Error loading video: ", video_path)
                continue
            stft = Video().from_frames_stft(video_path)
            if stft.data_stft.shape != (75, 480, 48, 3):
                print("Video ",video_path," has incorrect stft shape ",str(stft.data_stft.shape))
                continue
            # print('video_path', video_path)# "datasets/train/s1/aaaa.mpg"
            video_list.append(video_path)
            # print(video_list)
        # for v in video_list:
        #     v = v.replace('\\', '/')
        # video_list
        return video_list

    def enumerate_align_hash(self, video_list):
        align_hash = {}
        for video_path in video_list:
            video_id = os.path.splitext(video_path)[0].split(os.sep)[-1]
            # print('video_id = ', video_id)
            align_path = os.path.join(self.align_path, video_id)+".align"
            # print('align_path = ', align_path)
            align_hash[video_id] = Align(self.absolute_max_string_len, text_to_labels).from_file(align_path)
        return align_hash

    def build_dataset(self):
        print(self.get_cache_path())
        print(os.path.isfile(self.get_cache_path()))
        if os.path.isfile(self.get_cache_path()):
            print("\nLoading dataset list from cache...")
            with open (self.get_cache_path(), 'rb') as fp:
                self.train_list, self.val_list, self.align_hash = pickle.load(fp)
                # print('trainpath', os.path.join(self.train_path, '*', '*'))
        else:
            print("\nEnumerating dataset list from disk...")
            self.train_list = self.enumerate_videos(os.path.join(self.train_path, '*', '*'))
            # print('train_list', self.train_list)
            # for i in self.train_list:
            #     i = i.replace('\\', '/')
            # print('train_list...', self.train_list)
            self.val_list   = self.enumerate_videos(os.path.join(self.val_path, '*', '*'))
            # print(self.train_list + self.val_list) #['datasets\\train\\test\\bbaf2n.mpg', 'datasets\\train\\test\\bbaf3s.mpg', 'datasets\\val\\test\\sbwo1s.mpg', 'datasets\\val\\test\\sbwo2p.mpg']
            self.align_hash = self.enumerate_align_hash(self.train_list + self.val_list)
            print('align_hash = ', self.align_hash)
            # {'bbaf2n': <lipnet.lipreading.aligns.Align object at 0x000002CE53B1BE10>..}
            with open(self.get_cache_path(), 'wb') as fp:
                pickle.dump((self.train_list, self.val_list, self.align_hash), fp)

        print("Found {} videos for training.".format(self.training_size))
        print("Found {} videos for validation.".format(self.validation_size))
        print("")

        np.random.shuffle(self.train_list)

    def get_align(self, _id):
        return self.align_hash[_id]

    def get_batch(self, index, size, train):
        if train:
            video_list = self.train_list
        else:
            video_list = self.val_list
        # print(video_list, index, size) 0, 50
        X_data_path = get_list_safe(video_list, index, size)
        X_data = []
        Y_data = []
        label_length = []
        input_length = []
        source_str = []
        # print('X_data_path', X_data_path)
        for path in X_data_path:
            # video = Video().from_frames(path)
            stft = Video().from_frames_stft(path)
            # print(path.split(os.sep)[-1]) 'bbaf3s.mpg'
            # print('PATH,', path)
            align = self.get_align(os.path.splitext(path)[0].split(os.sep)[-1])
            # video_unpadded_length = video.length
            # if self.curriculum is not None:
            #     video, align, video_unpadded_length = self.curriculum.apply(video, align)
            X_data.append(stft.data_stft)
            Y_data.append(align.padded_label)#[3, 2, 5, 26, ,,-1,-1,-1,] length=32
            label_length.append(align.label_length) # CHANGED [A] -> A, CHECK!
            # input_length.append([video_unpadded_length - 2]) # 2 first frame discarded
            input_length.append(75) # Just use the video padded length to avoid CTC No path found error (v_len < a_len)
            source_str.append(align.sentence) # CHANGED [A] -> A, CHECK!

        source_str = np.array(source_str)
        label_length = np.array(label_length)
        input_length = np.array(input_length)
        Y_data = np.array(Y_data)
        X_data = np.array(X_data).astype(np.float32) / 255 # Normalize image data to [0,1], TODO: mean normalization over training data
        # print('Y_data = ', Y_data)

        inputs = {'the_input': X_data,
                  'the_labels': Y_data,
                  'input_length': input_length,
                  'label_length': label_length,
                
                  'source_str': source_str  # used for visualization only
                  }
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function

        return (inputs, outputs)

    @threadsafe_generator
    def next_train(self):
        r = np.random.RandomState(self.random_seed)
        while 1:
            # print ("SI: {}, SE: {}".format(self.cur_train_index.value, self.shared_train_epoch.value))
            with self.cur_train_index.get_lock(), self.shared_train_epoch.get_lock():
                cur_train_index = self.cur_train_index.value
                self.cur_train_index.value += self.minibatch_size
                # Shared epoch increment on start or index >= training in epoch
                if cur_train_index >= self.steps_per_epoch * self.minibatch_size:
                    cur_train_index = 0
                    self.shared_train_epoch.value += 1
                    self.cur_train_index.value = self.minibatch_size
                if self.shared_train_epoch.value < 0:
                    self.shared_train_epoch.value += 1
                # Shared index overflow
                if self.cur_train_index.value >= self.training_size:
                    self.cur_train_index.value = self.cur_train_index.value % self.minibatch_size
                # Calculate differences between process and shared epoch
                epoch_differences = self.shared_train_epoch.value - self.process_train_epoch
            if epoch_differences > 0:
                self.process_train_epoch += epoch_differences
                # for i in range(epoch_differences):
                #     r.shuffle(self.train_list) # Catch up
                # print ("GENERATOR EPOCH {}".format(self.process_train_epoch))
                # print (self.train_list[0])
            # print ("PI: {}, SI: {}, SE: {}".format(cur_train_index, self.cur_train_index.value, self.shared_train_epoch.value))
            if self.curriculum is not None and self.curriculum.epoch != self.process_train_epoch:
                self.update_curriculum(self.process_train_epoch, train=True)
            # print ("Train [{},{}] {}:{}".format(self.process_train_epoch, epoch_differences, cur_train_index,cur_train_index+self.minibatch_size))
            ret = self.get_batch(cur_train_index, self.minibatch_size, train=True)
            # if epoch_differences > 0:
            #     print ("GENERATOR EPOCH {} - {}:{}".format(self.process_train_epoch, cur_train_index, cur_train_index + self.minibatch_size))
            #     print (ret[0]['source_str'])
            #     print ("-------------------")
            yield ret

    @threadsafe_generator
    def next_val(self):
        while 1:
            with self.cur_val_index.get_lock():
                cur_val_index = self.cur_val_index.value
                self.cur_val_index.value += self.minibatch_size
                if self.cur_val_index.value >= self.validation_size:
                    self.cur_val_index.value = self.cur_val_index.value % self.minibatch_size
            if self.curriculum is not None and self.curriculum.epoch != self.process_epoch:
                self.update_curriculum(self.process_epoch, train=False)
            print ("Val [{}] {}:{}".format(self.process_epoch, cur_val_index,cur_val_index+self.minibatch_size))
            ret = self.get_batch(cur_val_index, self.minibatch_size, train=False)
            yield ret

    def on_train_begin(self, logs={}):
        with self.cur_train_index.get_lock():
            self.cur_train_index.value = 0
        with self.cur_val_index.get_lock():
            self.cur_val_index.value = 0

    def on_epoch_begin(self, epoch, logs={}):
        self.process_epoch = epoch

    def update_curriculum(self, epoch, train=True):
        self.curriculum.update(epoch, train=train)
        print("Epoch {}: {}".format(epoch, self.curriculum))



class BasicGenerator2(keras.callbacks.Callback):
    def __init__(self, dataset_path, minibatch_size, img_c, img_w, img_h, frames_n, absolute_max_string_len=30, **kwargs):
        self.dataset_path   = dataset_path
        self.minibatch_size = minibatch_size
        self.blank_label    = self.get_output_size() - 1
        self.img_c          = img_c
        self.img_w          = img_w
        self.img_h          = img_h
        self.frames_n       = frames_n
        self.absolute_max_string_len = absolute_max_string_len
        self.cur_train_index = multiprocessing.Value('i', 0)#int: i 初期値0
        self.cur_val_index   = multiprocessing.Value('i', 0)
        self.curriculum      = kwargs.get('curriculum', None)
        self.random_seed     = kwargs.get('random_seed', 13)
        self.vtype               = kwargs.get('vtype', 'mouth')
        self.face_predictor_path = kwargs.get('face_predictor_path', None)
        self.steps_per_epoch     = kwargs.get('steps_per_epoch', None)
        self.validation_steps    = kwargs.get('validation_steps', None)
        # Process epoch is used by non-training generator (e.g: validation)
        # because each epoch uses different validation data enqueuer
        # Will be updated on epoch_begin
        self.process_epoch       = -1
        # Maintain separate process train epoch because fit_generator only use
        # one enqueuer for the entire training, training enqueuer can contain
        # max_q_size batch data ahead than the current batch data which might be
        # in the epoch different with current actual epoch
        # Will be updated on next_train()
        self.shared_train_epoch  = multiprocessing.Value('i', -1)
        self.process_train_epoch = -1
        self.process_train_index = -1
        self.process_val_index   = -1

    def build(self, **kwargs):
        self.train_path     = os.path.join(self.dataset_path, 'train')
        self.val_path       = os.path.join(self.dataset_path, 'val')
        self.align_path     = os.path.join(self.dataset_path, 'align')
        self.build_dataset()
        # Set steps to dataset size if not set
        self.steps_per_epoch  = self.default_training_steps if self.steps_per_epoch is None else self.steps_per_epoch
        self.validation_steps = self.default_validation_steps if self.validation_steps is None else self.validation_steps
        return self

    @property
    def training_size(self):
        return len(self.train_list)

    @property
    def default_training_steps(self):
        return self.training_size // self.minibatch_size

    @property
    def validation_size(self):
        return len(self.val_list)

    @property
    def default_validation_steps(self):
        return self.validation_size // self.minibatch_size

    def get_output_size(self):
        return 28

    def get_cache_path(self):
        return self.dataset_path.rstrip(os.sep) + '.cache' #os.sep = "\\", rstrip(s) = 右側の文字sを消す、引数なしなら空白削除

    def enumerate_videos(self, path):
        video_list = []
        # print('video_path', video_path)# "datasets/train/s1/lbwo9p datasets/..."
        for video_path in glob.glob(path):
            try:
                if os.path.isfile(video_path):
                    video = Video(self.vtype, self.face_predictor_path).from_video(video_path)
                else:
                    video = Video(self.vtype, self.face_predictor_path).from_frames(video_path)
            except AttributeError as err:
                raise err
            except:
                print("Error loading video: ", video_path)
                continue

            if K.image_data_format() == 'channels_first' and video.data.shape != (self.img_c,self.frames_n,self.img_w,self.img_h):
                print("Video ",video_path," has incorrect shape ",str(video.data.shape),", must be ",str((self.img_c,self.frames_n,self.img_w,self.img_h)),"")
                continue
            if K.image_data_format() != 'channels_first' and video.data.shape != (self.frames_n,self.img_w,self.img_h,self.img_c):
                print("Video ",video_path," has incorrect shape ",str(video.data.shape),", must be ",str((self.frames_n,self.img_w,self.img_h,self.img_c))+"")
                continue

            # stft = Video().from_frames_stft(video_path)
            # if stft.data_stft.shape != (75, 480, 48, 3):
            #     print("Video ",video_path," has incorrect stft shape ",str(stft.data_stft.shape))
            #     continue

            wav = Audio().from_wav(video_path)
            if wav.shape != (131328,):
                print("Audio ",video_path," has incorrect wav shape ",str(wav.shape))
                continue

            video_list.append(video_path)
            # print(video_list)
        # for v in video_list:
        #     v = v.replace('\\', '/')
        # video_list
        return video_list

    def enumerate_align_hash(self, video_list):
        align_hash = {}
        for video_path in video_list:
            video_id = os.path.splitext(video_path)[0].split(os.sep)[-1]
            # print('video_id = ', video_id)
            align_path = os.path.join(self.align_path, video_id)+".align"
            # print('align_path = ', align_path)
            align_hash[video_id] = Align(self.absolute_max_string_len, text_to_labels).from_file(align_path)
        return align_hash

    def build_dataset(self):
        print(self.get_cache_path())
        print(os.path.isfile(self.get_cache_path()))
        if os.path.isfile(self.get_cache_path()):
            print("\nLoading dataset list from cache...")
            with open (self.get_cache_path(), 'rb') as fp:
                self.train_list, self.val_list, self.align_hash = pickle.load(fp)
                # print('trainpath', os.path.join(self.train_path, '*', '*'))
        else:
            print("\nEnumerating dataset list from disk...")
            self.train_list = self.enumerate_videos(os.path.join(self.train_path, '*', '*'))
            # print('train_list', self.train_list)
            # for i in self.train_list:
            #     i = i.replace('\\', '/')
            # print('train_list...', self.train_list)
            self.val_list   = self.enumerate_videos(os.path.join(self.val_path, '*', '*'))
            # print(self.train_list + self.val_list) #['datasets\\train\\test\\bbaf2n.mpg', 'datasets\\train\\test\\bbaf3s.mpg', 'datasets\\val\\test\\sbwo1s.mpg', 'datasets\\val\\test\\sbwo2p.mpg']
            self.align_hash = self.enumerate_align_hash(self.train_list + self.val_list)
            print('align_hash = ', self.align_hash)
            # {'bbaf2n': <lipnet.lipreading.aligns.Align object at 0x000002CE53B1BE10>..}
            with open(self.get_cache_path(), 'wb') as fp:
                pickle.dump((self.train_list, self.val_list, self.align_hash), fp)

        print("Found {} videos for training.".format(self.training_size))
        print("Found {} videos for validation.".format(self.validation_size))
        print("")

        np.random.shuffle(self.train_list)

    def get_align(self, _id):
        return self.align_hash[_id]

    def get_batch(self, index, size, train):
        if train:
            video_list = self.train_list
        else:
            video_list = self.val_list
        # print(video_list, index, size) 0, 50
        X_data_path = get_list_safe(video_list, index, size)#size分もってくる
        X_data = []
        X_data2 = []
        Y_data = []
        label_length = []
        input_length = []
        source_str = []
        # print('X_data_path', X_data_path)
        for path in X_data_path:
            video = Video().from_frames(path)
            wav = Audio().from_wav(path)
            # print(path.split(os.sep)[-1]) 'bbaf3s.mpg'
            # print('PATH,', path)
            align = self.get_align(os.path.splitext(path)[0].split(os.sep)[-1])
            video_unpadded_length = video.length
            # if self.curriculum is not None:
            #     video, align, video_unpadded_length = self.curriculum.apply(video, align)
            X_data.append(video.data)
            X_data2.append(wav)
            Y_data.append(align.padded_label)#[3, 2, 5, 26, ,,-1,-1,-1,] length=32
            label_length.append(align.label_length) # CHANGED [A] -> A, CHECK!
            # input_length.append([video_unpadded_length - 2]) # 2 first frame discarded
            input_length.append(video.length) # Just use the video padded length to avoid CTC No path found error (v_len < a_len)
            source_str.append(align.sentence) # CHANGED [A] -> A, CHECK!

        source_str = np.array(source_str)
        label_length = np.array(label_length)
        input_length = np.array(input_length)
        Y_data = np.array(Y_data)
        X_data = np.array(X_data).astype(np.float32) / 255 # Normalize image data to [0,1], TODO: mean normalization over training data
        X_data2 = np.array(X_data2).astype(np.float32) / 255
        # print('Y_data = ', Y_data)

        inputs = {'the_input': X_data, 
                    'the_input2': X_data2,
                  'the_labels': Y_data,
                  'input_length': input_length,
                  'label_length': label_length,
                
                  'source_str': source_str  # used for visualization only
                  }
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function

        return (inputs, outputs)

    dataset_path1 = os.path.join('pickle_data')
    @threadsafe_generator
    def next_train(self):
        #pickleデータ使用時、ジェネレータがうまくいってないので不使用なう
        """
        for p in glob.glob(os.path.join('pickle_data', 'train', '*')):
            with open (p, 'rb') as fp:
                z = pickle.load(fp)
            X_data = [r[0] for r in z]
            X_data2 = [r[1] for r in z]
            Y_data = [r[2] for r in z]
            input_length = [r[3] for r in z]
            label_length = [r[4] for r in z]
            source_str = [r[5] for r in z]

            X_data = np.array(X_data).astype(np.float32) / 255 # Normalize image data to [0,1], TODO: mean normalization over training data
            X_data2 = np.array(X_data2).astype(np.float32) / 255
            Y_data = np.array(Y_data)
            input_length = np.array(input_length)
            label_length = np.array(label_length)
            source_str = np.array(source_str)

            inputs = {'the_input': X_data, 
                    'the_input2': X_data2,
                    'the_labels': Y_data,
                    'input_length': input_length,
                    'label_length': label_length,
                
                    'source_str': source_str  # used for visualization only
                    }
            outputs = {'ctc': np.zeros([5])}  # dummy data for dummy loss function

            yield (inputs, outputs)

        """
        r = np.random.RandomState(self.random_seed)
        while 1:
            # print ("SI: {}, SE: {}".format(self.cur_train_index.value, self.shared_train_epoch.value))
            with self.cur_train_index.get_lock(), self.shared_train_epoch.get_lock():
                cur_train_index = self.cur_train_index.value
                self.cur_train_index.value += self.minibatch_size
                # Shared epoch increment on start or index >= training in epoch
                if cur_train_index >= self.steps_per_epoch * self.minibatch_size:
                    cur_train_index = 0
                    self.shared_train_epoch.value += 1
                    self.cur_train_index.value = self.minibatch_size
                if self.shared_train_epoch.value < 0:
                    self.shared_train_epoch.value += 1
                # Shared index overflow
                if self.cur_train_index.value >= self.training_size:
                    self.cur_train_index.value = self.cur_train_index.value % self.minibatch_size
                # Calculate differences between process and shared epoch
                epoch_differences = self.shared_train_epoch.value - self.process_train_epoch
            if epoch_differences > 0:
                self.process_train_epoch += epoch_differences
                # for i in range(epoch_differences):
                #     r.shuffle(self.train_list) # Catch up
                # print ("GENERATOR EPOCH {}".format(self.process_train_epoch))
                # print (self.train_list[0])
            # print ("PI: {}, SI: {}, SE: {}".format(cur_train_index, self.cur_train_index.value, self.shared_train_epoch.value))
            if self.curriculum is not None and self.curriculum.epoch != self.process_train_epoch:
                self.update_curriculum(self.process_train_epoch, train=True)
            # print ("Train [{},{}] {}:{}".format(self.process_train_epoch, epoch_differences, cur_train_index,cur_train_index+self.minibatch_size))
            ret = self.get_batch(cur_train_index, self.minibatch_size, train=True)
            # if epoch_differences > 0:
            #     print ("GENERATOR EPOCH {} - {}:{}".format(self.process_train_epoch, cur_train_index, cur_train_index + self.minibatch_size))
            #     print (ret[0]['source_str'])
            #     print ("-------------------")
            yield ret
        

    @threadsafe_generator
    def next_val(self):
        """
        for p in glob.glob(os.path.join('pickle_data', 'val', '*')):
            with open (p, 'rb') as fp:
                z = pickle.load(fp)
            X_data = [r[0] for r in z]
            X_data2 = [r[1] for r in z]
            Y_data = [r[2] for r in z]
            input_length = [r[3] for r in z]
            label_length = [r[4] for r in z]
            source_str = [r[5] for r in z]

            X_data = np.array(X_data).astype(np.float32) / 255 # Normalize image data to [0,1], TODO: mean normalization over training data
            X_data2 = np.array(X_data2).astype(np.float32) / 255
            Y_data = np.array(Y_data)
            input_length = np.array(input_length)
            label_length = np.array(label_length)
            source_str = np.array(source_str)

            inputs = {'the_input': X_data, 
                    'the_input2': X_data2,
                    'the_labels': Y_data,
                    'input_length': input_length,
                    'label_length': label_length,
                
                    'source_str': source_str  # used for visualization only
                    }
            outputs = {'ctc': np.zeros([5])}  # dummy data for dummy loss function

            yield (inputs, outputs)
        """
        while 1:
            with self.cur_val_index.get_lock():
                cur_val_index = self.cur_val_index.value
                self.cur_val_index.value += self.minibatch_size
                if self.cur_val_index.value >= self.validation_size:
                    self.cur_val_index.value = self.cur_val_index.value % self.minibatch_size
            if self.curriculum is not None and self.curriculum.epoch != self.process_epoch:
                self.update_curriculum(self.process_epoch, train=False)
            print ("Val [{}] {}:{}".format(self.process_epoch, cur_val_index,cur_val_index+self.minibatch_size))
            ret = self.get_batch(cur_val_index, self.minibatch_size, train=False)
            yield ret
        

    def on_train_begin(self, logs={}):
        with self.cur_train_index.get_lock():
            self.cur_train_index.value = 0
        with self.cur_val_index.get_lock():
            self.cur_val_index.value = 0

    def on_epoch_begin(self, epoch, logs={}):
        self.process_epoch = epoch

    def update_curriculum(self, epoch, train=True):
        self.curriculum.update(epoch, train=train)
        print("Epoch {}: {}".format(epoch, self.curriculum))