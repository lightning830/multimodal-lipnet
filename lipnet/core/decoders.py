from tensorflow.keras import backend as K
import numpy as np

def _decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
    """Decodes the output of a softmax.
    Can use either greedy search (also known as best path) 貪欲な探索
    or a constrained dictionary search. 制約付き辞書検索
    # Arguments 引数
        y_pred: tensor `(samples, time_steps, num_categories)`
            containing the prediction, or output of the softmax.
        input_length: tensor `(samples, )` containing the sequence length for
            each batch item in `y_pred`.
        greedy: perform much faster best-path search if `true`.
            This does not use a dictionary.
        beam_width: if `greedy` is `false`: a beam search decoder will be used
            with a beam of this width.
        top_paths: if `greedy` is `false`,
            how many of the most probable paths will be returned.
    # Returns
        Tuple:
            List: if `greedy` is `true`, returns a list of one element that
                contains the decoded sequence.
                If `false`, returns the `top_paths` most probable
                decoded sequences.
                Important: blank labels are returned as `-1`.
            Tensor `(top_paths, )` that contains
                the log probability of each decoded sequence.
    """
    decoded = K.ctc_decode(y_pred=y_pred, input_length=input_length,
                           greedy=greedy, beam_width=beam_width, top_paths=top_paths)
    # paths = [path.eval(session=K.get_session()) for path in decoded[0]]
    paths = decoded[0]
    # paths = K.get_value(paths[0])
    # print('paths = ', paths)
    # logprobs  = decoded[1].eval(session=K.get_session())
    logprobs = decoded[1]
    # print('logprobs = ', logprobs)

    return (paths, logprobs)

def decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1, **kwargs):
    language_model = kwargs.get('language_model', None)

    paths, logprobs = _decode(y_pred=y_pred, input_length=input_length,
                              greedy=greedy, beam_width=beam_width, top_paths=top_paths)
    if language_model is not None:
        # TODO: compute using language model
        raise NotImplementedError("Language model search is not implemented yet")
    else:
        # simply output highest probability sequence
        # paths has been sorted from the start
        result = K.get_value(paths[0])
        # print("result",result)
    return result

class Decoder(object):
    def __init__(self, greedy=True, beam_width=100, top_paths=1, **kwargs):
        self.greedy         = greedy
        self.beam_width     = beam_width
        self.top_paths      = top_paths
        self.language_model = kwargs.get('language_model', None)
        self.postprocessors = kwargs.get('postprocessors', [])

    def decode(self, y_pred, input_length):
        decoded = decode(y_pred, input_length, greedy=self.greedy, beam_width=self.beam_width,
                         top_paths=self.top_paths, language_model=self.language_model)
        preprocessed = []
        print('decoded', decoded)#tf.Tensor([[18  4 19 26 22  7  8 19  4 26 22  8 19  7 26  1 26 19 22 14 26 18 14 13]], shape=(1, 24), dtype=int64)
        # print('postprocessors', self.postprocessors)
        for output in decoded:
            out = output
            # print('out', output) 1回しか回ってない
            for postprocessor in self.postprocessors:#それぞれの関数に代入している
                out = postprocessor(out)#outを連続的に代入している
                # print('out=', out)
                # bin bred by c sie son -> bin red by c six soon
            # print('out', out) = bin red by c six soon
            preprocessed.append(out)
        # print('preprocessed', preprocessed)
        return preprocessed