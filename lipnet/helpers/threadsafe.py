import threading

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.

    イテレータ/ジェネレータを受け取り、与えられたイテレータ/ジェネレータの `next` メソッドの呼び出しをシリアライズすることでスレッドセーフにします。
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    ジェネレータ関数を受け取り、スレッドセーフにするデコレータ。
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g