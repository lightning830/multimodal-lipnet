import re
import string
from collections import Counter

# Source: https://github.com/commonsense/metanl/blob/master/metanl/token_utils.py
def untokenize(words):
    """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    """
    text = ' '.join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)#削除？
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
         "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()

# Source: https://stackoverflow.com/questions/367155/splitting-a-string-into-words-and-punctuation
def tokenize(text):#単語ごとに(記号は全分解)分解してリストで返す
    # print('tokenize=', re.findall(r"\w+|[^\w\s]", text, re.UNICODE))
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    #\w = 単語構成文字[a-zA-Z_0-9]
    #\s = 空白文字:[ \t\n\x0B\f\r]

# Source: http://norvig.com/spell-correct.html (with some modifications)
class Spell(object):
    def __init__(self, path):
        self.dictionary = Counter(list(string.punctuation) + self.words(open(path).read()))#string.punctuation=句読点たち
        print('dictionary=',self.dictionary)

    def words(self, text):
        return re.findall(r'\w+', text.lower())

    def P(self, word, N=None):
        "Probability of `word`."
        if N is None:
            N = sum(self.dictionary.values())
        # print('P=',self.dictionary[word] / N)
        # print('N=', N)
        return self.dictionary[word] / N

    def correction(self, word):#word = 'bin'
        "Most probable spelling correction for word."
        return max(self.candidates(word), key=self.P)#Pが最大のものを返す

    def candidates(self, word):
        "Generate possible spelling corrections for word."
        #左ほど優先で単語を出力,出力があったら終了
        # print('candidates=', (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word]))
        return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])

    def known(self, words):
        "The subset of `words` that appear in the dictionary of WORDS."
        # print('words=', words)
        # print('set=', set(w for w in words if w in self.dictionary))
        #wordsの単語がdictionaryにあるならそれを返す
        return set(w for w in words if w in self.dictionary)

    def edits1(self, word):
        "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]#1文字消したもののリスト
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]#順番入れ替え
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]#Rの一番左を削除してL+c+Rの群を作成
        inserts    = [L + c + R               for L, R in splits for c in letters]#分割したものの間に一文字入れる
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))#edits1したものをさらにedits1する

    # Correct words
    def corrections(self, words):
        # print('corrections=', [self.correction(word) for word in words])
        return [self.correction(word) for word in words]

    # Correct sentence
    def sentence(self, sentence):
        return untokenize(self.corrections(tokenize(sentence)))