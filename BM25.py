import math
import jieba
import numpy as np

class BM25():
    def __init__(self, docs):
        self.D = len(docs)
        self.avgld = np.mean([len(doc) for doc in docs])
        self.docs = docs
        self.f = []
        self.df = {}
        self.idf = {}
        self.k1 = 1.5
        self.b = 0.75
        self.init()
    
    def init(self):
        for doc in self.docs:
            tmp = {}
            for word in doc:
                tmp[word] = tmp.get(word, 0) + 1
            self.f.append(tmp)
            for k in tmp.keys():
                self.df[k] = self.df.get(k, 0) + 1
        for k, v in self.df.items():
            self.idf[k] = math.log(self.D - v + 0.5) - math.log(v + 0.5)
    
    def sim(self, doc, index):
        score = 0
        for word in doc:
            if word not in self.f[index]:
                continue
            d = len(self.docs[index])
            score += (self.idf[word]*self.f[index][word]*(self.k1 + 1)) / (self.f[index][word] + self.k1 * (1 - self.b + self.b * d / self.avgld))
        
        return score
    
    def simall(self, doc):
        scores = []
        for index in range(self.D):
            score = self.sim(doc, index)
            scores.append(score)
        return scores

if __name__ == '__main__':
    with open('test.txt', 'r', encoding='utf8') as f:
        sents = f.read().split("\n")
    with open("stopwords/sichuan.txt", "r", encoding='utf8') as f:
        stopwords = f.read().split("\n")
    
    doc = []
    for sent in sents:
        words = list(jieba.cut(sent))
        words = [word for word in words if word not in stopwords]
        doc.append(words)
    print(doc)
    s = BM25(doc)
    print(s.f)
    print(s.idf)
    print(s.simall(['自然语言', '计算机科学', '领域', '人工智能', '领域']))

