'''
This is the entry point for the genetic algorithm.
The command line argument -nodisplay prevents
the weights from being displayed.
'''
from audio import *
from ann import ANN
from random import shuffle,random,randint
from constants import *
INITIAL_SIZE = 10
TRAINING_ITERATIONS = 2000

pitcher = Pitchify(WINDOW[0], WINDOW[1], NOTES[0], NOTES[1], 6)

def split_phrase(data, num):
    size = num*2
    data = [data[i:i+size] for i in range(0, len(data), size)][1:-1]
    data = [pitcher.transform(dft(unpack(d))) for d in data]
    m = max(d.max() for d in data)
    for d in data:
        d /= m
        d = np.sqrt(d)
    return data

def my_exp(val):
    return exp(val)

class Gene:
    def __init__(self, lr, gs, rs, hl):
        self.learn_rate = lr
        self.gaus_scale = gs
        self.rand_scale = rs
        self.hidden_layers = hl
    def get_parameters(self):
        return my_exp(-self.learn_rate),my_exp(-self.gaus_scale),\
                my_exp(-self.rand_scale),[int(my_exp(hl)) for hl in self.hidden_layers]
    def get_net(self):
        lr,gs,rs,hl = self.get_parameters()
        return ANN([NOTES[1]-NOTES[0]]+hl+[len(names)],\
            activation='tanh+ax', gaus_scale=gs,\
            rand_scale=rs,learning_rate=lr)

def random_layer_count():
    return randint(1, 3)

def random_scale():
    return -log(1.-.9*random())

def random_layer():
    return [2*random_scale()+1 for _ in range(random_layer_count())]

def random_gene():
    return Gene(random_scale()*.5, random_scale()*.5, random_scale()*.5, random_layer())

def mutate():
    probs = ((-1,1),(0,10),(1,1))
    tot = sum(w for v,w in probs)
    val = tot*random()
    for v,w in probs:
        val -= w
        if val <= 0:
            return v*.32

def reproduce(g1, g2):
    lr = g1.learn_rate + g2.learn_rate
    lr /= 2
    gs = g1.gaus_scale + g2.gaus_scale
    gs /= 2
    rs = g1.rand_scale + g2.rand_scale
    rs /= 2
    np = len(g1.hidden_layers)
    split1 = 1 if np <= 1 else randint(1, np)
    np = len(g2.hidden_layers)
    split2 = 1 if np <= 1 else randint(1, np)
    if randint(0, 1):
        layers = g1.hidden_layers[:split1] + g2.hidden_layers[split2-1:]
    else:
        layers = g2.hidden_layers[:split2] + g1.hidden_layers[split1-1:]
    return Gene(lr, gs, rs, layers)

def copy(g):
    return Gene(g.learn_rate, g.gaus_scale, g.rand_scale, list(g.hidden_layers))

def mutate_gene(g):
    lr = g.learn_rate + mutate()
    gs = g.gaus_scale + mutate()
    rs = g.rand_scale + mutate()
    hl = [l + mutate() for l in g.hidden_layers]
    return Gene(lr, gs, rs, hl)

def repopulate(best, second):
    REPRODUCTIONS = 2
    BEST_MUTATIONS = 3
    SECOND_MUTATIONS = 2
    NEW = 1
    genes = [best, second]
    genes += [reproduce(best, second) for _ in range(REPRODUCTIONS)]
    genes += [mutate_gene(best) for _ in range(BEST_MUTATIONS)]
    genes += [mutate_gene(second) for _ in range(SECOND_MUTATIONS)]
    genes += [random_gene() for _ in range(NEW)]
    return genes

def test(n, samples, testing, shortest):
    for i in range(TRAINING_ITERATIONS):
        these_samples = []
        for s in samples.values():
            shuffle(s)
            these_samples += s[:shortest]
        shuffle(these_samples)
        lrf = len(samples)*shortest
        sm = sum(n.train(i,o,1/lrf) for i,o in these_samples)/len(these_samples)
        if not NO_DISPLAY:
            for j in range(n.get_layer_count()):
                cv2.imshow('Layer %d'%j, cv2.resize(n.get_layer_image(j), LAYER_DIM, interpolation=cv2.INTER_NEAREST))
        cv2.waitKey(30)
        if sm < TERMINATION_VALUE:
            break
    for j in range(n.get_layer_count()):
        cv2.destroyWindow('Layer %d'%j)
    sm = 0
    results = []
    for cl,test in testing:
        right = np.zeros(len(NAMES), dtype=np.float32)
        nidx = NAMES.index(cl)
        for d,_ in test:
            v = n.forward(d)
            idx = max(enumerate(v), key=lambda i:i[1])[0]
            right[idx] += 1
        right /= len(test)
        results.append((cl,right[nidx],right))
    return results

def rank(genes, samples):
    training,testing = dict(),[]
    for cls,samps in samples.items():
        part = len(samps)//PARTS
        testing.append((cls,samps[:part]))
        training[cls] = samps[part:]
    shortest = min(len(arr) for arr in training.values())
    results = [(g,test(g.get_net(), samples, testing, shortest)) for g in genes]
    results.sort(key=lambda v:-min(d[1] for d in v[1]))
    return results

if __name__ == '__main__':
    import cv2
    names = NAMES
    samples = dict()
    individuals = dict()
    for i,name in enumerate(names):
        samples[name] = []
        data = open("samples/%s.raw"%name, 'rb').read()
        data = split_phrase(data, SIZE)
        out = np.zeros(len(names))
        out[:] = -1.
        out[i] = 1.
        tmp = []
        for d in data:
            if d.mean() > VOLUME_THRESHOLD:
                tmp.append(d)
                samples[name].append((d,out))
        individuals[name] = np.array(tmp)
    shortest = min(len(s) for s in samples.values())
    print("Sample count:", sum(len(s) for s in samples.values()))
    if not NO_DISPLAY:
        for name in names:
            cv2.imshow("Samples for %s"%name, individuals[name])
    genes = [random_gene() for _ in range(INITIAL_SIZE)]
    i = 0
    while True:
        i += 1
        print("Generation", i)
        ranks = rank(genes, samples)
        best,second = ranks[0],ranks[1]
        print('Best params',best[0].get_parameters())
        print('Best results %r'%best[1])
        print('Best fitness %r'%min(d[1] for d in best[1]))
        print('Second params',second[0].get_parameters())
        print('Second results %r'%second[1])
        print('Second fitness %r'%min(d[1] for d in second[1]))
        genes = repopulate(best[0], second[0])

