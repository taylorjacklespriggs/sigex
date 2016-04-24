'''
This is the entry point for the ANN evaluation.
'''
from audio import *
from ann import ANN
from random import shuffle,random,randint
from constants import *

TRAINING_ITERATIONS = 200
PARTS = 10

pitcher = Pitchify(WINDOW[0], WINDOW[1], NOTES[0], NOTES[1], 6)

# net params
lr,gs,rs,hl = (1.2983004452161748, 1.019611668144157, 0.88720045809462, [81, 129])
LEARNING_RATE = lr
HIDDEN_LAYERS = hl
GAUS_SCALE = gs
RAND_SCALE = rs

def split_phrase(data, num):
    size = num*2
    data = [data[i:i+size] for i in range(0, len(data), size)][1:-1]
    data = [pitcher.transform(dft(unpack(d))) for d in data]
    m = max(d.max() for d in data)
    for d in data:
        d /= m
    return data

def test(n, samples, testing, shortest):
    percent = 0
    for i in range(TRAINING_ITERATIONS):
        new_p = 100*i//TRAINING_ITERATIONS
        if new_p != percent:
            percent = new_p
            print("Training %d%% complete"%percent)
        these_samples = []
        for s in samples.values():
            shuffle(s)
            these_samples += s[:shortest]
        shuffle(these_samples)
        lrf = len(samples)*shortest
        sm = sum(n.train(i,o,1/lrf) for i,o in these_samples)/len(these_samples)
        for j in range(n.get_layer_count()):
            layer = cv2.resize(n.get_layer_image(j), LAYER_DIM, interpolation=cv2.INTER_NEAREST)
            layer *= 255
            layer = layer.astype(np.uint8)
            cv2.imshow('Layer %d'%j, layer)
            cv2.imwrite('layer%d.jpg'%j, layer)
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

def evaluate(n, samples):
    training,testing = dict(),[]
    for cls,samps in samples.items():
        part = len(samps)//PARTS
        testing.append((cls,samps[:part]))
        training[cls] = samps[part:]
    shortest = min(len(arr) for arr in training.values())
    results = test(n, samples, testing, shortest)
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
    print("Sample count:", sum(len(s) for s in samples.values()))
    for name in names:
        cv2.imshow("Samples for %s"%name, individuals[name])
    n = ANN([NOTES[1]-NOTES[0]]+HIDDEN_LAYERS+[len(names)],\
            activation='tanh+ax', gaus_scale=GAUS_SCALE,\
            rand_scale=RAND_SCALE,learning_rate=LEARNING_RATE)
    print(evaluate(n, samples))
    input('Press enter to evaluate with a microphone')
    m = Microphone(SIZE)
    mx = 0
    mean = np.zeros(len(names))
    fade = 0.05
    length = 300
    idx = 0
    show = np.zeros((length, NOTES[1]-NOTES[0]))
    res_idx = 0
    res_length = length
    res_show = np.zeros((length, len(names)))
    while True:
        dft = pitcher.transform(m.get_fft())
        tmp = dft.max()
        mx = max(mx, tmp)
        dft /= mx
        show[idx] = dft
        idx = (idx+1)%length
        show[idx] = 1
        mean *= 1.-fade
        if dft.mean() > VOLUME_THRESHOLD:
            result = n.forward(dft)
            result += 1
            result /= 2
            res_show[res_idx] = result
            res_idx = (res_idx+1)%res_length
            res_show[res_idx] = 1
            mean += result
            val = max(enumerate(mean), key=lambda v:v[1])[0]
            print(names[val])
            print(mean)
            cv2.imshow('Guess', cv2.resize(res_show, (len(names)*100,length), interpolation=cv2.INTER_NEAREST))
        cv2.imshow('Voice', show)
        cv2.waitKey(1)

