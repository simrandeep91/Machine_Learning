import skimage
import skimage.io
import skimage.transform
import numpy as np

def print_prob(p, file_path, label):
    synset = [l.strip() for l in open(file_path).readlines()]

    np.set_printoptions(threshold='nan')
    i = 0;
    pr = np.zeros((label.shape[0]));
#    print(pr.shape)
    for prob in p:
        # print prob
        pred = np.argsort(prob)[::-1]

        # Get top1 label
        top1 = synset[pred[0]]
#        print("Top1: ", top1, prob[pred[0]])
        pr[i] = top1;
        i = i+1;
        # Get top5 label
#       top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
#       print("Top5: ", top5)

    cl = 0;
#    print(pr)
#    print(label)
    for k in range(0, pr.shape[0]):
#        print("Prob");
#        print(pr[k])
#        print(label[k])
        if pr[k] == label[k]:
            cl = cl+1

    cl = (cl*1.0)/pr.shape[0]
#    print("Small dataset Accuracy")
#    print(cl)
    return pr



def guess_print_prob(p, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]
    np.set_printoptions(threshold='nan')
    i = 0;
    pr = np.zeros((p.shape[0]));
    for prob in p:
        pred = np.argsort(prob)[::-1]
        # Get top1 label
        top1 = synset[pred[0]]
#        print("Top1: ", top1, prob[pred[0]])
        pr[i] = top1;
        i = i+1;
        # Get top5 label
#       top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    return pr

