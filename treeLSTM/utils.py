import logging
import os
import torch as th
from collections import OrderedDict
from tqdm import tqdm
import numpy as np

NAME_VAR = 'main'

def set_main_logger_settings(log_dir, name):
    global NAME_VAR

    NAME_VAR = name

    logger = logging.getLogger(NAME_VAR)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s: %(message)s")

    # file logger
    fh = logging.FileHandler(os.path.join(log_dir, NAME_VAR) + '.log', mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def get_new_logger(name):
    global NAME_VAR
    logger = logging.getLogger(NAME_VAR+'.'+name)
    #logger.setLevel(logging.DEBUG)
    #formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s: %(message)s")
    return logger


def load_vocabulary(data_dir, logger, extra = False):
    object_file = os.path.join(data_dir, 'vocab.pkl')
    text_file = os.path.join(data_dir, 'vocab.txt')
    if os.path.exists(object_file):
        # load vocab file
        vocab = th.load(object_file)
    else:
        # create vocab file
        vocab = OrderedDict()
        logger.debug('Loading vocabulary.')
        with open(text_file, encoding='utf-8') as vf:
            for line in tqdm(vf.readlines(), desc='Loading vocabulary: '):
                line = line.strip()
                vocab[line] = len(vocab)
        if extra:
            #adding POStag to vocab
            text_file = os.path.join(data_dir, 'POStags.txt')
            with open(text_file, encoding='utf-8') as vf:
                for line in tqdm(vf.readlines(), desc='Loading vocabulary POStag: '):
                    line = line.strip()
                    vocab[line] = len(vocab)

        th.save(vocab, object_file)
    logger.info('Vocabulary loaded.')
    return vocab


def load_embeddings(data_dir, pretrained_emb_file, vocab, logger, extra = False):
    object_file = os.path.join(data_dir, 'pretrained_emb.pkl')
    if os.path.exists(object_file):
        pretrained_emb = th.load(object_file)
    else:
        # filter glove
        glove_emb = {}
        logger.debug('Loading pretrained embeddings.')
        with open(pretrained_emb_file, 'r', encoding='utf-8') as pf:
            for line in tqdm(pf.readlines(), desc='Loading pretrained embeddings:'):
                sp = line.split(' ')
                if sp[0] in vocab:
                    glove_emb[sp[0].lower()] = np.array([float(x) for x in sp[1:]])

        if extra:
            #adding POStag to embeddings
            emb_file = os.path.join(data_dir, 'POStag_emb.txt')
            with open(emb_file , 'r', encoding='utf-8') as pf:
                for line in tqdm(pf.readlines(), desc='Loading pretrained embeddings POStag:'):
                    sp = line.split(' ')
                    if sp[0] in vocab:
                        glove_emb[sp[0].lower()] = np.array([float(x) for x in sp[1:]])

        # initialize with glove
        pretrained_emb = []
        fail_cnt = 0
        for line in vocab.keys():
            if not line.lower() in glove_emb:
                fail_cnt += 1
            pretrained_emb.append(glove_emb.get(line.lower(), np.random.uniform(-0.05, 0.05, 300)))

        logger.info('Miss word in GloVe {0:.4f}'.format(1.0 * fail_cnt / len(pretrained_emb)))
        pretrained_emb = th.tensor(np.stack(pretrained_emb, 0)).float()
        th.save(pretrained_emb, object_file)

    logger.info('Pretrained embeddings loaded.')
    return pretrained_emb


def print_graphs(list):
    import matplotlib.pyplot as plt
    import networkx as nx
    n = len(list)
    print("")
    for k in range(len(list)):
        g = list[k]
        plt.subplot(k+1,n,k+1)
        for i in range(len(g.nodes)):
            print(i, g.nodes[i].data)#['x'], g.nodes[i].data['y'], g.nodes[i].data['mask'], g.nodes[i].data['h'])
        print("-")
        nx.draw(list[k].to_networkx(), with_labels=True)
    plt.show()


def print_graphs2(list):
  import matplotlib.pyplot as plt
  import networkx as nx
  n = len(list)
  for k in range(len(list)):
    g = list[k]
    plt.subplot(k + 1, n, k + 1)
    G = g.to_networkx(node_attrs=['x'])
    pos = nx.spring_layout(G, k=len(G.nodes)*10, fixed = [0], scale = 100)
    #print(g.edges()[0])
    #print(g.edges()[1])
    nx.draw(G, pos, node_size = 1000)
    node_labels = nx.get_node_attributes(G, 'x')
    for k in node_labels:
      node_labels[k] = str(node_labels[k].item())+ "(" + str(k) + ")"
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    edge_labels = nx.get_edge_attributes(G, 'w')
    nx.draw_networkx_edge_labels(G, pos, labels=edge_labels)

  plt.show()