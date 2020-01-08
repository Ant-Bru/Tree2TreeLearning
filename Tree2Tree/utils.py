from TreeDecoder.models import TreeDecoder
from treeLSTM.dataset import TreeDataset
from TreeDecoder.cells import *
import networkx as nx
import dgl
import torch.nn.functional as F
from nltk.corpus.reader import BracketParseCorpusReader
from torch.utils.data import DataLoader
from tqdm import tqdm


# TODO: modfy the dataset class according to TreeDataset
# TODO: embeddigns anf vocabulray must be loaded outside the class and sahred amogn test/train/dev
# TODO: check out to load embeddings (remove lower on emdeggings key)

class Tree2TreeDataset():

    def __init__(self, input, target):
        self.input = input
        self.target = target
        self.__bridging__()

    def __bridging__(self):

        for i in len(self.input.data):
            input_g = self.input.data[i]
            target_g = self.output.data[i]

            root_id_input = 0 #[i for i in range(input_g.number_of_nodes()) if input.out_degree(i) == 0]
            root_id_target = 0 #[i for i in range(input_g.number_of_nodes()) if input.out_degree(i) == 0]

            #h_root_a = h_a_tree[root_id_a]
            #h_root_b = h_b_tree[root_id_b]

            target_g.add_edge(root_id_input, root_id_target)