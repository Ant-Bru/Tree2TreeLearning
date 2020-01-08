import torch.nn as nn
from treeLSTM.utils import print_graphs
from collections import namedtuple


class Tree2Tree(nn.Module):

    def __init__(self,
                 encoder,
                 decoder):
        super(Tree2Tree, self).__init__()
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, g_enc, x_enc, mask_enc, g_dec):

        self.encoder.forward(g_enc, x_enc , mask_enc) #tree encoding

        #print("ENCODED TREE ------------------")
        #print_graphs([g_enc])

        #print("DECODE TREE 1 ------------------")
        #self.print_elimination_graphs([g_dec], True)

        root_ids =  [i for i in range(g_enc.number_of_nodes()) if g_enc.out_degree(i) == 0]
        encs = g_enc.ndata['h'][root_ids] #root encoding for decoder

        #print("ENCS ROOT", encs)

        return self.decoder.forward(g_dec, encs)

    def print_elimination_graphs(self, list, target = False):
        import matplotlib.pyplot as plt
        import networkx as nx
        n = len(list)
        for k in range(len(list)):
            g = list[k]
            plt.subplot(k + 1, n, k + 1)
            G = g.to_networkx(node_attrs=['y'])
            pos = nx.spring_layout(G, k=len(G.nodes) * 10, fixed=[0], scale=100)
            # print(g.edges()[0])
            # print(g.edges()[1])
            nx.draw(G, pos, node_size=1000)
            node_labels = nx.get_node_attributes(G, 'y')
            for k in node_labels:
                node_labels[k] = str(node_labels[k].item()) + "(" + str(k) + ")"
            nx.draw_networkx_labels(G, pos, labels=node_labels)
            edge_labels = nx.get_edge_attributes(G, 'w')
            nx.draw_networkx_edge_labels(G, pos, labels=edge_labels)

        plt.show()


class Tree2Tree2(nn.Module):

    def __init__(self,
                 encoder,
                 decoder):
        super(Tree2Tree2, self).__init__()
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, g_enc, x_enc, mask_enc):

        self.encoder.forward(g_enc, x_enc , mask_enc) #tree encoding

        #print("ENCODED TREE ------------------")
        #print_graphs([g_enc])

        #print("DECODE TREE 1 ------------------")
        #self.print_elimination_graphs([g_dec], True)

        root_ids =  [i for i in range(g_enc.number_of_nodes()) if g_enc.out_degree(i) == 0]
        encs = g_enc.ndata['h'][root_ids] #root encoding for decoder

        #print("ENCS ROOT", encs)

        return self.decoder.forward(encs)

    def print_elimination_graphs(self, list, target = False):
        import matplotlib.pyplot as plt
        import networkx as nx
        n = len(list)
        for k in range(len(list)):
            g = list[k]
            plt.subplot(k + 1, n, k + 1)
            G = g.to_networkx(node_attrs=['y'])
            pos = nx.spring_layout(G, k=len(G.nodes) * 10, fixed=[0], scale=100)
            # print(g.edges()[0])
            # print(g.edges()[1])
            nx.draw(G, pos, node_size=1000)
            node_labels = nx.get_node_attributes(G, 'y')
            for k in node_labels:
                node_labels[k] = str(node_labels[k].item()) + "(" + str(k) + ")"
            nx.draw_networkx_labels(G, pos, labels=node_labels)
            edge_labels = nx.get_edge_attributes(G, 'w')
            nx.draw_networkx_edge_labels(G, pos, labels=edge_labels)

        plt.show()