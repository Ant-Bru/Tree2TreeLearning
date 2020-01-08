import torch.nn as nn
import dgl
import torch as th
import torch.nn.functional as F
import array as arr
from collections import deque


class TreeDecoder(nn.Module):
    def __init__(self,
                 h_size,
                 max_output_degree,
                 max_depth,
                 cell):
        super(TreeDecoder, self).__init__()
        self.h_size = h_size
        self.cell = cell
        self.max_outdegree = max_output_degree
        self.max_depth = max_depth


    def spread_encs(self, g, encs):

        root_ids = [i for i in range(g.number_of_nodes()) if g.in_degree(i) == 0]

        g.ndata['enc'] = th.zeros(len(g.nodes()),self.h_size)

        #set encs to roots
        for i in range(len(encs)):
            g.ndata['enc'][root_ids[i]] = encs[i]

        #spread to the whole trees
        for i in range(len(root_ids)-1):
            g.ndata['enc'][root_ids[i]+1:root_ids[i+1]] = encs[i]
        g.ndata['enc'][root_ids[-1] + 1:] = encs[-1]


    def forward(self, g, encs):

        #self.training = False #DA TOGLIERE SIMULO

        if self.training:
            #print("TRAIN----")

            self.spread_encs(g, encs)

            # topological order
            topo_nodes = dgl.topological_nodes_generator(g)

            roots = topo_nodes[0:1]
            others = topo_nodes[1:]

            #root training computations
            g.register_message_func(self.cell.message_func)
            g.register_reduce_func(self.cell.reduce_func)
            g.register_apply_node_func(self.cell.apply_node_func_root)
            g.prop_nodes(roots)
            #print("--------------------ROOT COMPUTED-------------")

            #other nodes training computations
            g.register_apply_node_func(self.cell.apply_node_func)
            g.prop_nodes(others)
            #print("--------------------ALL COMPUTED-------------")

        else:
            #print("EVAL----")
            trees = []
            features = {'parent_h': th.zeros(1, 1, self.h_size), 'parent_output': th.zeros(1, 1, self.cell.num_classes)}
            if str(self.cell) == 'DRNNCell':
                features['sibling_h'] = th.zeros(1, 1, self.h_size)
                features['sibling_output'] = th.zeros(1, 1, self.cell.num_classes)
            #print(features)
            #input("srthfjyg")
            #create only root trees without labels
            for i in range(len(encs)):
                tree = dgl.DGLGraph()
                tree.add_nodes(1, features)
                #print(tree.ndata)
                trees.append(tree)
            #print("#ROOTS", len(trees))
            g = dgl.batch(trees) #batch them
            g.ndata['enc'] = encs #set root encs
            g.ndata['pos'] = th.zeros(len((g.nodes())),10)
            g.ndata['depth'] = th.zeros(len((g.nodes())),10)

            #roots cumputations
            topo_nodes = dgl.topological_nodes_generator(g)
            g.register_message_func(self.cell.message_func)
            g.register_reduce_func(self.cell.reduce_func)
            g.register_apply_node_func(self.cell.apply_node_func_root)
            g.prop_nodes(topo_nodes)


            #print("--------------------ROOT (lvl 0): h, label, probs COMPUTED-------------")

            trees = dgl.unbatch(g) #unbatch to deal with single trees expansions

            nodes_id = []
            #single trees expansions
            for i in range(len(trees)):
                nodes_id.append(self.cell.expand(trees[i], 0))

            positions = [i for i in range(len(trees))]
            #print("ALBERI", positions)
            final_trees = [None] * len(trees)

            #print("TREES", len(trees), trees)
            #print("FINAL", final_trees)

            #print("NODI da elaborare", nodes_id)

            #nodes_id, filtered_trees, positions = self.filter(nodes_id, trees, positions, final_trees) #<---------- qui mi perdo gli alberi che non devo espandere


            self.filter2(nodes_id, trees, positions, final_trees)

            #print("POS FILTRATI", positions)

            #print("NODI da elaborare FILTRATI", nodes_id)

            #print("TREES FILTRATI", len(trees))
            #print("FINAL FILTRATI", final_trees)

            #for t in final_trees:
                #if t is not None:
                    #t.ndata['parent_h'] = th.zeros(t.number_of_nodes(), 1 ,self.h_size)
                    #t.ndata['parent_output'] = th.zeros(t.number_of_nodes(), 1, self.h_size)
                    #s= ""
                    #for k in t.ndata:
                        #s+= "   "+str(k)
                    #print(s)
                #else:
                    #print(None)

            #input("---------------------------")

            #print("POSITIONS", positions)
            depth = 0

            #loop expansions of the lower levels
            while nodes_id:
                #print("DEPTH", depth)
                # tree_nodes_id[0] = []
                # print("NODES IDS", tree_nodes_id)
                # print("TREES", trees)
                #nodes_id, filtered_trees, positions = self.filter(nodes_id, trees, positions, roots) #<---------- qui mi perdo gli alberi che non devo espandere



                #for i in range(len(pos)):
                    #positions[i] = positions[pos[i]]
                #print("NODI da elaborare filtrati", nodes_id)
                #print("Degli alberi", positions)
                # print("TREES", trees)
                # input("---------")

                g = dgl.batch(trees)  # batch again to computes new nodes data
                batch_nodes_id = self.tree_node_id_to_batch_node_id(trees, nodes_id)  # ids mapping
                #print("RESI per BATCH", batch_nodes_id)
                # input("-------------")

                g.register_message_func(self.cell.message_func)
                g.register_reduce_func(self.cell.reduce_func)
                g.register_apply_node_func(self.cell.apply_node_func)
                g.prop_nodes(batch_nodes_id)

                depth += 1

                #print("--------------------lvl "+str(depth)+" NODES: h, label, probs COMPUTED-------------")

                if depth < self.max_depth:

                    tree_nodes_id = nodes_id.copy()
                    trees = dgl.unbatch(g) #unbatch to deal with single trees expansions

                    #print("BATCH IDS to expand", batch_nodes_id)
                    #print("NODE IDS to expand", tree_nodes_id)

                    nodes_id = []
                    # single trees expansions
                    for i in range(len(trees)):
                        tree_ids = []
                        for j in range(len(tree_nodes_id[i])):
                            id = tree_nodes_id[i][j]
                            tree_ids+=self.cell.expand(trees[i], id)
                        nodes_id.append(tree_ids)
                    #print("NODI da elaborare", nodes_id)

                    #print("TREES", len(trees), trees)
                    #print("FINAL", final_trees)

                    #print("NODI da elaborare", nodes_id)

                    # nodes_id, filtered_trees, positions = self.filter(nodes_id, trees, positions, final_trees) #<---------- qui mi perdo gli alberi che non devo espandere

                    self.filter2(nodes_id, trees, positions, final_trees)

                    #print("POS FILTRATI", positions)

                    #print("NODI da elaborare FILTRATI", nodes_id)

                    #print("TREES FILTRATI", len(trees))
                    #print("FINAL FILTRATI", final_trees)

                else:
                    #devo mettere in final quelli rimasti in trees
                    for i in range(len(trees)):
                        final_trees[positions[i]] = trees[i]
                        #del (trees[positions.index(positions[i])])
                        #del (nodes_id[positions.index(positions[i])])
                        #positions.remove(positions[i])
                    break
                #input("-------------------------")
            #print(roots[1])
            #print(final_trees)
            g = dgl.batch(final_trees)

        return g


    # tree node id to batch node id mapping
    def tree_node_id_to_batch_node_id(self, trees, tree_nodes_id):
        l = []
        l.append(tree_nodes_id[0])
        c = trees[0].number_of_nodes()
        for i in range(1, len(tree_nodes_id)):
            l.append([int(x+c) for x in tree_nodes_id[i]])
            c += trees[i].number_of_nodes()
        return l

    def to_process(self, node_ids):
        s = 0
        for ids in node_ids:
            s+= len(ids)
        return s > 0


    def filter2(self, node_ids, trees, pos, final):
        for i in range(len(node_ids) - 1, -1, -1):
            if len(node_ids[i]) == 0:
                final[pos[i]] = trees[i]
                del(trees[pos.index(pos[i])])
                del(node_ids[pos.index(pos[i])])
                pos.remove(pos[i])



    def filter(self, node_ids, trees, pos, roots):
        #print("FILTERING")
        new_ids = []
        new_trees = []
        positions = []
        for i in range(len(node_ids)):
            if len(node_ids[i]) > 0:
                new_ids.append(node_ids[i])
                new_trees.append(trees[i])
                positions.append(pos[i])
            else:
                roots[pos[i]] = trees[pos[i]]
        return new_ids, new_trees, positions

class Attention(nn.Module):
    """
    Inputs:
        last_hidden: (batch_size, hidden_size)
        encoder_outputs: (batch_size, max_time, hidden_size)
    Returns:
        attention_weights: (batch_size, max_time)
    """
    def __init__(self, batch_size, hidden_size, method="dot", mlp=False):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if method == 'dot':
            pass
        elif method == 'general':
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        elif method == "concat":
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
            self.va = nn.Parameter(th.FloatTensor(batch_size, hidden_size))
        elif method == 'bahdanau':
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
            self.Ua = nn.Linear(hidden_size, hidden_size, bias=False)
            self.va = nn.Parameter(th.FloatTensor(batch_size, hidden_size))
        else:
            raise NotImplementedError

        self.mlp = mlp
        if mlp:
            self.phi = nn.Linear(hidden_size, hidden_size, bias=False)
            self.psi = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, last_hidden, encoder_outputs, seq_len=None):
        batch_size, seq_lens, _ = encoder_outputs.size()
        if self.mlp:
            last_hidden = self.phi(last_hidden)
            encoder_outputs = self.psi(encoder_outputs)

        attention_energies = self._score(last_hidden, encoder_outputs, self.method)

        if seq_len is not None:
            pass
            #attention_energies = mask_3d(attention_energies, seq_len, -float('inf'))  <-- DECOMMENTARE

        return F.softmax(attention_energies, -1)

    def _score(self, last_hidden, encoder_outputs, method):
        """
        Computes an attention score
        :param last_hidden: (batch_size, hidden_dim)
        :param encoder_outputs: (batch_size, max_time, hidden_dim)
        :param method: str (`dot`, `general`, `concat`, `bahdanau`)
        :return: a score (batch_size, max_time)
        """

        assert encoder_outputs.size()[-1] == self.hidden_size

        if method == 'dot':
            last_hidden = last_hidden.unsqueeze(-1)
            return encoder_outputs.bmm(last_hidden).squeeze(-1)

        elif method == 'general':
            x = self.Wa(last_hidden)
            x = x.unsqueeze(-1)
            return encoder_outputs.bmm(x).squeeze(-1)

        elif method == "concat":
            x = last_hidden.unsqueeze(1)
            x = F.tanh(self.Wa(th.cat((x, encoder_outputs), 1)))
            return x.bmm(self.va.unsqueeze(2)).squeeze(-1)

        elif method == "bahdanau":
            x = last_hidden.unsqueeze(1)
            out = F.tanh(self.Wa(x) + self.Ua(encoder_outputs))
            return out.bmm(self.va.unsqueeze(2)).squeeze(-1)

        else:
            raise NotImplementedError

    def extra_repr(self):
        return 'score={}, mlp_preprocessing={}'.format(
            self.method, self.mlp)


class SeqDecoder(nn.Module):
    def __init__(self, h_size, out_size, attention = False):
        super(SeqDecoder, self).__init__()
        self.hidden_size = h_size
        self.rnn = nn.GRU(
            input_size=h_size,
            hidden_size=h_size,
            num_layers=1,
            dropout=0,
            bidirectional= False,
            batch_first=True)
        if attention:
            self.attention = Attention(
                h_size,
                method=None, #config.get("attention_score", "dot"),
                mlp=None #config.get("attention_mlp_pre", False))
            )

        #self.gpu = config.get("gpu", False)
        #self.decoder_output_fn = F.log_softmax if config.get('loss', 'NLL') == 'NLL' else None

    def forward(self, encs, seqs):
        if self.training:
            return self.forward_train(encs, seqs)
        else:
            return self.forward_sample(encs, seqs)



    def forward_train(self, encs, targets):

        # TARGETS ha le sequenze gold che devo apprendere

        rnn_output, rnn_hidden = self.rnn(targets, encs.unsqueeze(0))
        output = rnn_output.squeeze(1)
        output = self.character_distribution(output)

        if self.decoder_output_fn:
            output = self.decoder_output_fn(output, -1)

        return output, rnn_hidden.squeeze(0)


    def forward_sample(self, encs, empties):

        # EMPTIES ha il primo nodo vuoto della sequenza da generare
        # oppure non lo passo e genero la sequenza dal nulla
        #assegnando come stato precedente encs e come input il vettore nullo

        rnn_output, rnn_hidden = self.rnn(empties, encs.unsqueeze(0))
        output = rnn_output.squeeze(1)
        output = self.character_distribution(output)

        if self.decoder_output_fn:
            output = self.decoder_output_fn(output, -1)

        return output, rnn_hidden.squeeze(0)


class TreeDecoder2(nn.Module):
    def __init__(self,
                 h_size,
                 output_size):
        super(TreeDecoder2, self).__init__()
        self.h_size = h_size
        self.linear1 = nn.Linear(h_size, output_size*2)
        self.linear2 = nn.Linear(output_size*2, output_size)


    def forward(self, encs):
        return th.sigmoid(self.linear2(self.linear1(encs)))