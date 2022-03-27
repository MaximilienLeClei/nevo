# Copyright 2022 Maximilien Le Clei.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from nets.dynamic.base import DynamicNetBase
from utils.functions.misc import deterministic_set, find_sublist_index


# Description of this Recurrent Network of Dynamic Complexity (pages 18-21)
# https://papyrus.bib.umontreal.ca/xmlui/bitstream/handle/1866/26072/Le_Clei_Maximilien_2021_memoire.pdf#page=18

class Net(DynamicNetBase):

    def __init__(self, arg_input, arg_output):

        self.d_input = arg_input if isinstance(arg_input, int) else None
        self.input_net = arg_input if not isinstance(arg_input, int) else None

        self.d_output = arg_output if isinstance(arg_output, int) else None
        self.output_net = arg_output if not isinstance(arg_output,
                                                       int) else None

        self.nodes = {'all': [], 'input': [], 'hidden': [], 'output': [],
                      'receiving': [], 'emitting': [], 'being pruned': [],
                      'layered': []}

        self.nb_nodes_grown = 0

        self.architectural_mutations = [self.grow_node,
                                        self.prune_node,
                                        self.grow_connection,
                                        self.prune_connection]

    def initialize_architecture(self):

        if self.d_input != None:
            for _ in range(self.d_input):
                self.grow_node('input')

        if self.d_output != None:
            for _ in range(self.d_output):
                self.grow_node('output')

    def mutate_parameters(self):

        for node in self.nodes['hidden'] + self.nodes['output']:
            node.mutate_parameters()

    def grow_node(self, type='hidden'):

        if type == 'input':

            new_input_node = Node('input', self.nb_nodes_grown)
            self.nb_nodes_grown += 1

            self.nodes['all'].append(new_input_node)
            self.nodes['input'].append(new_input_node)
            self.nodes['receiving'].append(new_input_node)

            if len(self.nodes['layered']) == 0:
                self.nodes['layered'].append([])

            self.nodes['layered'][0].append(new_input_node)

            return new_input_node

        elif type == 'output':

            new_output_node = Node('output', self.nb_nodes_grown)
            self.nb_nodes_grown += 1

            self.nodes['all'].append(new_output_node)
            self.nodes['output'].append(new_output_node)

            while len(self.nodes['layered']) < 2:
                self.nodes['layered'].append([])

            self.nodes['layered'][-1].append(new_output_node)

            return new_output_node
            
        else: # type == 'hidden'

            potential_in_nodes = deterministic_set(self.nodes['receiving'])

            in_node_1 = np.random.choice(potential_in_nodes)

            potential_in_nodes.remove(in_node_1)

            if len(potential_in_nodes) != 0:
                in_node_2 = np.random.choice(potential_in_nodes)

            out_node = np.random.choice(self.nodes['hidden'] +
                                        self.nodes['output'])

            new_hidden_node = Node('hidden', self.nb_nodes_grown)
            self.nb_nodes_grown += 1

            self.grow_connection(in_node_1, new_hidden_node)

            if len(potential_in_nodes) != 0:
                self.grow_connection(in_node_2, new_hidden_node)

            self.grow_connection(new_hidden_node, out_node)

            in_node_1_layer = find_sublist_index(in_node_1,
                                                 self.nodes['layered'])
            out_node_layer = find_sublist_index(out_node,
                                                self.nodes['layered'])

            layer_difference = out_node_layer - in_node_1_layer
            
            self.nodes['all'].append(new_hidden_node)
            self.nodes['hidden'].append(new_hidden_node)

            if abs(layer_difference) > 1:

                self.nodes['layered'][in_node_1_layer + 
                    np.sign(layer_difference)].append(new_hidden_node)
            
            else:

                if layer_difference == 1:
                    latest_layer = out_node_layer
                else: # layer_difference == -1 or layer_difference == 0:
                    latest_layer = in_node_1_layer

                self.nodes['layered'].insert(latest_layer, [])
                self.nodes['layered'][latest_layer].append(new_hidden_node)

    def grow_connection(self, in_node=None, out_node=None):

        if in_node == None:

            potential_in_nodes = deterministic_set(self.nodes['receiving'])

            for node in self.nodes['being pruned']:
                while node in potential_in_nodes:
                    potential_in_nodes.remove(node)

            if out_node != None:
                for node in out_node.in_nodes:
                    potential_in_nodes.remove(node)

            if len(potential_in_nodes) == 0:
                return

            in_node = np.random.choice(potential_in_nodes)

        if out_node == None:

            potential_out_nodes = self.nodes['hidden'] + self.nodes['output']

            for node in self.nodes['being pruned']:
                while node in potential_out_nodes:
                    potential_out_nodes.remove(node)

            for node in in_node.out_nodes:
                potential_out_nodes.remove(node)

            if len(potential_out_nodes) == 0:
                return

            out_node = np.random.choice(potential_out_nodes)
        
        in_node.connect_to(out_node)

        self.nodes['receiving'].append(out_node)
        self.nodes['emitting'].append(in_node)
            
    def prune_node(self, node=None):

        if node == None:

            if len(self.nodes['hidden']) == 0:
                return

            node = np.random.choice(self.nodes['hidden'])

        if node in self.nodes['being pruned']:
            return

        self.nodes['being pruned'].append(node)

        for out_node in node.out_nodes.copy():
            self.prune_connection(node, out_node, node)

        for in_node in node.in_nodes.copy():
            self.prune_connection(in_node, node, node)

        for key in self.nodes:

            if key == 'layered':

                node_layer = find_sublist_index(node, self.nodes['layered'])
                self.nodes['layered'][node_layer].remove(node)

                if (node_layer != 0
                    and node_layer != len(self.nodes['layered']) - 1): 
                    if self.nodes['layered'][node_layer] == []:
                        self.nodes['layered'].remove(
                            self.nodes['layered'][node_layer])
            else:
                while node in self.nodes[key]:
                    self.nodes[key].remove(node)   
          
    def prune_connection(self,
                         in_node=None,
                         out_node=None,
                         calling_node=None):

        if in_node == None:

            if len(self.nodes['emitting']) == 0:
                return

            in_node = np.random.choice(self.nodes['emitting'])

        if out_node == None:

            out_node = np.random.choice(in_node.out_nodes)

        connection_was_already_pruned = in_node.disconnect_from(out_node)

        if connection_was_already_pruned:
            return
        
        self.nodes['receiving'].remove(out_node)
        self.nodes['emitting'].remove(in_node)

        if in_node != calling_node:

            if in_node not in self.nodes['emitting']:

                if in_node in self.nodes['input']:
                    if self.input_net != None:
                        self.grow_connection(in_node=in_node)

                elif in_node in self.nodes['hidden']:
                    self.prune_node(in_node)

        if out_node != calling_node:

            if out_node not in self.nodes['receiving']:

                if out_node in self.nodes['output']:
                    if self.output_net != None:
                        self.grow_connection(out_node=out_node)

                elif out_node in self.nodes['hidden']:
                    self.prune_node(out_node)

        for node in [in_node, out_node]:

            if (node != calling_node
                and node not in self.nodes['being pruned']):

                if node in self.nodes['hidden']:
                    if node.in_nodes == [node] or node.out_nodes == [node]:
                        self.prune_node(node)

                elif node in self.nodes['output']:
                    if node.in_nodes == [node] and node.out_nodes == [node]:
                        self.prune_connection(node, node)

    # This method is for use when combining with other networks.
    def handle(self, source, action, i=None): 

        if source == self.input_net:

            if action == '+ node':

                new_input_node = self.grow_node('input')
                self.grow_connection(new_input_node, None)

                if self.output_net != None:
                    for output_node in self.nodes['output']:
                        if output_node not in self.nodes['receiving']:
                            self.grow_connection(None, output_node)

            else: # action == '- node':
                self.prune_node(self.nodes['input'][i])

        else: # source == self.output_net:

            if action == '+ node':

                new_output_node = self.grow_node('output')
                self.grow_connection(None, new_output_node)

                for input_node in self.nodes['input']:
                    if input_node not in self.nodes['emitting']:
                        self.grow_connection(input_node, None)

            else: # action == '- node':
                self.prune_node(self.nodes['output'][i])

    def reset(self):

        for node in self.nodes['all']:
            node.output = np.array([0])

    def __call__(self, x):

        for x_i, node in zip(x, self.nodes['input']):
            node.output = x_i
            
        for layer in range(1, len(self.nodes['layered'])):

            for node in self.nodes['layered'][layer]:
                node.compute()

            for node in self.nodes['layered'][layer]:
                node.update()

        return [ node.output for node in self.nodes['output'] ]


class Node:

    def __init__(self, type, id):

        self.id = id

        self.in_nodes  = []
        self.out_nodes = []
    
        self.output = np.array([0])

        self.type = type

        if self.type != 'input':
            self.initialize_parameters()

    def __repr__(self):

        in_node_ids = tuple([node.id for node in self.in_nodes])
        out_node_ids = tuple([node.id for node in self.out_nodes])

        if self.type == 'input':
            return str(('x',)) + '->' + str(self.id) + '->' + \
                   str(out_node_ids)
        elif self.type == 'hidden':
            return str(in_node_ids) + '->' + str(self.id) + '->' + \
                   str(out_node_ids)
        else: # self.type == 'output':
            return str(in_node_ids) + '->' + str(self.id) + '->' + \
                   str(('y',) + out_node_ids)
            
    def initialize_parameters(self):

        self.weights = np.empty(0)

        if self.type == 'hidden':
            self.bias = np.random.randn(1)
        else: # self.type == 'output':
            self.bias = np.zeros(1)

    def mutate_parameters(self):
        
        self.weights += 0.01 * np.random.randn(*self.weights.shape)
        self.bias += 0.01 * np.random.randn()

    def connect_to(self, node):
        
        new_weight = np.random.randn(1)
        node.weights = np.concatenate((node.weights, new_weight))
        
        self.out_nodes.append(node)
        node.in_nodes.append(self)

    def disconnect_from(self, node):

        if self not in node.in_nodes:
            return True

        i = node.in_nodes.index(self)
        node.weights = np.concatenate((node.weights[:i], node.weights[i+1:]))

        self.out_nodes.remove(node)
        node.in_nodes.remove(self)

        return False

    def compute(self):

        x = np.array([node.output for node in self.in_nodes]).squeeze()

        x = np.dot(x, self.weights) + self.bias

        x = np.clip(x, 0, 2**31-1)

        self.future_output = x

    def update(self):

        self.output = self.future_output