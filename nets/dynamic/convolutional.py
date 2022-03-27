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
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.dynamic.base import DynamicNetBase
from utils.functions.convolutional import avg_pool, compute_padding
from utils.functions.convolutional import neg, torch_cat
from utils.functions.misc import find_sublist_index


# Description of this Convolutional Network of Dynamic Complexity (pages 22-26)
# https://papyrus.bib.umontreal.ca/xmlui/bitstream/handle/1866/26072/Le_Clei_Maximilien_2021_memoire.pdf#page=22

# This excerpt only explains the case where the network is evolved to perform
# convolutions (direction='forward'). It can also be evolved to perform
# transpose convolutions (direction='backward').

class Net(DynamicNetBase):

    def __init__(self, d_input, direction='forward'):

        self.d_input = np.array(d_input)

        self.output_net = None

        self.direction = direction

        self.nodes = {'all': [], 'input': [], 'hidden': [], 'output': [],
                      'branching': [], 'convolutional': [], 'expanded': [],
                      'layered': []}

        self.nb_nodes_grown = 0

        self.architectural_mutations = [self.grow_branch,
                                        self.prune_branch,
                                        self.expand_node,
                                        self.contract_node]

    def initialize_architecture(self):

        self.grow_node()
        self.grow_branch()

    def mutate_parameters(self):

        for node in self.nodes['convolutional']:
            node.mutate_parameters()

    def grow_branch(self):

        base_node = np.random.choice(
            self.nodes['hidden'] + [self.nodes['input']])

        self.nodes['branching'].append(base_node)

        new_node = self.grow_node(base_node)
        
        while new_node.type != 'output':
            new_node = self.grow_node(new_node)

    def grow_node(self, base_node=None):

        if base_node == None:

            input_node = Node(
                self.d_input, self.direction, self.nb_nodes_grown)

            self.nb_nodes_grown += 1

            self.nodes['all'].append(input_node)

            self.nodes['input'] = input_node

            if self.direction == 'backward':
                self.nodes['convolutional'].append(input_node)
            
            self.nodes['layered'].append([input_node])

            return input_node

        else:
            
            new_node = Node(base_node, self.direction, self.nb_nodes_grown)
            self.nb_nodes_grown += 1

            self.nodes['all'].append(new_node)

            if new_node.type == 'hidden':
                self.nodes['hidden'].append(new_node)
            else:
                self.nodes['output'].append(new_node)
                self.output_net.handle(self, '+ node')

            if new_node.function == 'convolve':
                self.nodes['convolutional'].append(new_node)

            base_node_layer = find_sublist_index(
                base_node, self.nodes['layered'])

            if base_node_layer == len(self.nodes['layered']) - 1:
                self.nodes['layered'].append([])

            self.nodes['layered'][base_node_layer+1].append(new_node)

            return new_node

    def prune_branch(self):

        branching_node = np.random.choice(self.nodes['branching'])

        out_node = np.random.choice(branching_node.out_nodes)

        self.prune_node(out_node)

        self.nodes['branching'].remove(branching_node)

        if len(self.nodes['output']) == 0:
            self.grow_branch()

    def prune_node(self, node):

        for i in range(len(node.out_nodes) - 1, -1, -1):
            self.prune_node(node.out_nodes[i])

        node.in_node.disconnect_from(node)

        for key in self.nodes:

            if key == 'input':
                pass

            elif key == 'layered':
                node_layer = find_sublist_index(node, self.nodes['layered'])
                self.nodes['layered'][node_layer].remove(node)

            else:

                while node in self.nodes[key]:

                    if key == 'output':
                        self.output_net.handle(
                            self, '- node', self.nodes['output'].index(node))

                    self.nodes[key].remove(node)

    def expand_node(self):

        expandable_nodes = self.nodes['convolutional'].copy()

        if self.direction == 'forward':
            
            for output_node in self.nodes['output']:

                if output_node in self.nodes['convolutional']:
                    expandable_nodes.remove(output_node)
                
                else:

                    if output_node.in_node in self.nodes['convolutional']:
                        expandable_nodes.remove(output_node.in_node)

        else: # self.direction == 'backward':

            expandable_nodes.remove(self.nodes['input'])

        if len(expandable_nodes) == 0:
            return

        node = np.random.choice(expandable_nodes)

        node.expand()

        self.nodes['expanded'].append(node)

    def contract_node(self):

        if len(self.nodes['expanded']) == 0:
            return

        node = np.random.choice(self.nodes['expanded'])

        node.contract()

        self.nodes['expanded'].remove(node)

    def setup_to_run(self):
        
        for node in self.nodes['all']:
            node.setup_torch_operation()

    def setup_to_save(self):
    
        for node in self.nodes['all']:
            node.delete_torch_operation()

        self.reset()

    def reset(self):
        
        for node in self.nodes['all']:
            
            node.input = None
            node.output = None

    def __call__(self, x):

        if self.direction == 'forward':

            self.nodes['input'].input = x

            for layer in range(len(self.nodes['layered'])):

                for node in self.nodes['layered'][layer]:

                    node.compute()

            return [ node.output for node in self.nodes['output'] ]

        else: # self.direction == 'backward':

            for x_i, output_node_i in zip(x, self.nodes['output']):
                output_node_i.output = x_i

            for layer in range(len(self.nodes['layered']) - 1, -1, -1):

                for node in self.nodes['layered'][layer]:

                    node.compute()

            return self.nodes['input'].input


class Node:

    def __init__(self, in_arg, direction, id):

        self.id = id

        self.in_node = None
        self.out_nodes = []

        self.direction = direction

        self.input = None
        self.output = None

        if isinstance(in_arg, np.ndarray):

            self.type = 'input'

            self.d_in = in_arg

            self.d = len(self.d_in) - 1

            if self.direction == 'forward':

                self.function = 'none'

                self.d_out = np.copy(self.d_in)

            else: # self.direction == 'backward':

                self.function = 'convolve'

                self.d_out = np.append(0, self.d_in[1:])

                self.kernel_size = (1,) * self.d
                self.padding = (0,0) * self.d

        else: # isinstance(in_arg, Node):

            self.in_node = in_arg

            self.d = self.in_node.d

            if self.direction == 'forward':
                self.d_in = np.copy(self.in_node.d_out)

            pool_factor = self.in_node.sample_new_pool_factor()

            if pool_factor != None:

                self.function = 'pool'

                self.pool_factor = pool_factor

                if self.direction == 'backward':
                    self.d_in = np.append(0, self.in_node.d_out[1:])

                self.kernel_size = (self.pool_factor,) * self.d
                self.padding = None

                self.d_out = np.append(
                    self.d_in[0], self.d_in[1:] // self.pool_factor)

                if np.all(np.less_equal(self.d_out, 1)):

                    self.type = 'output'

                    if self.direction == 'backward':
                        self.d_in[0] += 1
                        self.d_out[0] += 1

                else: # np.any(np.greater(self.d_out, 1)):
                
                    self.type = 'hidden'

            else: # pool_factor == None:

                self.function = 'convolve'

                if self.direction == 'backward':
                    self.d_in = np.append(1, self.in_node.d_out[1:])

                self.kernel_size = (3,) * self.d
                self.padding = compute_padding(self.d_in)

                if self.direction == 'forward':
                    self.d_out = np.append(
                        1, np.maximum(self.d_in[1:], 3) - 2)
                else: # self.direction == 'backward':
                    self.d_out = np.append(
                        0, np.maximum(self.d_in[1:], 3) - 2)

                if np.all(np.equal(self.d_out[1:], 1)):

                    self.type = 'output'

                    if self.direction == 'backward':
                        self.d_out[0] += 1

                else: # np.any(np.greater(self.d_out[1:], 1)):
                
                    self.type = 'hidden'

            self.in_node.append(self)

        if self.function == 'convolve':
            self.initialize_parameters()

    def __repr__(self):

        out_nodes_ids = tuple([out_node.id for out_node in self.out_nodes])

        if self.type == 'input':
            return str(('x',)) + '->' + str(self.id) + '->' + \
                   str(out_nodes_ids)
        elif self.type == 'hidden':
            return str(self.in_node.id) + '->' + str(self.id) + '->' + \
                   str(out_nodes_ids)
        else: # self.type == 'output':
            return str(self.in_node.id) + '->' + str(self.id) + '->y'        

    def initialize_parameters(self):

        d_weights = (self.d_out[0], self.d_in[0]) + self.kernel_size

        if self.direction == 'forward':
            d_biases = self.d_out[0]
        else: # self.direction == 'backward': 
            d_biases = self.d_in[0]

        self.weights = np.random.randn(*d_weights)
        self.biases = np.random.randn(d_biases) if self.type != 'input' else \
                      np.zeros(d_biases)

    def mutate_parameters(self):
        
        self.weights += 0.01 * np.random.randn(*self.weights.shape)
        self.biases += 0.01 * np.random.randn(*self.biases.shape)   

    def setup_torch_operation(self):

        if self.function == 'convolve':

            if self.direction == 'forward':

                if self.d == 1:
                    self.conv = nn.Conv1d
                elif self.d == 2:
                    self.conv = nn.Conv2d
                else: # self.d == 3:
                    self.conv = nn.Conv3d

                op = self.conv = self.conv(
                    self.d_in[0], self.d_out[0], self.kernel_size)

            else: # self.direction == 'backward':

                if self.d == 1:
                    self.conv_T = nn.ConvTranspose1d
                elif self.d == 2:
                    self.conv_T = nn.ConvTranspose2d
                else: # self.d == 3:
                    self.conv_T = nn.ConvTranspose3d

                op = self.conv_T = self.conv_T(
                    self.d_in[0], self.d_out[0], self.kernel_size)

            op.weight.requires_grad = False
            op.bias.requires_grad = False

            op.weight.data = torch.Tensor(self.weights)
            op.bias.data = torch.Tensor(self.biases)

        elif self.function == 'pool':

            if self.direction == 'forward':

                if self.d == 1:
                    self.pool = nn.AvgPool1d(self.kernel_size)
                elif self.d == 2:
                    self.pool = avg_pool # CV2 nn.AvgPool2d(self.kernel_size)
                else: # self.d == 3:
                    self.pool = nn.AvgPool3d(self.kernel_size)

            else: # self.direction == 'backward':
                self.pool_T = F.interpolate

    def delete_torch_operation(self):

        if self.function == 'convolve':

            if self.direction == 'forward':
                self.conv = None
            else: # self.direction == 'backward':
                self.conv_T = None

        else: # self.function == 'pool':

            if self.direction == 'forward':
                self.pool = None
            else: # self.direction == 'backward':
                self.pool_T = None

    def sample_new_pool_factor(self):

        if self.function == 'pool':
            return None
        
        gcd = np.gcd.reduce(self.d_out[1:])
            
        if gcd == 1:
            return None

        all_pool_factors = []

        for i in range(1, gcd):
            if gcd % i == 0:
                all_pool_factors.append(gcd//i)

        current_out_nodes_pool_factors = []

        for out_node in self.out_nodes:
            if out_node.function == 'pool':
                current_out_nodes_pool_factors.append(out_node.pool_factor)

        for pool_factor in all_pool_factors:
            if pool_factor not in current_out_nodes_pool_factors:
                return pool_factor

        return None

    def append(self, out_node):
        
        self.out_nodes.append(out_node)

        if self.direction == 'backward':
            
            if out_node.function == 'convolve' or out_node.type == 'output':

                if self.function == 'convolve':

                    self.d_out[0] += out_node.d_in[0]

                    node = self

                else: # self.function == 'pool':
                    
                    self.d_out[0] += out_node.d_in[0]
                    self.d_in[0] += out_node.d_in[0]
                    self.in_node.d_out[0] += out_node.d_in[0]

                    node = self.in_node

                new_weights = np.random.randn(
                    *(out_node.d_in[0], node.d_in[0]) + node.kernel_size)
                node.weights = np.concatenate((node.weights, new_weights), 0)

    def disconnect_from(self, out_node):

        if self.direction == 'backward':

            if out_node.function == 'convolve' or out_node.type == 'output':

                i = self.find_index(out_node, 'start')
                j = self.find_index(out_node, 'end')

                if self.function == 'convolve':

                    self.d_out[0] -= out_node.d_in[0]

                    node = self

                else: # self.function == 'pool':

                    k = self.in_node.find_index(self, 'start')
                    i += k
                    j += k

                    self.d_out[0] -= out_node.d_in[0]
                    self.d_in[0] -= out_node.d_in[0]
                    self.in_node.d_out[0] -= out_node.d_in[0]

                    node = self.in_node

                node.weights = np.concatenate(
                    (node.weights[:i], node.weights[j:]), 0)

        self.out_nodes.remove(out_node)

    def find_index(self, node_to_find, relative_index):

        index = 0

        for out_node in self.out_nodes:

            if out_node != node_to_find:
                index += out_node.d_in[0]

            else:
                if relative_index == 'start':
                    relative_index = 0
                elif relative_index == 'end':
                    relative_index = node_to_find.d_in[0]
                    
                return index + relative_index

    def expand(self, calling_node=None):

        if self.direction == 'forward':

            if calling_node == None:

                self.d_out[0] += 1

                new_weights = np.random.randn(
                    *(1, self.d_in[0]) + self.kernel_size)
                self.weights = np.concatenate((self.weights, new_weights), 0)

                new_bias = np.random.randn(1)
                self.biases = np.concatenate((self.biases, new_bias), 0)

                for out_node in self.out_nodes:
                    out_node.expand(self)

            else: # calling_node != None:

                self.d_in[0] += 1

                if self.function == 'pool':

                    self.d_out[0] += 1

                    for out_node in self.out_nodes:
                        out_node.expand(self)

                else: # self.function == 'convolve'
                    
                    new_weights = np.random.randn(
                        *(self.d_out[0], 1) + self.kernel_size)
                    self.weights = np.concatenate(
                        (self.weights, new_weights), 1)

        else: # self.direction == 'backward':

            if calling_node == None:

                self.d_in[0] += 1

                new_weights = np.random.randn(
                    *(self.d_out[0], 1) + self.kernel_size)
                self.weights = np.concatenate((self.weights, new_weights), 1)

                new_bias = np.random.randn(1)
                self.biases = np.concatenate((self.biases, new_bias), 0)

                self.in_node.expand(self)

            else: # calling_node != None:

                self.d_out[0] += 1

                if self.function == 'pool':

                    self.d_in[0] += 1

                    self.in_node.expand(self)

                else: # self.function == 'convolve':

                    i = self.find_index(calling_node, 'end') - 1

                    new_weights = np.random.randn(
                        *(1, self.d_in[0]) + self.kernel_size)
                    self.weights = np.concatenate(
                        (self.weights[:i], new_weights, self.weights[i:]), 0)
                                                   
    def contract(self, calling_node=None, i=None):

        if self.direction == 'forward':

            if calling_node == None:
                
                i = np.random.randint(self.d_out[0])

                self.d_out[0] -= 1

                self.weights = np.concatenate(
                    (self.weights[:i], self.weights[i+1:]), 0)
                self.biases = np.concatenate(
                    (self.biases[:i], self.biases[i+1:]), 0)

                for out_node in self.out_nodes:
                    out_node.contract(self, i)

            else: # calling_node != None:
                
                self.d_in[0] -= 1

                if self.function == 'pool':

                    self.d_out[0] -= 1

                    for out_node in self.out_nodes:
                        out_node.contract(self, i)

                else: # self.function == 'convolve':

                    self.weights = np.concatenate(
                        (self.weights[:,:i], self.weights[:,i+1:]), 1)
                
        else: # self.direction == 'backward':

            if calling_node == None:

                i = np.random.randint(self.d_in[0])
                
                self.d_in[0] -= 1

                self.weights = np.concatenate(
                    (self.weights[:,:i], self.weights[:,i+1:]), 1)
                self.biases = np.concatenate(
                    (self.biases[:i], self.biases[i+1:]), 0)

                self.in_node.contract(self, i)
                
            else: # calling_node != None

                self.d_out[0] -= 1

                if self.function == 'pool':

                    self.d_in[0] -= 1

                    self.in_node.contract(self, i)

                else: # self.function == 'convolve':

                    i = self.find_index(calling_node, i)

                    self.weights = np.concatenate(
                        (self.weights[:i], self.weights[i+1:]), 0)

    def compute(self):
        
        if self.direction == 'forward':

            x = self.input if self.type == 'input' else self.in_node.output

            if self.function == 'none':
                pass

            elif self.function == 'pool':
                x = self.pool(x) if self.d != 2 else \
                    self.pool(x, self.pool_factor)

            else: # self.function == 'convolve':
                x = F.pad(x, self.padding)
                x = self.conv(x)
                x = F.relu(x)

            self.output = x
                
        else: # self.direction == 'backward':

            x = self.output if self.type == 'output' else \
                torch_cat([out_node.input for out_node in self.out_nodes], 1)

            if self.function == 'pool':
                x = self.pool_T(x, scale_factor=self.kernel_size)
                
            else: # self.function == 'convolve':
                x = self.conv_T(x)
                x = F.pad(x, neg(self.padding))
                x = F.relu(x)

            self.input = x