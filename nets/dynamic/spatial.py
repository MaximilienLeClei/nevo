import random

import numpy as np
from nets.dynamic.base import DynamicNetBase


class BiasVector:
    def __init__(self):

        self.data = np.empty((0, 1))

        self.nodes = {"pos": []}

    def append_input(self, input_node_pos):

        self.nodes["pos"].append(input_node_pos)

        self.data = np.vstack((self.data, 0))

    def append(self, node_pos):

        self.nodes["pos"].append(node_pos)

        self.data = np.vstack((self.data, np.random.randn(1)))

    def remove(self, node_pos):

        node_index = self.nodes["pos"].index(node_pos)

        self.nodes["pos"].remove(node_pos)

        self.data = np.delete(self.data, node_index, axis=0)

    def mutate(self):

        self.data += 0.01 * (self.data != 0)

    def to_vector(self):

        return self.data


class WeightMatrix:
    def __init__(self):

        self.row = np.array([])
        self.col = np.array([])
        self.data = np.array([])

        self.nodes = {"pos": []}

    def append_input(self, input_node_pos):

        self.nodes["pos"].append(input_node_pos)

    def append(self, in_node_pos, out_node_pos=None):

        in_node_index = self.nodes["pos"].index(in_node_pos)

        if out_node_pos not in self.nodes["pos"]:
            self.nodes["pos"].append(out_node_pos)

        out_node_index = self.nodes["pos"].index(out_node_pos)

        self.row = np.append(self.row, in_node_index)
        self.col = np.append(self.col, out_node_index)
        self.data = np.append(self.data, np.random.randn(1))

    def remove(self, node_pos):

        node_index = self.nodes["pos"].index(node_pos)
        self.nodes["pos"].remove(node_pos)

        self.data = np.delete(self.data, np.where(self.row == node_index))
        self.col = np.delete(self.col, np.where(self.row == node_index))
        self.row = np.delete(self.row, np.where(self.row == node_index))

        self.data = np.delete(self.data, np.where(self.col == node_index))
        self.row = np.delete(self.row, np.where(self.col == node_index))
        self.col = np.delete(self.col, np.where(self.col == node_index))

        self.row -= self.row > node_index
        self.col -= self.col > node_index

    def mutate(self):

        self.data += 0.01 * np.random.randn(*self.data.shape)

    def to_sparse_matrix(self):

        return coo_matrix(
            (self.data, (self.row, self.col)),
            shape=(len(self.nodes["pos"]), len(self.nodes["pos"])),
        )


def node_dict(nb_input_spaces, d_output):

    if not isinstance(d_output, int):
        d_output = 0

    nodes = {}

    nodes["pos"] = {}

    nodes["pos"]["all"] = []
    nodes["pos"]["input"] = []
    nodes["pos"]["hidden"] = []
    nodes["pos"]["hidden & output"] = []
    nodes["pos"]["output"] = [None] * d_output

    nodes["pos"]["visible"] = []
    nodes["pos"]["visible inputs"] = []
    nodes["pos"]["visible spaced inputs"] = np.empty((nb_input_spaces, 0)).tolist()

    nodes["pos"]["being pruned"] = []
    nodes["pos"]["in"] = {}
    nodes["pos"]["out"] = {}
    nodes["pos"]["connected with to"] = {}

    nodes["input space"] = {}

    return nodes


def generate_pos(n_0_pos, n_1_pos):

    direction = np.random.choice(["forward", "backward"])

    forward_point, backward_point = compute_furthest_z_equally_distant_points(
        n_0_pos, n_1_pos
    )

    if backward_point[-1] <= 0 or direction == "forward":
        n_2_pos = forward_point
    else:  # backward_point[-1] > 0 and direction == 'backward'
        n_2_pos = backward_point

    n_2_pos += 1e-2 * np.random.randn(3)

    n_2_pos = np.around(n_2_pos, decimals=4)

    return tuple(n_2_pos)


def compute_nb_input_spaces(d_input):

    input_height, input_width = d_input

    if input_height != input_width:
        raise Exception("Input height needs to be equal to input width.")

    if int(bin(input_height)[3:]) != 0:
        raise Exception("Input height needs to be a power of 2.")

    return (len(bin(input_height)[2:]) - 1) * 2 + 1


def compute_input_space_dimensions(d_input):

    nb_input_spaces = compute_nb_input_spaces(d_input)

    height, width = d_input

    height_sizes = [
        int(height / (2 ** (k // 2 + k % 2))) for k in range(nb_input_spaces)
    ]  # [8, 4, 4, 2, 2, 1, 1]
    width_sizes = [
        int(width / (2 ** (k // 2))) for k in range(nb_input_spaces)
    ]  # [8, 8, 4, 4, 2, 2, 1]

    return zip(height_sizes, width_sizes)


def compute_nb_input_spaces_and_d_input(x):

    total = 0
    nb_input_spaces = 0

    found = False

    d_input = [1, 1]

    while not found:

        if nb_input_spaces > 0:

            if nb_input_spaces % 2 == 0:
                d_input[1] *= 2
            else:  # nb_input_spaces % 2 == 1:
                d_input[0] *= 2

        total += 2**nb_input_spaces

        nb_input_spaces += 1

        if len(x) == total:
            found = True

    return nb_input_spaces, d_input


def reshape_inputs(x):

    X = x.flatten("C")

    nb_input_spaces = compute_nb_input_spaces(x.shape)

    for i in range(nb_input_spaces - 1):

        height, width = x.shape

        if i % 2 == 0:

            x = cv2.resize(x, (width, height // 2))
            X = np.concatenate((X, x.flatten("C")))

        else:  # i % 2 == 1:

            x = cv2.resize(x, (width // 2, height))
            X = np.concatenate((X, x.flatten("C")))

    return np.expand_dims(X, 1)


def reshape_outputs(x):

    nb_input_spaces, d_input = compute_nb_input_spaces_and_d_input(x)

    height, width = d_input

    X = np.zeros(d_input)

    i = j = k = 0
    l = height * width

    for s in range(nb_input_spaces):

        X += cv2.resize(
            x[k : k + l].reshape(height // 2**j, width // 2**i),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

        if s % 2 == 0:
            j += 1
        else:  # s % 2 == 1:
            i += 1

        k += l
        l //= 2

    return X


def compute_furthest_z_equally_distant_points(A, B):

    A, B = np.copy(A), np.copy(B)

    if A[-1] < B[-1]:
        A, B = B, A

    elif A[-1] == B[-1]:
        A[-1] += 1e-6

    if np.array_equal(A[:-1], B[:-1]):
        B[:-1] += 1e-6

    D = np.hstack((B[:-1], A[-1]))

    AB = B - A
    DB = B - D

    cross_AB_DB = np.cross(AB, DB)
    k = cross_AB_DB / np.linalg.norm(cross_AB_DB)
    v = AB

    v_rot = (
        v * np.cos(2 * np.pi / 6)
        + np.cross(k, v) * np.sin(2 * np.pi / 6)
        + v * np.dot(k, v) * (1 - np.cos(2 * np.pi / 6))
    )

    return B - v_rot, A + v_rot


class Net(DynamicNetBase):
    def __init__(self, d_input, d_output=None):

        self.d_input = d_input
        self.d_output = d_input if d_output == None else d_output

        self.nb_input_spaces = compute_nb_input_spaces(self.d_input)

        self.height, self.width = d_input

        self.nodes = node_dict(self.nb_input_spaces, self.d_output)

        self.nb_passes = 1

        self.weights = WeightMatrix()
        self.biases = BiasVector()

        self.h = np.empty((0, 1))

        self.architectural_mutations = [self.grow_node, self.prune_node]

    def initialize_architecture(self):

        self.grow_input_nodes()
        self.grow_node()

    def mutate_parameters(self):

        self.weights.mutate()
        self.biases.mutate()

        self.nb_passes = one_percent_change(self.nb_passes)

    def grow_input_nodes(self):

        for input_space, (height_size, width_size) in enumerate(
            compute_input_space_dimensions(self.d_input)
        ):

            for i in range(height_size):
                for j in range(width_size):

                    n_2_x = (j + 0.5) * (self.width / width_size)
                    n_2_y = self.height - (i + 0.5) * (self.height / height_size)
                    n_2_pos = (n_2_x, n_2_y, 0)

                    self.nodes["pos"]["all"].append(n_2_pos)
                    self.nodes["pos"]["input"].append(n_2_pos)

                    self.nodes["pos"]["in"][n_2_pos] = []
                    self.nodes["pos"]["out"][n_2_pos] = []
                    self.nodes["pos"]["connected with to"][n_2_pos] = []

                    self.nodes["input space"][n_2_pos] = input_space

                    self.weights.append_input(n_2_pos)
                    self.biases.append_input(n_2_pos)

                    if input_space >= self.nb_input_spaces - 2:

                        self.nodes["pos"]["visible"].append(n_2_pos)
                        self.nodes["pos"]["visible inputs"].append(n_2_pos)
                        self.nodes["pos"]["visible spaced inputs"][input_space].append(
                            n_2_pos
                        )

                    if self.height != height_size or self.width != width_size:

                        if height_size != width_size:
                            n_0_index = (
                                len(self.nodes["pos"]["all"])
                                - (height_size * width_size * 2)
                                + i * width_size
                                - 1
                            )
                            n_1_index = (
                                len(self.nodes["pos"]["all"])
                                - (height_size * width_size * 2)
                                + (i + 1) * width_size
                                - 1
                            )
                        else:  # height_size == width_size:
                            n_0_index = (
                                len(self.nodes["pos"]["all"])
                                - (height_size * width_size * 2)
                                + (i * height_size + j)
                                - 1
                            )
                            n_1_index = (
                                len(self.nodes["pos"]["all"])
                                - (height_size * width_size * 2)
                                + (i * height_size + j)
                            )

                        n_0_pos, n_1_pos = (
                            self.nodes["pos"]["all"][n_0_index],
                            self.nodes["pos"]["all"][n_1_index],
                        )

                        self.nodes["pos"]["in"][n_2_pos].extend([n_0_pos, n_1_pos])
                        self.nodes["pos"]["out"][n_0_pos].append(n_2_pos)
                        self.nodes["pos"]["out"][n_1_pos].append(n_2_pos)

    def grow_node(self):

        n_0_pos = random.choice(self.nodes["pos"]["visible"])

        n_1_pos = self.find_node_closest_to(n_0_pos, "connect with")

        if n_1_pos == None:
            return

        n_2_pos = generate_pos(n_0_pos, n_1_pos)

        self.nodes["pos"]["all"].append(n_2_pos)
        self.nodes["pos"]["hidden"].append(n_2_pos)
        self.nodes["pos"]["hidden & output"].append(n_2_pos)
        self.nodes["pos"]["visible"].append(n_2_pos)

        self.nodes["pos"]["in"][n_2_pos] = [n_0_pos, n_1_pos]
        self.nodes["pos"]["out"][n_0_pos].append(n_2_pos)
        self.nodes["pos"]["out"][n_1_pos].append(n_2_pos)
        self.nodes["pos"]["out"][n_2_pos] = []
        self.nodes["pos"]["connected with to"][n_0_pos].append([n_1_pos, n_2_pos])
        self.nodes["pos"]["connected with to"][n_1_pos].append([n_0_pos, n_2_pos])
        self.nodes["pos"]["connected with to"][n_2_pos] = []

        self.h = np.vstack((self.h, 0))

        if None in self.nodes["pos"]["output"]:
            self.swap_hidden_and_output_node(n_2_pos)

        elif n_0_pos in self.nodes["pos"]["output"]:
            self.swap_hidden_and_output_node(n_2_pos, n_0_pos)

        elif n_1_pos in self.nodes["pos"]["output"]:
            self.swap_hidden_and_output_node(n_2_pos, n_1_pos)

        self.weights.append(n_0_pos, n_2_pos)
        self.weights.append(n_1_pos, n_2_pos)

        n_3_pos = self.find_node_closest_to(n_2_pos, "connect to")

        if n_3_pos != None:

            self.weights.append(n_2_pos, n_3_pos)

            self.nodes["pos"]["out"][n_2_pos].append(n_3_pos)
            self.nodes["pos"]["in"][n_3_pos].append(n_2_pos)

        for in_node_pos in [n_0_pos, n_1_pos]:

            if in_node_pos in self.nodes["pos"]["input"]:

                for in_node_in_node_pos in self.nodes["pos"]["in"][in_node_pos]:

                    if in_node_in_node_pos not in self.nodes["pos"]["visible"]:

                        self.nodes["pos"]["visible"].append(in_node_in_node_pos)
                        self.nodes["pos"]["visible inputs"].append(in_node_in_node_pos)
                        input_space = self.nodes["input space"][in_node_in_node_pos]
                        self.nodes["pos"]["visible spaced inputs"][input_space].append(
                            in_node_in_node_pos
                        )

        self.biases.append(n_2_pos)

    def prune_node(self, node_pos=None):

        if node_pos == None:

            if len(self.nodes["pos"]["hidden & output"]) == 0:
                return

            node_pos = random.choice(self.nodes["pos"]["hidden & output"])

        if node_pos in self.nodes["pos"]["being pruned"]:
            return

        self.nodes["pos"]["being pruned"].append(node_pos)

        if node_pos in self.nodes["pos"]["output"]:
            self.swap_hidden_and_output_node(original_output_node_pos=node_pos)

        for i in range(len(self.nodes["pos"]["out"][node_pos]) - 1, -1, -1):
            self.prune_connection(
                node_pos, self.nodes["pos"]["out"][node_pos][i], node_pos
            )

        for i in range(len(self.nodes["pos"]["in"][node_pos]) - 1, -1, -1):
            self.prune_connection(
                self.nodes["pos"]["in"][node_pos][i], node_pos, node_pos
            )

        self.weights.remove(node_pos)
        self.biases.remove(node_pos)

        self.h = np.delete(
            self.h,
            self.nodes["pos"]["hidden & output"].index(node_pos),
            axis=0,
        )

        self.nodes["pos"]["all"].remove(node_pos)
        self.nodes["pos"]["hidden"].remove(node_pos)
        self.nodes["pos"]["hidden & output"].remove(node_pos)
        self.nodes["pos"]["visible"].remove(node_pos)
        self.nodes["pos"]["being pruned"].remove(node_pos)

        del self.nodes["pos"]["in"][node_pos]
        del self.nodes["pos"]["out"][node_pos]
        del self.nodes["pos"]["connected with to"][node_pos]

        if len(self.nodes["pos"]["hidden & output"]) == 0:
            self.grow_node()

    def prune_connection(self, in_node_pos, out_node_pos, calling_node_pos):

        if not out_node_pos in self.nodes["pos"]["out"][in_node_pos]:
            return

        self.nodes["pos"]["out"][in_node_pos].remove(out_node_pos)
        self.nodes["pos"]["in"][out_node_pos].remove(in_node_pos)

        for in_node_connected_with_pos, in_node_connected_to_pos in self.nodes["pos"][
            "connected with to"
        ][in_node_pos]:
            if in_node_connected_to_pos == out_node_pos:
                nodes_connected_with_to_pos = [
                    in_node_connected_with_pos,
                    in_node_connected_to_pos,
                ]
                self.nodes["pos"]["connected with to"][in_node_pos].remove(
                    nodes_connected_with_to_pos
                )

        if calling_node_pos == in_node_pos:

            if out_node_pos in self.nodes["pos"]["hidden & output"]:

                if len(self.nodes["pos"]["in"][out_node_pos]) == 0:

                    self.prune_node(out_node_pos)

        else:  # calling_node_pos == out_node_pos:

            if in_node_pos in self.nodes["pos"]["hidden"]:

                if len(self.nodes["pos"]["out"][in_node_pos]) == 0:

                    self.prune_node(in_node_pos)

            elif in_node_pos in self.nodes["pos"]["input"]:

                for in_node_out_node_pos in self.nodes["pos"]["out"][in_node_pos]:

                    if in_node_out_node_pos in self.nodes["pos"]["hidden & output"]:

                        return

                for in_node_in_node_pos in self.nodes["pos"]["in"][in_node_pos]:

                    if in_node_in_node_pos in self.nodes["pos"]["input"]:

                        if len(self.nodes["pos"]["out"][in_node_in_node_pos]) == 1:

                            self.nodes["pos"]["visible"].remove(in_node_in_node_pos)
                            self.nodes["pos"]["visible inputs"].remove(
                                in_node_in_node_pos
                            )
                            input_space = self.nodes["input space"][in_node_in_node_pos]
                            self.nodes["pos"]["visible spaced inputs"][
                                input_space
                            ].remove(in_node_in_node_pos)

    def find_node_closest_to(self, node_pos, action):

        if node_pos in self.nodes["pos"]["input"]:
            input_space = self.nodes["input space"][node_pos]
            potential_nodes_pos = self.nodes["pos"]["visible spaced inputs"][
                input_space
            ].copy()

        else:  # node_pos in self.nodes['pos']['hidden & output']:
            potential_nodes_pos = self.nodes["pos"]["hidden & output"].copy()

        if self.d_output == self.d_input and action == "connect to":
            potential_nodes_pos.extend(self.nodes["pos"]["visible inputs"])

        equal_node_pos = np.equal(potential_nodes_pos, node_pos).all(axis=1)
        potential_nodes_pos = np.delete(
            potential_nodes_pos, np.argwhere(equal_node_pos), axis=0
        )

        if action == "connect with":

            for node_connected_with_pos, _ in self.nodes["pos"]["connected with to"][
                node_pos
            ]:
                equal_connected_with = (
                    potential_nodes_pos == node_connected_with_pos
                ).all(axis=1)
                potential_nodes_pos = np.delete(
                    potential_nodes_pos,
                    np.argwhere(equal_connected_with),
                    axis=0,
                )

        else:  # action == 'connect to':

            for out_node_pos in self.nodes["pos"]["out"][node_pos]:
                equal_out_node_pos = (potential_nodes_pos == out_node_pos).all(axis=1)
                potential_nodes_pos = np.delete(
                    potential_nodes_pos,
                    np.argwhere(equal_out_node_pos),
                    axis=0,
                )

        if potential_nodes_pos.size == 0:
            return

        broadcasted_node_pos = np.ones([len(potential_nodes_pos), 3]) * node_pos
        squared_distances = np.sum(
            (broadcasted_node_pos - potential_nodes_pos) ** 2, axis=1
        )
        closest_nodes_indices = np.argwhere(
            np.equal(squared_distances, squared_distances.min())
        ).squeeze(1)

        random_closest_node_index = random.choice(closest_nodes_indices)

        return tuple(potential_nodes_pos[random_closest_node_index])

    def swap_hidden_and_output_node(
        self, original_hidden_node_pos=None, original_output_node_pos=None
    ):

        if original_hidden_node_pos == None:

            original_output_node_in_hidden_nodes_pos = []

            for original_output_node_in_node_pos in self.nodes["pos"]["in"][
                original_output_node_pos
            ]:
                if original_output_node_in_node_pos in self.nodes["pos"]["hidden"]:
                    original_output_node_in_hidden_nodes_pos.append(
                        original_output_node_in_node_pos
                    )

            if len(original_output_node_in_hidden_nodes_pos) > 0:
                original_hidden_node_pos = random.choice(
                    original_output_node_in_hidden_nodes_pos
                )

        if original_hidden_node_pos != None:

            self.nodes["pos"]["hidden"].remove(original_hidden_node_pos)

        if original_output_node_pos != None:

            self.nodes["pos"]["hidden"].append(original_output_node_pos)

            output_index = self.nodes["pos"]["output"].index(original_output_node_pos)

        else:

            output_index = random.choice(
                np.where(np.array(self.nodes["pos"]["output"]) == None)[0]
            )

        self.nodes["pos"]["output"][output_index] = original_hidden_node_pos

    def setup_to_run(self):

        self.w = self.weights.to_sparse_matrix()
        self.b = self.biases.to_vector()

    def setup_to_save(self):

        self.w = None
        self.b = None

    def reset(self):

        self.h = np.zeros((len(self.nodes["pos"]["hidden & output"]), 1))

    def __call__(self, x):

        X = reshape_inputs(x)

        for _ in range(self.nb_passes):

            x = np.concatenate((X, self.h))

            x = self.w @ x + self.b

            x = np.clip(x, 0, 2**31 - 1)

            self.h = x[len(self.nodes["pos"]["input"]) :]

        if self.d_output == self.d_input:

            out = reshape_outputs(x[: len(self.nodes["pos"]["input"])])

        else:

            out = np.zeros(self.d_output)

            for i in range(self.d_output):
                if self.nodes["pos"]["output"][i] != None:
                    out[i] = x[
                        self.nodes["pos"]["all"].index(self.nodes["pos"]["output"][i])
                    ]

        return out
