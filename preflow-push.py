import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.widgets as widget
import matplotlib.patches as mpatches
import threading
from networkx.algorithms.flow import preflow_push, build_residual_network

DEFAULT_NODE_COLOR = "#7dabf5"
RED_NODE_COLOR = '#ff8375'
GREEN_NODE_COLOR = '#5df252'
YELLOW_NODE_COLOR = '#faea5f'
WHITE_NODE_COLOR = '#a3a3a3'

animation_delay = 0.25
is_running = False
is_paused = False
G = None
edges = []
pos = []


def my_draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
    rad=0
):
    """Draw edge labels.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5*pos_1 + 0.5*pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0,1), (-1,0)])
        ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
        ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
        ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
        bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items

ev = threading.Event()

def run(event):
    global is_running
    if is_running == True:
        return
    is_running = True
    global G, ev
    preflow_push_visualize(G, 's', 't', ev)
    plt.show()

def reset(event):
    global is_running, res
    ev.set()
    plt.pause(0.1)
    is_running = False
    ax.cla()
    excessax.cla()
    excessax.axis('off')

    ax.set_title('Maximum Flow Graph')

    nx.draw_networkx_nodes(res, pos=pos, ax=ax, node_color=DEFAULT_NODE_COLOR, node_size=320)
    nx.draw_networkx_labels(res, pos=pos, ax=ax)
    labels = nx.get_edge_attributes(res,'capacity')
    # outlines = [plt.Circle(pos[node], 0.055, color='black') for node in res.nodes()]
    # for p in outlines:
    #     ax.add_patch(p)
    # nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels, ax=netax)
    draw_network_edges(res, ax=ax, Gpos=pos, zero=True, label_size=8)

    ev.clear()
    plt.draw()

def pause(event):
    global is_paused
    if is_paused:
        is_paused = False
    else:
        is_paused = True

def set_delay(event):
    global animation_delay
    global text_delay
    delay = text_delay.text
    if delay.replace('.', '', 1).isdigit() == False:
        return
    animation_delay = float(delay)
    print(animation_delay)

def set_example1(event):
    global is_running
    if is_running:
        return
    global G, pos, ax, netax, edges, res
    edges = [('s', 1, {"capacity": 16}), ('s', 3, {"capacity": 13}), (1, 2, {"capacity": 12}), (3, 1, {"capacity": 4}),
                (2, 3, {"capacity": 9}), (3, 4, {"capacity": 14}), (4, 2, {"capacity": 7}), (4, 't', {"capacity": 4}),
                (2, 't', {"capacity": 4})]
    G = nx.DiGraph()
    G.add_nodes_from(['s', 't'])
    G.add_edges_from(edges)
    res = build_residual_network(G, capacity='capacity')
    pos = nx.spring_layout(res, seed=5)

    ax.cla()
    netax.cla()
    ax.set_title('Maximum Flow Graph')
    netax.set_title('Original Network G')

    nx.draw_networkx(G, pos=pos, ax=netax, node_color=DEFAULT_NODE_COLOR, node_size=320)
    draw_network_edges(G, ax=netax, Gpos=pos, label_size=8)

    nx.draw_networkx_nodes(res, pos=pos, ax=ax, node_color=DEFAULT_NODE_COLOR, node_size=320)
    nx.draw_networkx_labels(res, pos=pos, ax=ax)
    draw_network_edges(res, ax=ax, Gpos=pos, zero=True, label_size=8)
        
def set_example2(event):
    global is_running
    if is_running:
        return
    global G, pos, ax, netax, edges, res
    edges = [('s', 1, {"capacity": 2}), ('s', 2, {"capacity": 1}), (1, 2, {"capacity": 3}), (1, 't', {"capacity": 1}),
            (2, 't', {"capacity": 2})]
    G = nx.DiGraph()
    G.add_nodes_from(['s', 't'])
    G.add_edges_from(edges)
    res = build_residual_network(G, capacity='capacity')
    pos = nx.spring_layout(res, seed=123771164)
    
    ax.cla()
    netax.cla()
    ax.set_title('Maximum Flow Graph')
    netax.set_title('Original Network G')

    nx.draw_networkx(G, pos=pos, ax=netax, node_color=DEFAULT_NODE_COLOR, node_size=320)
    draw_network_edges(G, ax=netax, Gpos=pos, label_size=8)

    nx.draw_networkx_nodes(res, pos=pos, ax=ax, node_color=DEFAULT_NODE_COLOR, node_size=320)
    nx.draw_networkx_labels(res, pos=pos, ax=ax)
    draw_network_edges(res, ax=ax, Gpos=pos, zero=True, label_size=8)

def set_example3(event):
    global is_running
    if is_running:
        return
    global G, pos, ax, netax, edges, res
    edges = [('s', 1, {"capacity": 23}), ('s', 2, {"capacity": 12}), (1, 2, {"capacity": 10}), (1, 't', {"capacity": 19}),
            (2, 4, {"capacity": 3}), (3, 4, {"capacity": 14}), (4, 1, {"capacity": 6}), (5, 3, {"capacity": 28}), (1, 5, {"capacity": 4})]
    G = nx.DiGraph()
    G.add_nodes_from(['s', 't'])
    G.add_edges_from(edges)
    res = build_residual_network(G, capacity='capacity')
    pos = nx.spring_layout(res, seed=46585)
    
    ax.cla()
    netax.cla()
    ax.set_title('Maximum Flow Graph')
    netax.set_title('Original Network G')

    nx.draw_networkx(G, pos=pos, ax=netax, node_color=DEFAULT_NODE_COLOR, node_size=320)
    draw_network_edges(G, ax=netax, Gpos=pos, label_size=8)

    nx.draw_networkx_nodes(res, pos=pos, ax=ax, node_color=DEFAULT_NODE_COLOR, node_size=320)
    nx.draw_networkx_labels(res, pos=pos, ax=ax)
    draw_network_edges(res, ax=ax, Gpos=pos, zero=True, label_size=8)

def set_example4(event):
    global is_running
    if is_running:
        return
    global G, pos, ax, netax, edges, res
    edges = [('s', 1, {"capacity": 23}), (1, 2, {"capacity": 14}), (2, 3, {"capacity": 52}), (3, 4, {"capacity": 9}), (4, 't', {"capacity": 18})]
    G = nx.DiGraph()
    G.add_nodes_from(['s', 't'])
    G.add_edges_from(edges)
    res = build_residual_network(G, capacity='capacity')
    pos = nx.spring_layout(res, seed=3984561)

    ax.cla()
    netax.cla()
    ax.set_title('Maximum Flow Graph')
    netax.set_title('Original Network G')

    nx.draw_networkx(G, pos=pos, ax=netax, node_color=DEFAULT_NODE_COLOR, node_size=320)
    draw_network_edges(G, ax=netax, Gpos=pos, label_size=8)

    nx.draw_networkx_nodes(res, pos=pos, ax=ax, node_color=DEFAULT_NODE_COLOR, node_size=320)
    nx.draw_networkx_labels(res, pos=pos, ax=ax)
    draw_network_edges(res, ax=ax, Gpos=pos, zero=True, label_size=8)

fig, ax = plt.subplots()
ax.axis('equal')
fig.set_figwidth(15)
fig.subplots_adjust(bottom=0.2, right=0.4, left=0.01)

edges = [('s', 1, {"capacity": 16}), ('s', 3, {"capacity": 13}), (1, 2, {"capacity": 12}), (3, 1, {"capacity": 4}),
            (2, 3, {"capacity": 9}), (3, 4, {"capacity": 14}), (4, 2, {"capacity": 7}), (4, 't', {"capacity": 4}),
            (2, 't', {"capacity": 4})]
G = nx.DiGraph()
G.add_nodes_from(['s', 't'])
G.add_edges_from(edges)
# netax = fig.add_axes([0.41, 0.2, 0.38, 0.68])
netax = fig.add_axes([0.61, 0.2, 0.38, 0.68])
netax.axis('equal')
# labelax = fig.add_axes([0.8, 0.2, 0.17, 0.68])
labelax = fig.add_axes([0.41, 0.2, 0.17, 0.68])
labelax.axis('equal')

labelax.yaxis.tick_right()
labelax.set_xticks([])
labelax.set_yticks([1,10])

excessax = fig.add_axes([0.01, 0.1, 0.2, 0.075])
excessax.axis('off')

ax.set_title('Maximum Flow Graph')
netax.set_title('Original Network G')
labelax.set_title('Heights')

res = build_residual_network(G, capacity='capacity')
pos = nx.spring_layout(res, seed=5)

def animate_frame(wait=1, pre_animate=lambda:(), ax=None):
    ax.cla()
    pre_animate()
    plt.pause(wait)
    plt.draw()

def draw_network_edges(M, zero=False, ax=None, label_size=10, Gpos=pos, label=True):
    edge_capacities = nx.get_edge_attributes(M,'capacity')
    edge_flows = nx.get_edge_attributes(M, 'flow')
    # print("Capacity:", edge_capacities)
    # print("Flow:", edge_flows)

    curved_edges = []
    straight_edges = []
    # print(M.edges())
    if zero:
        curved_edges = [edge for edge in M.edges() if (reversed(edge) in M.edges())]
        straight_edges = list(set(M.edges()) - set(curved_edges))
    else:
        curved_edges = [edge for edge in M.edges() if ((reversed(edge) in M.edges()) and edge_capacities[edge] != 0 and edge_capacities[tuple(reversed(edge))] != 0)]
        straight_edges = [edge for edge in list(set(G.edges()) - set(curved_edges)) if edge_capacities[edge] != 0]
    # print("Curved:", curved_edges)
    # print("Straight:", straight_edges)

    nx.draw_networkx_edges(M, Gpos, ax=ax, edgelist=straight_edges)
    arc_rad = 0.25
    nx.draw_networkx_edges(M, Gpos, ax=ax, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}')
    # print(edge_flows)

    if label == True:
        if zero:
            curved_edge_labels = {edge: str(0 if edge_flows.get(edge) == None else edge_flows.get(edge)) + '/' + str(edge_capacities[edge]) for edge in curved_edges}
            straight_edge_labels = {edge: str(0 if edge_flows.get(edge) == None else edge_flows.get(edge)) + '/' + str(edge_capacities[edge]) for edge in straight_edges}
        else:
            curved_edge_labels = {edge: str(0 if edge_flows.get(edge) == None else edge_flows.get(edge)) + '/' + str(edge_capacities[edge]) for edge in curved_edges if edge_capacities[edge] != 0}
            straight_edge_labels = {edge: str(0 if edge_flows.get(edge) == None else edge_flows.get(edge)) + '/' + str(edge_capacities[edge]) for edge in straight_edges if edge_capacities[edge] != 0}
        my_draw_networkx_edge_labels(M, Gpos, ax=ax, edge_labels=curved_edge_labels,rotate=False,rad=arc_rad, font_size=label_size)
        nx.draw_networkx_edge_labels(M, Gpos, ax=ax, edge_labels=straight_edge_labels,rotate=False, font_size=label_size)
    plt.draw()

def update_network(G, zero=False, ax=None, new_attributes=dict()):
    # print(new_attributes)
    nx.set_edge_attributes(G, new_attributes)

    pos = nx.spring_layout(G, seed=5)
    nx.draw_networkx_nodes(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)

    draw_network_edges(G, ax=ax)


nx.draw_networkx(G, pos=pos, ax=netax, node_color=DEFAULT_NODE_COLOR, node_size=320)
# labels = nx.get_edge_attributes(G,'capacity')
# nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels, ax=netax)
draw_network_edges(G, ax=netax, Gpos=pos, label_size=8)

nx.draw_networkx_nodes(res, pos=pos, ax=ax, node_color=DEFAULT_NODE_COLOR, node_size=320)
nx.draw_networkx_labels(res, pos=pos, ax=ax)
# outlines = [plt.Circle(pos[node], 0.01+0.02*len(G.nodes()), color='black') for node in res.nodes()]
# for p in outlines:
#     ax.add_patch(p)
# nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels, ax=netax)
draw_network_edges(res, ax=ax, Gpos=pos, zero=True, label_size=8)
# print(res.edges())

def preflow_push_visualize(G, s, t, ev):

    def update():
        ax.set_title('Maximum Flow Graph')
        new_edge_attributes = {edge: {"capacity": capacity[edge], "flow": flow[edge]} for edge in res_edges}
        nx.set_edge_attributes(res, new_edge_attributes)

        
        no_excess_nodes = [v for v in res_nodes if excess[v] == 0 or v == s or v == t]
        black_nodes = list(set(res_nodes)-set(no_excess_nodes)-set(relabeling_nodes)-set(pushing_nodes)-({curr_node} if curr_node != None and curr_node not in relabeling_nodes else set()))
        # print(black_nodes)
        nx.draw_networkx_nodes(res, pos, ax=ax, nodelist=black_nodes, node_color=DEFAULT_NODE_COLOR)
        nx.draw_networkx_nodes(res, pos, ax=ax, nodelist=no_excess_nodes, node_color=WHITE_NODE_COLOR)
        nx.draw_networkx_nodes(res, pos, ax=ax, nodelist=relabeling_nodes, node_color=RED_NODE_COLOR)
        nx.draw_networkx_nodes(res, pos, ax=ax, nodelist=pushing_nodes, node_color=GREEN_NODE_COLOR)
        if curr_node != None and curr_node not in relabeling_nodes:
            nx.draw_networkx_nodes(res, pos, ax=ax, nodelist=[curr_node], node_color=YELLOW_NODE_COLOR)
        nx.draw_networkx_labels(res, pos, ax=ax)

        # new_node_attributes = {node: excess[node] for node in res_nodes}
        # nx.draw_networkx_labels(res, {n:[pos[n][0]-0.05, pos[n][1]+0.05] for n in res_nodes}, labels=new_node_attributes, ax=ax, horizontalalignment='right')
        # nx.set_node_attributes(res, new_node_attributes)

        # outlines = [plt.Circle(pos[node], 0.01+0.02*len(G.nodes()), color='black') for node in res_nodes]
        
        # for p in outlines:
        #     ax.add_patch(p)

        labelax.cla()
        labelax.set_title('Heights')
        label_G = nx.DiGraph()
        label_G.add_nodes_from([(n, {'height': height[n]}) for n in res_nodes])
        label_G.add_edges_from(G_edges)
        nx.set_edge_attributes(label_G, new_edge_attributes)
        label_pos = {n: [pos[n][0], height[n]] for n in res_nodes}
        node_list = [n for n in res_nodes]
        node_list.sort(key=lambda n: pos[n][0])
        x = -1
        increment = 6/len(res_nodes)
        for n in node_list:
            label_pos[n] = [x, label_pos[n][1]]
            x += increment

        nx.draw_networkx_nodes(label_G, pos=label_pos, ax=labelax, node_color=DEFAULT_NODE_COLOR, node_size=1200/len(res_nodes))
        nx.draw_networkx_labels(label_G, pos=label_pos, ax=labelax, font_size=70/len(res_nodes))
        
        draw_network_edges(label_G, Gpos=label_pos, ax=labelax, label=False)

        # nx.draw_networkx_edges(label_G, label_pos, ax=labelax, edgelist=res_edges)
        # print(pos)
        # print(label_pos)

        excessax.cla()
        excessax.axis('off')
        excessax.text(0, 0.5, 'excess: '+str({node: excess[node] for node in res_nodes}), fontsize=16)
        excessax.text(0, -0.3, 'height: '+str({node: height[node] for node in res_nodes}), fontsize=16)
        # excessax.text(0, -0.3, 'delay: '+str(animation_delay), fontsize=10)
        draw_network_edges(res, ax=ax, Gpos=pos, label_size=8)

    global is_running, is_paused, animation_delay

    res = build_residual_network(G, capacity='capacity')

    G_edges = G.edges()

    res_nodes = res.nodes()
    res_edges = res.edges()

    forward_edges = [e for e in res.edges() if e in G_edges]
    backward_edges = list(set(res_edges) - set(forward_edges))
    # print("Forward:", forward_edges)
    # print("Backward:", backward_edges)

    n = len(res_nodes)
    height = {s: n}
    capacity = nx.get_edge_attributes(res, 'capacity')
    res_capacity = capacity.copy()
    capacity = {e: capacity[e] if capacity[e] != 0 else capacity[tuple(reversed(e))] for e in res_edges}
    flow = dict()
    excess = {v: 0 for v in res_nodes}

    relabeling_nodes = []
    pushing_nodes = []
    curr_node = None

    def push(v, w, e, amount):
        while is_paused:
            plt.pause(1)

        flow[e] += amount
        flow[tuple(reversed(e))] -= amount
        excess[v] -= amount
        excess[w] += amount
        

    def calculate_excess(v):
        ex = sum(flow[edge] if edge[1] == v else -flow[edge] if edge[0] == v else 0 for edge in res_edges)
        return ex

    for e in res_edges:
        flow[e] = 0

    for v in res_nodes:
        if s != v:
            height[v] = 0
        excess[v] = calculate_excess(v)

    # Begin animation
    animate_frame(wait=animation_delay, pre_animate=update, ax=ax)

    push_edges = forward_edges.copy()
    curr_node = s
    for e in res_edges:
        if ev.is_set() or is_running == False:
            excessax.cla()
            return
        if e[0] == s:
            # Saturate all edges from s
            # flow[e] = capacity[e]
            pushing_nodes.append(e[1])
            reversed_edge = tuple(reversed(e))
            push(s, e[1], e, capacity[e])
            if flow[e] == capacity[e] and e in push_edges:
                push_edges.remove(e)
            if flow[reversed_edge] == 0 and reversed_edge in push_edges:
                push_edges.remove(reversed_edge)
            if flow[e] < capacity[e] and e not in push_edges:
                push_edges.append(e)
            if flow[reversed_edge] < 0 and reversed_edge not in push_edges:
                push_edges.append(reversed_edge)
            animate_frame(wait=animation_delay, pre_animate=update, ax=ax)
            pushing_nodes.remove(e[1])
    
    def relabel(v):
        while is_paused:
            plt.pause(1)
        animate_frame(wait=animation_delay, pre_animate=update, ax=ax)
        relabeling_nodes.append(v)
        height[v] += 1
        animate_frame(wait=animation_delay, pre_animate=update, ax=ax)
        relabeling_nodes.remove(v)
        # animate_frame()

    # print("Heights:", height)
    # print("Capacity:", capacity)
    # print("Flow:", flow)
    # print("Excess:", excess)

    # excess[v] = flow[into v] - flow[out of v]
    r = [v for (v, ex) in excess.items() if ex > 0 and v != s and v != t]
    # i = 0
    while len(r) > 0:
        if ev.is_set() or is_running == False:
            excessax.cla()
            return
        # print(excess)
        v = r.pop(0)
        
        curr_node = v
        v_edges = [e for e in push_edges if e[0] == v and height[v] > height[e[1]]]
        # print(v_edges)
        pushed_flow = False
        visited = False
        for e in v_edges:
            if excess[v] == 0:
                break
            w = e[1]
            # print(e, "-", str(flow[e]) + "/" + str(capacity[e]), "| height[v] = " + str(height[v]) + " - height[w] = " + str(height[w]))
            # if (height[v] == height[w] + 1 or height[v] == height[w]) and flow[e] < capacity[e]:
            if height[v] == height[w] + 1 and flow[e] < capacity[e]:
                pushing_nodes.append(w)
                reversed_edge = tuple(reversed(e))
                max_possible = min(excess[v], capacity[e] - flow[e] if e in forward_edges else -flow[e])
                # print("> Pushed", max_possible, "flow through", e, "("+(str(capacity[e])+"-"+str(flow[e]) if e in forward_edges else str(-flow[e])) +")")
                # print("Flow", e, "=", flow[e], "| Flow", reversed_edge, "=", flow[reversed_edge])
                # print("Excess", v, "=", excess[v], "| Excess", w, "=", excess[w])
                # flow[tuple(reversed(e))] += max_possible
                # flow[e] = capacity[e] - flow[tuple(reversed(e))]

                flow[e] += max_possible
                flow[reversed_edge] -= max_possible
                excess[v] -= max_possible
                excess[w] += max_possible

                res_capacity[e] -= max_possible
                res_capacity[reversed_edge] += max_possible

                if flow[e] == capacity[e] and e in push_edges:
                    push_edges.remove(e)
                if flow[reversed_edge] == 0 and reversed_edge in push_edges:
                    push_edges.remove(reversed_edge)
                if flow[e] < capacity[e] and e not in push_edges:
                    push_edges.append(e)
                if flow[reversed_edge] < 0 and reversed_edge not in push_edges:
                    push_edges.append(reversed_edge)

                if excess[w] > 0 and w != t and w not in r:
                    r.append(w)
                # print("New flow", e, "=", flow[e], "| New flow", reversed_edge, "=", flow[reversed_edge])
                # print("New excess", v, "=", excess[v], "| New excess", w, "=", excess[w])
                
                pushed_flow = True
                animate_frame(wait=animation_delay, pre_animate=update, ax=ax)
                pushing_nodes.remove(w)
                visited = True
        # if len(r) == 0 and v != t and excess[v] > 0:
        if len(r) == 0:
            r = [v for (v, ex) in excess.items() if ex > 0 and v != s and v != t]
            if len(r) == 0:
                break
            v = r.pop()
            curr_node = v
            relabel(v)
            r.insert(0, v)
            visited = True
            # print("Relabeling", v, "to height", height[v])
        
        # if visited == False:
        #    r.insert(len(r), v)

        # print(r)
        # print("Heights:", height)
        # print("Capacity:", capacity)
        # print("Flow:", flow)
        # print("Excess:", excess)
        # r = 
        # i += 1
    
    print(r)

    print("Heights:", height)
    print("Capacity:", capacity)
    print("Flow:", flow)
    print("Excess:", excess)

    # new_edge_attributes = {edge: {"capacity": capacity[edge], "flow": flow[edge]} for edge in res_edges}
    # netax.cla()
    # netax.set_title('Original Network G')   
    # nx.draw_networkx(G, pos=pos, ax=netax, node_color=DEFAULT_NODE_COLOR, node_size=320)
    # nx.set_edge_attributes(G, new_edge_attributes)
    # draw_network_edges(G, ax=netax, label_size=8)

    print("Algorithm Complete")

    # animate_frame(wait=0, pre_animate=update, ax=ax)
    is_running = False

    ax.cla()
    new_edge_attributes = {edge: {"capacity": capacity[edge], "flow": flow[edge]} for edge in res_edges}
    ax.set_title('Maximum Flow Graph')   
    nx.draw_networkx(G, pos=pos, ax=ax, node_color=DEFAULT_NODE_COLOR, node_size=320)
    nx.set_edge_attributes(G, new_edge_attributes)
    draw_network_edges(G, ax=ax, Gpos=pos, label_size=8)
    plt.draw()

# pfp = preflow_push(G, 's', 't')
# nx.draw_networkx_nodes(pfp, pos=pos, ax=netax, node_color=DEFAULT_NODE_COLOR, node_size=320)
# nx.draw_networkx_labels(pfp, pos=pos, ax=netax)
# draw_network_edges(pfp, ax=netax, zero=False, label_size=8)

menu_start_x = 0.4
axrun = fig.add_axes([menu_start_x, 0.1, 0.1, 0.075])
axreset = fig.add_axes([menu_start_x+0.11, 0.1, 0.1, 0.075])
axpause = fig.add_axes([menu_start_x+0.22, 0.1, 0.1, 0.075])
axex1 = fig.add_axes([menu_start_x, 0.03, 0.025, 0.05])
axex2 = fig.add_axes([menu_start_x+0.03, 0.03, 0.025, 0.05])
axex3 = fig.add_axes([menu_start_x+0.03*2, 0.03, 0.025, 0.05])
axex4 = fig.add_axes([menu_start_x+0.03*3, 0.03, 0.025, 0.05])
# axrun = fig.add_axes([menu_start_x+0.44, 0.1, 0.1, 0.075])
# axreset = fig.add_axes([menu_start_x+0.33, 0.1, 0.1, 0.075])
# axstep = fig.add_axes([menu_start_x+0.22, 0.1, 0.1, 0.075])
axbox = fig.add_axes([menu_start_x+0.44, 0.1, 0.1, 0.075])
brun = widget.Button(axrun, 'Run')
breset = widget.Button(axreset, 'Reset')
bpause = widget.Button(axpause, 'Pause')
bex1 = widget.Button(axex1, 'Ex1')
bex2 = widget.Button(axex2, 'Ex2')
bex3 = widget.Button(axex3, 'Ex3')
bex4 = widget.Button(axex4, 'Ex4')
# bstep = widget.Button(axstep, 'Step')
text_delay = widget.TextBox(axbox, "Delay", initial=animation_delay)

breset.on_clicked(reset)
brun.on_clicked(run)
bpause.on_clicked(pause)
text_delay.on_submit(set_delay)
bex1.on_clicked(set_example1)
bex2.on_clicked(set_example2)
bex3.on_clicked(set_example3)
bex4.on_clicked(set_example4)
plt.show()