import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.widgets as widget
from networkx.algorithms.flow import preflow_push, build_residual_network

# Source code for networkx.algorithms.flow.preflowpush:
# https://networkx.org/documentation/stable/_modules/networkx/algorithms/flow/preflowpush.html 

"""
Highest-label preflow-push algorithm for maximum flow problems.
"""

from collections import deque
from itertools import islice

import networkx as nx

from networkx.utils import arbitrary_element
from networkx.algorithms.flow.utils import (
    CurrentEdge,
    GlobalRelabelThreshold,
    Level,
    build_residual_network,
    detect_unboundedness,
)

__all__ = ["preflow_push"]


def preflow_push_impl(G, s, t, capacity, residual, global_relabel_freq, value_only):
    """Implementation of the highest-label preflow-push algorithm."""
    if s not in G:
        raise nx.NetworkXError(f"node {str(s)} not in graph")
    if t not in G:
        raise nx.NetworkXError(f"node {str(t)} not in graph")
    if s == t:
        raise nx.NetworkXError("source and sink are the same node")

    if global_relabel_freq is None:
        global_relabel_freq = 0
    if global_relabel_freq < 0:
        raise nx.NetworkXError("global_relabel_freq must be nonnegative.")

    if residual is None:
        R = build_residual_network(G, capacity)
    else:
        R = residual

    detect_unboundedness(R, s, t)

    R_nodes = R.nodes
    R_pred = R.pred
    R_succ = R.succ

    # Initialize/reset the residual network.
    for u in R:
        R_nodes[u]["excess"] = 0
        for e in R_succ[u].values():
            e["flow"] = 0

    def reverse_bfs(src):
        """Perform a reverse breadth-first search from src in the residual
        network.
        """
        heights = {src: 0}
        q = deque([(src, 0)])
        while q:
            u, height = q.popleft()
            height += 1
            for v, attr in R_pred[u].items():
                if v not in heights and attr["flow"] < attr["capacity"]:
                    heights[v] = height
                    q.append((v, height))
        return heights

    # Initialize heights of the nodes.
    heights = reverse_bfs(t)

    if s not in heights:
        # t is not reachable from s in the residual network. The maximum flow
        # must be zero.
        R.graph["flow_value"] = 0
        return R

    n = len(R)
    # max_height represents the height of the highest level below level n with
    # at least one active node.
    max_height = max(heights[u] for u in heights if u != s)
    heights[s] = n

    grt = GlobalRelabelThreshold(n, R.size(), global_relabel_freq)

    # Initialize heights and 'current edge' data structures of the nodes.
    for u in R:
        R_nodes[u]["height"] = heights[u] if u in heights else n + 1
        R_nodes[u]["curr_edge"] = CurrentEdge(R_succ[u])

    def push(u, v, flow):
        """Push flow units of flow from u to v."""
        R_succ[u][v]["flow"] += flow
        R_succ[v][u]["flow"] -= flow
        R_nodes[u]["excess"] -= flow
        R_nodes[v]["excess"] += flow

    # The maximum flow must be nonzero now. Initialize the preflow by
    # saturating all edges emanating from s.
    for u, attr in R_succ[s].items():
        flow = attr["capacity"]
        if flow > 0:
            push(s, u, flow)

    # Partition nodes into levels.
    levels = [Level() for i in range(2 * n)]
    for u in R:
        if u != s and u != t:
            level = levels[R_nodes[u]["height"]]
            if R_nodes[u]["excess"] > 0:
                level.active.add(u)
            else:
                level.inactive.add(u)

    def activate(v):
        """Move a node from the inactive set to the active set of its level."""
        if v != s and v != t:
            level = levels[R_nodes[v]["height"]]
            if v in level.inactive:
                level.inactive.remove(v)
                level.active.add(v)

    def relabel(u):
        """Relabel a node to create an admissible edge."""
        grt.add_work(len(R_succ[u]))
        return (
            min(
                R_nodes[v]["height"]
                for v, attr in R_succ[u].items()
                if attr["flow"] < attr["capacity"]
            )
            + 1
        )

    def discharge(u, is_phase1):
        """Discharge a node until it becomes inactive or, during phase 1 (see
        below), its height reaches at least n. The node is known to have the
        largest height among active nodes.
        """
        height = R_nodes[u]["height"]
        curr_edge = R_nodes[u]["curr_edge"]
        # next_height represents the next height to examine after discharging
        # the current node. During phase 1, it is capped to below n.
        next_height = height
        levels[height].active.remove(u)
        while True:
            v, attr = curr_edge.get()
            if height == R_nodes[v]["height"] + 1 and attr["flow"] < attr["capacity"]:
                flow = min(R_nodes[u]["excess"], attr["capacity"] - attr["flow"])
                push(u, v, flow)
                activate(v)
                if R_nodes[u]["excess"] == 0:
                    # The node has become inactive.
                    levels[height].inactive.add(u)
                    break
            try:
                curr_edge.move_to_next()
            except StopIteration:
                # We have run off the end of the adjacency list, and there can
                # be no more admissible edges. Relabel the node to create one.
                height = relabel(u)
                if is_phase1 and height >= n - 1:
                    # Although the node is still active, with a height at least
                    # n - 1, it is now known to be on the s side of the minimum
                    # s-t cut. Stop processing it until phase 2.
                    levels[height].active.add(u)
                    break
                # The first relabel operation after global relabeling may not
                # increase the height of the node since the 'current edge' data
                # structure is not rewound. Use height instead of (height - 1)
                # in case other active nodes at the same level are missed.
                next_height = height
        R_nodes[u]["height"] = height
        return next_height

    def gap_heuristic(height):
        """Apply the gap heuristic."""
        # Move all nodes at levels (height + 1) to max_height to level n + 1.
        for level in islice(levels, height + 1, max_height + 1):
            for u in level.active:
                R_nodes[u]["height"] = n + 1
            for u in level.inactive:
                R_nodes[u]["height"] = n + 1
            levels[n + 1].active.update(level.active)
            level.active.clear()
            levels[n + 1].inactive.update(level.inactive)
            level.inactive.clear()

    def global_relabel(from_sink):
        """Apply the global relabeling heuristic."""
        src = t if from_sink else s
        heights = reverse_bfs(src)
        if not from_sink:
            # s must be reachable from t. Remove t explicitly.
            del heights[t]
        max_height = max(heights.values())
        if from_sink:
            # Also mark nodes from which t is unreachable for relabeling. This
            # serves the same purpose as the gap heuristic.
            for u in R:
                if u not in heights and R_nodes[u]["height"] < n:
                    heights[u] = n + 1
        else:
            # Shift the computed heights because the height of s is n.
            for u in heights:
                heights[u] += n
            max_height += n
        del heights[src]
        for u, new_height in heights.items():
            old_height = R_nodes[u]["height"]
            if new_height != old_height:
                if u in levels[old_height].active:
                    levels[old_height].active.remove(u)
                    levels[new_height].active.add(u)
                else:
                    levels[old_height].inactive.remove(u)
                    levels[new_height].inactive.add(u)
                R_nodes[u]["height"] = new_height
        return max_height

    # Phase 1: Find the maximum preflow by pushing as much flow as possible to
    # t.

    height = max_height
    while height > 0:
        # Discharge active nodes in the current level.
        while True:
            level = levels[height]
            if not level.active:
                # All active nodes in the current level have been discharged.
                # Move to the next lower level.
                height -= 1
                break
            # Record the old height and level for the gap heuristic.
            old_height = height
            old_level = level
            u = arbitrary_element(level.active)
            height = discharge(u, True)
            if grt.is_reached():
                # Global relabeling heuristic: Recompute the exact heights of
                # all nodes.
                height = global_relabel(True)
                max_height = height
                grt.clear_work()
            elif not old_level.active and not old_level.inactive:
                # Gap heuristic: If the level at old_height is empty (a 'gap'),
                # a minimum cut has been identified. All nodes with heights
                # above old_height can have their heights set to n + 1 and not
                # be further processed before a maximum preflow is found.
                gap_heuristic(old_height)
                height = old_height - 1
                max_height = height
            else:
                # Update the height of the highest level with at least one
                # active node.
                max_height = max(max_height, height)

    # A maximum preflow has been found. The excess at t is the maximum flow
    # value.
    if value_only:
        R.graph["flow_value"] = R_nodes[t]["excess"]
        return R

    # Phase 2: Convert the maximum preflow into a maximum flow by returning the
    # excess to s.

    # Relabel all nodes so that they have accurate heights.
    height = global_relabel(False)
    grt.clear_work()

    # Continue to discharge the active nodes.
    while height > n:
        # Discharge active nodes in the current level.
        while True:
            level = levels[height]
            if not level.active:
                # All active nodes in the current level have been discharged.
                # Move to the next lower level.
                height -= 1
                break
            u = arbitrary_element(level.active)
            height = discharge(u, False)
            if grt.is_reached():
                # Global relabeling heuristic.
                height = global_relabel(False)
                grt.clear_work()

    R.graph["flow_value"] = R_nodes[t]["excess"]
    return R

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

def animate_frame(wait=1, pre_animate=lambda:(), ax=None):
    ax.cla()
    pre_animate()
    plt.pause(wait)
    plt.draw()

def draw_network_edges(G, zero=False, ax=None):
    edge_capacities = nx.get_edge_attributes(G,'capacity')
    edge_flows = nx.get_edge_attributes(G, 'flow')
    # print("Capacity:", edge_capacities)
    # print("Flow:", edge_flows)

    curved_edges = []
    straight_edges = []
    if zero:
        curved_edges = [edge for edge in G.edges() if (reversed(edge) in G.edges())]
        straight_edges = list(set(G.edges()) - set(curved_edges))
    else:
        curved_edges = [edge for edge in G.edges() if ((reversed(edge) in G.edges()) and edge_capacities[edge] != 0 and edge_capacities[tuple(reversed(edge))] != 0)]
        straight_edges = [edge for edge in list(set(res.edges()) - set(curved_edges)) if edge_capacities[edge] != 0]
    # print("Curved:", curved_edges)
    # print("Straight:", straight_edges)

    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=straight_edges)
    arc_rad = 0.25
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}')
    # print(edge_flows)

    if zero:
        curved_edge_labels = {edge: str(0 if edge_flows.get(edge) == None else edge_flows.get(edge)) + '/' + str(edge_capacities[edge]) for edge in curved_edges}
        straight_edge_labels = {edge: str(0 if edge_flows.get(edge) == None else edge_flows.get(edge)) + '/' + str(edge_capacities[edge]) for edge in straight_edges}
    else:
        curved_edge_labels = {edge: str(0 if edge_flows.get(edge) == None else edge_flows.get(edge)) + '/' + str(edge_capacities[edge]) for edge in curved_edges if edge_capacities[edge] != 0}
        straight_edge_labels = {edge: str(0 if edge_flows.get(edge) == None else edge_flows.get(edge)) + '/' + str(edge_capacities[edge]) for edge in straight_edges if edge_capacities[edge] != 0}
    my_draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=curved_edge_labels,rotate=False,rad=arc_rad)
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=straight_edge_labels,rotate=False)
    plt.draw()

def update_network(G, zero=False, ax=None, new_attributes=dict()):
    # print(new_attributes)
    nx.set_edge_attributes(G, new_attributes)

    pos = nx.spring_layout(G, seed=5)
    nx.draw_networkx_nodes(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)

    draw_network_edges(G, ax=ax)

def preflow_push_visualize(G, s, t):
    res = build_residual_network(G, capacity='capacity')

    G_edges = G.edges()

    res_nodes = res.nodes()
    res_edges = res.edges()

    forward_edges = [e for e in res.edges() if e in G_edges]
    backward_edges = list(set(res_edges) - set(forward_edges))
    print("Forward:", forward_edges)
    print("Backward:", backward_edges)

    n = len(res_nodes)
    height = {s: n}
    capacity = nx.get_edge_attributes(res, 'capacity')
    res_capacity = capacity.copy()
    capacity = {e: capacity[e] if capacity[e] != 0 else capacity[tuple(reversed(e))] for e in res_edges}
    print(capacity)
    flow = dict()
    excess = {v: 0 for v in res_nodes}

    def push(v, w, e, amount):
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

    push_edges = forward_edges.copy()
    for e in res_edges:
        if e[0] == s:
            # Saturate all edges from s
            # flow[e] = capacity[e]
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
    
    def relabel(v):
        height[v] += 1
        # animate_frame()

    print("Heights:", height)
    print("Capacity:", capacity)
    print("Flow:", flow)
    print("Excess:", excess)

    excessax = fig.add_axes([0.01, 0.1, 0.2, 0.075])

    # excess[v] = flow[into v] - flow[out of v]
    r = [v for (v, ex) in excess.items() if ex > 0 and v != s and v != t]
    i = 0
    while len(r) > 0 and i < 1000:
        # print(excess)
        v = r.pop(0)
        v_edges = [e for e in push_edges if e[0] == v and height[v] > height[e[1]]]
        # print(v_edges)
        pushed_flow = False
        for e in v_edges:
            if excess[v] == 0:
                break
            w = e[1]
            print(e, "-", str(flow[e]) + "/" + str(capacity[e]), "| height[v] = " + str(height[v]) + " - height[w] = " + str(height[w]))
            if height[v] == height[w] + 1 and flow[e] < capacity[e]:
                reversed_edge = tuple(reversed(e))
                max_possible = min(excess[v], capacity[e] - flow[e] if e in forward_edges else -flow[e])
                print("> Pushed", max_possible, "flow through", e, "("+(str(capacity[e])+"-"+str(flow[e]) if e in forward_edges else str(-flow[e])) +")")
                print("Flow", e, "=", flow[e], "| Flow", reversed_edge, "=", flow[reversed_edge])
                print("Excess", v, "=", excess[v], "| Excess", w, "=", excess[w])
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

                if excess[w] > 0 and w != t:
                    r.append(w)
                print("New flow", e, "=", flow[e], "| New flow", reversed_edge, "=", flow[reversed_edge])
                print("New excess", v, "=", excess[v], "| New excess", w, "=", excess[w])
                
                # if e in forward_edges:
                #     max_possible = min(excess[v], capacity[e] - flow[e])
                #     print("> Pushed", max_possible, "flow through", e)
                #     print("Flow", e, "=", flow[e], "| Flow", tuple(reversed(e)), "=", flow[tuple(reversed(e))])
                #     # flow[tuple(reversed(e))] += max_possible
                #     # flow[e] = capacity[e] - flow[tuple(reversed(e))]
                #     flow[tuple(reversed(e))] -= max_possible
                #     flow[e] += max_possible
                #     excess[v] -= max_possible
                #     excess[w] += max_possible

                #     if flow[e] == 0:
                #         push_edges.remove(e)
                #     if flow[tuple(reversed(e))] > 0:
                #         push_edges.append(tuple(reversed(e)))

                #     if w != t:
                #         r.append(w)
                #     print("New flow:", flow[e])
                #     print("New flow of backedge:", flow[tuple(reversed(e))])
                #     print("New excess:", excess[v])
                # else:
                #     max_possible = min(excess[v], flow[e])
                #     print("> Decreased", max_possible, "flow through", e)
                #     # flow[e] -= max_possible
                #     # flow[tuple(reversed(e))] = capacity[e] - flow[e]
                #     flow[e] += max_possible
                #     flow[tuple(reversed(e))] -= max_possible
                #     excess[v] -= max_possible
                #     excess[w] += max_possible

                #     if flow[e] == 0:
                #         push_edges.remove(e)
                #     if flow[tuple(reversed(e))] > 0:
                #         push_edges.append(tuple(reversed(e)))

                #     if w != t:
                #         r.append(v)
                #     print("New flow:", flow[tuple(reversed(e))])
                #     print("New excess:", excess[v])
                pushed_flow = True

                def update():
                    new_edge_attributes = {edge: {"capacity": capacity[edge], "flow": flow[edge]} for edge in res_edges}
                    nx.set_edge_attributes(res, new_edge_attributes)

                    pos = nx.spring_layout(res, seed=5)
                    nx.draw_networkx_nodes(res, pos, ax=ax)
                    nx.draw_networkx_labels(res, pos, ax=ax)

                    # new_node_attributes = {node: excess[node] for node in res_nodes}
                    # nx.draw_networkx_labels(res, {n:[pos[n][0]-0.05, pos[n][1]+0.05] for n in res_nodes}, labels=new_node_attributes, ax=ax, horizontalalignment='right')
                    # nx.set_node_attributes(res, new_node_attributes)

                    excessax.cla()
                    excessax.axis('off')
                    excessax.text(0, 0.4, 'excess: '+str({node: excess[node] for node in res_nodes}))
                    excessax.text(0, -0.1, 'height: '+str({node: height[node] for node in res_nodes}))

                    draw_network_edges(res, ax=ax)
                animate_frame(wait=0.25, pre_animate=update, ax=ax)
        if len(r) == 0 and v != t and excess[v] > 0:
            r = [v for (v, ex) in excess.items() if ex > 0 and v != s and v != t]
            v = r.pop()
            relabel(v)
            r.insert(0, v)
            print("Relabeling", v, "to height", height[v])
        # print(r)
        # print("Heights:", height)
        # print("Capacity:", capacity)
        # print("Flow:", flow)
        # print("Excess:", excess)
        # r = 
        i += 1


    print("Heights:", height)
    print("Capacity:", capacity)
    print("Flow:", flow)
    print("Excess:", excess)

    print("Algorithm Complete")

    new_attributes = {edge: {"capacity": capacity[edge], "flow": flow[edge]} for edge in res_edges}
    # print(new_attributes)
    nx.set_edge_attributes(res, new_attributes)

    pos = nx.spring_layout(res, seed=5)
    nx.draw_networkx_nodes(res, pos, ax=ax)
    nx.draw_networkx_labels(res, pos, ax=ax)

    draw_network_edges(res, ax=ax)
    plt.draw()
    
    


def reset(event):
    plt.draw()



fig, ax = plt.subplots()
fig.set_figwidth(10)
fig.subplots_adjust(bottom=0.2, right=0.5, left=0.01)
menu_start_x = 0.4
axrun = fig.add_axes([menu_start_x+0.44, 0.1, 0.1, 0.075])
axreset = fig.add_axes([menu_start_x+0.33, 0.1, 0.1, 0.075])
axstep = fig.add_axes([menu_start_x+0.22, 0.1, 0.1, 0.075])
axbox = fig.add_axes([menu_start_x, 0.1, 0.2, 0.075])
brun = widget.Button(axrun, 'Run')
breset = widget.Button(axreset, 'Reset')
bstep = widget.Button(axstep, 'Step')
text_delay = widget.TextBox(axbox, "Delay")

breset.on_clicked(reset)



edges = [('s', 1, {"capacity": 16}), ('s', 3, {"capacity": 13}), (1, 2, {"capacity": 12}), (3, 1, {"capacity": 4}),
            (2, 3, {"capacity": 9}), (3, 4, {"capacity": 14}), (4, 2, {"capacity": 7}), (4, 't', {"capacity": 4}),
            (4, 't', {"capacity": 4})]
G = nx.DiGraph()
G.add_nodes_from(['s', 't'])
netax = fig.add_axes([0.51, 0.2, 0.48, 0.68])
# edges.append((3, 1))
# edges.append((2, 4))
# edges.append(('s', 2))
# edges.append((1, 't'))
# edges.append((4, 1))
# edges.append((2, 3))

G.add_edges_from(edges)
# pos = nx.spring_layout(G)
# nx.draw_networkx(G, pos=pos, ax=ax)
# labels = nx.get_edge_attributes(G,'capacity')
# nx.draw_networkx_edge_labels(G,pos,edge_labels=labels, ax=ax)


# Drawing multiple edges between two nodes with networkx
# https://stackoverflow.com/questions/22785849/drawing-multiple-edges-between-two-nodes-with-networkx
# res = preflow_push(build_residual_network(G, capacity='capacity'), 's', 't')
res = build_residual_network(G, capacity='capacity')
pos = nx.spring_layout(res, seed=5)

# Draw original network G
netax.set_title('Original Network G')
# nx.draw_networkx(G, pos, ax=netax)
# network_capacity = nx.get_edge_attributes(G, "capacity")
# network_capacity = {edge: '0/' + str(network_capacity[edge]) for edge in G.edges() if network_capacity[edge] != 0}
# nx.draw_networkx_edge_labels(G, pos, edge_labels=network_capacity, rotate=False)
res = preflow_push(G, 's', 't')



nx.draw_networkx_nodes(res, pos, ax=netax)
nx.draw_networkx_labels(res, pos, ax=netax)
draw_network_edges(res, ax=netax)

ax.set_title('Maximum Flow Graph')
# nx.draw_networkx_nodes(res, pos, ax=ax)
# nx.draw_networkx_labels(res, pos, ax=ax)


# edge_capacities = nx.get_edge_attributes(res,'capacity')
# edge_flows = nx.get_edge_attributes(res, 'flow')
# print(edge_capacities, edge_flows)

# curved_edges = [edge for edge in res.edges() if ((reversed(edge) in res.edges()) and edge_capacities[edge] != 0 and edge_capacities[tuple(reversed(edge))] != 0)]
# # curved_edges = [edge for edge in res.edges() if (reversed(edge) in res.edges())]
# straight_edges = [edge for edge in list(set(res.edges()) - set(curved_edges)) if edge_capacities[edge] != 0]
# print(curved_edges, straight_edges)


# nx.draw_networkx_edges(res, pos, ax=ax, edgelist=straight_edges)
# arc_rad = 0.25
# nx.draw_networkx_edges(res, pos, ax=ax, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}')
# # print(edge_flows)
# # curved_edge_labels = {edge: str(0 if edge_flows.get(edge) == None else edge_flows.get(edge)) + '/' + str(edge_capacities[edge]) for edge in curved_edges}
# # straight_edge_labels = {edge: str(0 if edge_flows.get(edge) == None else edge_flows.get(edge)) + '/' + str(edge_capacities[edge]) for edge in straight_edges}
# curved_edge_labels = {edge: str(0 if edge_flows.get(edge) == None else edge_flows.get(edge)) + '/' + str(edge_capacities[edge]) for edge in curved_edges if edge_capacities[edge] != 0}
# straight_edge_labels = {edge: str(0 if edge_flows.get(edge) == None else edge_flows.get(edge)) + '/' + str(edge_capacities[edge]) for edge in straight_edges if edge_capacities[edge] != 0}
# my_draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=curved_edge_labels,rotate=False,rad=arc_rad)
# nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=straight_edge_labels,rotate=False)


preflow_push_visualize(G, 's', 't')
# fig.savefig("3.png", bbox_inches='tight',pad_inches=0)
# flow_label = nx.get_edge_attributes(res, 'flow')
# capacity_label = nx.get_edge_attributes(res, 'capacity')
# formatted_edge_labels = {(elem[0],elem[1]):capacity_label[elem] for elem in capacity_label if capacity_label[elem] != 0}
# print(formatted_edge_labels)
# edge_labels = my_draw_networkx_edge_labels(res,pos,edge_labels=formatted_edge_labels, ax=ax, font_size=10)

# R = preflow_push_impl(G, 's', 't', capacity='capacity', residual=None, global_relabel_freq=1, value_only=False)
# R.graph["algorithm"] = "preflow_push"
# pos = nx.spring_layout(R)
# labels = nx.get_edge_attributes(R,'flow')
# nx.draw_networkx(R, ax=ax)
# nx.draw_networkx_edge_labels(R,pos,edge_labels=labels, ax=ax)
plt.show()