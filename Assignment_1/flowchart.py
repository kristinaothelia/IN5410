from graphviz import Digraph, Graph
#import pygraphviz

# last pages of this pdf has many things, all colors, shapes etc. 
# https://www.graphviz.org/pdf/dotguide.pdf
# https://www.graphviz.org/doc/info/attrs.html
# maybe google dpi or something for better quality?


dot = Digraph(comment='flowchart_assignment1', format='png', engine='dot')
#dot.edge_attr.update(arrowhead='vee', arrowsize='2')
dot.node_attr.update(fontsize='15', fontname='helvetica')
dot.edge_attr.update(concentrate='true') 
dot.graph_attr.update(splines='ortho') # nodesep='1' # ranksep='1'

# if we want a black, horizontal line, can use something like this
#dot.node('L', "", shape='rectangle', style='filled', fillcolor='black', width='5', height='0.3')

dot.node('S', 'Start', shape='circle', style='filled', fillcolor='grey80')
dot.node('R', 'Read energy_use.xlsx', shape='box', style='filled', fillcolor = 'cornflowerblue', fontcolor='white')
dot.node('T', 'Choosing task', shape='box')

dot.edge('S', 'R', constraint='true', label=' hei', color='red')
dot.edge('R', 'T', constraint='true') # constraint = false, horizontal instead of vertical


dot.node('struct1', '<f0> Task 1|<f1> Task 2|<f2> Task 3', shape='record')
dot.edge('T', 'struct1')


dot.node('A1', 'Select appliances:\n EV, laundry machine and dishwasher', shape='box')
dot.node('A2', 'Select appliances:\n all appliances (shift + non-shift + EV)', shape='box')
dot.node('A3', 'Select appliances:\n all non-shift and some shiftable', shape='box')

dot.edge('struct1:f0', 'A1', constraint='true')
dot.edge('struct1:f1', 'A2', constraint='true')
dot.edge('struct1:f2', 'A3', constraint='true', xlabel='Price=RTP \n Households: 30')

dot.node('p1', 'Price=ToU', shape='circle', fixedsize='true', width='1.1', height='1.1')
dot.node('p2', 'Price=RTP', shape='circle', fixedsize='true', width='1.1', height='1.1')

dot.edge('A1', 'p1', constraint='true')
dot.edge('A2', 'p2', constraint='true')


dot.node('Lin', 'Linprog calculation', shape='box')


# Creates invisible arrows for nicer plot
dot.edge('A1', 'Lin', constraint='true', style='invis')
dot.edge('A2', 'Lin', constraint='true', style='invis')

#dot.graph_attr.update(splines='ortho', nodesep='0.1') 
dot.edge('p1', 'Lin', constraint='false', nodesep='0.1')
dot.edge('p2', 'Lin', constraint='false', nodesep='0.1')

dot.node('I1', 'Plot==True', shape='diamond', fixedsize='true', width='1.2', height='1.2')

dot.edge('Lin', 'I1', constraint='true', nodesep='0.1')

dot.node('Plot1', 'Creates and shows plots', shape='box', nodesep='10')
dot.node('Calc1', 'Prints minimized cost', shape='box')

# Creates invisible arrows for nicer plot
dot.edge('p1', 'Plot1', constraint='true', style='invis')
dot.edge('p2', 'Calc1', constraint='true', style='invis')

dot.edge('I1', 'Plot1', constraint='false', xlabel='   yes')
dot.edge('I1', 'Calc1', constraint='false', xlabel='   no')


dot.node('E', 'EV: yes/no', shape='box')
dot.node('RA', 'Add shiftable appliances (>= 2)', shape='box')

dot.edge('A3', 'E', constraint='true')
dot.edge('E', 'RA', constraint='true') #minlen='0.01'


dot.node('Lin3', 'Linprog calculation', shape='box')
dot.edge('RA', 'Lin3', constraint='true', rank='false', len='0.0001')

dot.node('I2', 'Plot==True', shape='diamond', fixedsize='true', width='1.2', height='1.2')
dot.edge('Lin3', 'I2', constraint='true')

dot.node('Plot3', 'Creates and shows plots', shape='box')
dot.node('Calc3', 'Prints minimized cost', shape='box')

#dot.node('X', '', shape='plaintext')
#dot.edge('I2', 'X', constraint='true', weight='3')

dot.edge('I2', 'Plot3', constraint='true', weight='3')
dot.edge('I2', 'Calc3', constraint='true', weight='2')


print(dot.source)
dot.render('flowchart_assignment1', view=True)
