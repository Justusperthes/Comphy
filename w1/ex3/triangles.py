import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plots
import matplotlib.pyplot as plt
import tkinter

plt.ion()

class Triangle:
    # constructor
    def __init__(self, color, edge_points=None, cm=(0,0), triangle=None, cross=None):
        # attributes (instance variables)
        self.cm = cm
        if edge_points == None:
            self.edge_points = self.default_equilateral_triangle(cm)
            #self.edge_points = self.calculate_edge_points(self.set_edge_points())
        else:
            self.edge_points = edge_points
        self.color = color #string color
        #
    
    # setter methods
    def set_color(self, new_color):
        self.color = new_color

    def set_edge_points(self, edge_points):
        self.edge_points = edge_points
    
    # def move(r):
    #     #move the triangle r = vec(r)

    # getter methods
    def get_color(self):
        return self.color

    def get_edges(self):
        return self.edge_points

    # aux methods and drawing methods
    def default_equilateral_triangle(self, cm):
        side_length = 1 
        height = np.sqrt(3) / 2 * side_length
        #vertices
        vertex1 = (cm[0], float(cm[1] + (2/3) * height))
        vertex2 = (cm[0] - side_length / 2, float(cm[1] - (1/3) * height))
        vertex3 = (cm[0] + side_length / 2, float(cm[1] - (1/3) * height))
        return [vertex1, vertex2, vertex3]
    
    def calculate_edge_points(self):
        edge_points = []
        for i in range(0,2):
            vertex = self.edge_points[i][i]
            edge_points.append(vertex)
        return edge_points
    
    def draw(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        triangle = plt.Polygon(self.edge_points, color=self.color)
        ax.add_patch(triangle)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal', 'box')
        plt.grid(True)
        plt.show()


my_triangle = Triangle("blue")
print(my_triangle.get_color())
my_triangle.set_color("red")
print(my_triangle.get_color())
vertices = [(1,1),(0,1),(0,0)]
my_other_triangle = Triangle("purple",vertices)
my_other_triangle.get_edges()
print(my_triangle.get_edges())
print(my_other_triangle.get_edges())

my_other_triangle.draw()