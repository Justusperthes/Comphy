#!/usr/bin/env python
# coding: utf-8

# Week 4: Clustering 

# In[2]:





# Exercise 18: ...

# In[139]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

# Load and reshape position data
pos_flat = np.loadtxt('lj10clusters.txt')
positions = pos_flat.reshape(-1, pos_flat.shape[1] // 2, 2)

class DistanceMoments:
    
    def __init__(self, color='C4'):
        self.xwidth = 1
        self.color = color
        self.bin_centers = range(2)
    
    # Calculate mean and standard deviation of pairwise distances
    def descriptor(self, pos):
        all_distances = pdist(pos)
        mean = np.mean(all_distances)
        std = np.std(all_distances)
        return np.array([mean, std])
    
    # Draw both the atom configuration and the bar plot of mean and std dev
    def draw(self, pos, ax, ax2):
        # Bar plot of mean and std dev
        vector = self.descriptor(pos)
        ax.bar(self.bin_centers, vector, width=0.8 * self.xwidth, color=self.color)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.set_ylim([0, 2.3])
        
        xticklabels = ['mu', 'sigma']
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels)
        ax.set_title(self.__class__.__name__)
        
        # Scatter plot of atom positions (atom configuration)
        ax2.scatter(pos[:, 0], pos[:, 1], color='blue', s=200, label="Atoms")
        ax2.set_title("Atom Configuration")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_aspect('equal')
        ax2.grid(True)


# In[48]:


distance_moments = DistanceMoments()

# Number of clusters to plot
#n_clusters = positions.shape[0]
n_clusters = 5

n_cols = 2  # Each cluster will use two columns (one for the bar chart and one for the scatter plot)
n_rows = n_clusters  # Number of clusters

fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 5 * n_clusters))

for i in range(n_clusters):
    ax_bar = axes[i, 0]  
    ax_scatter = axes[i, 1]  
    distance_moments.draw(positions[i], ax_bar, ax_scatter)

plt.tight_layout()
plt.show() 


# 4.4 Descriptor space: plot the 38 clusters in (mu,sigma)-coordinate system.

# In[49]:


distance_moments = DistanceMoments()

mu_values = []
sigma_values = []
colors = []

color_range = [
    (1.6, 1.69, 'red'),
    (1.69,1.78, 'blue'),
    (1.78, 1.85, 'purple'),
    (1.85, 1.9, 'green'),
    (1.9, 2.1, 'pink'),
    (2.1, 1000, 'cyan')
]

def get_color(mu):
    for lower, upper, color in color_range:
        if lower <= mu < upper:
            return color
    return 'black'

# Loop through each cluster and compute mu and sigma
for pos in positions:
    mu, sigma = distance_moments.descriptor(pos)
    mu_values.append(mu)
    sigma_values.append(sigma)
    colors.append(get_color(mu))
# print(colors)

# Plot mu vs sigma
plt.figure(figsize=(10, 6))
plt.scatter(mu_values, sigma_values, color=colors, s=100)
plt.xlabel('Mean (mu)')
plt.ylabel('Standard Deviation (sigma)')
plt.title('Descriptor space')
plt.grid(True)
plt.legend()
plt.show()


# Exercise 19: 5.1 Extreme neighbor count

# In[50]:


from scipy.spatial.distance import pdist,squareform

class ExtremeNeighborCount():
    
    def __init__(self, sigma, color='C5'):
        self.xwidth = 1
        self.color = color
        self.bin_centers = range(2)
        self.sigma = sigma
        
        #compute cutoff
        A = 1.2
        r_min = 2**(1/6)*sigma
        self.cutoff = A*r_min
    
    def descriptor(self,pos):
        distance_matrix = squareform(pdist(pos))
        connectivity_matrix = (distance_matrix < self.cutoff).astype(int) #entry 1 if atoms are within cutoff, else 0
        np.fill_diagonal( connectivity_matrix, 0 )
        neighbor_count = np.sum(connectivity_matrix, axis=1) #sum connectivity_matrix rows to get number of neighbours
        Nlowest = np.min(neighbor_count) #min number of neighbours
        Nhighest = np.max(neighbor_count) #max ...
        return np.array([Nlowest,Nhighest])

    def draw(self,pos,ax, ax2):
        vector = self.descriptor(pos)
        ax.bar(self.bin_centers,vector,width=0.8 * self.xwidth,color=self.color)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.set_ylim([0,7])

        xticklabels = ['$N_{lowest}$','$N_{highest}$']
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels)
        ax.set_title(self.__class__.__name__)

        # Scatter plot of atom positions (atom configuration)
        ax2.scatter(pos[:, 0], pos[:, 1], color='blue', s=200, label="Atoms")
        ax2.set_title("Atom Configuration")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_aspect('equal')
        ax2.grid(True)


# In[51]:


distance_moments = DistanceMoments()
mu,sigma = distance_moments.descriptor(positions[0])
enc = ExtremeNeighborCount(sigma = sigma)
enc.descriptor(positions[0])

fig, axes = plt.subplots(3, 2, figsize=(10, 10))

for i in range(3):
    ax_bar = axes[i, 0]  
    ax_scatter = axes[i, 1]  
    enc.draw(positions[i],ax_bar,ax_scatter)


plt.tight_layout()
plt.show() 


# 5.3 Descriptor space 

# In[52]:


distance_moments = DistanceMoments()
enc = ExtremeNeighborCount(sigma = sigma)
enc.descriptor(positions[0])

N_low_vals = []
N_high_vals = []
colors = []
mu_values = []

color_range = [
    (1.6, 1.69, 'red'),
    (1.69,1.78, 'blue'),
    (1.78, 1.85, 'purple'),
    (1.85, 1.9, 'green'),
    (1.9, 2.1, 'pink'),
    (2.1, 1000, 'cyan')
]

def get_color(mu):
    for lower, upper, color in color_range:
        if lower <= mu < upper:
            return color
    return 'black'


print(colors)

# Loop through each cluster and compute mu and sigma
for pos in positions:
    mu, sigma = distance_moments.descriptor(pos)
    mu_values.append(mu)
    colors.append(get_color(mu))
    Nlow, Nhigh = enc.descriptor(pos)
    N_low_vals.append(Nlow)
    N_high_vals.append(Nhigh)
    mu = distance_moments.descriptor

# for pos in positions:
#     mu, sigma = distance_moments.descriptor(pos)
#     mu_values.append(mu)
#     sigma_values.append(sigma)
#     colors.append(get_color(mu))
    
print(colors)
# Plot low vs high
plt.figure(figsize=(10, 6))
plt.scatter(N_low_vals, N_high_vals, color=colors, s=100)
plt.xlabel('N_lowest')
plt.ylabel('N_highest')
plt.title('Descriptor space')
plt.grid(True)
plt.legend()
plt.show()


# Ex 19, 5.4 Inspect atomic clysters

# In[104]:


# taken from Bjork's ipynb zip

fig, ax = plt.subplots(figsize=(12,20))
a = 5
sorted_indices = [int(i) for i in np.argsort(clusters)]
#sorted_indices = range(len(clusters))
disps = {}
for i,(atomic_cluster,cluster,j) in enumerate(zip(np.array(atomic_clusters)[sorted_indices],
                                      clusters[sorted_indices],sorted_indices)):
    atomic_cluster_copy = atomic_cluster.copy()
    disp = disps.get(cluster,0)
    disps[cluster] = disp + 1
    atomic_cluster_copy.set_positions(atomic_cluster.get_positions() + \
                                      9*np.array([(i%a),(i//a)]) +0*np.array([disp,cluster]),
                                ignore_b=True)
    atomic_cluster_copy.draw(ax,size=200,color=cmap(cluster),alpha=alpha,energy_title=False)
    ax.text(9*(i%a)-2,9*(i//a)-4,str(j) + ':' + str(atomic_cluster.descriptor))
ax.set_aspect('equal')
ax.set_title('Clustering with ' + atomic_cluster.descriptor_method.__class__.__name__) 
ax.set_xticks([])
ax.set_yticks([])


# Ex 20, 6.2 Interatomic distance distribution

# the descriptor in each representation always throws away some information

# In[138]:


#implementation of interatomic distance distribution
class PairDistances():
    
    def __init__(self, color='C1'):
        self.xwidth = 0.5
        self.color = color
        self.bin_edges = np.arange(0,7.01,self.xwidth)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) /2
    
    def descriptor(self,pos):
        pairwise_distances = pdist(pos)
        # print(pairwise_distances)
        hist, _ = np.histogram(pairwise_distances, bins=self.bin_edges)
        
        # pairwise_distances = []
        # for i in range 
        #     pairwise_distances_manual = np.sqrt((pos[1][0]-pos[0][0])**2+(pos[1][1]-pos[1][0])**2)
        # ...
        
        return hist
    
    def draw(self,pos,ax):
        vector = self.descriptor(pos)
        ax.bar(self.bin_centers,vector,width=0.8 * self.xwidth,color=self.color)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.set_title(self.__class__.__name__)


# In[96]:


# try it out

pair_distances = PairDistances()
fig, ax = plt.subplots(3,2)
ax_flat = ax.flatten()
for i in range(6):
    pair_distances.draw(positions[i], ax_flat[i])
plt.show()



# Ex 20, 6.3 Coordination Numer profile

# In[136]:


class CoordinationNumbers():
    
    def __init__(self,sigma, color='C2'):
        self.xwidth = 1
        self.color = color
        self.sigma = sigma
    
    def descriptor(self, sigma, pos):
        r_cut = 1.2 * 2**(1/6)*sigma # r_cut = A*r_min
        pairwise_distances_matrix = squareform(pdist(pos))
        coordination_numbers = np.sum((pairwise_distances_matrix < r_cut) & (pairwise_distances_matrix > 0), axis=1)
        hist, _ = np.histogram(coordination_numbers, bins=np.arange(9), range=(0, 8))
        return hist
    
    def draw(self,pos,ax):
        vector = self.descriptor(sigma, pos)
        N = len(vector)
        xcenters = np.linspace(0,N-1,N) * self.xwidth
        ax.bar(xcenters,vector,width=0.8 * self.xwidth,color=self.color)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.set_title(self.__class__.__name__)


# In[140]:


# Example of plotting
fig, ax = plt.subplots(3,2)
ax_flat = ax.flatten()
# sigma = 1.0  # Example value for sigma
sigma_values = []

for pos in positions:
    mu, sigma = distance_moments.descriptor(pos)
    sigma_values.append(sigma)


for i in range(6):
    coord_number = CoordinationNumbers(sigma_values[i])
    coord_number.draw(positions[i], ax_flat[i])  # Plot for the first cluster
plt.show()


# Ex 20, 6.4 Connectivity Graph Spectrum

# In[ ]:


class ConnectivityGraphSpectrum():
    
    def __init__(self, color='C3'):
        self.xwidth = 1
        self.color = color
    
    def descriptor(self,pos):
        #YOUR CODE
        return "hello"
    
    def draw(self,pos,ax):
        vector = self.descriptor(pos)
        N = len(vector)
        xcenters = np.linspace(0,N-1,N) * self.xwidth
        ax.bar(xcenters,vector,width=0.8 * self.xwidth,color=self.color)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.set_title(self.__class__.__name__)


# Ex 20, 6.5 Coulomb Matrix Spectrum descriptors

# In[125]:


class CoulombMatrixSpectrum():
    
    def __init__(self, color='C4'):
        self.xwidth = 1
        self.color = color
    
    def descriptor(self,pos):
        N = pos.shape[0]
        coulomb_matrix = np.zeros((N, N))
        
        # Fill the Coulomb matrix
        for i in range(N):
            for j in range(N):
                if i == j:
                    coulomb_matrix[i, j] = 1.0  # Diagonal elements = 1
                else:
                    # Off-diagonal = 1 / r_ij 
                    # r_ij = distance between atoms i and j
                    distance = np.linalg.norm(pos[i] - pos[j])
                    coulomb_matrix[i, j] = 1.0 / distance
        
        # Compute eigenvalues of Coulomb matrix
        eigenvalues = np.linalg.eigvalsh(coulomb_matrix)
        
        return np.sort(eigenvalues)

    def draw(self,pos,ax):
        vector = self.descriptor(pos)
        N = len(vector)
        xcenters = np.linspace(0,N-1,N) * self.xwidth
        ax.bar(xcenters,vector,width=0.8 * self.xwidth,color=self.color)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.set_ylim([-2,8])
        ax.set_title(self.__class__.__name__)


# In[143]:


# Example usage
fig, ax = plt.subplots(3,2)
ax_flat = ax.flatten()

coulomb_matrix_spectrum = CoulombMatrixSpectrum()
for i in range(6):
    coulomb_matrix_spectrum.draw(positions[i], ax_flat[i])

plt.show()


