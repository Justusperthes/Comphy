import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import fmin

class LennardJones():
    def __init__(self,eps0=5,sigma=2**(-1/6)):
        self.eps0 = eps0
        self.sigma = sigma
        
    def _V(self,r):
        return 4 * self.eps0 * ( (self.sigma/r)**12 - (self.sigma/r)**6 )

    def _dV_dr(self, r):
        return -4 * self.eps0 * (12 * (self.sigma / r)**12 - 6 * (self.sigma / r)**6) / r    

    def energy(self, pos):
        return np.sum(self._V(pdist(pos)))
    
    def forces(self, pos):
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(r, np.inf)
        force_magnitude = -self._dV_dr(r)
        forces = np.sum(force_magnitude[..., np.newaxis] * diff / \
                        r[..., np.newaxis], axis=1)
        return forces



class AtomicCluster():

    def __init__(self, calc, N=None, pos=None, static=None, box_half_width=4, wrap=False):
        self.calc = calc
        self.wrap_enforced = wrap
        assert (N is not None and pos is None) or \
               (N is None and pos is not None), 'You must specify either N or pos'
        if pos is not None:
            self.pos = np.array(pos)*1.
            self.N = len(pos)
        else:
            self.N = N
            self.pos = 2*box_half_width*np.random.rand(N,2) - box_half_width
        if static is not None:
            assert len(static) == self.N, 'static must be N long'
            self.static = static
        else:
            self.static = [False for _ in range(self.N)]
        self.filter = np.array([self.static,self.static]).T
        self.indices_dynamic_atoms = [i for i,static in enumerate(self.static) if not static]
        self.box_half_width = box_half_width
        self.plot_artists = {}

    def copy(self):
        return AtomicCluster(self.calc,
                             pos=self.pos,
                             static=self.static,
                             box_half_width=self.box_half_width,
                             wrap=self.wrap_enforced)
        
    @property
    def energy(self):
        return self.calc.energy(self.pos)

    def forces(self):
        forces = self.calc.forces(self.pos)
        return np.where(self.filter,0,forces)

    def wrap(self):
        if self.wrap_enforced:
            self.pos = (self.pos + self.box_half_width) % (2*self.box_half_width) \
                - self.box_half_width
        
    def rattle_one_pos(self,A=0.01):
        index = np.random.choice(self.indices_dynamic_atoms)
        self.pos[index,:] = self.pos[index,:] + A*np.random.randn(2)
        self.wrap()
        
    def rattle_all_pos(self,A=1):
        delta = A*np.random.randn(self.N,2)
        delta = np.where(self.filter,0,delta)
        self.pos += delta
        self.wrap()

    def set_positions(self,pos):
        self.pos = pos
        
    def get_positions(self):
        return self.pos.copy()
        
    def draw(self,ax,size=100):
        if self.plot_artists.get(ax,None) is None:
            colors = ['C1' if s else 'C0' for s in self.static]
            self.plot_artists[ax] = ax.scatter(self.pos[:,0],self.pos[:,1],s=size,c=colors)
        else:
            self.plot_artists[ax].set_offsets(self.pos)
        ax.set_title(f'E={self.energy:8.3f}')

def relax_for_basin_hopping(cluster,steps=100):
    
    test = cluster.copy()
    def energy_of_alpha(alpha,p):
        test.set_positions(cluster.get_positions() + alpha * p)
        return test.energy
    
    for i in range(steps):
        f = cluster.forces()
        fnorm = np.linalg.norm(f)
        if fnorm < 0.05:
            break
        p = f/fnorm
        
        alpha_opt = fmin(lambda alpha: energy_of_alpha(alpha,p), 0.1, disp=False)
            
        cluster.set_positions(cluster.get_positions() + alpha_opt * p)
