import numpy as np
import scipy as sp
import itertools
from copy import copy, deepcopy
import ase 
import ase.io as aseio
import pickle

from tqdm import tqdm
import rascal
from rascal.representations import SphericalExpansion as SEXP
from rascal.representations import SphericalInvariants as SOAP
from rascal.neighbourlist.structure_manager import (
        mask_center_atoms_by_species, mask_center_atoms_by_id)
from rascal.utils import (get_radial_basis_covariance, get_radial_basis_pca, 
                          get_radial_basis_projections, get_optimal_radial_basis_hypers )
from scipy.optimize import minimize

"""
This is a (pretty rough) script that starts from the "quasi-degenerate manifold"
of Parsaeifard & Goedecker and demonstrates its relations with the degenerate 
pairs from Pozdnyakov et al. (2020). Some basic naming conventions because there
is too much "degeneracy" to avoid confusion. We call the manifolds that are 
formed by degenerate pairs "Donald" manifolds. The points where the degenerate
Donald pairs cross are part of "Doubly Donald" manifolds. The "P&G quasi-degenerate
manifolds" (that we show to be "orbits" around Doubly-Donald points) are
"Boris" manifolds. 

Michele Ceriotti 2022
"""

### "torsional energy" functions from Parsaeifard & Goedecker
def normalized_cross_product(a,b):
    # | x  y  z  |
    # | a0 a1 a2 |
    # | b0 b1 b2 |
    # = x(a1b2-a2b1) - y(a0b2-a2b0) + z(a0b1-a1b0)
    nhat = np.array([a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]])
    norm = np.dot(nhat,nhat)
    return nhat/np.sqrt(norm)

def torsion_angle(rxyz): # rxyz is an np.array of shape (nat,3) where the first atom is C
   perms = [[1,2,3],[1,2,4],[2,3,4],[1,3,4]]
   d = 0.
   for it1, perm in enumerate(perms):
        A = rxyz[perm[0]]
        B = rxyz[perm[1]]
        C = rxyz[perm[2]]
        nhat = normalized_cross_product(B-A, C-A)
        for it2 in range(it1+1, len(perms)):
            A_prime = rxyz[perms[it2][0]]
            B_prime = rxyz[perms[it2][1]]
            C_prime = rxyz[perms[it2][2]]
            nhat_prime = normalized_cross_product(B_prime-A_prime, C_prime-A_prime)
            d += np.dot(nhat, nhat_prime)**2
   return d
bohr2a = 0.529177
def fourbody(struc, conversion=1.0/bohr2a):    
    return torsion_angle(struc.positions*conversion)

### converts structures from Parsaeifard & Goedecker to a standard format
def fix_structures(frames, conversion=bohr2a):
    for i, f in enumerate(frames):            
        mask_center_atoms_by_id(f, [0])
        f.positions *= conversion
        f.info["fourbody"] = fourbody(f)
        f.cell = [100,100,100]
        f.positions += 50
        f.wrap(eps=1e-12)
    return frames

print("Loading train structures (for basis optimization)")
frames_train = aseio.read('data/train_without_traj.xyz', ":")
nframes_train = len(frames_train)
fix_structures(frames_train, bohr2a);

# these are kind of over-converged SOAP parameters. 
# max_radial = 4 and max_angular = 4 give overall similar results 
# IF used with an optimal radial basis
Hsoap = {
    'soap_type': 'PowerSpectrum',
    'interaction_cutoff': 2.5,
    'max_radial': 8,
    'max_angular': 8,
    'gaussian_sigma_constant': 0.2,
    'gaussian_sigma_type': 'Constant',
    'cutoff_smooth_width': 0.0,
    'radial_basis': 'GTO',
    'normalize' : False
}

# this finds the data-driven optimal basis, as per https://aip.scitation.org/doi/abs/10.1063/5.0057229
Hsoap = get_optimal_radial_basis_hypers(Hsoap, frames_train[::10], expanded_max_radial=20)

# this is just to avoid getting C density coefficients, given it's always the same. 
nnl = np.array(list(itertools.product(range(Hsoap['max_radial']), range(Hsoap['max_radial']), 
                                      range(Hsoap['max_angular']+1))), dtype=int)
Hsoap['coefficient_subselection']= {       
        "a" : np.ones(Hsoap['max_radial']**2*(Hsoap['max_angular']+1), dtype=int).tolist(),
        "b" : np.ones(Hsoap['max_radial']**2*(Hsoap['max_angular']+1), dtype=int).tolist(),
        "n1": nnl[:,0].tolist(),
        "n2": nnl[:,1].tolist(),
        "l":  nnl[:,2].tolist()
    }

soap = SOAP(**Hsoap)
Hsoap["compute_gradients"] = True
gsoap = SOAP(**Hsoap)


#### SOME HELPER FUNCTIONS
def soap_analytical_deri(struc, rep):
    ds = []
    try:
        struc.arrays.pop("center_atoms_mask")
    except:
        pass
    nneigh = struc.positions.shape[0]-1
    tsoap = rep.transform(struc)
    x = tsoap.get_features(rep)[0]
    ij = tsoap.get_gradients_info()
    xij = tsoap.get_features_gradient(rep).reshape((ij.shape[0], 3, -1))
    dx = np.zeros((nneigh, 3, len(x)))
    for i in range(nneigh):
        dx[ij[i+1,2]-1] = xij[i+1]
    return x, dx.reshape(3*nneigh,-1)

def norot(struc, dir):
    # takes a set of orthogonal directions and eliminates the rotations of the given environment around its center
    px = struc.positions[1:] - struc.positions[0]
    ux, sx, vtx = np.linalg.svd(px)
    rv = np.asarray([np.cross(px, vtx[i]).flatten() for i in range(3)]).T
    rv/= np.sqrt((rv**2).sum(axis=0))    
    odir, _ = np.linalg.qr(dir - rv@(rv.T@dir))
    return  odir, rv

def norad(struc, dir):
    # takes a set of orthogonal directions and eliminates the displacements along the radial direction
    px = struc.positions[1:] - struc.positions[0]
    rv = np.zeros((px.shape[1]*px.shape[0], px.shape[0]))
    for i in range(len(px)):
        rv[3*i:3*(i+1),i] = px[i]/np.sqrt(px[i]@px[i])
    odir, _ = np.linalg.qr(dir - rv@(rv.T@dir))
    return odir, rv

obase = np.diag(1+np.linspace(0,1,12))+np.multiply.outer(np.linspace(0,1,12),np.linspace(0,1,12))
def get_jval_norot(struc, rep):
    x, dx = soap_analytical_deri(struc, rep=rep)
    onorot, rv = norot(struc, obase[:,:9])
    dxnorot = onorot.T @ dx    
    return x, dx, onorot, sp.linalg.svdvals(dxnorot)


def pos2struc(pos):
    struc = ase.Atoms("CH4")
    struc.positions[1:] = pos.reshape((-1,3)) 
    mask_center_atoms_by_id(struc, [0])
    struc.cell = [100,100,100]
    struc.positions += 50    
    return struc

# gets the (rotationally purified) Jacobian condition number for a set of H positions
def get_jcn(pos):
    _, _, _, jeva = get_jval_norot(pos2struc(pos), gsoap)
    return jeva[-1]/jeva[0]

# loss that is optimized to find a doubly-donald point close to the PG boris manifold
def close_donald_loss(pos, ref, alpha=0.5):
    dq = np.linalg.norm(pos-ref)
    cn = get_jcn(pos)
    print(f'Donald search: CN {cn} Distance to ref: {dq}', end='\r')
    return dq*alpha+(1-alpha)*cn

# loads the boris manifold from P&G
manifold = aseio.read('data/pg_boris.xyz', ':')
fix_structures(manifold);

# finds the structure at the center of boris
mean_manifold = manifold[0].copy()
for a in manifold[1:]:
    mean_manifold.positions += a.positions
mean_manifold.positions /= len(manifold)
mean_manifold.positions -= mean_manifold.positions[0]
mean_manifold.positions += 50

# optimizes to find the double-donald.
# NB: this would also look for symmetric points that have 
# a physical reason to have a zero singular value. luckily 
# P&G boris is already very close to dd
print("Looking for the Double Donald")
x0 = (mean_manifold.positions[1:]-mean_manifold.positions[0]).flatten()
opt = minimize(close_donald_loss, args=(x0, 1e-5), 
               x0=x0, tol=1e-12, method="Nelder-Mead", options=dict(maxiter=1000))

# ... and finishes off relaxing the "being close to PG center" requirement
opt = minimize(close_donald_loss, args=(x0, 1e-12), 
               x0=opt['x'], tol=1e-16, method="Nelder-Mead", options=dict(maxiter=2000))            

# and here it is, in all its degenerate glory
doubledonald = pos2struc(opt.x)

### Now we repeat the construction by B&P, but we explicitly set the starting
### point at a predefined distance from the double-donald. 1.0 is the distance
### of the original boris, but we will see there are many boris

# re-define SOAP so we can experiment with different parameters. 
# these changes would also shift the position of double-donald by a tiny bit,
# but it doesn't matter because we look for a "consistent" boris, and the only 
# thing that would change is the position of the starting point of the orbit
boris_family = {}
for s in [0.5, 0.2, 0.1]:
    Hsoap = {
        'soap_type': 'PowerSpectrum',
        'interaction_cutoff': 2.5,
        'max_radial': 8,
        'max_angular': 8,
        'gaussian_sigma_constant': s,
        'gaussian_sigma_type': 'Constant',
        'cutoff_smooth_width': 0.0,
        'radial_basis': 'GTO',
        'normalize' : False
    }
    soap_def=f"s{s}-n8-l8"
    Hsoap = get_optimal_radial_basis_hypers(Hsoap, frames_train[::10], expanded_max_radial=20)
    nnl = np.array(list(itertools.product(range(Hsoap['max_radial']), range(Hsoap['max_radial']), 
                                        range(Hsoap['max_angular']+1))), dtype=int)
    Hsoap['coefficient_subselection']= {       
            "a" : np.ones(Hsoap['max_radial']**2*(Hsoap['max_angular']+1), dtype=int).tolist(),
            "b" : np.ones(Hsoap['max_radial']**2*(Hsoap['max_angular']+1), dtype=int).tolist(),
            "n1": nnl[:,0].tolist(),
            "n2": nnl[:,1].tolist(),
            "l":  nnl[:,2].tolist()
        }
    soap = SOAP(**Hsoap)
    Hsoap["compute_gradients"] = True
    gsoap = SOAP(**Hsoap)

    # this is the distance from double-donald
    for ddd in [1.0, 0.1, 0.5, 1.5]:
        print(f"Looking for a Boris: SOAP {soap_def}, orbit distance {ddd}")
        frame = manifold[0].copy()
        frame.positions = doubledonald.positions + ddd* (frame.positions-doubledonald.positions)
        
        # initialize the orbit
        my_boris = [frame.copy()]
        uold = np.zeros(12); uold[0] = 1
        x0, dx0 = soap_analytical_deri(frame, rep=gsoap)
        for i in range(20000):
            x, dx = soap_analytical_deri(frame, rep=gsoap)
            onorot, rv = norot(frame, obase[:,:9])
            dxnorot = onorot.T @ dx  
            u, s, v = sp.linalg.svd(dxnorot)
            frame.info["singular_values"] = s
            frame.info["fourbody"] = fourbody(frame)
            
            # this is the displacement along the "quasi-singular direction"
            ufull = onorot @ u[:,-1]
            # the direction of the singular vector is random - we want to keep moving in the same direction
            ufull *= np.sign(ufull@uold)    
            uold = ufull            
            # deform by a small finite amount
            du = 1e-4
            frame.positions[1:] += ufull.reshape((4,3))*du
            if i%100==0: # we take small steps but save one in 100 frames                
                print(" Local CN: ", s[0]/s[-1], " dist to start ", np.linalg.norm(x-x0), end='\r')            
                my_boris.append(frame.copy())

        boris_family[(ddd, soap_def)]=my_boris

        # dumps the boris orbits
        pickle.dump(boris_family, open("data/quasiconstant-manifolds.pickle", "wb"))
