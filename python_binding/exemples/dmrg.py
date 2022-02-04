#%%
import quantit as qtt
import torch
#%%

#Parameters that we initialize as torch tensors. When using tensors for parameters, they must be rank one.
# Using torch tensor rather than plain number mean that we could potentially differentiate dependent tensors with respect to these values.
U = torch.full((),4)
J = torch.full((),-1)
mu = torch.full((),2)
Hu = qtt.operators.Hubbard(U,mu,10)
He = qtt.operators.Heisenberg(J,10)
dmrg_opt = qtt.algorithms.dmrg_options()
# The dmrg_options structure bundle together all the adjustement knobs of DMRG
# The most important one are certainly the cutoff for the singular values and the convergence criterion for the energy
print("cutoff: ", dmrg_opt.cutoff, " Energy convergence: ", dmrg_opt.convergence_criterion)
# The maximum and minimum bond dimension can be controlled as well, by default the bond dimension is essentially unlimited.
print("maximum bond: ", dmrg_opt.maximum_bond)
#%%
#DMRG can then be called on the resulting MPO, a random starting state is automatically generated and optimized.
(Hu_E, Hu_psi) = qtt.algorithms.dmrg(Hu,dmrg_opt)
(He_E, He_psi) = qtt.algorithms.dmrg(He,dmrg_opt)
#A random state just like the one created internally by DMRG can be created in a few different ways
state_Hu = qtt.networks.random_MPS(   10 ,        4      ,        4)
#                                  length, bond dimension, physical dimensions
state_V = qtt.networks.random_MPS(      4        ,      [4,2,4]  )
#                                  bond dimension, Length & physical dimensions
state_MPO = qtt.networks.random_MPS(     4        ,         He)
#                                  bond dimension, MPO for length and physical dimensions
#DMRG can also be called with an externally built MPS.
He_E2 = qtt.algorithm.dmrg(He,state_MPO)
# %%

#And now with conservation laws and block tensors!
#Declare the shape for the physical indices
#Here we consider a basic one dimensionnal hubbard model,
# both charge and spin are conserved.
# Both quantities can be mapped onto integer in ]-\inf,\inf[, so a pair of integer (ZZ) does the trick.
cval = qtt.conserved.ZZ
#In order to construct the MPO, we ust supply the conserved quantity of the physical indices in the form of a(n empty) btensor.
FermionShape = qtt.btensor([[(1,cval(0,0)),(1,cval(1,1)),(1,cval(1,-1)),(1,cval(2,0))]],cval(0,0))
#                             empty state, one spin up  , one spin down, double occupied, a constraint for the tensor unimportant here
Hu = qtt.operators.Hubbard(U,mu,10,FermionShape)
#compared with DMRG without block tensor, we must adittionnally specify what constraint the state we optimze should respect.
#In this case we have DMRG solve for the state with 10 particles and 0 net spin and lowest energy.
# For the half-filled Hubbard model with 10 sites, that is the ground state.
#%%
(Hu_E, Hu_psi) = qtt.algorithms.dmrg(Hu,cval(10,0),dmrg_opt)

#%%
#Similarly to the case using torch tensors, we can generate a random MPS outside of DMRG.
#Again we have to specify the constraint for this state, as well as the physical dimensions.
state_Hu = qtt.networks.random_MPS(   10 ,        4      ,        FermionShape, cval(10,0))
#                                  length, bond dimension, physical dimensions, constraint
HardCoreBoson_Shape = qtt.btensor([[(1,cval(1,1)),(1,cval(1,-1))]],cval(0,0))
state_V = qtt.networks.random_MPS(        4      ,      [FermionShape,HardCoreBoson_Shape,FermionShape], cval(3,1)  )
#                                  bond dimension, Length & physical dimensions              , constraint
state_MPO = qtt.networks.random_MPS(     4        ,         Hu                            , cval(10,0))
#                                  bond dimension, MPO for length and physical dimensions, constraint
# %%
#sladarabgads
# %%
