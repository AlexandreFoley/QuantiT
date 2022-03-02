
#%%


# %%
import quantit as qtt
import torch
# constant values for the Hubbard hamiltonian
U = torch.full([],8)
mu = torch.full([],4)
t = torch.full([],-1)


#%%
c_up,c_dn,F,id = qtt.operators.fermions()
c_dag_up = c_up.conj().permute([1,0])
c_dag_dn = c_dn.conj().permute([1,0])
n_up = torch.matmul(c_dag_up,c_up)
n_dn = torch.matmul(c_dag_dn,c_dn)
n_tot = n_up+n_dn
#%% We will construct the translationnaly invariant tensor for a 1D hubbard model
Hubbard_tensor = torch.zeros(6,4,6,4) #This can be understood as a 6 by 6 matrix of operators (themselves of size 4 by 4)
#We put the local term on the hamiltonian in the bottom left corner of the matrix of operator
# The idea behind this construction is that the actual hamiltonian is constructed in the bottom left corner of the matrix of operator 
# built by multiplying all the tensor of the MPO together.
Hubbard_tensor[5,:,0,:] = torch.tensordot(n_up,n_dn,dims=([1],[0]))*U - mu*n_tot #The local term, e-e interaction and chemical potential
Hubbard_tensor[0,:,0,:] = torch.eye(4)
Hubbard_tensor[5,:,5,:] = torch.eye(4)
Hubbard_tensor[1,:,0,:] = c_up
Hubbard_tensor[2,:,0,:] = c_dn
Hubbard_tensor[3,:,0,:] = c_dag_up
Hubbard_tensor[4,:,0,:] = c_dag_dn
Hubbard_tensor[5,:,1,:] = torch.matmul(c_dag_up,F)
Hubbard_tensor[5,:,2,:] = torch.matmul(c_dag_dn,F)
Hubbard_tensor[5,:,3,:] = torch.matmul(F,c_up)
Hubbard_tensor[5,:,4,:] = torch.matmul(F,c_dn)
Hubbard_Hamil = qtt.networks.MPO(10,Hubbard_tensor) # build a 10 site 1D hubbard model by repeating the supplied tensor
#Adjustement to the edges for finite sized systems
Hubbard_tensor[0] = Hubbard_tensor[0][5:,:,:,:] 
Hubbard_tensor[9] = Hubbard_tensor[9][:,:,:,5:] 
# %%
#With btensor
cval = qtt.conserved.ZZ
LHilSpace = qtt.btensor([[(1,cval(0,0)),(1,cval(1,1)),(1,cval(1,-1)),(1,cval(2,0))]],cval(0,0))
# c_up,c_dn,F,id = qtt.operators.fermions(qtt.shape_from([LHilSpace,LHilSpace.conj()]))
# %%
# %%
