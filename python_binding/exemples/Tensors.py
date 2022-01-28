#%%
import quantit as qtt
import torch

#%%

# quantit offer several factory function to generate tensors,
# and several possible conservation laws.
# Currently The following consevations law can be used without writing any C++ extensions:
qtt.conserved.Z
qtt.conserved.ZC2
qtt.conserved.ZC3
qtt.conserved.ZC4
qtt.conserved.ZC5
qtt.conserved.ZC6
qtt.conserved.ZZ
qtt.conserved.ZZC2
qtt.conserved.ZZC3
qtt.conserved.ZZC4
qtt.conserved.ZZC5
qtt.conserved.ZZC6
qtt.conserved.C2
qtt.conserved.C3
qtt.conserved.C4
qtt.conserved.C5
qtt.conserved.C6
qtt.conserved.C2C2
qtt.conserved.C2C3
qtt.conserved.C2C4
qtt.conserved.C2C5
qtt.conserved.C2C6
# The Z stand for the natural numbers, It is good for conserved that value can be mapped onto the positive or negative values, zero included, such as angular momentum and particle number.
# The C stands for cyclical, and the number following it is the length of the cycle.
# for exemple C6 can take values from 0 to 5, and could be used to quantify the momentum of a particle on a 6 sites periodic system, such as an hexagonal plaquette.
# qtt.conserved.ZZ means that our conservation law can be specified by 2 integers. For exemple, this particular one is pertinent when both particle and spin is conserved.

# Those are so-called Abelian conservation laws, because they relate to abelian symmetry groups.
# The key property of such conserved quantity is that the tensor product of two states with well defined conserved quantities also has a well defined conserved conserved quantities.
# Such is not the case for every conserved quantities. For exemple, the total angular momentum doesn't, and isn't described by an Abelian group.

# A btensor (block tensor) defined by quantity is built of blocks of the same rank has the overall tensor. In essence,
# the full tensor is subdivided into sections along each of its dimensions. Each of the section has an associated conserved quantity and size.
# The blocks are formed by the intersection of the section in each dimensions. Each block has a one conserved quantity on each of its dimensions.
# Only the blocks that respect a selection rule are allowed to have non-zero elements.
# A block respect the selection rule if the conserved quantities of each of its dimensions sums up to a predetermined value.
# Forbiden blocks are not explicitly stored and permitted blocks of zeros need not be explicitly stored.
#%%
# to build a block tensor we must specify its shape:
# Each dimension of the tensor is defined by a list of conserved quantites and sizes.
# Let's consider a vector as an exemple:
Z = qtt.conserved.Z
V = qtt.sparse_zeros([[(2, Z(0)), (3, Z(1)), (1, Z(-1))]], Z(1))
# The tensor thus constructed is empty, it has 6 elements and the selection rules is Z(1).
# The first 2 element of the vector are associated with the conserved quantity Z(0), the next three Z(1) and the last one with Z(-1)
# Only the 3 middle elements can ever differ from zero.
FermionShape = qtt.btensor([[(1, Z(0)), (2, Z(1)), (1, Z(2))]], Z(0))
# Fermions shape describe the shape a ket on a single site could take
# This second exemple call a constructor of btensor instead of a factory function,
# and it build a vector in the local hilbert space of electrons
# This constructor is equivalent to sparse_zero.
# This vector has 4 elements, one with 0 particles, 2 with one particles, and one with two particles
#%%
# Conjugation also inverse the conserved quantities.
ConjFermionShape = FermionShape.conj()
# Conjugating this tensor lends us the shape a bra should have.
print([x for x in ConjFermionShape.sections_quantity(0)])
# btensor shape can be composed to create new more complicated shapes.
#%%
# The shape of tensor can be composed into tensors of higher rank.
# From the previous two vector shape, we can construct the shape of an operator in that Hilbert space.
Hilb_operatorShape = qtt.shape_from([FermionShape, ConjFermionShape])

print("Hilb_operator")
#%%
# There's a few different way to construct a tensor:
# block by block
c_up = qtt.shape_from([Hilb_operatorShape]) #We create a new empty tensor with the same shape as Hilb_operatorShape
c_up.set_selection_rule_(Z(-1)) #we set the selection rule to -1: 
# only the blocks that reduce the number of particle by one can be non-zero
c_up.blocks[0, 1] = torch.Tensor([[1, 0]]) #we populate with the correct value the two relevent blocks
c_up.blocks[1, 2] = torch.Tensor([[0], [1]])
print("c_up", c_up)
# or by converting a full tensor into a block tensor
c_dn = qtt.from_torch_tensor_like(
    Hilb_operatorShape.set_selection_rule(Z(-1)),
    torch.Tensor([[0, 0, 1, 0], [0, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0]]),
)
print("c_dn", c_dn)
# A downside of this second method is that null block that are allowed by the selection rule could be present.
# it's not a concern in this particular case.

c_up_dag = c_up.conj().t_()# t_ permute that last two dimensions of a tensor, it transpose matrices
c_dn_dag = c_dn.conj().t_()# the underscore signifies that the operation is done "in-place" no new tensors created by the operation (conj already created a new one that t_ acts on )


print(c_up_dag)
print(c_dn_dag)

#%%
# here we construct the number operator by performing the multiplication of two operator using tensordot
n_up = qtt.tensordot(c_up_dag,c_up,dims=([1],[0]))
# we can also use matmul for such simple case
n_dn = c_dn_dag.bmm(c_dn)

print(n_up)
print(n_dn)

# As we can see, the selection rule of the result is the sum of the selection rules of the tensor constracted.
# This is expected: the number operator doesn't change the number of particle in a state, it therefore has selection rule Z(0)
# %%
