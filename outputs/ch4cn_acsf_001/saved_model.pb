��
�2�2
,
Abs
x"T
y"T"
Ttype:

2	
/
Acos
x"T
y"T"
Ttype:

2	
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	��
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�

BatchDataset
input_dataset

batch_size	

handle"
output_types
list(type)(0" 
output_shapeslist(shape)(0
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
,
Cos
x"T
y"T"
Ttype:

2
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
B
Equal
x"T
y"T
z
"
Ttype:
2	
�
,
Exp
x"T
y"T"
Ttype:

2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
p
GatherNd
params"Tparams
indices"Tindices
output"Tparams"
Tparamstype"
Tindicestype:
2	
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
:
InvertPermutation
x"T
y"T"
Ttype0:
2	
�
Iterator

handle"
shared_namestring"
	containerstring"
output_types
list(type)(0" 
output_shapeslist(shape)(0�
�
IteratorGetNext
iterator

components2output_types"
output_types
list(type)(0" 
output_shapeslist(shape)(0�
C
IteratorToStringHandle
resource_handle
string_handle�
:
Less
x"T
y"T
z
"
Ttype:
2	
\
ListDiff
x"T
y"T
out"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
$

LogicalAnd
x

y

z
�


LogicalNot
x

y

,
MakeIterator
dataset
iterator�
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
B
MatrixSetDiag

input"T
diagonal"T
output"T"	
Ttype
;
Maximum
x"T
y"T
z"T"
Ttype:

2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
;
Minimum
x"T
y"T
z"T"
Ttype:

2	�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
6
Pow
x"T
y"T
z"T"
Ttype:

2	
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
)
Rank

input"T

output"	
Ttype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
s
	ScatterNd
indices"Tindices
updates"T
shape"Tindices
output"T"	
Ttype"
Tindicestype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
�
ShuffleDataset
input_dataset
buffer_size	
seed		
seed2	

handle"$
reshuffle_each_iterationbool("
output_types
list(type)(0" 
output_shapeslist(shape)(0
0
Sigmoid
x"T
y"T"
Ttype:

2
=
SigmoidGrad
y"T
dy"T
z"T"
Ttype:

2
/
Sign
x"T
y"T"
Ttype:

2	
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
-
Sqrt
x"T
y"T"
Ttype:

2
1
Square
x"T
y"T"
Ttype:

2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
TensorSliceDataset

components2Toutput_types

handle"
Toutput_types
list(type)(0" 
output_shapeslist(shape)(0�
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
f
TopKV2

input"T
k
values"T
indices"
sortedbool("
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �
E
Where

input"T	
index	"%
Ttype0
:
2	
"serve*1.9.02v1.9.0-0-g25c197e023��
}
Data/DescriptorsPlaceholder*
dtype0*,
_output_shapes
:����������
*!
shape:����������

v
Data/Atomic-numbersPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
r
Data/PropertiesPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
P
Data/bufferPlaceholder*
dtype0	*
_output_shapes
:*
shape:
K
	Data/seedConst*
value	B	 R *
dtype0	*
_output_shapes
: 
L

Data/seed2Const*
value	B	 R *
dtype0	*
_output_shapes
: 
Q
Data/batch_sizeConst*
dtype0	*
_output_shapes
: *
value	B	 R|
�
Data/IteratorIterator*
	container *
_output_shapes
: *
output_types
2*
shared_name *Q
output_shapes@
>:����������
:���������:���������
\
Data/IteratorToStringHandleIteratorToStringHandleData/Iterator*
_output_shapes
: 
�
Data/IteratorGetNextIteratorGetNextData/Iterator*Q
output_shapes@
>:����������
:���������:���������*R
_output_shapes@
>:����������
:���������:���������*
output_types
2
o
Weights/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"F   s  
b
Weights/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
d
Weights/truncated_normal/stddevConst*
valueB
 *U�<*
dtype0*
_output_shapes
: 
�
(Weights/truncated_normal/TruncatedNormalTruncatedNormalWeights/truncated_normal/shape*

seed *
T0*
dtype0*
seed2 *
_output_shapes
:	F�

�
Weights/truncated_normal/mulMul(Weights/truncated_normal/TruncatedNormalWeights/truncated_normal/stddev*
_output_shapes
:	F�
*
T0
�
Weights/truncated_normalAddWeights/truncated_normal/mulWeights/truncated_normal/mean*
T0*
_output_shapes
:	F�

�
Weights/weight_in
VariableV2*
dtype0*
	container *
_output_shapes
:	F�
*
shape:	F�
*
shared_name 
�
Weights/weight_in/AssignAssignWeights/weight_inWeights/truncated_normal*
T0*$
_class
loc:@Weights/weight_in*
validate_shape(*
_output_shapes
:	F�
*
use_locking(
�
Weights/weight_in/readIdentityWeights/weight_in*
T0*$
_class
loc:@Weights/weight_in*
_output_shapes
:	F�

Z
Weights/zerosConst*
valueBF*    *
dtype0*
_output_shapes
:F
{
Weights/bias_in
VariableV2*
dtype0*
	container *
_output_shapes
:F*
shape:F*
shared_name 
�
Weights/bias_in/AssignAssignWeights/bias_inWeights/zeros*
use_locking(*
T0*"
_class
loc:@Weights/bias_in*
validate_shape(*
_output_shapes
:F
z
Weights/bias_in/readIdentityWeights/bias_in*
T0*"
_class
loc:@Weights/bias_in*
_output_shapes
:F
q
 Weights/truncated_normal_1/shapeConst*
valueB"h   F   *
dtype0*
_output_shapes
:
d
Weights/truncated_normal_1/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
f
!Weights/truncated_normal_1/stddevConst*
valueB
 *g��=*
dtype0*
_output_shapes
: 
�
*Weights/truncated_normal_1/TruncatedNormalTruncatedNormal Weights/truncated_normal_1/shape*
T0*
dtype0*
seed2 *
_output_shapes

:hF*

seed 
�
Weights/truncated_normal_1/mulMul*Weights/truncated_normal_1/TruncatedNormal!Weights/truncated_normal_1/stddev*
T0*
_output_shapes

:hF
�
Weights/truncated_normal_1AddWeights/truncated_normal_1/mulWeights/truncated_normal_1/mean*
T0*
_output_shapes

:hF
�
Weights/weight_hidden_1
VariableV2*
shape
:hF*
shared_name *
dtype0*
	container *
_output_shapes

:hF
�
Weights/weight_hidden_1/AssignAssignWeights/weight_hidden_1Weights/truncated_normal_1*
T0**
_class 
loc:@Weights/weight_hidden_1*
validate_shape(*
_output_shapes

:hF*
use_locking(
�
Weights/weight_hidden_1/readIdentityWeights/weight_hidden_1*
T0**
_class 
loc:@Weights/weight_hidden_1*
_output_shapes

:hF
\
Weights/zeros_1Const*
valueBh*    *
dtype0*
_output_shapes
:h
�
Weights/bias_hidden_1
VariableV2*
shape:h*
shared_name *
dtype0*
	container *
_output_shapes
:h
�
Weights/bias_hidden_1/AssignAssignWeights/bias_hidden_1Weights/zeros_1*
use_locking(*
T0*(
_class
loc:@Weights/bias_hidden_1*
validate_shape(*
_output_shapes
:h
�
Weights/bias_hidden_1/readIdentityWeights/bias_hidden_1*
T0*(
_class
loc:@Weights/bias_hidden_1*
_output_shapes
:h
q
 Weights/truncated_normal_2/shapeConst*
valueB"   h   *
dtype0*
_output_shapes
:
d
Weights/truncated_normal_2/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
f
!Weights/truncated_normal_2/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *���=
�
*Weights/truncated_normal_2/TruncatedNormalTruncatedNormal Weights/truncated_normal_2/shape*
dtype0*
seed2 *
_output_shapes

:h*

seed *
T0
�
Weights/truncated_normal_2/mulMul*Weights/truncated_normal_2/TruncatedNormal!Weights/truncated_normal_2/stddev*
T0*
_output_shapes

:h
�
Weights/truncated_normal_2AddWeights/truncated_normal_2/mulWeights/truncated_normal_2/mean*
T0*
_output_shapes

:h
�
Weights/weight_out
VariableV2*
dtype0*
	container *
_output_shapes

:h*
shape
:h*
shared_name 
�
Weights/weight_out/AssignAssignWeights/weight_outWeights/truncated_normal_2*
T0*%
_class
loc:@Weights/weight_out*
validate_shape(*
_output_shapes

:h*
use_locking(
�
Weights/weight_out/readIdentityWeights/weight_out*
T0*%
_class
loc:@Weights/weight_out*
_output_shapes

:h
\
Weights/zeros_2Const*
dtype0*
_output_shapes
:*
valueB*    
|
Weights/bias_out
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
�
Weights/bias_out/AssignAssignWeights/bias_outWeights/zeros_2*
use_locking(*
T0*#
_class
loc:@Weights/bias_out*
validate_shape(*
_output_shapes
:
}
Weights/bias_out/readIdentityWeights/bias_out*
T0*#
_class
loc:@Weights/bias_out*
_output_shapes
:
q
 Weights/truncated_normal_3/shapeConst*
valueB"F   s  *
dtype0*
_output_shapes
:
d
Weights/truncated_normal_3/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
f
!Weights/truncated_normal_3/stddevConst*
valueB
 *U�<*
dtype0*
_output_shapes
: 
�
*Weights/truncated_normal_3/TruncatedNormalTruncatedNormal Weights/truncated_normal_3/shape*
T0*
dtype0*
seed2 *
_output_shapes
:	F�
*

seed 
�
Weights/truncated_normal_3/mulMul*Weights/truncated_normal_3/TruncatedNormal!Weights/truncated_normal_3/stddev*
_output_shapes
:	F�
*
T0
�
Weights/truncated_normal_3AddWeights/truncated_normal_3/mulWeights/truncated_normal_3/mean*
T0*
_output_shapes
:	F�

�
Weights/weight_in_1
VariableV2*
dtype0*
	container *
_output_shapes
:	F�
*
shape:	F�
*
shared_name 
�
Weights/weight_in_1/AssignAssignWeights/weight_in_1Weights/truncated_normal_3*
validate_shape(*
_output_shapes
:	F�
*
use_locking(*
T0*&
_class
loc:@Weights/weight_in_1
�
Weights/weight_in_1/readIdentityWeights/weight_in_1*
T0*&
_class
loc:@Weights/weight_in_1*
_output_shapes
:	F�

\
Weights/zeros_3Const*
valueBF*    *
dtype0*
_output_shapes
:F
}
Weights/bias_in_1
VariableV2*
dtype0*
	container *
_output_shapes
:F*
shape:F*
shared_name 
�
Weights/bias_in_1/AssignAssignWeights/bias_in_1Weights/zeros_3*
validate_shape(*
_output_shapes
:F*
use_locking(*
T0*$
_class
loc:@Weights/bias_in_1
�
Weights/bias_in_1/readIdentityWeights/bias_in_1*
T0*$
_class
loc:@Weights/bias_in_1*
_output_shapes
:F
q
 Weights/truncated_normal_4/shapeConst*
valueB"h   F   *
dtype0*
_output_shapes
:
d
Weights/truncated_normal_4/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
f
!Weights/truncated_normal_4/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *g��=
�
*Weights/truncated_normal_4/TruncatedNormalTruncatedNormal Weights/truncated_normal_4/shape*
T0*
dtype0*
seed2 *
_output_shapes

:hF*

seed 
�
Weights/truncated_normal_4/mulMul*Weights/truncated_normal_4/TruncatedNormal!Weights/truncated_normal_4/stddev*
T0*
_output_shapes

:hF
�
Weights/truncated_normal_4AddWeights/truncated_normal_4/mulWeights/truncated_normal_4/mean*
T0*
_output_shapes

:hF
�
Weights/weight_hidden_1_1
VariableV2*
dtype0*
	container *
_output_shapes

:hF*
shape
:hF*
shared_name 
�
 Weights/weight_hidden_1_1/AssignAssignWeights/weight_hidden_1_1Weights/truncated_normal_4*
use_locking(*
T0*,
_class"
 loc:@Weights/weight_hidden_1_1*
validate_shape(*
_output_shapes

:hF
�
Weights/weight_hidden_1_1/readIdentityWeights/weight_hidden_1_1*
_output_shapes

:hF*
T0*,
_class"
 loc:@Weights/weight_hidden_1_1
\
Weights/zeros_4Const*
valueBh*    *
dtype0*
_output_shapes
:h
�
Weights/bias_hidden_1_1
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:h*
shape:h
�
Weights/bias_hidden_1_1/AssignAssignWeights/bias_hidden_1_1Weights/zeros_4*
use_locking(*
T0**
_class 
loc:@Weights/bias_hidden_1_1*
validate_shape(*
_output_shapes
:h
�
Weights/bias_hidden_1_1/readIdentityWeights/bias_hidden_1_1*
T0**
_class 
loc:@Weights/bias_hidden_1_1*
_output_shapes
:h
q
 Weights/truncated_normal_5/shapeConst*
valueB"   h   *
dtype0*
_output_shapes
:
d
Weights/truncated_normal_5/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
f
!Weights/truncated_normal_5/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
*Weights/truncated_normal_5/TruncatedNormalTruncatedNormal Weights/truncated_normal_5/shape*

seed *
T0*
dtype0*
seed2 *
_output_shapes

:h
�
Weights/truncated_normal_5/mulMul*Weights/truncated_normal_5/TruncatedNormal!Weights/truncated_normal_5/stddev*
T0*
_output_shapes

:h
�
Weights/truncated_normal_5AddWeights/truncated_normal_5/mulWeights/truncated_normal_5/mean*
T0*
_output_shapes

:h
�
Weights/weight_out_1
VariableV2*
shape
:h*
shared_name *
dtype0*
	container *
_output_shapes

:h
�
Weights/weight_out_1/AssignAssignWeights/weight_out_1Weights/truncated_normal_5*
use_locking(*
T0*'
_class
loc:@Weights/weight_out_1*
validate_shape(*
_output_shapes

:h
�
Weights/weight_out_1/readIdentityWeights/weight_out_1*
T0*'
_class
loc:@Weights/weight_out_1*
_output_shapes

:h
\
Weights/zeros_5Const*
valueB*    *
dtype0*
_output_shapes
:
~
Weights/bias_out_1
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
�
Weights/bias_out_1/AssignAssignWeights/bias_out_1Weights/zeros_5*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*%
_class
loc:@Weights/bias_out_1
�
Weights/bias_out_1/readIdentityWeights/bias_out_1*
T0*%
_class
loc:@Weights/bias_out_1*
_output_shapes
:
q
 Weights/truncated_normal_6/shapeConst*
valueB"F   s  *
dtype0*
_output_shapes
:
d
Weights/truncated_normal_6/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
f
!Weights/truncated_normal_6/stddevConst*
valueB
 *U�<*
dtype0*
_output_shapes
: 
�
*Weights/truncated_normal_6/TruncatedNormalTruncatedNormal Weights/truncated_normal_6/shape*
dtype0*
seed2 *
_output_shapes
:	F�
*

seed *
T0
�
Weights/truncated_normal_6/mulMul*Weights/truncated_normal_6/TruncatedNormal!Weights/truncated_normal_6/stddev*
_output_shapes
:	F�
*
T0
�
Weights/truncated_normal_6AddWeights/truncated_normal_6/mulWeights/truncated_normal_6/mean*
T0*
_output_shapes
:	F�

�
Weights/weight_in_2
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:	F�
*
shape:	F�

�
Weights/weight_in_2/AssignAssignWeights/weight_in_2Weights/truncated_normal_6*
use_locking(*
T0*&
_class
loc:@Weights/weight_in_2*
validate_shape(*
_output_shapes
:	F�

�
Weights/weight_in_2/readIdentityWeights/weight_in_2*
_output_shapes
:	F�
*
T0*&
_class
loc:@Weights/weight_in_2
\
Weights/zeros_6Const*
valueBF*    *
dtype0*
_output_shapes
:F
}
Weights/bias_in_2
VariableV2*
shape:F*
shared_name *
dtype0*
	container *
_output_shapes
:F
�
Weights/bias_in_2/AssignAssignWeights/bias_in_2Weights/zeros_6*
use_locking(*
T0*$
_class
loc:@Weights/bias_in_2*
validate_shape(*
_output_shapes
:F
�
Weights/bias_in_2/readIdentityWeights/bias_in_2*
T0*$
_class
loc:@Weights/bias_in_2*
_output_shapes
:F
q
 Weights/truncated_normal_7/shapeConst*
dtype0*
_output_shapes
:*
valueB"h   F   
d
Weights/truncated_normal_7/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
f
!Weights/truncated_normal_7/stddevConst*
valueB
 *g��=*
dtype0*
_output_shapes
: 
�
*Weights/truncated_normal_7/TruncatedNormalTruncatedNormal Weights/truncated_normal_7/shape*
dtype0*
seed2 *
_output_shapes

:hF*

seed *
T0
�
Weights/truncated_normal_7/mulMul*Weights/truncated_normal_7/TruncatedNormal!Weights/truncated_normal_7/stddev*
_output_shapes

:hF*
T0
�
Weights/truncated_normal_7AddWeights/truncated_normal_7/mulWeights/truncated_normal_7/mean*
_output_shapes

:hF*
T0
�
Weights/weight_hidden_1_2
VariableV2*
shape
:hF*
shared_name *
dtype0*
	container *
_output_shapes

:hF
�
 Weights/weight_hidden_1_2/AssignAssignWeights/weight_hidden_1_2Weights/truncated_normal_7*
use_locking(*
T0*,
_class"
 loc:@Weights/weight_hidden_1_2*
validate_shape(*
_output_shapes

:hF
�
Weights/weight_hidden_1_2/readIdentityWeights/weight_hidden_1_2*
T0*,
_class"
 loc:@Weights/weight_hidden_1_2*
_output_shapes

:hF
\
Weights/zeros_7Const*
valueBh*    *
dtype0*
_output_shapes
:h
�
Weights/bias_hidden_1_2
VariableV2*
shape:h*
shared_name *
dtype0*
	container *
_output_shapes
:h
�
Weights/bias_hidden_1_2/AssignAssignWeights/bias_hidden_1_2Weights/zeros_7*
T0**
_class 
loc:@Weights/bias_hidden_1_2*
validate_shape(*
_output_shapes
:h*
use_locking(
�
Weights/bias_hidden_1_2/readIdentityWeights/bias_hidden_1_2*
T0**
_class 
loc:@Weights/bias_hidden_1_2*
_output_shapes
:h
q
 Weights/truncated_normal_8/shapeConst*
valueB"   h   *
dtype0*
_output_shapes
:
d
Weights/truncated_normal_8/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
f
!Weights/truncated_normal_8/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
*Weights/truncated_normal_8/TruncatedNormalTruncatedNormal Weights/truncated_normal_8/shape*
T0*
dtype0*
seed2 *
_output_shapes

:h*

seed 
�
Weights/truncated_normal_8/mulMul*Weights/truncated_normal_8/TruncatedNormal!Weights/truncated_normal_8/stddev*
T0*
_output_shapes

:h
�
Weights/truncated_normal_8AddWeights/truncated_normal_8/mulWeights/truncated_normal_8/mean*
T0*
_output_shapes

:h
�
Weights/weight_out_2
VariableV2*
dtype0*
	container *
_output_shapes

:h*
shape
:h*
shared_name 
�
Weights/weight_out_2/AssignAssignWeights/weight_out_2Weights/truncated_normal_8*
use_locking(*
T0*'
_class
loc:@Weights/weight_out_2*
validate_shape(*
_output_shapes

:h
�
Weights/weight_out_2/readIdentityWeights/weight_out_2*
T0*'
_class
loc:@Weights/weight_out_2*
_output_shapes

:h
\
Weights/zeros_8Const*
valueB*    *
dtype0*
_output_shapes
:
~
Weights/bias_out_2
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
�
Weights/bias_out_2/AssignAssignWeights/bias_out_2Weights/zeros_8*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*%
_class
loc:@Weights/bias_out_2
�
Weights/bias_out_2/readIdentityWeights/bias_out_2*
_output_shapes
:*
T0*%
_class
loc:@Weights/bias_out_2
l
Model/zeros_like/ShapeShapeData/IteratorGetNext:1*
T0*
out_type0*
_output_shapes
:
[
Model/zeros_like/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Model/zeros_likeFillModel/zeros_like/ShapeModel/zeros_like/Const*
T0*

index_type0*'
_output_shapes
:���������
M
Model/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
V
Model/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B : 
r
Model/ExpandDims
ExpandDimsModel/ConstModel/ExpandDims/dim*
_output_shapes
:*

Tdim0*
T0
p
Model/EqualEqualData/IteratorGetNext:1Model/ExpandDims*
T0*'
_output_shapes
:���������
S
Model/WhereWhereModel/Equal*'
_output_shapes
:���������*
T0

`

Model/CastCastModel/Where*

DstT0*'
_output_shapes
:���������*

SrcT0	
�
Model/GatherNdGatherNdData/IteratorGetNext
Model/Cast*
Tindices0*
Tparams0*(
_output_shapes
:����������

U
Model/transpose/RankRankWeights/weight_in/read*
T0*
_output_shapes
: 
W
Model/transpose/sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
h
Model/transpose/subSubModel/transpose/RankModel/transpose/sub/y*
T0*
_output_shapes
: 
]
Model/transpose/Range/startConst*
value	B : *
dtype0*
_output_shapes
: 
]
Model/transpose/Range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
�
Model/transpose/RangeRangeModel/transpose/Range/startModel/transpose/RankModel/transpose/Range/delta*
_output_shapes
:*

Tidx0
m
Model/transpose/sub_1SubModel/transpose/subModel/transpose/Range*
T0*
_output_shapes
:
�
Model/transpose	TransposeWeights/weight_in/readModel/transpose/sub_1*
Tperm0*
T0*
_output_shapes
:	�
F
c
Model/Tensordot/ShapeShapeModel/GatherNd*
_output_shapes
:*
T0*
out_type0
V
Model/Tensordot/RankConst*
dtype0*
_output_shapes
: *
value	B :
^
Model/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:
`
Model/Tensordot/GreaterEqual/yConst*
dtype0*
_output_shapes
: *
value	B : 
�
Model/Tensordot/GreaterEqualGreaterEqualModel/Tensordot/axesModel/Tensordot/GreaterEqual/y*
_output_shapes
:*
T0
n
Model/Tensordot/CastCastModel/Tensordot/GreaterEqual*

SrcT0
*

DstT0*
_output_shapes
:
k
Model/Tensordot/mulMulModel/Tensordot/CastModel/Tensordot/axes*
T0*
_output_shapes
:
X
Model/Tensordot/Less/yConst*
value	B : *
dtype0*
_output_shapes
: 
o
Model/Tensordot/LessLessModel/Tensordot/axesModel/Tensordot/Less/y*
T0*
_output_shapes
:
h
Model/Tensordot/Cast_1CastModel/Tensordot/Less*

DstT0*
_output_shapes
:*

SrcT0

k
Model/Tensordot/addAddModel/Tensordot/axesModel/Tensordot/Rank*
T0*
_output_shapes
:
n
Model/Tensordot/mul_1MulModel/Tensordot/Cast_1Model/Tensordot/add*
T0*
_output_shapes
:
m
Model/Tensordot/add_1AddModel/Tensordot/mulModel/Tensordot/mul_1*
_output_shapes
:*
T0
]
Model/Tensordot/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
]
Model/Tensordot/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Model/Tensordot/rangeRangeModel/Tensordot/range/startModel/Tensordot/RankModel/Tensordot/range/delta*
_output_shapes
:*

Tidx0
�
Model/Tensordot/ListDiffListDiffModel/Tensordot/rangeModel/Tensordot/add_1*
T0*
out_idx0*2
_output_shapes 
:���������:���������
_
Model/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot/GatherV2GatherV2Model/Tensordot/ShapeModel/Tensordot/ListDiffModel/Tensordot/GatherV2/axis*
Tparams0*#
_output_shapes
:���������*
Taxis0*
Tindices0
a
Model/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot/GatherV2_1GatherV2Model/Tensordot/ShapeModel/Tensordot/add_1Model/Tensordot/GatherV2_1/axis*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0
_
Model/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
Model/Tensordot/ProdProdModel/Tensordot/GatherV2Model/Tensordot/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
a
Model/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
Model/Tensordot/Prod_1ProdModel/Tensordot/GatherV2_1Model/Tensordot/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
]
Model/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot/concatConcatV2Model/Tensordot/GatherV2_1Model/Tensordot/GatherV2Model/Tensordot/concat/axis*
T0*
N*#
_output_shapes
:���������*

Tidx0
_
Model/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot/concat_1ConcatV2Model/Tensordot/ListDiffModel/Tensordot/add_1Model/Tensordot/concat_1/axis*
N*#
_output_shapes
:���������*

Tidx0*
T0
�
Model/Tensordot/stackPackModel/Tensordot/ProdModel/Tensordot/Prod_1*
N*
_output_shapes
:*
T0*

axis 
�
Model/Tensordot/transpose	TransposeModel/GatherNdModel/Tensordot/concat_1*0
_output_shapes
:������������������*
Tperm0*
T0
�
Model/Tensordot/ReshapeReshapeModel/Tensordot/transposeModel/Tensordot/stack*0
_output_shapes
:������������������*
T0*
Tshape0
q
 Model/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
�
Model/Tensordot/transpose_1	TransposeModel/transpose Model/Tensordot/transpose_1/perm*
_output_shapes
:	�
F*
Tperm0*
T0
p
Model/Tensordot/Reshape_1/shapeConst*
valueB"s  F   *
dtype0*
_output_shapes
:
�
Model/Tensordot/Reshape_1ReshapeModel/Tensordot/transpose_1Model/Tensordot/Reshape_1/shape*
T0*
Tshape0*
_output_shapes
:	�
F
�
Model/Tensordot/MatMulMatMulModel/Tensordot/ReshapeModel/Tensordot/Reshape_1*
T0*
transpose_a( *'
_output_shapes
:���������F*
transpose_b( 
a
Model/Tensordot/Const_2Const*
valueB:F*
dtype0*
_output_shapes
:
_
Model/Tensordot/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot/concat_2ConcatV2Model/Tensordot/GatherV2Model/Tensordot/Const_2Model/Tensordot/concat_2/axis*
T0*
N*#
_output_shapes
:���������*

Tidx0
�
Model/TensordotReshapeModel/Tensordot/MatMulModel/Tensordot/concat_2*
T0*
Tshape0*'
_output_shapes
:���������F
i
	Model/AddAddModel/TensordotWeights/bias_in/read*
T0*'
_output_shapes
:���������F
U
Model/SigmoidSigmoid	Model/Add*
T0*'
_output_shapes
:���������F
]
Model/transpose_1/RankRankWeights/weight_hidden_1/read*
T0*
_output_shapes
: 
Y
Model/transpose_1/sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
n
Model/transpose_1/subSubModel/transpose_1/RankModel/transpose_1/sub/y*
T0*
_output_shapes
: 
_
Model/transpose_1/Range/startConst*
dtype0*
_output_shapes
: *
value	B : 
_
Model/transpose_1/Range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Model/transpose_1/RangeRangeModel/transpose_1/Range/startModel/transpose_1/RankModel/transpose_1/Range/delta*

Tidx0*
_output_shapes
:
s
Model/transpose_1/sub_1SubModel/transpose_1/subModel/transpose_1/Range*
_output_shapes
:*
T0
�
Model/transpose_1	TransposeWeights/weight_hidden_1/readModel/transpose_1/sub_1*
_output_shapes

:Fh*
Tperm0*
T0
d
Model/Tensordot_1/ShapeShapeModel/Sigmoid*
T0*
out_type0*
_output_shapes
:
X
Model/Tensordot_1/RankConst*
value	B :*
dtype0*
_output_shapes
: 
`
Model/Tensordot_1/axesConst*
valueB:*
dtype0*
_output_shapes
:
b
 Model/Tensordot_1/GreaterEqual/yConst*
dtype0*
_output_shapes
: *
value	B : 
�
Model/Tensordot_1/GreaterEqualGreaterEqualModel/Tensordot_1/axes Model/Tensordot_1/GreaterEqual/y*
T0*
_output_shapes
:
r
Model/Tensordot_1/CastCastModel/Tensordot_1/GreaterEqual*

SrcT0
*

DstT0*
_output_shapes
:
q
Model/Tensordot_1/mulMulModel/Tensordot_1/CastModel/Tensordot_1/axes*
T0*
_output_shapes
:
Z
Model/Tensordot_1/Less/yConst*
dtype0*
_output_shapes
: *
value	B : 
u
Model/Tensordot_1/LessLessModel/Tensordot_1/axesModel/Tensordot_1/Less/y*
_output_shapes
:*
T0
l
Model/Tensordot_1/Cast_1CastModel/Tensordot_1/Less*

SrcT0
*

DstT0*
_output_shapes
:
q
Model/Tensordot_1/addAddModel/Tensordot_1/axesModel/Tensordot_1/Rank*
T0*
_output_shapes
:
t
Model/Tensordot_1/mul_1MulModel/Tensordot_1/Cast_1Model/Tensordot_1/add*
T0*
_output_shapes
:
s
Model/Tensordot_1/add_1AddModel/Tensordot_1/mulModel/Tensordot_1/mul_1*
T0*
_output_shapes
:
_
Model/Tensordot_1/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
_
Model/Tensordot_1/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Model/Tensordot_1/rangeRangeModel/Tensordot_1/range/startModel/Tensordot_1/RankModel/Tensordot_1/range/delta*

Tidx0*
_output_shapes
:
�
Model/Tensordot_1/ListDiffListDiffModel/Tensordot_1/rangeModel/Tensordot_1/add_1*
T0*
out_idx0*2
_output_shapes 
:���������:���������
a
Model/Tensordot_1/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_1/GatherV2GatherV2Model/Tensordot_1/ShapeModel/Tensordot_1/ListDiffModel/Tensordot_1/GatherV2/axis*
Tparams0*#
_output_shapes
:���������*
Taxis0*
Tindices0
c
!Model/Tensordot_1/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_1/GatherV2_1GatherV2Model/Tensordot_1/ShapeModel/Tensordot_1/add_1!Model/Tensordot_1/GatherV2_1/axis*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0
a
Model/Tensordot_1/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
Model/Tensordot_1/ProdProdModel/Tensordot_1/GatherV2Model/Tensordot_1/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
c
Model/Tensordot_1/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
Model/Tensordot_1/Prod_1ProdModel/Tensordot_1/GatherV2_1Model/Tensordot_1/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
_
Model/Tensordot_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_1/concatConcatV2Model/Tensordot_1/GatherV2_1Model/Tensordot_1/GatherV2Model/Tensordot_1/concat/axis*
T0*
N*#
_output_shapes
:���������*

Tidx0
a
Model/Tensordot_1/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
Model/Tensordot_1/concat_1ConcatV2Model/Tensordot_1/ListDiffModel/Tensordot_1/add_1Model/Tensordot_1/concat_1/axis*

Tidx0*
T0*
N*#
_output_shapes
:���������
�
Model/Tensordot_1/stackPackModel/Tensordot_1/ProdModel/Tensordot_1/Prod_1*
N*
_output_shapes
:*
T0*

axis 
�
Model/Tensordot_1/transpose	TransposeModel/SigmoidModel/Tensordot_1/concat_1*
T0*0
_output_shapes
:������������������*
Tperm0
�
Model/Tensordot_1/ReshapeReshapeModel/Tensordot_1/transposeModel/Tensordot_1/stack*
T0*
Tshape0*0
_output_shapes
:������������������
s
"Model/Tensordot_1/transpose_1/permConst*
dtype0*
_output_shapes
:*
valueB"       
�
Model/Tensordot_1/transpose_1	TransposeModel/transpose_1"Model/Tensordot_1/transpose_1/perm*
_output_shapes

:Fh*
Tperm0*
T0
r
!Model/Tensordot_1/Reshape_1/shapeConst*
valueB"F   h   *
dtype0*
_output_shapes
:
�
Model/Tensordot_1/Reshape_1ReshapeModel/Tensordot_1/transpose_1!Model/Tensordot_1/Reshape_1/shape*
T0*
Tshape0*
_output_shapes

:Fh
�
Model/Tensordot_1/MatMulMatMulModel/Tensordot_1/ReshapeModel/Tensordot_1/Reshape_1*
transpose_a( *'
_output_shapes
:���������h*
transpose_b( *
T0
c
Model/Tensordot_1/Const_2Const*
valueB:h*
dtype0*
_output_shapes
:
a
Model/Tensordot_1/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_1/concat_2ConcatV2Model/Tensordot_1/GatherV2Model/Tensordot_1/Const_2Model/Tensordot_1/concat_2/axis*
T0*
N*#
_output_shapes
:���������*

Tidx0
�
Model/Tensordot_1ReshapeModel/Tensordot_1/MatMulModel/Tensordot_1/concat_2*
T0*
Tshape0*'
_output_shapes
:���������h
s
Model/Add_1AddModel/Tensordot_1Weights/bias_hidden_1/read*
T0*'
_output_shapes
:���������h
Y
Model/Sigmoid_1SigmoidModel/Add_1*
T0*'
_output_shapes
:���������h
X
Model/transpose_2/RankRankWeights/weight_out/read*
T0*
_output_shapes
: 
Y
Model/transpose_2/sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
n
Model/transpose_2/subSubModel/transpose_2/RankModel/transpose_2/sub/y*
T0*
_output_shapes
: 
_
Model/transpose_2/Range/startConst*
dtype0*
_output_shapes
: *
value	B : 
_
Model/transpose_2/Range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Model/transpose_2/RangeRangeModel/transpose_2/Range/startModel/transpose_2/RankModel/transpose_2/Range/delta*
_output_shapes
:*

Tidx0
s
Model/transpose_2/sub_1SubModel/transpose_2/subModel/transpose_2/Range*
_output_shapes
:*
T0
�
Model/transpose_2	TransposeWeights/weight_out/readModel/transpose_2/sub_1*
T0*
_output_shapes

:h*
Tperm0
f
Model/Tensordot_2/ShapeShapeModel/Sigmoid_1*
T0*
out_type0*
_output_shapes
:
X
Model/Tensordot_2/RankConst*
value	B :*
dtype0*
_output_shapes
: 
`
Model/Tensordot_2/axesConst*
dtype0*
_output_shapes
:*
valueB:
b
 Model/Tensordot_2/GreaterEqual/yConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_2/GreaterEqualGreaterEqualModel/Tensordot_2/axes Model/Tensordot_2/GreaterEqual/y*
_output_shapes
:*
T0
r
Model/Tensordot_2/CastCastModel/Tensordot_2/GreaterEqual*

DstT0*
_output_shapes
:*

SrcT0

q
Model/Tensordot_2/mulMulModel/Tensordot_2/CastModel/Tensordot_2/axes*
T0*
_output_shapes
:
Z
Model/Tensordot_2/Less/yConst*
value	B : *
dtype0*
_output_shapes
: 
u
Model/Tensordot_2/LessLessModel/Tensordot_2/axesModel/Tensordot_2/Less/y*
T0*
_output_shapes
:
l
Model/Tensordot_2/Cast_1CastModel/Tensordot_2/Less*

SrcT0
*

DstT0*
_output_shapes
:
q
Model/Tensordot_2/addAddModel/Tensordot_2/axesModel/Tensordot_2/Rank*
T0*
_output_shapes
:
t
Model/Tensordot_2/mul_1MulModel/Tensordot_2/Cast_1Model/Tensordot_2/add*
T0*
_output_shapes
:
s
Model/Tensordot_2/add_1AddModel/Tensordot_2/mulModel/Tensordot_2/mul_1*
_output_shapes
:*
T0
_
Model/Tensordot_2/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
_
Model/Tensordot_2/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Model/Tensordot_2/rangeRangeModel/Tensordot_2/range/startModel/Tensordot_2/RankModel/Tensordot_2/range/delta*

Tidx0*
_output_shapes
:
�
Model/Tensordot_2/ListDiffListDiffModel/Tensordot_2/rangeModel/Tensordot_2/add_1*
T0*
out_idx0*2
_output_shapes 
:���������:���������
a
Model/Tensordot_2/GatherV2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
Model/Tensordot_2/GatherV2GatherV2Model/Tensordot_2/ShapeModel/Tensordot_2/ListDiffModel/Tensordot_2/GatherV2/axis*#
_output_shapes
:���������*
Taxis0*
Tindices0*
Tparams0
c
!Model/Tensordot_2/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_2/GatherV2_1GatherV2Model/Tensordot_2/ShapeModel/Tensordot_2/add_1!Model/Tensordot_2/GatherV2_1/axis*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0
a
Model/Tensordot_2/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
Model/Tensordot_2/ProdProdModel/Tensordot_2/GatherV2Model/Tensordot_2/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
c
Model/Tensordot_2/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
Model/Tensordot_2/Prod_1ProdModel/Tensordot_2/GatherV2_1Model/Tensordot_2/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
_
Model/Tensordot_2/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_2/concatConcatV2Model/Tensordot_2/GatherV2_1Model/Tensordot_2/GatherV2Model/Tensordot_2/concat/axis*

Tidx0*
T0*
N*#
_output_shapes
:���������
a
Model/Tensordot_2/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_2/concat_1ConcatV2Model/Tensordot_2/ListDiffModel/Tensordot_2/add_1Model/Tensordot_2/concat_1/axis*

Tidx0*
T0*
N*#
_output_shapes
:���������
�
Model/Tensordot_2/stackPackModel/Tensordot_2/ProdModel/Tensordot_2/Prod_1*
N*
_output_shapes
:*
T0*

axis 
�
Model/Tensordot_2/transpose	TransposeModel/Sigmoid_1Model/Tensordot_2/concat_1*0
_output_shapes
:������������������*
Tperm0*
T0
�
Model/Tensordot_2/ReshapeReshapeModel/Tensordot_2/transposeModel/Tensordot_2/stack*
T0*
Tshape0*0
_output_shapes
:������������������
s
"Model/Tensordot_2/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
�
Model/Tensordot_2/transpose_1	TransposeModel/transpose_2"Model/Tensordot_2/transpose_1/perm*
T0*
_output_shapes

:h*
Tperm0
r
!Model/Tensordot_2/Reshape_1/shapeConst*
valueB"h      *
dtype0*
_output_shapes
:
�
Model/Tensordot_2/Reshape_1ReshapeModel/Tensordot_2/transpose_1!Model/Tensordot_2/Reshape_1/shape*
_output_shapes

:h*
T0*
Tshape0
�
Model/Tensordot_2/MatMulMatMulModel/Tensordot_2/ReshapeModel/Tensordot_2/Reshape_1*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:���������
c
Model/Tensordot_2/Const_2Const*
dtype0*
_output_shapes
:*
valueB:
a
Model/Tensordot_2/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_2/concat_2ConcatV2Model/Tensordot_2/GatherV2Model/Tensordot_2/Const_2Model/Tensordot_2/concat_2/axis*

Tidx0*
T0*
N*#
_output_shapes
:���������
�
Model/Tensordot_2ReshapeModel/Tensordot_2/MatMulModel/Tensordot_2/concat_2*
T0*
Tshape0*'
_output_shapes
:���������
n
Model/Add_2AddModel/Tensordot_2Weights/bias_out/read*
T0*'
_output_shapes
:���������
s
Model/SqueezeSqueezeModel/Add_2*
squeeze_dims

���������*
T0*#
_output_shapes
:���������
a
Model/ShapeShapeData/IteratorGetNext:1*
T0*
out_type0*
_output_shapes
:
�
Model/ScatterNd	ScatterNd
Model/CastModel/SqueezeModel/Shape*
Tindices0*
T0*'
_output_shapes
:���������
g
Model/Add_3AddModel/zeros_likeModel/ScatterNd*
T0*'
_output_shapes
:���������
O
Model/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
X
Model/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
x
Model/ExpandDims_1
ExpandDimsModel/Const_1Model/ExpandDims_1/dim*
T0*
_output_shapes
:*

Tdim0
t
Model/Equal_1EqualData/IteratorGetNext:1Model/ExpandDims_1*
T0*'
_output_shapes
:���������
W
Model/Where_1WhereModel/Equal_1*'
_output_shapes
:���������*
T0

d
Model/Cast_1CastModel/Where_1*

SrcT0	*

DstT0*'
_output_shapes
:���������
�
Model/GatherNd_1GatherNdData/IteratorGetNextModel/Cast_1*
Tindices0*
Tparams0*(
_output_shapes
:����������

Y
Model/transpose_3/RankRankWeights/weight_in_1/read*
T0*
_output_shapes
: 
Y
Model/transpose_3/sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
n
Model/transpose_3/subSubModel/transpose_3/RankModel/transpose_3/sub/y*
T0*
_output_shapes
: 
_
Model/transpose_3/Range/startConst*
value	B : *
dtype0*
_output_shapes
: 
_
Model/transpose_3/Range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Model/transpose_3/RangeRangeModel/transpose_3/Range/startModel/transpose_3/RankModel/transpose_3/Range/delta*
_output_shapes
:*

Tidx0
s
Model/transpose_3/sub_1SubModel/transpose_3/subModel/transpose_3/Range*
T0*
_output_shapes
:
�
Model/transpose_3	TransposeWeights/weight_in_1/readModel/transpose_3/sub_1*
T0*
_output_shapes
:	�
F*
Tperm0
g
Model/Tensordot_3/ShapeShapeModel/GatherNd_1*
T0*
out_type0*
_output_shapes
:
X
Model/Tensordot_3/RankConst*
value	B :*
dtype0*
_output_shapes
: 
`
Model/Tensordot_3/axesConst*
valueB:*
dtype0*
_output_shapes
:
b
 Model/Tensordot_3/GreaterEqual/yConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_3/GreaterEqualGreaterEqualModel/Tensordot_3/axes Model/Tensordot_3/GreaterEqual/y*
T0*
_output_shapes
:
r
Model/Tensordot_3/CastCastModel/Tensordot_3/GreaterEqual*

SrcT0
*

DstT0*
_output_shapes
:
q
Model/Tensordot_3/mulMulModel/Tensordot_3/CastModel/Tensordot_3/axes*
_output_shapes
:*
T0
Z
Model/Tensordot_3/Less/yConst*
value	B : *
dtype0*
_output_shapes
: 
u
Model/Tensordot_3/LessLessModel/Tensordot_3/axesModel/Tensordot_3/Less/y*
T0*
_output_shapes
:
l
Model/Tensordot_3/Cast_1CastModel/Tensordot_3/Less*

SrcT0
*

DstT0*
_output_shapes
:
q
Model/Tensordot_3/addAddModel/Tensordot_3/axesModel/Tensordot_3/Rank*
T0*
_output_shapes
:
t
Model/Tensordot_3/mul_1MulModel/Tensordot_3/Cast_1Model/Tensordot_3/add*
T0*
_output_shapes
:
s
Model/Tensordot_3/add_1AddModel/Tensordot_3/mulModel/Tensordot_3/mul_1*
_output_shapes
:*
T0
_
Model/Tensordot_3/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
_
Model/Tensordot_3/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Model/Tensordot_3/rangeRangeModel/Tensordot_3/range/startModel/Tensordot_3/RankModel/Tensordot_3/range/delta*

Tidx0*
_output_shapes
:
�
Model/Tensordot_3/ListDiffListDiffModel/Tensordot_3/rangeModel/Tensordot_3/add_1*
T0*
out_idx0*2
_output_shapes 
:���������:���������
a
Model/Tensordot_3/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_3/GatherV2GatherV2Model/Tensordot_3/ShapeModel/Tensordot_3/ListDiffModel/Tensordot_3/GatherV2/axis*#
_output_shapes
:���������*
Taxis0*
Tindices0*
Tparams0
c
!Model/Tensordot_3/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
Model/Tensordot_3/GatherV2_1GatherV2Model/Tensordot_3/ShapeModel/Tensordot_3/add_1!Model/Tensordot_3/GatherV2_1/axis*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0
a
Model/Tensordot_3/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
Model/Tensordot_3/ProdProdModel/Tensordot_3/GatherV2Model/Tensordot_3/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
c
Model/Tensordot_3/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
Model/Tensordot_3/Prod_1ProdModel/Tensordot_3/GatherV2_1Model/Tensordot_3/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
_
Model/Tensordot_3/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_3/concatConcatV2Model/Tensordot_3/GatherV2_1Model/Tensordot_3/GatherV2Model/Tensordot_3/concat/axis*
N*#
_output_shapes
:���������*

Tidx0*
T0
a
Model/Tensordot_3/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_3/concat_1ConcatV2Model/Tensordot_3/ListDiffModel/Tensordot_3/add_1Model/Tensordot_3/concat_1/axis*

Tidx0*
T0*
N*#
_output_shapes
:���������
�
Model/Tensordot_3/stackPackModel/Tensordot_3/ProdModel/Tensordot_3/Prod_1*
T0*

axis *
N*
_output_shapes
:
�
Model/Tensordot_3/transpose	TransposeModel/GatherNd_1Model/Tensordot_3/concat_1*
T0*0
_output_shapes
:������������������*
Tperm0
�
Model/Tensordot_3/ReshapeReshapeModel/Tensordot_3/transposeModel/Tensordot_3/stack*
T0*
Tshape0*0
_output_shapes
:������������������
s
"Model/Tensordot_3/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
�
Model/Tensordot_3/transpose_1	TransposeModel/transpose_3"Model/Tensordot_3/transpose_1/perm*
_output_shapes
:	�
F*
Tperm0*
T0
r
!Model/Tensordot_3/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"s  F   
�
Model/Tensordot_3/Reshape_1ReshapeModel/Tensordot_3/transpose_1!Model/Tensordot_3/Reshape_1/shape*
T0*
Tshape0*
_output_shapes
:	�
F
�
Model/Tensordot_3/MatMulMatMulModel/Tensordot_3/ReshapeModel/Tensordot_3/Reshape_1*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:���������F
c
Model/Tensordot_3/Const_2Const*
dtype0*
_output_shapes
:*
valueB:F
a
Model/Tensordot_3/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_3/concat_2ConcatV2Model/Tensordot_3/GatherV2Model/Tensordot_3/Const_2Model/Tensordot_3/concat_2/axis*

Tidx0*
T0*
N*#
_output_shapes
:���������
�
Model/Tensordot_3ReshapeModel/Tensordot_3/MatMulModel/Tensordot_3/concat_2*
T0*
Tshape0*'
_output_shapes
:���������F
o
Model/Add_4AddModel/Tensordot_3Weights/bias_in_1/read*'
_output_shapes
:���������F*
T0
Y
Model/Sigmoid_2SigmoidModel/Add_4*
T0*'
_output_shapes
:���������F
_
Model/transpose_4/RankRankWeights/weight_hidden_1_1/read*
T0*
_output_shapes
: 
Y
Model/transpose_4/sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
n
Model/transpose_4/subSubModel/transpose_4/RankModel/transpose_4/sub/y*
T0*
_output_shapes
: 
_
Model/transpose_4/Range/startConst*
value	B : *
dtype0*
_output_shapes
: 
_
Model/transpose_4/Range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Model/transpose_4/RangeRangeModel/transpose_4/Range/startModel/transpose_4/RankModel/transpose_4/Range/delta*
_output_shapes
:*

Tidx0
s
Model/transpose_4/sub_1SubModel/transpose_4/subModel/transpose_4/Range*
T0*
_output_shapes
:
�
Model/transpose_4	TransposeWeights/weight_hidden_1_1/readModel/transpose_4/sub_1*
Tperm0*
T0*
_output_shapes

:Fh
f
Model/Tensordot_4/ShapeShapeModel/Sigmoid_2*
T0*
out_type0*
_output_shapes
:
X
Model/Tensordot_4/RankConst*
value	B :*
dtype0*
_output_shapes
: 
`
Model/Tensordot_4/axesConst*
valueB:*
dtype0*
_output_shapes
:
b
 Model/Tensordot_4/GreaterEqual/yConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_4/GreaterEqualGreaterEqualModel/Tensordot_4/axes Model/Tensordot_4/GreaterEqual/y*
_output_shapes
:*
T0
r
Model/Tensordot_4/CastCastModel/Tensordot_4/GreaterEqual*

SrcT0
*

DstT0*
_output_shapes
:
q
Model/Tensordot_4/mulMulModel/Tensordot_4/CastModel/Tensordot_4/axes*
T0*
_output_shapes
:
Z
Model/Tensordot_4/Less/yConst*
value	B : *
dtype0*
_output_shapes
: 
u
Model/Tensordot_4/LessLessModel/Tensordot_4/axesModel/Tensordot_4/Less/y*
T0*
_output_shapes
:
l
Model/Tensordot_4/Cast_1CastModel/Tensordot_4/Less*

SrcT0
*

DstT0*
_output_shapes
:
q
Model/Tensordot_4/addAddModel/Tensordot_4/axesModel/Tensordot_4/Rank*
_output_shapes
:*
T0
t
Model/Tensordot_4/mul_1MulModel/Tensordot_4/Cast_1Model/Tensordot_4/add*
_output_shapes
:*
T0
s
Model/Tensordot_4/add_1AddModel/Tensordot_4/mulModel/Tensordot_4/mul_1*
_output_shapes
:*
T0
_
Model/Tensordot_4/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
_
Model/Tensordot_4/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Model/Tensordot_4/rangeRangeModel/Tensordot_4/range/startModel/Tensordot_4/RankModel/Tensordot_4/range/delta*
_output_shapes
:*

Tidx0
�
Model/Tensordot_4/ListDiffListDiffModel/Tensordot_4/rangeModel/Tensordot_4/add_1*2
_output_shapes 
:���������:���������*
T0*
out_idx0
a
Model/Tensordot_4/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_4/GatherV2GatherV2Model/Tensordot_4/ShapeModel/Tensordot_4/ListDiffModel/Tensordot_4/GatherV2/axis*#
_output_shapes
:���������*
Taxis0*
Tindices0*
Tparams0
c
!Model/Tensordot_4/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_4/GatherV2_1GatherV2Model/Tensordot_4/ShapeModel/Tensordot_4/add_1!Model/Tensordot_4/GatherV2_1/axis*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0
a
Model/Tensordot_4/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
Model/Tensordot_4/ProdProdModel/Tensordot_4/GatherV2Model/Tensordot_4/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
c
Model/Tensordot_4/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
Model/Tensordot_4/Prod_1ProdModel/Tensordot_4/GatherV2_1Model/Tensordot_4/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
_
Model/Tensordot_4/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_4/concatConcatV2Model/Tensordot_4/GatherV2_1Model/Tensordot_4/GatherV2Model/Tensordot_4/concat/axis*
T0*
N*#
_output_shapes
:���������*

Tidx0
a
Model/Tensordot_4/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_4/concat_1ConcatV2Model/Tensordot_4/ListDiffModel/Tensordot_4/add_1Model/Tensordot_4/concat_1/axis*
T0*
N*#
_output_shapes
:���������*

Tidx0
�
Model/Tensordot_4/stackPackModel/Tensordot_4/ProdModel/Tensordot_4/Prod_1*
T0*

axis *
N*
_output_shapes
:
�
Model/Tensordot_4/transpose	TransposeModel/Sigmoid_2Model/Tensordot_4/concat_1*
Tperm0*
T0*0
_output_shapes
:������������������
�
Model/Tensordot_4/ReshapeReshapeModel/Tensordot_4/transposeModel/Tensordot_4/stack*
T0*
Tshape0*0
_output_shapes
:������������������
s
"Model/Tensordot_4/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
�
Model/Tensordot_4/transpose_1	TransposeModel/transpose_4"Model/Tensordot_4/transpose_1/perm*
_output_shapes

:Fh*
Tperm0*
T0
r
!Model/Tensordot_4/Reshape_1/shapeConst*
valueB"F   h   *
dtype0*
_output_shapes
:
�
Model/Tensordot_4/Reshape_1ReshapeModel/Tensordot_4/transpose_1!Model/Tensordot_4/Reshape_1/shape*
T0*
Tshape0*
_output_shapes

:Fh
�
Model/Tensordot_4/MatMulMatMulModel/Tensordot_4/ReshapeModel/Tensordot_4/Reshape_1*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:���������h
c
Model/Tensordot_4/Const_2Const*
valueB:h*
dtype0*
_output_shapes
:
a
Model/Tensordot_4/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_4/concat_2ConcatV2Model/Tensordot_4/GatherV2Model/Tensordot_4/Const_2Model/Tensordot_4/concat_2/axis*
T0*
N*#
_output_shapes
:���������*

Tidx0
�
Model/Tensordot_4ReshapeModel/Tensordot_4/MatMulModel/Tensordot_4/concat_2*
T0*
Tshape0*'
_output_shapes
:���������h
u
Model/Add_5AddModel/Tensordot_4Weights/bias_hidden_1_1/read*'
_output_shapes
:���������h*
T0
Y
Model/Sigmoid_3SigmoidModel/Add_5*
T0*'
_output_shapes
:���������h
Z
Model/transpose_5/RankRankWeights/weight_out_1/read*
T0*
_output_shapes
: 
Y
Model/transpose_5/sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
n
Model/transpose_5/subSubModel/transpose_5/RankModel/transpose_5/sub/y*
T0*
_output_shapes
: 
_
Model/transpose_5/Range/startConst*
dtype0*
_output_shapes
: *
value	B : 
_
Model/transpose_5/Range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
�
Model/transpose_5/RangeRangeModel/transpose_5/Range/startModel/transpose_5/RankModel/transpose_5/Range/delta*
_output_shapes
:*

Tidx0
s
Model/transpose_5/sub_1SubModel/transpose_5/subModel/transpose_5/Range*
T0*
_output_shapes
:
�
Model/transpose_5	TransposeWeights/weight_out_1/readModel/transpose_5/sub_1*
T0*
_output_shapes

:h*
Tperm0
f
Model/Tensordot_5/ShapeShapeModel/Sigmoid_3*
_output_shapes
:*
T0*
out_type0
X
Model/Tensordot_5/RankConst*
value	B :*
dtype0*
_output_shapes
: 
`
Model/Tensordot_5/axesConst*
valueB:*
dtype0*
_output_shapes
:
b
 Model/Tensordot_5/GreaterEqual/yConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_5/GreaterEqualGreaterEqualModel/Tensordot_5/axes Model/Tensordot_5/GreaterEqual/y*
T0*
_output_shapes
:
r
Model/Tensordot_5/CastCastModel/Tensordot_5/GreaterEqual*

SrcT0
*

DstT0*
_output_shapes
:
q
Model/Tensordot_5/mulMulModel/Tensordot_5/CastModel/Tensordot_5/axes*
T0*
_output_shapes
:
Z
Model/Tensordot_5/Less/yConst*
value	B : *
dtype0*
_output_shapes
: 
u
Model/Tensordot_5/LessLessModel/Tensordot_5/axesModel/Tensordot_5/Less/y*
T0*
_output_shapes
:
l
Model/Tensordot_5/Cast_1CastModel/Tensordot_5/Less*

DstT0*
_output_shapes
:*

SrcT0

q
Model/Tensordot_5/addAddModel/Tensordot_5/axesModel/Tensordot_5/Rank*
_output_shapes
:*
T0
t
Model/Tensordot_5/mul_1MulModel/Tensordot_5/Cast_1Model/Tensordot_5/add*
_output_shapes
:*
T0
s
Model/Tensordot_5/add_1AddModel/Tensordot_5/mulModel/Tensordot_5/mul_1*
T0*
_output_shapes
:
_
Model/Tensordot_5/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
_
Model/Tensordot_5/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Model/Tensordot_5/rangeRangeModel/Tensordot_5/range/startModel/Tensordot_5/RankModel/Tensordot_5/range/delta*

Tidx0*
_output_shapes
:
�
Model/Tensordot_5/ListDiffListDiffModel/Tensordot_5/rangeModel/Tensordot_5/add_1*
T0*
out_idx0*2
_output_shapes 
:���������:���������
a
Model/Tensordot_5/GatherV2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
Model/Tensordot_5/GatherV2GatherV2Model/Tensordot_5/ShapeModel/Tensordot_5/ListDiffModel/Tensordot_5/GatherV2/axis*
Tindices0*
Tparams0*#
_output_shapes
:���������*
Taxis0
c
!Model/Tensordot_5/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_5/GatherV2_1GatherV2Model/Tensordot_5/ShapeModel/Tensordot_5/add_1!Model/Tensordot_5/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
a
Model/Tensordot_5/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
Model/Tensordot_5/ProdProdModel/Tensordot_5/GatherV2Model/Tensordot_5/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
c
Model/Tensordot_5/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
Model/Tensordot_5/Prod_1ProdModel/Tensordot_5/GatherV2_1Model/Tensordot_5/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
_
Model/Tensordot_5/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_5/concatConcatV2Model/Tensordot_5/GatherV2_1Model/Tensordot_5/GatherV2Model/Tensordot_5/concat/axis*
T0*
N*#
_output_shapes
:���������*

Tidx0
a
Model/Tensordot_5/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_5/concat_1ConcatV2Model/Tensordot_5/ListDiffModel/Tensordot_5/add_1Model/Tensordot_5/concat_1/axis*
T0*
N*#
_output_shapes
:���������*

Tidx0
�
Model/Tensordot_5/stackPackModel/Tensordot_5/ProdModel/Tensordot_5/Prod_1*
T0*

axis *
N*
_output_shapes
:
�
Model/Tensordot_5/transpose	TransposeModel/Sigmoid_3Model/Tensordot_5/concat_1*0
_output_shapes
:������������������*
Tperm0*
T0
�
Model/Tensordot_5/ReshapeReshapeModel/Tensordot_5/transposeModel/Tensordot_5/stack*
T0*
Tshape0*0
_output_shapes
:������������������
s
"Model/Tensordot_5/transpose_1/permConst*
dtype0*
_output_shapes
:*
valueB"       
�
Model/Tensordot_5/transpose_1	TransposeModel/transpose_5"Model/Tensordot_5/transpose_1/perm*
_output_shapes

:h*
Tperm0*
T0
r
!Model/Tensordot_5/Reshape_1/shapeConst*
valueB"h      *
dtype0*
_output_shapes
:
�
Model/Tensordot_5/Reshape_1ReshapeModel/Tensordot_5/transpose_1!Model/Tensordot_5/Reshape_1/shape*
T0*
Tshape0*
_output_shapes

:h
�
Model/Tensordot_5/MatMulMatMulModel/Tensordot_5/ReshapeModel/Tensordot_5/Reshape_1*
transpose_a( *'
_output_shapes
:���������*
transpose_b( *
T0
c
Model/Tensordot_5/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
a
Model/Tensordot_5/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_5/concat_2ConcatV2Model/Tensordot_5/GatherV2Model/Tensordot_5/Const_2Model/Tensordot_5/concat_2/axis*
N*#
_output_shapes
:���������*

Tidx0*
T0
�
Model/Tensordot_5ReshapeModel/Tensordot_5/MatMulModel/Tensordot_5/concat_2*
T0*
Tshape0*'
_output_shapes
:���������
p
Model/Add_6AddModel/Tensordot_5Weights/bias_out_1/read*
T0*'
_output_shapes
:���������
u
Model/Squeeze_1SqueezeModel/Add_6*
T0*#
_output_shapes
:���������*
squeeze_dims

���������
c
Model/Shape_1ShapeData/IteratorGetNext:1*
T0*
out_type0*
_output_shapes
:
�
Model/ScatterNd_1	ScatterNdModel/Cast_1Model/Squeeze_1Model/Shape_1*
T0*'
_output_shapes
:���������*
Tindices0
d
Model/Add_7AddModel/Add_3Model/ScatterNd_1*
T0*'
_output_shapes
:���������
O
Model/Const_2Const*
value	B :*
dtype0*
_output_shapes
: 
X
Model/ExpandDims_2/dimConst*
value	B : *
dtype0*
_output_shapes
: 
x
Model/ExpandDims_2
ExpandDimsModel/Const_2Model/ExpandDims_2/dim*
T0*
_output_shapes
:*

Tdim0
t
Model/Equal_2EqualData/IteratorGetNext:1Model/ExpandDims_2*'
_output_shapes
:���������*
T0
W
Model/Where_2WhereModel/Equal_2*
T0
*'
_output_shapes
:���������
d
Model/Cast_2CastModel/Where_2*

SrcT0	*

DstT0*'
_output_shapes
:���������
�
Model/GatherNd_2GatherNdData/IteratorGetNextModel/Cast_2*(
_output_shapes
:����������
*
Tindices0*
Tparams0
Y
Model/transpose_6/RankRankWeights/weight_in_2/read*
T0*
_output_shapes
: 
Y
Model/transpose_6/sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
n
Model/transpose_6/subSubModel/transpose_6/RankModel/transpose_6/sub/y*
_output_shapes
: *
T0
_
Model/transpose_6/Range/startConst*
value	B : *
dtype0*
_output_shapes
: 
_
Model/transpose_6/Range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
�
Model/transpose_6/RangeRangeModel/transpose_6/Range/startModel/transpose_6/RankModel/transpose_6/Range/delta*
_output_shapes
:*

Tidx0
s
Model/transpose_6/sub_1SubModel/transpose_6/subModel/transpose_6/Range*
T0*
_output_shapes
:
�
Model/transpose_6	TransposeWeights/weight_in_2/readModel/transpose_6/sub_1*
_output_shapes
:	�
F*
Tperm0*
T0
g
Model/Tensordot_6/ShapeShapeModel/GatherNd_2*
T0*
out_type0*
_output_shapes
:
X
Model/Tensordot_6/RankConst*
value	B :*
dtype0*
_output_shapes
: 
`
Model/Tensordot_6/axesConst*
valueB:*
dtype0*
_output_shapes
:
b
 Model/Tensordot_6/GreaterEqual/yConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_6/GreaterEqualGreaterEqualModel/Tensordot_6/axes Model/Tensordot_6/GreaterEqual/y*
T0*
_output_shapes
:
r
Model/Tensordot_6/CastCastModel/Tensordot_6/GreaterEqual*

SrcT0
*

DstT0*
_output_shapes
:
q
Model/Tensordot_6/mulMulModel/Tensordot_6/CastModel/Tensordot_6/axes*
T0*
_output_shapes
:
Z
Model/Tensordot_6/Less/yConst*
value	B : *
dtype0*
_output_shapes
: 
u
Model/Tensordot_6/LessLessModel/Tensordot_6/axesModel/Tensordot_6/Less/y*
T0*
_output_shapes
:
l
Model/Tensordot_6/Cast_1CastModel/Tensordot_6/Less*

SrcT0
*

DstT0*
_output_shapes
:
q
Model/Tensordot_6/addAddModel/Tensordot_6/axesModel/Tensordot_6/Rank*
T0*
_output_shapes
:
t
Model/Tensordot_6/mul_1MulModel/Tensordot_6/Cast_1Model/Tensordot_6/add*
_output_shapes
:*
T0
s
Model/Tensordot_6/add_1AddModel/Tensordot_6/mulModel/Tensordot_6/mul_1*
T0*
_output_shapes
:
_
Model/Tensordot_6/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
_
Model/Tensordot_6/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Model/Tensordot_6/rangeRangeModel/Tensordot_6/range/startModel/Tensordot_6/RankModel/Tensordot_6/range/delta*
_output_shapes
:*

Tidx0
�
Model/Tensordot_6/ListDiffListDiffModel/Tensordot_6/rangeModel/Tensordot_6/add_1*
T0*
out_idx0*2
_output_shapes 
:���������:���������
a
Model/Tensordot_6/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_6/GatherV2GatherV2Model/Tensordot_6/ShapeModel/Tensordot_6/ListDiffModel/Tensordot_6/GatherV2/axis*#
_output_shapes
:���������*
Taxis0*
Tindices0*
Tparams0
c
!Model/Tensordot_6/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_6/GatherV2_1GatherV2Model/Tensordot_6/ShapeModel/Tensordot_6/add_1!Model/Tensordot_6/GatherV2_1/axis*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0
a
Model/Tensordot_6/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
Model/Tensordot_6/ProdProdModel/Tensordot_6/GatherV2Model/Tensordot_6/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
c
Model/Tensordot_6/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
Model/Tensordot_6/Prod_1ProdModel/Tensordot_6/GatherV2_1Model/Tensordot_6/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
_
Model/Tensordot_6/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_6/concatConcatV2Model/Tensordot_6/GatherV2_1Model/Tensordot_6/GatherV2Model/Tensordot_6/concat/axis*

Tidx0*
T0*
N*#
_output_shapes
:���������
a
Model/Tensordot_6/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
Model/Tensordot_6/concat_1ConcatV2Model/Tensordot_6/ListDiffModel/Tensordot_6/add_1Model/Tensordot_6/concat_1/axis*
T0*
N*#
_output_shapes
:���������*

Tidx0
�
Model/Tensordot_6/stackPackModel/Tensordot_6/ProdModel/Tensordot_6/Prod_1*
N*
_output_shapes
:*
T0*

axis 
�
Model/Tensordot_6/transpose	TransposeModel/GatherNd_2Model/Tensordot_6/concat_1*
T0*0
_output_shapes
:������������������*
Tperm0
�
Model/Tensordot_6/ReshapeReshapeModel/Tensordot_6/transposeModel/Tensordot_6/stack*
T0*
Tshape0*0
_output_shapes
:������������������
s
"Model/Tensordot_6/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
�
Model/Tensordot_6/transpose_1	TransposeModel/transpose_6"Model/Tensordot_6/transpose_1/perm*
_output_shapes
:	�
F*
Tperm0*
T0
r
!Model/Tensordot_6/Reshape_1/shapeConst*
valueB"s  F   *
dtype0*
_output_shapes
:
�
Model/Tensordot_6/Reshape_1ReshapeModel/Tensordot_6/transpose_1!Model/Tensordot_6/Reshape_1/shape*
_output_shapes
:	�
F*
T0*
Tshape0
�
Model/Tensordot_6/MatMulMatMulModel/Tensordot_6/ReshapeModel/Tensordot_6/Reshape_1*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:���������F
c
Model/Tensordot_6/Const_2Const*
valueB:F*
dtype0*
_output_shapes
:
a
Model/Tensordot_6/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_6/concat_2ConcatV2Model/Tensordot_6/GatherV2Model/Tensordot_6/Const_2Model/Tensordot_6/concat_2/axis*
T0*
N*#
_output_shapes
:���������*

Tidx0
�
Model/Tensordot_6ReshapeModel/Tensordot_6/MatMulModel/Tensordot_6/concat_2*
T0*
Tshape0*'
_output_shapes
:���������F
o
Model/Add_8AddModel/Tensordot_6Weights/bias_in_2/read*
T0*'
_output_shapes
:���������F
Y
Model/Sigmoid_4SigmoidModel/Add_8*
T0*'
_output_shapes
:���������F
_
Model/transpose_7/RankRankWeights/weight_hidden_1_2/read*
T0*
_output_shapes
: 
Y
Model/transpose_7/sub/yConst*
dtype0*
_output_shapes
: *
value	B :
n
Model/transpose_7/subSubModel/transpose_7/RankModel/transpose_7/sub/y*
T0*
_output_shapes
: 
_
Model/transpose_7/Range/startConst*
dtype0*
_output_shapes
: *
value	B : 
_
Model/transpose_7/Range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Model/transpose_7/RangeRangeModel/transpose_7/Range/startModel/transpose_7/RankModel/transpose_7/Range/delta*
_output_shapes
:*

Tidx0
s
Model/transpose_7/sub_1SubModel/transpose_7/subModel/transpose_7/Range*
T0*
_output_shapes
:
�
Model/transpose_7	TransposeWeights/weight_hidden_1_2/readModel/transpose_7/sub_1*
Tperm0*
T0*
_output_shapes

:Fh
f
Model/Tensordot_7/ShapeShapeModel/Sigmoid_4*
T0*
out_type0*
_output_shapes
:
X
Model/Tensordot_7/RankConst*
dtype0*
_output_shapes
: *
value	B :
`
Model/Tensordot_7/axesConst*
valueB:*
dtype0*
_output_shapes
:
b
 Model/Tensordot_7/GreaterEqual/yConst*
dtype0*
_output_shapes
: *
value	B : 
�
Model/Tensordot_7/GreaterEqualGreaterEqualModel/Tensordot_7/axes Model/Tensordot_7/GreaterEqual/y*
T0*
_output_shapes
:
r
Model/Tensordot_7/CastCastModel/Tensordot_7/GreaterEqual*

SrcT0
*

DstT0*
_output_shapes
:
q
Model/Tensordot_7/mulMulModel/Tensordot_7/CastModel/Tensordot_7/axes*
T0*
_output_shapes
:
Z
Model/Tensordot_7/Less/yConst*
value	B : *
dtype0*
_output_shapes
: 
u
Model/Tensordot_7/LessLessModel/Tensordot_7/axesModel/Tensordot_7/Less/y*
T0*
_output_shapes
:
l
Model/Tensordot_7/Cast_1CastModel/Tensordot_7/Less*

SrcT0
*

DstT0*
_output_shapes
:
q
Model/Tensordot_7/addAddModel/Tensordot_7/axesModel/Tensordot_7/Rank*
T0*
_output_shapes
:
t
Model/Tensordot_7/mul_1MulModel/Tensordot_7/Cast_1Model/Tensordot_7/add*
_output_shapes
:*
T0
s
Model/Tensordot_7/add_1AddModel/Tensordot_7/mulModel/Tensordot_7/mul_1*
T0*
_output_shapes
:
_
Model/Tensordot_7/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
_
Model/Tensordot_7/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Model/Tensordot_7/rangeRangeModel/Tensordot_7/range/startModel/Tensordot_7/RankModel/Tensordot_7/range/delta*
_output_shapes
:*

Tidx0
�
Model/Tensordot_7/ListDiffListDiffModel/Tensordot_7/rangeModel/Tensordot_7/add_1*2
_output_shapes 
:���������:���������*
T0*
out_idx0
a
Model/Tensordot_7/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_7/GatherV2GatherV2Model/Tensordot_7/ShapeModel/Tensordot_7/ListDiffModel/Tensordot_7/GatherV2/axis*
Tparams0*#
_output_shapes
:���������*
Taxis0*
Tindices0
c
!Model/Tensordot_7/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_7/GatherV2_1GatherV2Model/Tensordot_7/ShapeModel/Tensordot_7/add_1!Model/Tensordot_7/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
a
Model/Tensordot_7/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
Model/Tensordot_7/ProdProdModel/Tensordot_7/GatherV2Model/Tensordot_7/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
c
Model/Tensordot_7/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
Model/Tensordot_7/Prod_1ProdModel/Tensordot_7/GatherV2_1Model/Tensordot_7/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
_
Model/Tensordot_7/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_7/concatConcatV2Model/Tensordot_7/GatherV2_1Model/Tensordot_7/GatherV2Model/Tensordot_7/concat/axis*
T0*
N*#
_output_shapes
:���������*

Tidx0
a
Model/Tensordot_7/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_7/concat_1ConcatV2Model/Tensordot_7/ListDiffModel/Tensordot_7/add_1Model/Tensordot_7/concat_1/axis*
T0*
N*#
_output_shapes
:���������*

Tidx0
�
Model/Tensordot_7/stackPackModel/Tensordot_7/ProdModel/Tensordot_7/Prod_1*
N*
_output_shapes
:*
T0*

axis 
�
Model/Tensordot_7/transpose	TransposeModel/Sigmoid_4Model/Tensordot_7/concat_1*
T0*0
_output_shapes
:������������������*
Tperm0
�
Model/Tensordot_7/ReshapeReshapeModel/Tensordot_7/transposeModel/Tensordot_7/stack*0
_output_shapes
:������������������*
T0*
Tshape0
s
"Model/Tensordot_7/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
�
Model/Tensordot_7/transpose_1	TransposeModel/transpose_7"Model/Tensordot_7/transpose_1/perm*
T0*
_output_shapes

:Fh*
Tperm0
r
!Model/Tensordot_7/Reshape_1/shapeConst*
valueB"F   h   *
dtype0*
_output_shapes
:
�
Model/Tensordot_7/Reshape_1ReshapeModel/Tensordot_7/transpose_1!Model/Tensordot_7/Reshape_1/shape*
T0*
Tshape0*
_output_shapes

:Fh
�
Model/Tensordot_7/MatMulMatMulModel/Tensordot_7/ReshapeModel/Tensordot_7/Reshape_1*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:���������h
c
Model/Tensordot_7/Const_2Const*
valueB:h*
dtype0*
_output_shapes
:
a
Model/Tensordot_7/concat_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
Model/Tensordot_7/concat_2ConcatV2Model/Tensordot_7/GatherV2Model/Tensordot_7/Const_2Model/Tensordot_7/concat_2/axis*

Tidx0*
T0*
N*#
_output_shapes
:���������
�
Model/Tensordot_7ReshapeModel/Tensordot_7/MatMulModel/Tensordot_7/concat_2*'
_output_shapes
:���������h*
T0*
Tshape0
u
Model/Add_9AddModel/Tensordot_7Weights/bias_hidden_1_2/read*
T0*'
_output_shapes
:���������h
Y
Model/Sigmoid_5SigmoidModel/Add_9*'
_output_shapes
:���������h*
T0
Z
Model/transpose_8/RankRankWeights/weight_out_2/read*
T0*
_output_shapes
: 
Y
Model/transpose_8/sub/yConst*
dtype0*
_output_shapes
: *
value	B :
n
Model/transpose_8/subSubModel/transpose_8/RankModel/transpose_8/sub/y*
_output_shapes
: *
T0
_
Model/transpose_8/Range/startConst*
value	B : *
dtype0*
_output_shapes
: 
_
Model/transpose_8/Range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
�
Model/transpose_8/RangeRangeModel/transpose_8/Range/startModel/transpose_8/RankModel/transpose_8/Range/delta*
_output_shapes
:*

Tidx0
s
Model/transpose_8/sub_1SubModel/transpose_8/subModel/transpose_8/Range*
_output_shapes
:*
T0
�
Model/transpose_8	TransposeWeights/weight_out_2/readModel/transpose_8/sub_1*
_output_shapes

:h*
Tperm0*
T0
f
Model/Tensordot_8/ShapeShapeModel/Sigmoid_5*
T0*
out_type0*
_output_shapes
:
X
Model/Tensordot_8/RankConst*
value	B :*
dtype0*
_output_shapes
: 
`
Model/Tensordot_8/axesConst*
valueB:*
dtype0*
_output_shapes
:
b
 Model/Tensordot_8/GreaterEqual/yConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_8/GreaterEqualGreaterEqualModel/Tensordot_8/axes Model/Tensordot_8/GreaterEqual/y*
_output_shapes
:*
T0
r
Model/Tensordot_8/CastCastModel/Tensordot_8/GreaterEqual*

SrcT0
*

DstT0*
_output_shapes
:
q
Model/Tensordot_8/mulMulModel/Tensordot_8/CastModel/Tensordot_8/axes*
T0*
_output_shapes
:
Z
Model/Tensordot_8/Less/yConst*
value	B : *
dtype0*
_output_shapes
: 
u
Model/Tensordot_8/LessLessModel/Tensordot_8/axesModel/Tensordot_8/Less/y*
_output_shapes
:*
T0
l
Model/Tensordot_8/Cast_1CastModel/Tensordot_8/Less*

SrcT0
*

DstT0*
_output_shapes
:
q
Model/Tensordot_8/addAddModel/Tensordot_8/axesModel/Tensordot_8/Rank*
T0*
_output_shapes
:
t
Model/Tensordot_8/mul_1MulModel/Tensordot_8/Cast_1Model/Tensordot_8/add*
_output_shapes
:*
T0
s
Model/Tensordot_8/add_1AddModel/Tensordot_8/mulModel/Tensordot_8/mul_1*
_output_shapes
:*
T0
_
Model/Tensordot_8/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
_
Model/Tensordot_8/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Model/Tensordot_8/rangeRangeModel/Tensordot_8/range/startModel/Tensordot_8/RankModel/Tensordot_8/range/delta*
_output_shapes
:*

Tidx0
�
Model/Tensordot_8/ListDiffListDiffModel/Tensordot_8/rangeModel/Tensordot_8/add_1*
T0*
out_idx0*2
_output_shapes 
:���������:���������
a
Model/Tensordot_8/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_8/GatherV2GatherV2Model/Tensordot_8/ShapeModel/Tensordot_8/ListDiffModel/Tensordot_8/GatherV2/axis*#
_output_shapes
:���������*
Taxis0*
Tindices0*
Tparams0
c
!Model/Tensordot_8/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_8/GatherV2_1GatherV2Model/Tensordot_8/ShapeModel/Tensordot_8/add_1!Model/Tensordot_8/GatherV2_1/axis*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0
a
Model/Tensordot_8/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
Model/Tensordot_8/ProdProdModel/Tensordot_8/GatherV2Model/Tensordot_8/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
c
Model/Tensordot_8/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
Model/Tensordot_8/Prod_1ProdModel/Tensordot_8/GatherV2_1Model/Tensordot_8/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
_
Model/Tensordot_8/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_8/concatConcatV2Model/Tensordot_8/GatherV2_1Model/Tensordot_8/GatherV2Model/Tensordot_8/concat/axis*
T0*
N*#
_output_shapes
:���������*

Tidx0
a
Model/Tensordot_8/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_8/concat_1ConcatV2Model/Tensordot_8/ListDiffModel/Tensordot_8/add_1Model/Tensordot_8/concat_1/axis*
N*#
_output_shapes
:���������*

Tidx0*
T0
�
Model/Tensordot_8/stackPackModel/Tensordot_8/ProdModel/Tensordot_8/Prod_1*
T0*

axis *
N*
_output_shapes
:
�
Model/Tensordot_8/transpose	TransposeModel/Sigmoid_5Model/Tensordot_8/concat_1*
T0*0
_output_shapes
:������������������*
Tperm0
�
Model/Tensordot_8/ReshapeReshapeModel/Tensordot_8/transposeModel/Tensordot_8/stack*
T0*
Tshape0*0
_output_shapes
:������������������
s
"Model/Tensordot_8/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
�
Model/Tensordot_8/transpose_1	TransposeModel/transpose_8"Model/Tensordot_8/transpose_1/perm*
Tperm0*
T0*
_output_shapes

:h
r
!Model/Tensordot_8/Reshape_1/shapeConst*
valueB"h      *
dtype0*
_output_shapes
:
�
Model/Tensordot_8/Reshape_1ReshapeModel/Tensordot_8/transpose_1!Model/Tensordot_8/Reshape_1/shape*
T0*
Tshape0*
_output_shapes

:h
�
Model/Tensordot_8/MatMulMatMulModel/Tensordot_8/ReshapeModel/Tensordot_8/Reshape_1*
T0*
transpose_a( *'
_output_shapes
:���������*
transpose_b( 
c
Model/Tensordot_8/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
a
Model/Tensordot_8/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model/Tensordot_8/concat_2ConcatV2Model/Tensordot_8/GatherV2Model/Tensordot_8/Const_2Model/Tensordot_8/concat_2/axis*
T0*
N*#
_output_shapes
:���������*

Tidx0
�
Model/Tensordot_8ReshapeModel/Tensordot_8/MatMulModel/Tensordot_8/concat_2*'
_output_shapes
:���������*
T0*
Tshape0
q
Model/Add_10AddModel/Tensordot_8Weights/bias_out_2/read*
T0*'
_output_shapes
:���������
v
Model/Squeeze_2SqueezeModel/Add_10*
T0*#
_output_shapes
:���������*
squeeze_dims

���������
c
Model/Shape_2ShapeData/IteratorGetNext:1*
T0*
out_type0*
_output_shapes
:
�
Model/ScatterNd_2	ScatterNdModel/Cast_2Model/Squeeze_2Model/Shape_2*
T0*'
_output_shapes
:���������*
Tindices0
e
Model/Add_11AddModel/Add_7Model/ScatterNd_2*'
_output_shapes
:���������*
T0
i
Model/output/reduction_indicesConst*
dtype0*
_output_shapes
: *
valueB :
���������
�
Model/outputSumModel/Add_11Model/output/reduction_indices*

Tidx0*
	keep_dims(*
T0*'
_output_shapes
:���������
l
Cost_func/SubSubData/IteratorGetNext:2Model/output*
T0*'
_output_shapes
:���������
[
Cost_func/SquareSquareCost_func/Sub*
T0*'
_output_shapes
:���������
`
Cost_func/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
w
Cost_func/lossMeanCost_func/SquareCost_func/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
V
Cost_func/l2_lossConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
Cost_func/Square_1SquareWeights/weight_in/read*
T0*
_output_shapes
:	F�

b
Cost_func/Const_1Const*
valueB"       *
dtype0*
_output_shapes
:
y
Cost_func/SumSumCost_func/Square_1Cost_func/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
W
Cost_func/addAddCost_func/l2_lossCost_func/Sum*
T0*
_output_shapes
: 
c
Cost_func/Square_2SquareWeights/weight_hidden_1/read*
T0*
_output_shapes

:hF
b
Cost_func/Const_2Const*
valueB"       *
dtype0*
_output_shapes
:
{
Cost_func/Sum_1SumCost_func/Square_2Cost_func/Const_2*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
W
Cost_func/add_1AddCost_func/addCost_func/Sum_1*
T0*
_output_shapes
: 
T
Cost_func/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *�H5
W
Cost_func/mulMulCost_func/mul/xCost_func/add_1*
_output_shapes
: *
T0
V
Cost_func/add_2/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Y
Cost_func/add_2AddCost_func/add_2/xCost_func/mul*
_output_shapes
: *
T0
X
Cost_func/l2_loss_1Const*
dtype0*
_output_shapes
: *
valueB
 *    
`
Cost_func/Square_3SquareWeights/weight_in_1/read*
T0*
_output_shapes
:	F�

b
Cost_func/Const_3Const*
valueB"       *
dtype0*
_output_shapes
:
{
Cost_func/Sum_2SumCost_func/Square_3Cost_func/Const_3*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
]
Cost_func/add_3AddCost_func/l2_loss_1Cost_func/Sum_2*
T0*
_output_shapes
: 
e
Cost_func/Square_4SquareWeights/weight_hidden_1_1/read*
T0*
_output_shapes

:hF
b
Cost_func/Const_4Const*
valueB"       *
dtype0*
_output_shapes
:
{
Cost_func/Sum_3SumCost_func/Square_4Cost_func/Const_4*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
Y
Cost_func/add_4AddCost_func/add_3Cost_func/Sum_3*
_output_shapes
: *
T0
V
Cost_func/mul_1/xConst*
valueB
 *�H5*
dtype0*
_output_shapes
: 
[
Cost_func/mul_1MulCost_func/mul_1/xCost_func/add_4*
T0*
_output_shapes
: 
Y
Cost_func/add_5AddCost_func/add_2Cost_func/mul_1*
_output_shapes
: *
T0
X
Cost_func/l2_loss_2Const*
valueB
 *    *
dtype0*
_output_shapes
: 
`
Cost_func/Square_5SquareWeights/weight_in_2/read*
_output_shapes
:	F�
*
T0
b
Cost_func/Const_5Const*
valueB"       *
dtype0*
_output_shapes
:
{
Cost_func/Sum_4SumCost_func/Square_5Cost_func/Const_5*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
]
Cost_func/add_6AddCost_func/l2_loss_2Cost_func/Sum_4*
T0*
_output_shapes
: 
e
Cost_func/Square_6SquareWeights/weight_hidden_1_2/read*
_output_shapes

:hF*
T0
b
Cost_func/Const_6Const*
valueB"       *
dtype0*
_output_shapes
:
{
Cost_func/Sum_5SumCost_func/Square_6Cost_func/Const_6*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
Y
Cost_func/add_7AddCost_func/add_6Cost_func/Sum_5*
T0*
_output_shapes
: 
V
Cost_func/mul_2/xConst*
valueB
 *�H5*
dtype0*
_output_shapes
: 
[
Cost_func/mul_2MulCost_func/mul_2/xCost_func/add_7*
_output_shapes
: *
T0
Y
Cost_func/add_8AddCost_func/add_5Cost_func/mul_2*
T0*
_output_shapes
: 
X
Cost_func/add_9AddCost_func/lossCost_func/add_8*
T0*
_output_shapes
: 
V
Cost_func/l1_lossConst*
valueB
 *    *
dtype0*
_output_shapes
: 
V
Cost_func/AbsAbsWeights/weight_in/read*
T0*
_output_shapes
:	F�

b
Cost_func/Const_7Const*
valueB"       *
dtype0*
_output_shapes
:
v
Cost_func/Sum_6SumCost_func/AbsCost_func/Const_7*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
\
Cost_func/add_10AddCost_func/l1_lossCost_func/Sum_6*
T0*
_output_shapes
: 
]
Cost_func/Abs_1AbsWeights/weight_hidden_1/read*
_output_shapes

:hF*
T0
b
Cost_func/Const_8Const*
valueB"       *
dtype0*
_output_shapes
:
x
Cost_func/Sum_7SumCost_func/Abs_1Cost_func/Const_8*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
[
Cost_func/add_11AddCost_func/add_10Cost_func/Sum_7*
T0*
_output_shapes
: 
V
Cost_func/mul_3/xConst*
valueB
 *�`�5*
dtype0*
_output_shapes
: 
\
Cost_func/mul_3MulCost_func/mul_3/xCost_func/add_11*
_output_shapes
: *
T0
W
Cost_func/add_12/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
]
Cost_func/add_12AddCost_func/add_12/xCost_func/mul_3*
T0*
_output_shapes
: 
X
Cost_func/l1_loss_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
Cost_func/Abs_2AbsWeights/weight_in_1/read*
T0*
_output_shapes
:	F�

b
Cost_func/Const_9Const*
valueB"       *
dtype0*
_output_shapes
:
x
Cost_func/Sum_8SumCost_func/Abs_2Cost_func/Const_9*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
^
Cost_func/add_13AddCost_func/l1_loss_1Cost_func/Sum_8*
T0*
_output_shapes
: 
_
Cost_func/Abs_3AbsWeights/weight_hidden_1_1/read*
T0*
_output_shapes

:hF
c
Cost_func/Const_10Const*
valueB"       *
dtype0*
_output_shapes
:
y
Cost_func/Sum_9SumCost_func/Abs_3Cost_func/Const_10*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
[
Cost_func/add_14AddCost_func/add_13Cost_func/Sum_9*
T0*
_output_shapes
: 
V
Cost_func/mul_4/xConst*
valueB
 *�`�5*
dtype0*
_output_shapes
: 
\
Cost_func/mul_4MulCost_func/mul_4/xCost_func/add_14*
T0*
_output_shapes
: 
[
Cost_func/add_15AddCost_func/add_12Cost_func/mul_4*
T0*
_output_shapes
: 
X
Cost_func/l1_loss_2Const*
dtype0*
_output_shapes
: *
valueB
 *    
Z
Cost_func/Abs_4AbsWeights/weight_in_2/read*
_output_shapes
:	F�
*
T0
c
Cost_func/Const_11Const*
valueB"       *
dtype0*
_output_shapes
:
z
Cost_func/Sum_10SumCost_func/Abs_4Cost_func/Const_11*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
_
Cost_func/add_16AddCost_func/l1_loss_2Cost_func/Sum_10*
T0*
_output_shapes
: 
_
Cost_func/Abs_5AbsWeights/weight_hidden_1_2/read*
T0*
_output_shapes

:hF
c
Cost_func/Const_12Const*
valueB"       *
dtype0*
_output_shapes
:
z
Cost_func/Sum_11SumCost_func/Abs_5Cost_func/Const_12*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
\
Cost_func/add_17AddCost_func/add_16Cost_func/Sum_11*
_output_shapes
: *
T0
V
Cost_func/mul_5/xConst*
dtype0*
_output_shapes
: *
valueB
 *�`�5
\
Cost_func/mul_5MulCost_func/mul_5/xCost_func/add_17*
T0*
_output_shapes
: 
[
Cost_func/add_18AddCost_func/add_15Cost_func/mul_5*
T0*
_output_shapes
: 
[
Cost_func/add_19AddCost_func/add_9Cost_func/add_18*
_output_shapes
: *
T0
R
gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
X
gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  �?
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
I
0gradients/Cost_func/add_19_grad/tuple/group_depsNoOp^gradients/Fill
�
8gradients/Cost_func/add_19_grad/tuple/control_dependencyIdentitygradients/Fill1^gradients/Cost_func/add_19_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
�
:gradients/Cost_func/add_19_grad/tuple/control_dependency_1Identitygradients/Fill1^gradients/Cost_func/add_19_grad/tuple/group_deps*
_output_shapes
: *
T0*!
_class
loc:@gradients/Fill
r
/gradients/Cost_func/add_9_grad/tuple/group_depsNoOp9^gradients/Cost_func/add_19_grad/tuple/control_dependency
�
7gradients/Cost_func/add_9_grad/tuple/control_dependencyIdentity8gradients/Cost_func/add_19_grad/tuple/control_dependency0^gradients/Cost_func/add_9_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
�
9gradients/Cost_func/add_9_grad/tuple/control_dependency_1Identity8gradients/Cost_func/add_19_grad/tuple/control_dependency0^gradients/Cost_func/add_9_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
u
0gradients/Cost_func/add_18_grad/tuple/group_depsNoOp;^gradients/Cost_func/add_19_grad/tuple/control_dependency_1
�
8gradients/Cost_func/add_18_grad/tuple/control_dependencyIdentity:gradients/Cost_func/add_19_grad/tuple/control_dependency_11^gradients/Cost_func/add_18_grad/tuple/group_deps*
_output_shapes
: *
T0*!
_class
loc:@gradients/Fill
�
:gradients/Cost_func/add_18_grad/tuple/control_dependency_1Identity:gradients/Cost_func/add_19_grad/tuple/control_dependency_11^gradients/Cost_func/add_18_grad/tuple/group_deps*
_output_shapes
: *
T0*!
_class
loc:@gradients/Fill
|
+gradients/Cost_func/loss_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
�
%gradients/Cost_func/loss_grad/ReshapeReshape7gradients/Cost_func/add_9_grad/tuple/control_dependency+gradients/Cost_func/loss_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
s
#gradients/Cost_func/loss_grad/ShapeShapeCost_func/Square*
T0*
out_type0*
_output_shapes
:
�
"gradients/Cost_func/loss_grad/TileTile%gradients/Cost_func/loss_grad/Reshape#gradients/Cost_func/loss_grad/Shape*
T0*'
_output_shapes
:���������*

Tmultiples0
u
%gradients/Cost_func/loss_grad/Shape_1ShapeCost_func/Square*
T0*
out_type0*
_output_shapes
:
h
%gradients/Cost_func/loss_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
m
#gradients/Cost_func/loss_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
"gradients/Cost_func/loss_grad/ProdProd%gradients/Cost_func/loss_grad/Shape_1#gradients/Cost_func/loss_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
o
%gradients/Cost_func/loss_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
$gradients/Cost_func/loss_grad/Prod_1Prod%gradients/Cost_func/loss_grad/Shape_2%gradients/Cost_func/loss_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
i
'gradients/Cost_func/loss_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
%gradients/Cost_func/loss_grad/MaximumMaximum$gradients/Cost_func/loss_grad/Prod_1'gradients/Cost_func/loss_grad/Maximum/y*
_output_shapes
: *
T0
�
&gradients/Cost_func/loss_grad/floordivFloorDiv"gradients/Cost_func/loss_grad/Prod%gradients/Cost_func/loss_grad/Maximum*
T0*
_output_shapes
: 
�
"gradients/Cost_func/loss_grad/CastCast&gradients/Cost_func/loss_grad/floordiv*

SrcT0*

DstT0*
_output_shapes
: 
�
%gradients/Cost_func/loss_grad/truedivRealDiv"gradients/Cost_func/loss_grad/Tile"gradients/Cost_func/loss_grad/Cast*
T0*'
_output_shapes
:���������
s
/gradients/Cost_func/add_8_grad/tuple/group_depsNoOp:^gradients/Cost_func/add_9_grad/tuple/control_dependency_1
�
7gradients/Cost_func/add_8_grad/tuple/control_dependencyIdentity9gradients/Cost_func/add_9_grad/tuple/control_dependency_10^gradients/Cost_func/add_8_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
�
9gradients/Cost_func/add_8_grad/tuple/control_dependency_1Identity9gradients/Cost_func/add_9_grad/tuple/control_dependency_10^gradients/Cost_func/add_8_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
s
0gradients/Cost_func/add_15_grad/tuple/group_depsNoOp9^gradients/Cost_func/add_18_grad/tuple/control_dependency
�
8gradients/Cost_func/add_15_grad/tuple/control_dependencyIdentity8gradients/Cost_func/add_18_grad/tuple/control_dependency1^gradients/Cost_func/add_15_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
�
:gradients/Cost_func/add_15_grad/tuple/control_dependency_1Identity8gradients/Cost_func/add_18_grad/tuple/control_dependency1^gradients/Cost_func/add_15_grad/tuple/group_deps*
_output_shapes
: *
T0*!
_class
loc:@gradients/Fill
�
"gradients/Cost_func/mul_5_grad/MulMul:gradients/Cost_func/add_18_grad/tuple/control_dependency_1Cost_func/add_17*
_output_shapes
: *
T0
�
$gradients/Cost_func/mul_5_grad/Mul_1Mul:gradients/Cost_func/add_18_grad/tuple/control_dependency_1Cost_func/mul_5/x*
T0*
_output_shapes
: 
�
/gradients/Cost_func/mul_5_grad/tuple/group_depsNoOp#^gradients/Cost_func/mul_5_grad/Mul%^gradients/Cost_func/mul_5_grad/Mul_1
�
7gradients/Cost_func/mul_5_grad/tuple/control_dependencyIdentity"gradients/Cost_func/mul_5_grad/Mul0^gradients/Cost_func/mul_5_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/Cost_func/mul_5_grad/Mul*
_output_shapes
: 
�
9gradients/Cost_func/mul_5_grad/tuple/control_dependency_1Identity$gradients/Cost_func/mul_5_grad/Mul_10^gradients/Cost_func/mul_5_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/Cost_func/mul_5_grad/Mul_1*
_output_shapes
: 
�
%gradients/Cost_func/Square_grad/ConstConst&^gradients/Cost_func/loss_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
#gradients/Cost_func/Square_grad/MulMulCost_func/Sub%gradients/Cost_func/Square_grad/Const*
T0*'
_output_shapes
:���������
�
%gradients/Cost_func/Square_grad/Mul_1Mul%gradients/Cost_func/loss_grad/truediv#gradients/Cost_func/Square_grad/Mul*'
_output_shapes
:���������*
T0
q
/gradients/Cost_func/add_5_grad/tuple/group_depsNoOp8^gradients/Cost_func/add_8_grad/tuple/control_dependency
�
7gradients/Cost_func/add_5_grad/tuple/control_dependencyIdentity7gradients/Cost_func/add_8_grad/tuple/control_dependency0^gradients/Cost_func/add_5_grad/tuple/group_deps*
_output_shapes
: *
T0*!
_class
loc:@gradients/Fill
�
9gradients/Cost_func/add_5_grad/tuple/control_dependency_1Identity7gradients/Cost_func/add_8_grad/tuple/control_dependency0^gradients/Cost_func/add_5_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
�
"gradients/Cost_func/mul_2_grad/MulMul9gradients/Cost_func/add_8_grad/tuple/control_dependency_1Cost_func/add_7*
_output_shapes
: *
T0
�
$gradients/Cost_func/mul_2_grad/Mul_1Mul9gradients/Cost_func/add_8_grad/tuple/control_dependency_1Cost_func/mul_2/x*
T0*
_output_shapes
: 
�
/gradients/Cost_func/mul_2_grad/tuple/group_depsNoOp#^gradients/Cost_func/mul_2_grad/Mul%^gradients/Cost_func/mul_2_grad/Mul_1
�
7gradients/Cost_func/mul_2_grad/tuple/control_dependencyIdentity"gradients/Cost_func/mul_2_grad/Mul0^gradients/Cost_func/mul_2_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/Cost_func/mul_2_grad/Mul*
_output_shapes
: 
�
9gradients/Cost_func/mul_2_grad/tuple/control_dependency_1Identity$gradients/Cost_func/mul_2_grad/Mul_10^gradients/Cost_func/mul_2_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/Cost_func/mul_2_grad/Mul_1*
_output_shapes
: 
s
0gradients/Cost_func/add_12_grad/tuple/group_depsNoOp9^gradients/Cost_func/add_15_grad/tuple/control_dependency
�
8gradients/Cost_func/add_12_grad/tuple/control_dependencyIdentity8gradients/Cost_func/add_15_grad/tuple/control_dependency1^gradients/Cost_func/add_12_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
�
:gradients/Cost_func/add_12_grad/tuple/control_dependency_1Identity8gradients/Cost_func/add_15_grad/tuple/control_dependency1^gradients/Cost_func/add_12_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
�
"gradients/Cost_func/mul_4_grad/MulMul:gradients/Cost_func/add_15_grad/tuple/control_dependency_1Cost_func/add_14*
_output_shapes
: *
T0
�
$gradients/Cost_func/mul_4_grad/Mul_1Mul:gradients/Cost_func/add_15_grad/tuple/control_dependency_1Cost_func/mul_4/x*
T0*
_output_shapes
: 
�
/gradients/Cost_func/mul_4_grad/tuple/group_depsNoOp#^gradients/Cost_func/mul_4_grad/Mul%^gradients/Cost_func/mul_4_grad/Mul_1
�
7gradients/Cost_func/mul_4_grad/tuple/control_dependencyIdentity"gradients/Cost_func/mul_4_grad/Mul0^gradients/Cost_func/mul_4_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/Cost_func/mul_4_grad/Mul*
_output_shapes
: 
�
9gradients/Cost_func/mul_4_grad/tuple/control_dependency_1Identity$gradients/Cost_func/mul_4_grad/Mul_10^gradients/Cost_func/mul_4_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/Cost_func/mul_4_grad/Mul_1*
_output_shapes
: 
t
0gradients/Cost_func/add_17_grad/tuple/group_depsNoOp:^gradients/Cost_func/mul_5_grad/tuple/control_dependency_1
�
8gradients/Cost_func/add_17_grad/tuple/control_dependencyIdentity9gradients/Cost_func/mul_5_grad/tuple/control_dependency_11^gradients/Cost_func/add_17_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/Cost_func/mul_5_grad/Mul_1*
_output_shapes
: 
�
:gradients/Cost_func/add_17_grad/tuple/control_dependency_1Identity9gradients/Cost_func/mul_5_grad/tuple/control_dependency_11^gradients/Cost_func/add_17_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/Cost_func/mul_5_grad/Mul_1*
_output_shapes
: 
x
"gradients/Cost_func/Sub_grad/ShapeShapeData/IteratorGetNext:2*
T0*
out_type0*
_output_shapes
:
p
$gradients/Cost_func/Sub_grad/Shape_1ShapeModel/output*
T0*
out_type0*
_output_shapes
:
�
2gradients/Cost_func/Sub_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/Cost_func/Sub_grad/Shape$gradients/Cost_func/Sub_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
 gradients/Cost_func/Sub_grad/SumSum%gradients/Cost_func/Square_grad/Mul_12gradients/Cost_func/Sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
$gradients/Cost_func/Sub_grad/ReshapeReshape gradients/Cost_func/Sub_grad/Sum"gradients/Cost_func/Sub_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
"gradients/Cost_func/Sub_grad/Sum_1Sum%gradients/Cost_func/Square_grad/Mul_14gradients/Cost_func/Sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
n
 gradients/Cost_func/Sub_grad/NegNeg"gradients/Cost_func/Sub_grad/Sum_1*
_output_shapes
:*
T0
�
&gradients/Cost_func/Sub_grad/Reshape_1Reshape gradients/Cost_func/Sub_grad/Neg$gradients/Cost_func/Sub_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
-gradients/Cost_func/Sub_grad/tuple/group_depsNoOp%^gradients/Cost_func/Sub_grad/Reshape'^gradients/Cost_func/Sub_grad/Reshape_1
�
5gradients/Cost_func/Sub_grad/tuple/control_dependencyIdentity$gradients/Cost_func/Sub_grad/Reshape.^gradients/Cost_func/Sub_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/Cost_func/Sub_grad/Reshape*'
_output_shapes
:���������
�
7gradients/Cost_func/Sub_grad/tuple/control_dependency_1Identity&gradients/Cost_func/Sub_grad/Reshape_1.^gradients/Cost_func/Sub_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/Cost_func/Sub_grad/Reshape_1*'
_output_shapes
:���������
q
/gradients/Cost_func/add_2_grad/tuple/group_depsNoOp8^gradients/Cost_func/add_5_grad/tuple/control_dependency
�
7gradients/Cost_func/add_2_grad/tuple/control_dependencyIdentity7gradients/Cost_func/add_5_grad/tuple/control_dependency0^gradients/Cost_func/add_2_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
�
9gradients/Cost_func/add_2_grad/tuple/control_dependency_1Identity7gradients/Cost_func/add_5_grad/tuple/control_dependency0^gradients/Cost_func/add_2_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
�
"gradients/Cost_func/mul_1_grad/MulMul9gradients/Cost_func/add_5_grad/tuple/control_dependency_1Cost_func/add_4*
T0*
_output_shapes
: 
�
$gradients/Cost_func/mul_1_grad/Mul_1Mul9gradients/Cost_func/add_5_grad/tuple/control_dependency_1Cost_func/mul_1/x*
_output_shapes
: *
T0
�
/gradients/Cost_func/mul_1_grad/tuple/group_depsNoOp#^gradients/Cost_func/mul_1_grad/Mul%^gradients/Cost_func/mul_1_grad/Mul_1
�
7gradients/Cost_func/mul_1_grad/tuple/control_dependencyIdentity"gradients/Cost_func/mul_1_grad/Mul0^gradients/Cost_func/mul_1_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/Cost_func/mul_1_grad/Mul*
_output_shapes
: 
�
9gradients/Cost_func/mul_1_grad/tuple/control_dependency_1Identity$gradients/Cost_func/mul_1_grad/Mul_10^gradients/Cost_func/mul_1_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/Cost_func/mul_1_grad/Mul_1*
_output_shapes
: 
s
/gradients/Cost_func/add_7_grad/tuple/group_depsNoOp:^gradients/Cost_func/mul_2_grad/tuple/control_dependency_1
�
7gradients/Cost_func/add_7_grad/tuple/control_dependencyIdentity9gradients/Cost_func/mul_2_grad/tuple/control_dependency_10^gradients/Cost_func/add_7_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/Cost_func/mul_2_grad/Mul_1*
_output_shapes
: 
�
9gradients/Cost_func/add_7_grad/tuple/control_dependency_1Identity9gradients/Cost_func/mul_2_grad/tuple/control_dependency_10^gradients/Cost_func/add_7_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/Cost_func/mul_2_grad/Mul_1*
_output_shapes
: 
�
"gradients/Cost_func/mul_3_grad/MulMul:gradients/Cost_func/add_12_grad/tuple/control_dependency_1Cost_func/add_11*
T0*
_output_shapes
: 
�
$gradients/Cost_func/mul_3_grad/Mul_1Mul:gradients/Cost_func/add_12_grad/tuple/control_dependency_1Cost_func/mul_3/x*
_output_shapes
: *
T0
�
/gradients/Cost_func/mul_3_grad/tuple/group_depsNoOp#^gradients/Cost_func/mul_3_grad/Mul%^gradients/Cost_func/mul_3_grad/Mul_1
�
7gradients/Cost_func/mul_3_grad/tuple/control_dependencyIdentity"gradients/Cost_func/mul_3_grad/Mul0^gradients/Cost_func/mul_3_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/Cost_func/mul_3_grad/Mul*
_output_shapes
: 
�
9gradients/Cost_func/mul_3_grad/tuple/control_dependency_1Identity$gradients/Cost_func/mul_3_grad/Mul_10^gradients/Cost_func/mul_3_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/Cost_func/mul_3_grad/Mul_1*
_output_shapes
: 
t
0gradients/Cost_func/add_14_grad/tuple/group_depsNoOp:^gradients/Cost_func/mul_4_grad/tuple/control_dependency_1
�
8gradients/Cost_func/add_14_grad/tuple/control_dependencyIdentity9gradients/Cost_func/mul_4_grad/tuple/control_dependency_11^gradients/Cost_func/add_14_grad/tuple/group_deps*
_output_shapes
: *
T0*7
_class-
+)loc:@gradients/Cost_func/mul_4_grad/Mul_1
�
:gradients/Cost_func/add_14_grad/tuple/control_dependency_1Identity9gradients/Cost_func/mul_4_grad/tuple/control_dependency_11^gradients/Cost_func/add_14_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/Cost_func/mul_4_grad/Mul_1*
_output_shapes
: 
s
0gradients/Cost_func/add_16_grad/tuple/group_depsNoOp9^gradients/Cost_func/add_17_grad/tuple/control_dependency
�
8gradients/Cost_func/add_16_grad/tuple/control_dependencyIdentity8gradients/Cost_func/add_17_grad/tuple/control_dependency1^gradients/Cost_func/add_16_grad/tuple/group_deps*
_output_shapes
: *
T0*7
_class-
+)loc:@gradients/Cost_func/mul_5_grad/Mul_1
�
:gradients/Cost_func/add_16_grad/tuple/control_dependency_1Identity8gradients/Cost_func/add_17_grad/tuple/control_dependency1^gradients/Cost_func/add_16_grad/tuple/group_deps*
_output_shapes
: *
T0*7
_class-
+)loc:@gradients/Cost_func/mul_5_grad/Mul_1
~
-gradients/Cost_func/Sum_11_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
�
'gradients/Cost_func/Sum_11_grad/ReshapeReshape:gradients/Cost_func/add_17_grad/tuple/control_dependency_1-gradients/Cost_func/Sum_11_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
v
%gradients/Cost_func/Sum_11_grad/ConstConst*
valueB"h   F   *
dtype0*
_output_shapes
:
�
$gradients/Cost_func/Sum_11_grad/TileTile'gradients/Cost_func/Sum_11_grad/Reshape%gradients/Cost_func/Sum_11_grad/Const*

Tmultiples0*
T0*
_output_shapes

:hF
m
!gradients/Model/output_grad/ShapeShapeModel/Add_11*
_output_shapes
:*
T0*
out_type0
�
 gradients/Model/output_grad/SizeConst*
value	B :*4
_class*
(&loc:@gradients/Model/output_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients/Model/output_grad/addAddModel/output/reduction_indices gradients/Model/output_grad/Size*
T0*4
_class*
(&loc:@gradients/Model/output_grad/Shape*
_output_shapes
: 
�
gradients/Model/output_grad/modFloorModgradients/Model/output_grad/add gradients/Model/output_grad/Size*
T0*4
_class*
(&loc:@gradients/Model/output_grad/Shape*
_output_shapes
: 
�
#gradients/Model/output_grad/Shape_1Const*
valueB *4
_class*
(&loc:@gradients/Model/output_grad/Shape*
dtype0*
_output_shapes
: 
�
'gradients/Model/output_grad/range/startConst*
value	B : *4
_class*
(&loc:@gradients/Model/output_grad/Shape*
dtype0*
_output_shapes
: 
�
'gradients/Model/output_grad/range/deltaConst*
value	B :*4
_class*
(&loc:@gradients/Model/output_grad/Shape*
dtype0*
_output_shapes
: 
�
!gradients/Model/output_grad/rangeRange'gradients/Model/output_grad/range/start gradients/Model/output_grad/Size'gradients/Model/output_grad/range/delta*
_output_shapes
:*

Tidx0*4
_class*
(&loc:@gradients/Model/output_grad/Shape
�
&gradients/Model/output_grad/Fill/valueConst*
value	B :*4
_class*
(&loc:@gradients/Model/output_grad/Shape*
dtype0*
_output_shapes
: 
�
 gradients/Model/output_grad/FillFill#gradients/Model/output_grad/Shape_1&gradients/Model/output_grad/Fill/value*
T0*

index_type0*4
_class*
(&loc:@gradients/Model/output_grad/Shape*
_output_shapes
: 
�
)gradients/Model/output_grad/DynamicStitchDynamicStitch!gradients/Model/output_grad/rangegradients/Model/output_grad/mod!gradients/Model/output_grad/Shape gradients/Model/output_grad/Fill*
T0*4
_class*
(&loc:@gradients/Model/output_grad/Shape*
N*#
_output_shapes
:���������
�
%gradients/Model/output_grad/Maximum/yConst*
value	B :*4
_class*
(&loc:@gradients/Model/output_grad/Shape*
dtype0*
_output_shapes
: 
�
#gradients/Model/output_grad/MaximumMaximum)gradients/Model/output_grad/DynamicStitch%gradients/Model/output_grad/Maximum/y*
T0*4
_class*
(&loc:@gradients/Model/output_grad/Shape*#
_output_shapes
:���������
�
$gradients/Model/output_grad/floordivFloorDiv!gradients/Model/output_grad/Shape#gradients/Model/output_grad/Maximum*
_output_shapes
:*
T0*4
_class*
(&loc:@gradients/Model/output_grad/Shape
�
#gradients/Model/output_grad/ReshapeReshape7gradients/Cost_func/Sub_grad/tuple/control_dependency_1)gradients/Model/output_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
�
 gradients/Model/output_grad/TileTile#gradients/Model/output_grad/Reshape$gradients/Model/output_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:���������
�
 gradients/Cost_func/mul_grad/MulMul9gradients/Cost_func/add_2_grad/tuple/control_dependency_1Cost_func/add_1*
_output_shapes
: *
T0
�
"gradients/Cost_func/mul_grad/Mul_1Mul9gradients/Cost_func/add_2_grad/tuple/control_dependency_1Cost_func/mul/x*
T0*
_output_shapes
: 
}
-gradients/Cost_func/mul_grad/tuple/group_depsNoOp!^gradients/Cost_func/mul_grad/Mul#^gradients/Cost_func/mul_grad/Mul_1
�
5gradients/Cost_func/mul_grad/tuple/control_dependencyIdentity gradients/Cost_func/mul_grad/Mul.^gradients/Cost_func/mul_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/Cost_func/mul_grad/Mul*
_output_shapes
: 
�
7gradients/Cost_func/mul_grad/tuple/control_dependency_1Identity"gradients/Cost_func/mul_grad/Mul_1.^gradients/Cost_func/mul_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/Cost_func/mul_grad/Mul_1*
_output_shapes
: 
s
/gradients/Cost_func/add_4_grad/tuple/group_depsNoOp:^gradients/Cost_func/mul_1_grad/tuple/control_dependency_1
�
7gradients/Cost_func/add_4_grad/tuple/control_dependencyIdentity9gradients/Cost_func/mul_1_grad/tuple/control_dependency_10^gradients/Cost_func/add_4_grad/tuple/group_deps*
_output_shapes
: *
T0*7
_class-
+)loc:@gradients/Cost_func/mul_1_grad/Mul_1
�
9gradients/Cost_func/add_4_grad/tuple/control_dependency_1Identity9gradients/Cost_func/mul_1_grad/tuple/control_dependency_10^gradients/Cost_func/add_4_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/Cost_func/mul_1_grad/Mul_1*
_output_shapes
: 
q
/gradients/Cost_func/add_6_grad/tuple/group_depsNoOp8^gradients/Cost_func/add_7_grad/tuple/control_dependency
�
7gradients/Cost_func/add_6_grad/tuple/control_dependencyIdentity7gradients/Cost_func/add_7_grad/tuple/control_dependency0^gradients/Cost_func/add_6_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/Cost_func/mul_2_grad/Mul_1*
_output_shapes
: 
�
9gradients/Cost_func/add_6_grad/tuple/control_dependency_1Identity7gradients/Cost_func/add_7_grad/tuple/control_dependency0^gradients/Cost_func/add_6_grad/tuple/group_deps*
_output_shapes
: *
T0*7
_class-
+)loc:@gradients/Cost_func/mul_2_grad/Mul_1
}
,gradients/Cost_func/Sum_5_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
�
&gradients/Cost_func/Sum_5_grad/ReshapeReshape9gradients/Cost_func/add_7_grad/tuple/control_dependency_1,gradients/Cost_func/Sum_5_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
u
$gradients/Cost_func/Sum_5_grad/ConstConst*
valueB"h   F   *
dtype0*
_output_shapes
:
�
#gradients/Cost_func/Sum_5_grad/TileTile&gradients/Cost_func/Sum_5_grad/Reshape$gradients/Cost_func/Sum_5_grad/Const*

Tmultiples0*
T0*
_output_shapes

:hF
t
0gradients/Cost_func/add_11_grad/tuple/group_depsNoOp:^gradients/Cost_func/mul_3_grad/tuple/control_dependency_1
�
8gradients/Cost_func/add_11_grad/tuple/control_dependencyIdentity9gradients/Cost_func/mul_3_grad/tuple/control_dependency_11^gradients/Cost_func/add_11_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/Cost_func/mul_3_grad/Mul_1*
_output_shapes
: 
�
:gradients/Cost_func/add_11_grad/tuple/control_dependency_1Identity9gradients/Cost_func/mul_3_grad/tuple/control_dependency_11^gradients/Cost_func/add_11_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/Cost_func/mul_3_grad/Mul_1*
_output_shapes
: 
s
0gradients/Cost_func/add_13_grad/tuple/group_depsNoOp9^gradients/Cost_func/add_14_grad/tuple/control_dependency
�
8gradients/Cost_func/add_13_grad/tuple/control_dependencyIdentity8gradients/Cost_func/add_14_grad/tuple/control_dependency1^gradients/Cost_func/add_13_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/Cost_func/mul_4_grad/Mul_1*
_output_shapes
: 
�
:gradients/Cost_func/add_13_grad/tuple/control_dependency_1Identity8gradients/Cost_func/add_14_grad/tuple/control_dependency1^gradients/Cost_func/add_13_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/Cost_func/mul_4_grad/Mul_1*
_output_shapes
: 
}
,gradients/Cost_func/Sum_9_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
&gradients/Cost_func/Sum_9_grad/ReshapeReshape:gradients/Cost_func/add_14_grad/tuple/control_dependency_1,gradients/Cost_func/Sum_9_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
u
$gradients/Cost_func/Sum_9_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB"h   F   
�
#gradients/Cost_func/Sum_9_grad/TileTile&gradients/Cost_func/Sum_9_grad/Reshape$gradients/Cost_func/Sum_9_grad/Const*

Tmultiples0*
T0*
_output_shapes

:hF
~
-gradients/Cost_func/Sum_10_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
�
'gradients/Cost_func/Sum_10_grad/ReshapeReshape:gradients/Cost_func/add_16_grad/tuple/control_dependency_1-gradients/Cost_func/Sum_10_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
v
%gradients/Cost_func/Sum_10_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB"F   s  
�
$gradients/Cost_func/Sum_10_grad/TileTile'gradients/Cost_func/Sum_10_grad/Reshape%gradients/Cost_func/Sum_10_grad/Const*

Tmultiples0*
T0*
_output_shapes
:	F�

t
#gradients/Cost_func/Abs_5_grad/SignSignWeights/weight_hidden_1_2/read*
T0*
_output_shapes

:hF
�
"gradients/Cost_func/Abs_5_grad/mulMul$gradients/Cost_func/Sum_11_grad/Tile#gradients/Cost_func/Abs_5_grad/Sign*
T0*
_output_shapes

:hF
l
!gradients/Model/Add_11_grad/ShapeShapeModel/Add_7*
T0*
out_type0*
_output_shapes
:
t
#gradients/Model/Add_11_grad/Shape_1ShapeModel/ScatterNd_2*
T0*
out_type0*
_output_shapes
:
�
1gradients/Model/Add_11_grad/BroadcastGradientArgsBroadcastGradientArgs!gradients/Model/Add_11_grad/Shape#gradients/Model/Add_11_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/Model/Add_11_grad/SumSum gradients/Model/output_grad/Tile1gradients/Model/Add_11_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
#gradients/Model/Add_11_grad/ReshapeReshapegradients/Model/Add_11_grad/Sum!gradients/Model/Add_11_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
!gradients/Model/Add_11_grad/Sum_1Sum gradients/Model/output_grad/Tile3gradients/Model/Add_11_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
%gradients/Model/Add_11_grad/Reshape_1Reshape!gradients/Model/Add_11_grad/Sum_1#gradients/Model/Add_11_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
,gradients/Model/Add_11_grad/tuple/group_depsNoOp$^gradients/Model/Add_11_grad/Reshape&^gradients/Model/Add_11_grad/Reshape_1
�
4gradients/Model/Add_11_grad/tuple/control_dependencyIdentity#gradients/Model/Add_11_grad/Reshape-^gradients/Model/Add_11_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*6
_class,
*(loc:@gradients/Model/Add_11_grad/Reshape
�
6gradients/Model/Add_11_grad/tuple/control_dependency_1Identity%gradients/Model/Add_11_grad/Reshape_1-^gradients/Model/Add_11_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*8
_class.
,*loc:@gradients/Model/Add_11_grad/Reshape_1
q
/gradients/Cost_func/add_1_grad/tuple/group_depsNoOp8^gradients/Cost_func/mul_grad/tuple/control_dependency_1
�
7gradients/Cost_func/add_1_grad/tuple/control_dependencyIdentity7gradients/Cost_func/mul_grad/tuple/control_dependency_10^gradients/Cost_func/add_1_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/Cost_func/mul_grad/Mul_1*
_output_shapes
: 
�
9gradients/Cost_func/add_1_grad/tuple/control_dependency_1Identity7gradients/Cost_func/mul_grad/tuple/control_dependency_10^gradients/Cost_func/add_1_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/Cost_func/mul_grad/Mul_1*
_output_shapes
: 
q
/gradients/Cost_func/add_3_grad/tuple/group_depsNoOp8^gradients/Cost_func/add_4_grad/tuple/control_dependency
�
7gradients/Cost_func/add_3_grad/tuple/control_dependencyIdentity7gradients/Cost_func/add_4_grad/tuple/control_dependency0^gradients/Cost_func/add_3_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/Cost_func/mul_1_grad/Mul_1*
_output_shapes
: 
�
9gradients/Cost_func/add_3_grad/tuple/control_dependency_1Identity7gradients/Cost_func/add_4_grad/tuple/control_dependency0^gradients/Cost_func/add_3_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/Cost_func/mul_1_grad/Mul_1*
_output_shapes
: 
}
,gradients/Cost_func/Sum_3_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
�
&gradients/Cost_func/Sum_3_grad/ReshapeReshape9gradients/Cost_func/add_4_grad/tuple/control_dependency_1,gradients/Cost_func/Sum_3_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
u
$gradients/Cost_func/Sum_3_grad/ConstConst*
valueB"h   F   *
dtype0*
_output_shapes
:
�
#gradients/Cost_func/Sum_3_grad/TileTile&gradients/Cost_func/Sum_3_grad/Reshape$gradients/Cost_func/Sum_3_grad/Const*

Tmultiples0*
T0*
_output_shapes

:hF
}
,gradients/Cost_func/Sum_4_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
�
&gradients/Cost_func/Sum_4_grad/ReshapeReshape9gradients/Cost_func/add_6_grad/tuple/control_dependency_1,gradients/Cost_func/Sum_4_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
u
$gradients/Cost_func/Sum_4_grad/ConstConst*
valueB"F   s  *
dtype0*
_output_shapes
:
�
#gradients/Cost_func/Sum_4_grad/TileTile&gradients/Cost_func/Sum_4_grad/Reshape$gradients/Cost_func/Sum_4_grad/Const*
_output_shapes
:	F�
*

Tmultiples0*
T0
�
'gradients/Cost_func/Square_6_grad/ConstConst$^gradients/Cost_func/Sum_5_grad/Tile*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
%gradients/Cost_func/Square_6_grad/MulMulWeights/weight_hidden_1_2/read'gradients/Cost_func/Square_6_grad/Const*
T0*
_output_shapes

:hF
�
'gradients/Cost_func/Square_6_grad/Mul_1Mul#gradients/Cost_func/Sum_5_grad/Tile%gradients/Cost_func/Square_6_grad/Mul*
_output_shapes

:hF*
T0
s
0gradients/Cost_func/add_10_grad/tuple/group_depsNoOp9^gradients/Cost_func/add_11_grad/tuple/control_dependency
�
8gradients/Cost_func/add_10_grad/tuple/control_dependencyIdentity8gradients/Cost_func/add_11_grad/tuple/control_dependency1^gradients/Cost_func/add_10_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/Cost_func/mul_3_grad/Mul_1*
_output_shapes
: 
�
:gradients/Cost_func/add_10_grad/tuple/control_dependency_1Identity8gradients/Cost_func/add_11_grad/tuple/control_dependency1^gradients/Cost_func/add_10_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/Cost_func/mul_3_grad/Mul_1*
_output_shapes
: 
}
,gradients/Cost_func/Sum_7_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
&gradients/Cost_func/Sum_7_grad/ReshapeReshape:gradients/Cost_func/add_11_grad/tuple/control_dependency_1,gradients/Cost_func/Sum_7_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
u
$gradients/Cost_func/Sum_7_grad/ConstConst*
valueB"h   F   *
dtype0*
_output_shapes
:
�
#gradients/Cost_func/Sum_7_grad/TileTile&gradients/Cost_func/Sum_7_grad/Reshape$gradients/Cost_func/Sum_7_grad/Const*
T0*
_output_shapes

:hF*

Tmultiples0
}
,gradients/Cost_func/Sum_8_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
�
&gradients/Cost_func/Sum_8_grad/ReshapeReshape:gradients/Cost_func/add_13_grad/tuple/control_dependency_1,gradients/Cost_func/Sum_8_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
u
$gradients/Cost_func/Sum_8_grad/ConstConst*
valueB"F   s  *
dtype0*
_output_shapes
:
�
#gradients/Cost_func/Sum_8_grad/TileTile&gradients/Cost_func/Sum_8_grad/Reshape$gradients/Cost_func/Sum_8_grad/Const*

Tmultiples0*
T0*
_output_shapes
:	F�

t
#gradients/Cost_func/Abs_3_grad/SignSignWeights/weight_hidden_1_1/read*
T0*
_output_shapes

:hF
�
"gradients/Cost_func/Abs_3_grad/mulMul#gradients/Cost_func/Sum_9_grad/Tile#gradients/Cost_func/Abs_3_grad/Sign*
T0*
_output_shapes

:hF
o
#gradients/Cost_func/Abs_4_grad/SignSignWeights/weight_in_2/read*
_output_shapes
:	F�
*
T0
�
"gradients/Cost_func/Abs_4_grad/mulMul$gradients/Cost_func/Sum_10_grad/Tile#gradients/Cost_func/Abs_4_grad/Sign*
T0*
_output_shapes
:	F�

k
 gradients/Model/Add_7_grad/ShapeShapeModel/Add_3*
_output_shapes
:*
T0*
out_type0
s
"gradients/Model/Add_7_grad/Shape_1ShapeModel/ScatterNd_1*
_output_shapes
:*
T0*
out_type0
�
0gradients/Model/Add_7_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/Model/Add_7_grad/Shape"gradients/Model/Add_7_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/Model/Add_7_grad/SumSum4gradients/Model/Add_11_grad/tuple/control_dependency0gradients/Model/Add_7_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
"gradients/Model/Add_7_grad/ReshapeReshapegradients/Model/Add_7_grad/Sum gradients/Model/Add_7_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
 gradients/Model/Add_7_grad/Sum_1Sum4gradients/Model/Add_11_grad/tuple/control_dependency2gradients/Model/Add_7_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
$gradients/Model/Add_7_grad/Reshape_1Reshape gradients/Model/Add_7_grad/Sum_1"gradients/Model/Add_7_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������

+gradients/Model/Add_7_grad/tuple/group_depsNoOp#^gradients/Model/Add_7_grad/Reshape%^gradients/Model/Add_7_grad/Reshape_1
�
3gradients/Model/Add_7_grad/tuple/control_dependencyIdentity"gradients/Model/Add_7_grad/Reshape,^gradients/Model/Add_7_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*5
_class+
)'loc:@gradients/Model/Add_7_grad/Reshape
�
5gradients/Model/Add_7_grad/tuple/control_dependency_1Identity$gradients/Model/Add_7_grad/Reshape_1,^gradients/Model/Add_7_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/Model/Add_7_grad/Reshape_1*'
_output_shapes
:���������
�
)gradients/Model/ScatterNd_2_grad/GatherNdGatherNd6gradients/Model/Add_11_grad/tuple/control_dependency_1Model/Cast_2*
Tindices0*
Tparams0*#
_output_shapes
:���������
o
-gradients/Cost_func/add_grad/tuple/group_depsNoOp8^gradients/Cost_func/add_1_grad/tuple/control_dependency
�
5gradients/Cost_func/add_grad/tuple/control_dependencyIdentity7gradients/Cost_func/add_1_grad/tuple/control_dependency.^gradients/Cost_func/add_grad/tuple/group_deps*
_output_shapes
: *
T0*5
_class+
)'loc:@gradients/Cost_func/mul_grad/Mul_1
�
7gradients/Cost_func/add_grad/tuple/control_dependency_1Identity7gradients/Cost_func/add_1_grad/tuple/control_dependency.^gradients/Cost_func/add_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/Cost_func/mul_grad/Mul_1*
_output_shapes
: 
}
,gradients/Cost_func/Sum_1_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
�
&gradients/Cost_func/Sum_1_grad/ReshapeReshape9gradients/Cost_func/add_1_grad/tuple/control_dependency_1,gradients/Cost_func/Sum_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
u
$gradients/Cost_func/Sum_1_grad/ConstConst*
valueB"h   F   *
dtype0*
_output_shapes
:
�
#gradients/Cost_func/Sum_1_grad/TileTile&gradients/Cost_func/Sum_1_grad/Reshape$gradients/Cost_func/Sum_1_grad/Const*

Tmultiples0*
T0*
_output_shapes

:hF
}
,gradients/Cost_func/Sum_2_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
&gradients/Cost_func/Sum_2_grad/ReshapeReshape9gradients/Cost_func/add_3_grad/tuple/control_dependency_1,gradients/Cost_func/Sum_2_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
u
$gradients/Cost_func/Sum_2_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB"F   s  
�
#gradients/Cost_func/Sum_2_grad/TileTile&gradients/Cost_func/Sum_2_grad/Reshape$gradients/Cost_func/Sum_2_grad/Const*
T0*
_output_shapes
:	F�
*

Tmultiples0
�
'gradients/Cost_func/Square_4_grad/ConstConst$^gradients/Cost_func/Sum_3_grad/Tile*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
%gradients/Cost_func/Square_4_grad/MulMulWeights/weight_hidden_1_1/read'gradients/Cost_func/Square_4_grad/Const*
T0*
_output_shapes

:hF
�
'gradients/Cost_func/Square_4_grad/Mul_1Mul#gradients/Cost_func/Sum_3_grad/Tile%gradients/Cost_func/Square_4_grad/Mul*
_output_shapes

:hF*
T0
�
'gradients/Cost_func/Square_5_grad/ConstConst$^gradients/Cost_func/Sum_4_grad/Tile*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
%gradients/Cost_func/Square_5_grad/MulMulWeights/weight_in_2/read'gradients/Cost_func/Square_5_grad/Const*
T0*
_output_shapes
:	F�

�
'gradients/Cost_func/Square_5_grad/Mul_1Mul#gradients/Cost_func/Sum_4_grad/Tile%gradients/Cost_func/Square_5_grad/Mul*
_output_shapes
:	F�
*
T0
}
,gradients/Cost_func/Sum_6_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
�
&gradients/Cost_func/Sum_6_grad/ReshapeReshape:gradients/Cost_func/add_10_grad/tuple/control_dependency_1,gradients/Cost_func/Sum_6_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
u
$gradients/Cost_func/Sum_6_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB"F   s  
�
#gradients/Cost_func/Sum_6_grad/TileTile&gradients/Cost_func/Sum_6_grad/Reshape$gradients/Cost_func/Sum_6_grad/Const*
T0*
_output_shapes
:	F�
*

Tmultiples0
r
#gradients/Cost_func/Abs_1_grad/SignSignWeights/weight_hidden_1/read*
T0*
_output_shapes

:hF
�
"gradients/Cost_func/Abs_1_grad/mulMul#gradients/Cost_func/Sum_7_grad/Tile#gradients/Cost_func/Abs_1_grad/Sign*
T0*
_output_shapes

:hF
o
#gradients/Cost_func/Abs_2_grad/SignSignWeights/weight_in_1/read*
T0*
_output_shapes
:	F�

�
"gradients/Cost_func/Abs_2_grad/mulMul#gradients/Cost_func/Sum_8_grad/Tile#gradients/Cost_func/Abs_2_grad/Sign*
T0*
_output_shapes
:	F�

p
 gradients/Model/Add_3_grad/ShapeShapeModel/zeros_like*
T0*
out_type0*
_output_shapes
:
q
"gradients/Model/Add_3_grad/Shape_1ShapeModel/ScatterNd*
T0*
out_type0*
_output_shapes
:
�
0gradients/Model/Add_3_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/Model/Add_3_grad/Shape"gradients/Model/Add_3_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/Model/Add_3_grad/SumSum3gradients/Model/Add_7_grad/tuple/control_dependency0gradients/Model/Add_3_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
"gradients/Model/Add_3_grad/ReshapeReshapegradients/Model/Add_3_grad/Sum gradients/Model/Add_3_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
 gradients/Model/Add_3_grad/Sum_1Sum3gradients/Model/Add_7_grad/tuple/control_dependency2gradients/Model/Add_3_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
$gradients/Model/Add_3_grad/Reshape_1Reshape gradients/Model/Add_3_grad/Sum_1"gradients/Model/Add_3_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������

+gradients/Model/Add_3_grad/tuple/group_depsNoOp#^gradients/Model/Add_3_grad/Reshape%^gradients/Model/Add_3_grad/Reshape_1
�
3gradients/Model/Add_3_grad/tuple/control_dependencyIdentity"gradients/Model/Add_3_grad/Reshape,^gradients/Model/Add_3_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/Model/Add_3_grad/Reshape*'
_output_shapes
:���������
�
5gradients/Model/Add_3_grad/tuple/control_dependency_1Identity$gradients/Model/Add_3_grad/Reshape_1,^gradients/Model/Add_3_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/Model/Add_3_grad/Reshape_1*'
_output_shapes
:���������
�
)gradients/Model/ScatterNd_1_grad/GatherNdGatherNd5gradients/Model/Add_7_grad/tuple/control_dependency_1Model/Cast_1*
Tindices0*
Tparams0*#
_output_shapes
:���������
p
$gradients/Model/Squeeze_2_grad/ShapeShapeModel/Add_10*
T0*
out_type0*
_output_shapes
:
�
&gradients/Model/Squeeze_2_grad/ReshapeReshape)gradients/Model/ScatterNd_2_grad/GatherNd$gradients/Model/Squeeze_2_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
{
*gradients/Cost_func/Sum_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
�
$gradients/Cost_func/Sum_grad/ReshapeReshape7gradients/Cost_func/add_grad/tuple/control_dependency_1*gradients/Cost_func/Sum_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
s
"gradients/Cost_func/Sum_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB"F   s  
�
!gradients/Cost_func/Sum_grad/TileTile$gradients/Cost_func/Sum_grad/Reshape"gradients/Cost_func/Sum_grad/Const*

Tmultiples0*
T0*
_output_shapes
:	F�

�
'gradients/Cost_func/Square_2_grad/ConstConst$^gradients/Cost_func/Sum_1_grad/Tile*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
%gradients/Cost_func/Square_2_grad/MulMulWeights/weight_hidden_1/read'gradients/Cost_func/Square_2_grad/Const*
T0*
_output_shapes

:hF
�
'gradients/Cost_func/Square_2_grad/Mul_1Mul#gradients/Cost_func/Sum_1_grad/Tile%gradients/Cost_func/Square_2_grad/Mul*
_output_shapes

:hF*
T0
�
'gradients/Cost_func/Square_3_grad/ConstConst$^gradients/Cost_func/Sum_2_grad/Tile*
dtype0*
_output_shapes
: *
valueB
 *   @
�
%gradients/Cost_func/Square_3_grad/MulMulWeights/weight_in_1/read'gradients/Cost_func/Square_3_grad/Const*
_output_shapes
:	F�
*
T0
�
'gradients/Cost_func/Square_3_grad/Mul_1Mul#gradients/Cost_func/Sum_2_grad/Tile%gradients/Cost_func/Square_3_grad/Mul*
_output_shapes
:	F�
*
T0
k
!gradients/Cost_func/Abs_grad/SignSignWeights/weight_in/read*
T0*
_output_shapes
:	F�

�
 gradients/Cost_func/Abs_grad/mulMul#gradients/Cost_func/Sum_6_grad/Tile!gradients/Cost_func/Abs_grad/Sign*
_output_shapes
:	F�
*
T0
�
'gradients/Model/ScatterNd_grad/GatherNdGatherNd5gradients/Model/Add_3_grad/tuple/control_dependency_1
Model/Cast*
Tparams0*#
_output_shapes
:���������*
Tindices0
o
$gradients/Model/Squeeze_1_grad/ShapeShapeModel/Add_6*
T0*
out_type0*
_output_shapes
:
�
&gradients/Model/Squeeze_1_grad/ReshapeReshape)gradients/Model/ScatterNd_1_grad/GatherNd$gradients/Model/Squeeze_1_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
r
!gradients/Model/Add_10_grad/ShapeShapeModel/Tensordot_8*
T0*
out_type0*
_output_shapes
:
m
#gradients/Model/Add_10_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
1gradients/Model/Add_10_grad/BroadcastGradientArgsBroadcastGradientArgs!gradients/Model/Add_10_grad/Shape#gradients/Model/Add_10_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/Model/Add_10_grad/SumSum&gradients/Model/Squeeze_2_grad/Reshape1gradients/Model/Add_10_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
#gradients/Model/Add_10_grad/ReshapeReshapegradients/Model/Add_10_grad/Sum!gradients/Model/Add_10_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
!gradients/Model/Add_10_grad/Sum_1Sum&gradients/Model/Squeeze_2_grad/Reshape3gradients/Model/Add_10_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
%gradients/Model/Add_10_grad/Reshape_1Reshape!gradients/Model/Add_10_grad/Sum_1#gradients/Model/Add_10_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
,gradients/Model/Add_10_grad/tuple/group_depsNoOp$^gradients/Model/Add_10_grad/Reshape&^gradients/Model/Add_10_grad/Reshape_1
�
4gradients/Model/Add_10_grad/tuple/control_dependencyIdentity#gradients/Model/Add_10_grad/Reshape-^gradients/Model/Add_10_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/Model/Add_10_grad/Reshape*'
_output_shapes
:���������
�
6gradients/Model/Add_10_grad/tuple/control_dependency_1Identity%gradients/Model/Add_10_grad/Reshape_1-^gradients/Model/Add_10_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/Model/Add_10_grad/Reshape_1*
_output_shapes
:
�
'gradients/Cost_func/Square_1_grad/ConstConst"^gradients/Cost_func/Sum_grad/Tile*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
%gradients/Cost_func/Square_1_grad/MulMulWeights/weight_in/read'gradients/Cost_func/Square_1_grad/Const*
_output_shapes
:	F�
*
T0
�
'gradients/Cost_func/Square_1_grad/Mul_1Mul!gradients/Cost_func/Sum_grad/Tile%gradients/Cost_func/Square_1_grad/Mul*
T0*
_output_shapes
:	F�

m
"gradients/Model/Squeeze_grad/ShapeShapeModel/Add_2*
T0*
out_type0*
_output_shapes
:
�
$gradients/Model/Squeeze_grad/ReshapeReshape'gradients/Model/ScatterNd_grad/GatherNd"gradients/Model/Squeeze_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
q
 gradients/Model/Add_6_grad/ShapeShapeModel/Tensordot_5*
T0*
out_type0*
_output_shapes
:
l
"gradients/Model/Add_6_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
0gradients/Model/Add_6_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/Model/Add_6_grad/Shape"gradients/Model/Add_6_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/Model/Add_6_grad/SumSum&gradients/Model/Squeeze_1_grad/Reshape0gradients/Model/Add_6_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
"gradients/Model/Add_6_grad/ReshapeReshapegradients/Model/Add_6_grad/Sum gradients/Model/Add_6_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
 gradients/Model/Add_6_grad/Sum_1Sum&gradients/Model/Squeeze_1_grad/Reshape2gradients/Model/Add_6_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
$gradients/Model/Add_6_grad/Reshape_1Reshape gradients/Model/Add_6_grad/Sum_1"gradients/Model/Add_6_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

+gradients/Model/Add_6_grad/tuple/group_depsNoOp#^gradients/Model/Add_6_grad/Reshape%^gradients/Model/Add_6_grad/Reshape_1
�
3gradients/Model/Add_6_grad/tuple/control_dependencyIdentity"gradients/Model/Add_6_grad/Reshape,^gradients/Model/Add_6_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/Model/Add_6_grad/Reshape*'
_output_shapes
:���������
�
5gradients/Model/Add_6_grad/tuple/control_dependency_1Identity$gradients/Model/Add_6_grad/Reshape_1,^gradients/Model/Add_6_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/Model/Add_6_grad/Reshape_1*
_output_shapes
:
~
&gradients/Model/Tensordot_8_grad/ShapeShapeModel/Tensordot_8/MatMul*
T0*
out_type0*
_output_shapes
:
�
(gradients/Model/Tensordot_8_grad/ReshapeReshape4gradients/Model/Add_10_grad/tuple/control_dependency&gradients/Model/Tensordot_8_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
q
 gradients/Model/Add_2_grad/ShapeShapeModel/Tensordot_2*
T0*
out_type0*
_output_shapes
:
l
"gradients/Model/Add_2_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
0gradients/Model/Add_2_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/Model/Add_2_grad/Shape"gradients/Model/Add_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/Model/Add_2_grad/SumSum$gradients/Model/Squeeze_grad/Reshape0gradients/Model/Add_2_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
"gradients/Model/Add_2_grad/ReshapeReshapegradients/Model/Add_2_grad/Sum gradients/Model/Add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
 gradients/Model/Add_2_grad/Sum_1Sum$gradients/Model/Squeeze_grad/Reshape2gradients/Model/Add_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
$gradients/Model/Add_2_grad/Reshape_1Reshape gradients/Model/Add_2_grad/Sum_1"gradients/Model/Add_2_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0

+gradients/Model/Add_2_grad/tuple/group_depsNoOp#^gradients/Model/Add_2_grad/Reshape%^gradients/Model/Add_2_grad/Reshape_1
�
3gradients/Model/Add_2_grad/tuple/control_dependencyIdentity"gradients/Model/Add_2_grad/Reshape,^gradients/Model/Add_2_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/Model/Add_2_grad/Reshape*'
_output_shapes
:���������
�
5gradients/Model/Add_2_grad/tuple/control_dependency_1Identity$gradients/Model/Add_2_grad/Reshape_1,^gradients/Model/Add_2_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/Model/Add_2_grad/Reshape_1*
_output_shapes
:
~
&gradients/Model/Tensordot_5_grad/ShapeShapeModel/Tensordot_5/MatMul*
T0*
out_type0*
_output_shapes
:
�
(gradients/Model/Tensordot_5_grad/ReshapeReshape3gradients/Model/Add_6_grad/tuple/control_dependency&gradients/Model/Tensordot_5_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
.gradients/Model/Tensordot_8/MatMul_grad/MatMulMatMul(gradients/Model/Tensordot_8_grad/ReshapeModel/Tensordot_8/Reshape_1*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:���������h
�
0gradients/Model/Tensordot_8/MatMul_grad/MatMul_1MatMulModel/Tensordot_8/Reshape(gradients/Model/Tensordot_8_grad/Reshape*
transpose_b( *
T0*
transpose_a(*'
_output_shapes
:���������
�
8gradients/Model/Tensordot_8/MatMul_grad/tuple/group_depsNoOp/^gradients/Model/Tensordot_8/MatMul_grad/MatMul1^gradients/Model/Tensordot_8/MatMul_grad/MatMul_1
�
@gradients/Model/Tensordot_8/MatMul_grad/tuple/control_dependencyIdentity.gradients/Model/Tensordot_8/MatMul_grad/MatMul9^gradients/Model/Tensordot_8/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������h*
T0*A
_class7
53loc:@gradients/Model/Tensordot_8/MatMul_grad/MatMul
�
Bgradients/Model/Tensordot_8/MatMul_grad/tuple/control_dependency_1Identity0gradients/Model/Tensordot_8/MatMul_grad/MatMul_19^gradients/Model/Tensordot_8/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/Model/Tensordot_8/MatMul_grad/MatMul_1*
_output_shapes

:h
~
&gradients/Model/Tensordot_2_grad/ShapeShapeModel/Tensordot_2/MatMul*
T0*
out_type0*
_output_shapes
:
�
(gradients/Model/Tensordot_2_grad/ReshapeReshape3gradients/Model/Add_2_grad/tuple/control_dependency&gradients/Model/Tensordot_2_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
.gradients/Model/Tensordot_5/MatMul_grad/MatMulMatMul(gradients/Model/Tensordot_5_grad/ReshapeModel/Tensordot_5/Reshape_1*
T0*
transpose_a( *'
_output_shapes
:���������h*
transpose_b(
�
0gradients/Model/Tensordot_5/MatMul_grad/MatMul_1MatMulModel/Tensordot_5/Reshape(gradients/Model/Tensordot_5_grad/Reshape*
transpose_b( *
T0*
transpose_a(*'
_output_shapes
:���������
�
8gradients/Model/Tensordot_5/MatMul_grad/tuple/group_depsNoOp/^gradients/Model/Tensordot_5/MatMul_grad/MatMul1^gradients/Model/Tensordot_5/MatMul_grad/MatMul_1
�
@gradients/Model/Tensordot_5/MatMul_grad/tuple/control_dependencyIdentity.gradients/Model/Tensordot_5/MatMul_grad/MatMul9^gradients/Model/Tensordot_5/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/Model/Tensordot_5/MatMul_grad/MatMul*'
_output_shapes
:���������h
�
Bgradients/Model/Tensordot_5/MatMul_grad/tuple/control_dependency_1Identity0gradients/Model/Tensordot_5/MatMul_grad/MatMul_19^gradients/Model/Tensordot_5/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/Model/Tensordot_5/MatMul_grad/MatMul_1*
_output_shapes

:h
�
.gradients/Model/Tensordot_8/Reshape_grad/ShapeShapeModel/Tensordot_8/transpose*
_output_shapes
:*
T0*
out_type0
�
0gradients/Model/Tensordot_8/Reshape_grad/ReshapeReshape@gradients/Model/Tensordot_8/MatMul_grad/tuple/control_dependency.gradients/Model/Tensordot_8/Reshape_grad/Shape*
T0*
Tshape0*0
_output_shapes
:������������������
�
0gradients/Model/Tensordot_8/Reshape_1_grad/ShapeConst*
valueB"h      *
dtype0*
_output_shapes
:
�
2gradients/Model/Tensordot_8/Reshape_1_grad/ReshapeReshapeBgradients/Model/Tensordot_8/MatMul_grad/tuple/control_dependency_10gradients/Model/Tensordot_8/Reshape_1_grad/Shape*
T0*
Tshape0*
_output_shapes

:h
�
.gradients/Model/Tensordot_2/MatMul_grad/MatMulMatMul(gradients/Model/Tensordot_2_grad/ReshapeModel/Tensordot_2/Reshape_1*
T0*
transpose_a( *'
_output_shapes
:���������h*
transpose_b(
�
0gradients/Model/Tensordot_2/MatMul_grad/MatMul_1MatMulModel/Tensordot_2/Reshape(gradients/Model/Tensordot_2_grad/Reshape*
transpose_b( *
T0*
transpose_a(*'
_output_shapes
:���������
�
8gradients/Model/Tensordot_2/MatMul_grad/tuple/group_depsNoOp/^gradients/Model/Tensordot_2/MatMul_grad/MatMul1^gradients/Model/Tensordot_2/MatMul_grad/MatMul_1
�
@gradients/Model/Tensordot_2/MatMul_grad/tuple/control_dependencyIdentity.gradients/Model/Tensordot_2/MatMul_grad/MatMul9^gradients/Model/Tensordot_2/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������h*
T0*A
_class7
53loc:@gradients/Model/Tensordot_2/MatMul_grad/MatMul
�
Bgradients/Model/Tensordot_2/MatMul_grad/tuple/control_dependency_1Identity0gradients/Model/Tensordot_2/MatMul_grad/MatMul_19^gradients/Model/Tensordot_2/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/Model/Tensordot_2/MatMul_grad/MatMul_1*
_output_shapes

:h
�
.gradients/Model/Tensordot_5/Reshape_grad/ShapeShapeModel/Tensordot_5/transpose*
_output_shapes
:*
T0*
out_type0
�
0gradients/Model/Tensordot_5/Reshape_grad/ReshapeReshape@gradients/Model/Tensordot_5/MatMul_grad/tuple/control_dependency.gradients/Model/Tensordot_5/Reshape_grad/Shape*0
_output_shapes
:������������������*
T0*
Tshape0
�
0gradients/Model/Tensordot_5/Reshape_1_grad/ShapeConst*
valueB"h      *
dtype0*
_output_shapes
:
�
2gradients/Model/Tensordot_5/Reshape_1_grad/ReshapeReshapeBgradients/Model/Tensordot_5/MatMul_grad/tuple/control_dependency_10gradients/Model/Tensordot_5/Reshape_1_grad/Shape*
T0*
Tshape0*
_output_shapes

:h
�
<gradients/Model/Tensordot_8/transpose_grad/InvertPermutationInvertPermutationModel/Tensordot_8/concat_1*
T0*#
_output_shapes
:���������
�
4gradients/Model/Tensordot_8/transpose_grad/transpose	Transpose0gradients/Model/Tensordot_8/Reshape_grad/Reshape<gradients/Model/Tensordot_8/transpose_grad/InvertPermutation*
T0*'
_output_shapes
:���������h*
Tperm0
�
>gradients/Model/Tensordot_8/transpose_1_grad/InvertPermutationInvertPermutation"Model/Tensordot_8/transpose_1/perm*
T0*
_output_shapes
:
�
6gradients/Model/Tensordot_8/transpose_1_grad/transpose	Transpose2gradients/Model/Tensordot_8/Reshape_1_grad/Reshape>gradients/Model/Tensordot_8/transpose_1_grad/InvertPermutation*
_output_shapes

:h*
Tperm0*
T0
�
.gradients/Model/Tensordot_2/Reshape_grad/ShapeShapeModel/Tensordot_2/transpose*
_output_shapes
:*
T0*
out_type0
�
0gradients/Model/Tensordot_2/Reshape_grad/ReshapeReshape@gradients/Model/Tensordot_2/MatMul_grad/tuple/control_dependency.gradients/Model/Tensordot_2/Reshape_grad/Shape*
T0*
Tshape0*0
_output_shapes
:������������������
�
0gradients/Model/Tensordot_2/Reshape_1_grad/ShapeConst*
valueB"h      *
dtype0*
_output_shapes
:
�
2gradients/Model/Tensordot_2/Reshape_1_grad/ReshapeReshapeBgradients/Model/Tensordot_2/MatMul_grad/tuple/control_dependency_10gradients/Model/Tensordot_2/Reshape_1_grad/Shape*
T0*
Tshape0*
_output_shapes

:h
�
<gradients/Model/Tensordot_5/transpose_grad/InvertPermutationInvertPermutationModel/Tensordot_5/concat_1*
T0*#
_output_shapes
:���������
�
4gradients/Model/Tensordot_5/transpose_grad/transpose	Transpose0gradients/Model/Tensordot_5/Reshape_grad/Reshape<gradients/Model/Tensordot_5/transpose_grad/InvertPermutation*
T0*'
_output_shapes
:���������h*
Tperm0
�
>gradients/Model/Tensordot_5/transpose_1_grad/InvertPermutationInvertPermutation"Model/Tensordot_5/transpose_1/perm*
T0*
_output_shapes
:
�
6gradients/Model/Tensordot_5/transpose_1_grad/transpose	Transpose2gradients/Model/Tensordot_5/Reshape_1_grad/Reshape>gradients/Model/Tensordot_5/transpose_1_grad/InvertPermutation*
_output_shapes

:h*
Tperm0*
T0
�
*gradients/Model/Sigmoid_5_grad/SigmoidGradSigmoidGradModel/Sigmoid_54gradients/Model/Tensordot_8/transpose_grad/transpose*
T0*'
_output_shapes
:���������h
�
2gradients/Model/transpose_8_grad/InvertPermutationInvertPermutationModel/transpose_8/sub_1*
T0*
_output_shapes
:
�
*gradients/Model/transpose_8_grad/transpose	Transpose6gradients/Model/Tensordot_8/transpose_1_grad/transpose2gradients/Model/transpose_8_grad/InvertPermutation*
_output_shapes

:h*
Tperm0*
T0
�
<gradients/Model/Tensordot_2/transpose_grad/InvertPermutationInvertPermutationModel/Tensordot_2/concat_1*#
_output_shapes
:���������*
T0
�
4gradients/Model/Tensordot_2/transpose_grad/transpose	Transpose0gradients/Model/Tensordot_2/Reshape_grad/Reshape<gradients/Model/Tensordot_2/transpose_grad/InvertPermutation*
T0*'
_output_shapes
:���������h*
Tperm0
�
>gradients/Model/Tensordot_2/transpose_1_grad/InvertPermutationInvertPermutation"Model/Tensordot_2/transpose_1/perm*
T0*
_output_shapes
:
�
6gradients/Model/Tensordot_2/transpose_1_grad/transpose	Transpose2gradients/Model/Tensordot_2/Reshape_1_grad/Reshape>gradients/Model/Tensordot_2/transpose_1_grad/InvertPermutation*
T0*
_output_shapes

:h*
Tperm0
�
*gradients/Model/Sigmoid_3_grad/SigmoidGradSigmoidGradModel/Sigmoid_34gradients/Model/Tensordot_5/transpose_grad/transpose*
T0*'
_output_shapes
:���������h
�
2gradients/Model/transpose_5_grad/InvertPermutationInvertPermutationModel/transpose_5/sub_1*
T0*
_output_shapes
:
�
*gradients/Model/transpose_5_grad/transpose	Transpose6gradients/Model/Tensordot_5/transpose_1_grad/transpose2gradients/Model/transpose_5_grad/InvertPermutation*
T0*
_output_shapes

:h*
Tperm0
q
 gradients/Model/Add_9_grad/ShapeShapeModel/Tensordot_7*
T0*
out_type0*
_output_shapes
:
l
"gradients/Model/Add_9_grad/Shape_1Const*
valueB:h*
dtype0*
_output_shapes
:
�
0gradients/Model/Add_9_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/Model/Add_9_grad/Shape"gradients/Model/Add_9_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/Model/Add_9_grad/SumSum*gradients/Model/Sigmoid_5_grad/SigmoidGrad0gradients/Model/Add_9_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
"gradients/Model/Add_9_grad/ReshapeReshapegradients/Model/Add_9_grad/Sum gradients/Model/Add_9_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������h
�
 gradients/Model/Add_9_grad/Sum_1Sum*gradients/Model/Sigmoid_5_grad/SigmoidGrad2gradients/Model/Add_9_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
$gradients/Model/Add_9_grad/Reshape_1Reshape gradients/Model/Add_9_grad/Sum_1"gradients/Model/Add_9_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:h

+gradients/Model/Add_9_grad/tuple/group_depsNoOp#^gradients/Model/Add_9_grad/Reshape%^gradients/Model/Add_9_grad/Reshape_1
�
3gradients/Model/Add_9_grad/tuple/control_dependencyIdentity"gradients/Model/Add_9_grad/Reshape,^gradients/Model/Add_9_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/Model/Add_9_grad/Reshape*'
_output_shapes
:���������h
�
5gradients/Model/Add_9_grad/tuple/control_dependency_1Identity$gradients/Model/Add_9_grad/Reshape_1,^gradients/Model/Add_9_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/Model/Add_9_grad/Reshape_1*
_output_shapes
:h
�
*gradients/Model/Sigmoid_1_grad/SigmoidGradSigmoidGradModel/Sigmoid_14gradients/Model/Tensordot_2/transpose_grad/transpose*'
_output_shapes
:���������h*
T0
�
2gradients/Model/transpose_2_grad/InvertPermutationInvertPermutationModel/transpose_2/sub_1*
_output_shapes
:*
T0
�
*gradients/Model/transpose_2_grad/transpose	Transpose6gradients/Model/Tensordot_2/transpose_1_grad/transpose2gradients/Model/transpose_2_grad/InvertPermutation*
_output_shapes

:h*
Tperm0*
T0
q
 gradients/Model/Add_5_grad/ShapeShapeModel/Tensordot_4*
T0*
out_type0*
_output_shapes
:
l
"gradients/Model/Add_5_grad/Shape_1Const*
valueB:h*
dtype0*
_output_shapes
:
�
0gradients/Model/Add_5_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/Model/Add_5_grad/Shape"gradients/Model/Add_5_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/Model/Add_5_grad/SumSum*gradients/Model/Sigmoid_3_grad/SigmoidGrad0gradients/Model/Add_5_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
"gradients/Model/Add_5_grad/ReshapeReshapegradients/Model/Add_5_grad/Sum gradients/Model/Add_5_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������h
�
 gradients/Model/Add_5_grad/Sum_1Sum*gradients/Model/Sigmoid_3_grad/SigmoidGrad2gradients/Model/Add_5_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
$gradients/Model/Add_5_grad/Reshape_1Reshape gradients/Model/Add_5_grad/Sum_1"gradients/Model/Add_5_grad/Shape_1*
_output_shapes
:h*
T0*
Tshape0

+gradients/Model/Add_5_grad/tuple/group_depsNoOp#^gradients/Model/Add_5_grad/Reshape%^gradients/Model/Add_5_grad/Reshape_1
�
3gradients/Model/Add_5_grad/tuple/control_dependencyIdentity"gradients/Model/Add_5_grad/Reshape,^gradients/Model/Add_5_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/Model/Add_5_grad/Reshape*'
_output_shapes
:���������h
�
5gradients/Model/Add_5_grad/tuple/control_dependency_1Identity$gradients/Model/Add_5_grad/Reshape_1,^gradients/Model/Add_5_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/Model/Add_5_grad/Reshape_1*
_output_shapes
:h
~
&gradients/Model/Tensordot_7_grad/ShapeShapeModel/Tensordot_7/MatMul*
T0*
out_type0*
_output_shapes
:
�
(gradients/Model/Tensordot_7_grad/ReshapeReshape3gradients/Model/Add_9_grad/tuple/control_dependency&gradients/Model/Tensordot_7_grad/Shape*'
_output_shapes
:���������h*
T0*
Tshape0
q
 gradients/Model/Add_1_grad/ShapeShapeModel/Tensordot_1*
_output_shapes
:*
T0*
out_type0
l
"gradients/Model/Add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:h
�
0gradients/Model/Add_1_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/Model/Add_1_grad/Shape"gradients/Model/Add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/Model/Add_1_grad/SumSum*gradients/Model/Sigmoid_1_grad/SigmoidGrad0gradients/Model/Add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
"gradients/Model/Add_1_grad/ReshapeReshapegradients/Model/Add_1_grad/Sum gradients/Model/Add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������h
�
 gradients/Model/Add_1_grad/Sum_1Sum*gradients/Model/Sigmoid_1_grad/SigmoidGrad2gradients/Model/Add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
$gradients/Model/Add_1_grad/Reshape_1Reshape gradients/Model/Add_1_grad/Sum_1"gradients/Model/Add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:h

+gradients/Model/Add_1_grad/tuple/group_depsNoOp#^gradients/Model/Add_1_grad/Reshape%^gradients/Model/Add_1_grad/Reshape_1
�
3gradients/Model/Add_1_grad/tuple/control_dependencyIdentity"gradients/Model/Add_1_grad/Reshape,^gradients/Model/Add_1_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/Model/Add_1_grad/Reshape*'
_output_shapes
:���������h
�
5gradients/Model/Add_1_grad/tuple/control_dependency_1Identity$gradients/Model/Add_1_grad/Reshape_1,^gradients/Model/Add_1_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/Model/Add_1_grad/Reshape_1*
_output_shapes
:h
~
&gradients/Model/Tensordot_4_grad/ShapeShapeModel/Tensordot_4/MatMul*
T0*
out_type0*
_output_shapes
:
�
(gradients/Model/Tensordot_4_grad/ReshapeReshape3gradients/Model/Add_5_grad/tuple/control_dependency&gradients/Model/Tensordot_4_grad/Shape*'
_output_shapes
:���������h*
T0*
Tshape0
�
.gradients/Model/Tensordot_7/MatMul_grad/MatMulMatMul(gradients/Model/Tensordot_7_grad/ReshapeModel/Tensordot_7/Reshape_1*
transpose_a( *'
_output_shapes
:���������F*
transpose_b(*
T0
�
0gradients/Model/Tensordot_7/MatMul_grad/MatMul_1MatMulModel/Tensordot_7/Reshape(gradients/Model/Tensordot_7_grad/Reshape*
T0*
transpose_a(*'
_output_shapes
:���������h*
transpose_b( 
�
8gradients/Model/Tensordot_7/MatMul_grad/tuple/group_depsNoOp/^gradients/Model/Tensordot_7/MatMul_grad/MatMul1^gradients/Model/Tensordot_7/MatMul_grad/MatMul_1
�
@gradients/Model/Tensordot_7/MatMul_grad/tuple/control_dependencyIdentity.gradients/Model/Tensordot_7/MatMul_grad/MatMul9^gradients/Model/Tensordot_7/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/Model/Tensordot_7/MatMul_grad/MatMul*'
_output_shapes
:���������F
�
Bgradients/Model/Tensordot_7/MatMul_grad/tuple/control_dependency_1Identity0gradients/Model/Tensordot_7/MatMul_grad/MatMul_19^gradients/Model/Tensordot_7/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/Model/Tensordot_7/MatMul_grad/MatMul_1*
_output_shapes

:Fh
~
&gradients/Model/Tensordot_1_grad/ShapeShapeModel/Tensordot_1/MatMul*
T0*
out_type0*
_output_shapes
:
�
(gradients/Model/Tensordot_1_grad/ReshapeReshape3gradients/Model/Add_1_grad/tuple/control_dependency&gradients/Model/Tensordot_1_grad/Shape*'
_output_shapes
:���������h*
T0*
Tshape0
�
.gradients/Model/Tensordot_4/MatMul_grad/MatMulMatMul(gradients/Model/Tensordot_4_grad/ReshapeModel/Tensordot_4/Reshape_1*
T0*
transpose_a( *'
_output_shapes
:���������F*
transpose_b(
�
0gradients/Model/Tensordot_4/MatMul_grad/MatMul_1MatMulModel/Tensordot_4/Reshape(gradients/Model/Tensordot_4_grad/Reshape*
T0*
transpose_a(*'
_output_shapes
:���������h*
transpose_b( 
�
8gradients/Model/Tensordot_4/MatMul_grad/tuple/group_depsNoOp/^gradients/Model/Tensordot_4/MatMul_grad/MatMul1^gradients/Model/Tensordot_4/MatMul_grad/MatMul_1
�
@gradients/Model/Tensordot_4/MatMul_grad/tuple/control_dependencyIdentity.gradients/Model/Tensordot_4/MatMul_grad/MatMul9^gradients/Model/Tensordot_4/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/Model/Tensordot_4/MatMul_grad/MatMul*'
_output_shapes
:���������F
�
Bgradients/Model/Tensordot_4/MatMul_grad/tuple/control_dependency_1Identity0gradients/Model/Tensordot_4/MatMul_grad/MatMul_19^gradients/Model/Tensordot_4/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/Model/Tensordot_4/MatMul_grad/MatMul_1*
_output_shapes

:Fh
�
.gradients/Model/Tensordot_7/Reshape_grad/ShapeShapeModel/Tensordot_7/transpose*
T0*
out_type0*
_output_shapes
:
�
0gradients/Model/Tensordot_7/Reshape_grad/ReshapeReshape@gradients/Model/Tensordot_7/MatMul_grad/tuple/control_dependency.gradients/Model/Tensordot_7/Reshape_grad/Shape*
T0*
Tshape0*0
_output_shapes
:������������������
�
0gradients/Model/Tensordot_7/Reshape_1_grad/ShapeConst*
valueB"F   h   *
dtype0*
_output_shapes
:
�
2gradients/Model/Tensordot_7/Reshape_1_grad/ReshapeReshapeBgradients/Model/Tensordot_7/MatMul_grad/tuple/control_dependency_10gradients/Model/Tensordot_7/Reshape_1_grad/Shape*
T0*
Tshape0*
_output_shapes

:Fh
�
.gradients/Model/Tensordot_1/MatMul_grad/MatMulMatMul(gradients/Model/Tensordot_1_grad/ReshapeModel/Tensordot_1/Reshape_1*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:���������F
�
0gradients/Model/Tensordot_1/MatMul_grad/MatMul_1MatMulModel/Tensordot_1/Reshape(gradients/Model/Tensordot_1_grad/Reshape*
T0*
transpose_a(*'
_output_shapes
:���������h*
transpose_b( 
�
8gradients/Model/Tensordot_1/MatMul_grad/tuple/group_depsNoOp/^gradients/Model/Tensordot_1/MatMul_grad/MatMul1^gradients/Model/Tensordot_1/MatMul_grad/MatMul_1
�
@gradients/Model/Tensordot_1/MatMul_grad/tuple/control_dependencyIdentity.gradients/Model/Tensordot_1/MatMul_grad/MatMul9^gradients/Model/Tensordot_1/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������F*
T0*A
_class7
53loc:@gradients/Model/Tensordot_1/MatMul_grad/MatMul
�
Bgradients/Model/Tensordot_1/MatMul_grad/tuple/control_dependency_1Identity0gradients/Model/Tensordot_1/MatMul_grad/MatMul_19^gradients/Model/Tensordot_1/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/Model/Tensordot_1/MatMul_grad/MatMul_1*
_output_shapes

:Fh
�
.gradients/Model/Tensordot_4/Reshape_grad/ShapeShapeModel/Tensordot_4/transpose*
T0*
out_type0*
_output_shapes
:
�
0gradients/Model/Tensordot_4/Reshape_grad/ReshapeReshape@gradients/Model/Tensordot_4/MatMul_grad/tuple/control_dependency.gradients/Model/Tensordot_4/Reshape_grad/Shape*
T0*
Tshape0*0
_output_shapes
:������������������
�
0gradients/Model/Tensordot_4/Reshape_1_grad/ShapeConst*
valueB"F   h   *
dtype0*
_output_shapes
:
�
2gradients/Model/Tensordot_4/Reshape_1_grad/ReshapeReshapeBgradients/Model/Tensordot_4/MatMul_grad/tuple/control_dependency_10gradients/Model/Tensordot_4/Reshape_1_grad/Shape*
_output_shapes

:Fh*
T0*
Tshape0
�
<gradients/Model/Tensordot_7/transpose_grad/InvertPermutationInvertPermutationModel/Tensordot_7/concat_1*
T0*#
_output_shapes
:���������
�
4gradients/Model/Tensordot_7/transpose_grad/transpose	Transpose0gradients/Model/Tensordot_7/Reshape_grad/Reshape<gradients/Model/Tensordot_7/transpose_grad/InvertPermutation*
T0*'
_output_shapes
:���������F*
Tperm0
�
>gradients/Model/Tensordot_7/transpose_1_grad/InvertPermutationInvertPermutation"Model/Tensordot_7/transpose_1/perm*
T0*
_output_shapes
:
�
6gradients/Model/Tensordot_7/transpose_1_grad/transpose	Transpose2gradients/Model/Tensordot_7/Reshape_1_grad/Reshape>gradients/Model/Tensordot_7/transpose_1_grad/InvertPermutation*
T0*
_output_shapes

:Fh*
Tperm0
�
.gradients/Model/Tensordot_1/Reshape_grad/ShapeShapeModel/Tensordot_1/transpose*
_output_shapes
:*
T0*
out_type0
�
0gradients/Model/Tensordot_1/Reshape_grad/ReshapeReshape@gradients/Model/Tensordot_1/MatMul_grad/tuple/control_dependency.gradients/Model/Tensordot_1/Reshape_grad/Shape*
T0*
Tshape0*0
_output_shapes
:������������������
�
0gradients/Model/Tensordot_1/Reshape_1_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"F   h   
�
2gradients/Model/Tensordot_1/Reshape_1_grad/ReshapeReshapeBgradients/Model/Tensordot_1/MatMul_grad/tuple/control_dependency_10gradients/Model/Tensordot_1/Reshape_1_grad/Shape*
_output_shapes

:Fh*
T0*
Tshape0
�
<gradients/Model/Tensordot_4/transpose_grad/InvertPermutationInvertPermutationModel/Tensordot_4/concat_1*#
_output_shapes
:���������*
T0
�
4gradients/Model/Tensordot_4/transpose_grad/transpose	Transpose0gradients/Model/Tensordot_4/Reshape_grad/Reshape<gradients/Model/Tensordot_4/transpose_grad/InvertPermutation*
T0*'
_output_shapes
:���������F*
Tperm0
�
>gradients/Model/Tensordot_4/transpose_1_grad/InvertPermutationInvertPermutation"Model/Tensordot_4/transpose_1/perm*
T0*
_output_shapes
:
�
6gradients/Model/Tensordot_4/transpose_1_grad/transpose	Transpose2gradients/Model/Tensordot_4/Reshape_1_grad/Reshape>gradients/Model/Tensordot_4/transpose_1_grad/InvertPermutation*
T0*
_output_shapes

:Fh*
Tperm0
�
*gradients/Model/Sigmoid_4_grad/SigmoidGradSigmoidGradModel/Sigmoid_44gradients/Model/Tensordot_7/transpose_grad/transpose*'
_output_shapes
:���������F*
T0
�
2gradients/Model/transpose_7_grad/InvertPermutationInvertPermutationModel/transpose_7/sub_1*
_output_shapes
:*
T0
�
*gradients/Model/transpose_7_grad/transpose	Transpose6gradients/Model/Tensordot_7/transpose_1_grad/transpose2gradients/Model/transpose_7_grad/InvertPermutation*
T0*
_output_shapes

:hF*
Tperm0
�
<gradients/Model/Tensordot_1/transpose_grad/InvertPermutationInvertPermutationModel/Tensordot_1/concat_1*#
_output_shapes
:���������*
T0
�
4gradients/Model/Tensordot_1/transpose_grad/transpose	Transpose0gradients/Model/Tensordot_1/Reshape_grad/Reshape<gradients/Model/Tensordot_1/transpose_grad/InvertPermutation*
T0*'
_output_shapes
:���������F*
Tperm0
�
>gradients/Model/Tensordot_1/transpose_1_grad/InvertPermutationInvertPermutation"Model/Tensordot_1/transpose_1/perm*
T0*
_output_shapes
:
�
6gradients/Model/Tensordot_1/transpose_1_grad/transpose	Transpose2gradients/Model/Tensordot_1/Reshape_1_grad/Reshape>gradients/Model/Tensordot_1/transpose_1_grad/InvertPermutation*
Tperm0*
T0*
_output_shapes

:Fh
�
*gradients/Model/Sigmoid_2_grad/SigmoidGradSigmoidGradModel/Sigmoid_24gradients/Model/Tensordot_4/transpose_grad/transpose*
T0*'
_output_shapes
:���������F
�
2gradients/Model/transpose_4_grad/InvertPermutationInvertPermutationModel/transpose_4/sub_1*
_output_shapes
:*
T0
�
*gradients/Model/transpose_4_grad/transpose	Transpose6gradients/Model/Tensordot_4/transpose_1_grad/transpose2gradients/Model/transpose_4_grad/InvertPermutation*
_output_shapes

:hF*
Tperm0*
T0
q
 gradients/Model/Add_8_grad/ShapeShapeModel/Tensordot_6*
T0*
out_type0*
_output_shapes
:
l
"gradients/Model/Add_8_grad/Shape_1Const*
valueB:F*
dtype0*
_output_shapes
:
�
0gradients/Model/Add_8_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/Model/Add_8_grad/Shape"gradients/Model/Add_8_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/Model/Add_8_grad/SumSum*gradients/Model/Sigmoid_4_grad/SigmoidGrad0gradients/Model/Add_8_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
"gradients/Model/Add_8_grad/ReshapeReshapegradients/Model/Add_8_grad/Sum gradients/Model/Add_8_grad/Shape*'
_output_shapes
:���������F*
T0*
Tshape0
�
 gradients/Model/Add_8_grad/Sum_1Sum*gradients/Model/Sigmoid_4_grad/SigmoidGrad2gradients/Model/Add_8_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
$gradients/Model/Add_8_grad/Reshape_1Reshape gradients/Model/Add_8_grad/Sum_1"gradients/Model/Add_8_grad/Shape_1*
_output_shapes
:F*
T0*
Tshape0

+gradients/Model/Add_8_grad/tuple/group_depsNoOp#^gradients/Model/Add_8_grad/Reshape%^gradients/Model/Add_8_grad/Reshape_1
�
3gradients/Model/Add_8_grad/tuple/control_dependencyIdentity"gradients/Model/Add_8_grad/Reshape,^gradients/Model/Add_8_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/Model/Add_8_grad/Reshape*'
_output_shapes
:���������F
�
5gradients/Model/Add_8_grad/tuple/control_dependency_1Identity$gradients/Model/Add_8_grad/Reshape_1,^gradients/Model/Add_8_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/Model/Add_8_grad/Reshape_1*
_output_shapes
:F
�
gradients/AddNAddN"gradients/Cost_func/Abs_5_grad/mul'gradients/Cost_func/Square_6_grad/Mul_1*gradients/Model/transpose_7_grad/transpose*
T0*5
_class+
)'loc:@gradients/Cost_func/Abs_5_grad/mul*
N*
_output_shapes

:hF
�
(gradients/Model/Sigmoid_grad/SigmoidGradSigmoidGradModel/Sigmoid4gradients/Model/Tensordot_1/transpose_grad/transpose*'
_output_shapes
:���������F*
T0
�
2gradients/Model/transpose_1_grad/InvertPermutationInvertPermutationModel/transpose_1/sub_1*
T0*
_output_shapes
:
�
*gradients/Model/transpose_1_grad/transpose	Transpose6gradients/Model/Tensordot_1/transpose_1_grad/transpose2gradients/Model/transpose_1_grad/InvertPermutation*
T0*
_output_shapes

:hF*
Tperm0
q
 gradients/Model/Add_4_grad/ShapeShapeModel/Tensordot_3*
T0*
out_type0*
_output_shapes
:
l
"gradients/Model/Add_4_grad/Shape_1Const*
valueB:F*
dtype0*
_output_shapes
:
�
0gradients/Model/Add_4_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/Model/Add_4_grad/Shape"gradients/Model/Add_4_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/Model/Add_4_grad/SumSum*gradients/Model/Sigmoid_2_grad/SigmoidGrad0gradients/Model/Add_4_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
"gradients/Model/Add_4_grad/ReshapeReshapegradients/Model/Add_4_grad/Sum gradients/Model/Add_4_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������F
�
 gradients/Model/Add_4_grad/Sum_1Sum*gradients/Model/Sigmoid_2_grad/SigmoidGrad2gradients/Model/Add_4_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
$gradients/Model/Add_4_grad/Reshape_1Reshape gradients/Model/Add_4_grad/Sum_1"gradients/Model/Add_4_grad/Shape_1*
_output_shapes
:F*
T0*
Tshape0

+gradients/Model/Add_4_grad/tuple/group_depsNoOp#^gradients/Model/Add_4_grad/Reshape%^gradients/Model/Add_4_grad/Reshape_1
�
3gradients/Model/Add_4_grad/tuple/control_dependencyIdentity"gradients/Model/Add_4_grad/Reshape,^gradients/Model/Add_4_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/Model/Add_4_grad/Reshape*'
_output_shapes
:���������F
�
5gradients/Model/Add_4_grad/tuple/control_dependency_1Identity$gradients/Model/Add_4_grad/Reshape_1,^gradients/Model/Add_4_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/Model/Add_4_grad/Reshape_1*
_output_shapes
:F
�
gradients/AddN_1AddN"gradients/Cost_func/Abs_3_grad/mul'gradients/Cost_func/Square_4_grad/Mul_1*gradients/Model/transpose_4_grad/transpose*
N*
_output_shapes

:hF*
T0*5
_class+
)'loc:@gradients/Cost_func/Abs_3_grad/mul
~
&gradients/Model/Tensordot_6_grad/ShapeShapeModel/Tensordot_6/MatMul*
T0*
out_type0*
_output_shapes
:
�
(gradients/Model/Tensordot_6_grad/ReshapeReshape3gradients/Model/Add_8_grad/tuple/control_dependency&gradients/Model/Tensordot_6_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������F
m
gradients/Model/Add_grad/ShapeShapeModel/Tensordot*
T0*
out_type0*
_output_shapes
:
j
 gradients/Model/Add_grad/Shape_1Const*
valueB:F*
dtype0*
_output_shapes
:
�
.gradients/Model/Add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Model/Add_grad/Shape gradients/Model/Add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/Model/Add_grad/SumSum(gradients/Model/Sigmoid_grad/SigmoidGrad.gradients/Model/Add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
 gradients/Model/Add_grad/ReshapeReshapegradients/Model/Add_grad/Sumgradients/Model/Add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������F
�
gradients/Model/Add_grad/Sum_1Sum(gradients/Model/Sigmoid_grad/SigmoidGrad0gradients/Model/Add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
"gradients/Model/Add_grad/Reshape_1Reshapegradients/Model/Add_grad/Sum_1 gradients/Model/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:F
y
)gradients/Model/Add_grad/tuple/group_depsNoOp!^gradients/Model/Add_grad/Reshape#^gradients/Model/Add_grad/Reshape_1
�
1gradients/Model/Add_grad/tuple/control_dependencyIdentity gradients/Model/Add_grad/Reshape*^gradients/Model/Add_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/Model/Add_grad/Reshape*'
_output_shapes
:���������F
�
3gradients/Model/Add_grad/tuple/control_dependency_1Identity"gradients/Model/Add_grad/Reshape_1*^gradients/Model/Add_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/Model/Add_grad/Reshape_1*
_output_shapes
:F
�
gradients/AddN_2AddN"gradients/Cost_func/Abs_1_grad/mul'gradients/Cost_func/Square_2_grad/Mul_1*gradients/Model/transpose_1_grad/transpose*
T0*5
_class+
)'loc:@gradients/Cost_func/Abs_1_grad/mul*
N*
_output_shapes

:hF
~
&gradients/Model/Tensordot_3_grad/ShapeShapeModel/Tensordot_3/MatMul*
T0*
out_type0*
_output_shapes
:
�
(gradients/Model/Tensordot_3_grad/ReshapeReshape3gradients/Model/Add_4_grad/tuple/control_dependency&gradients/Model/Tensordot_3_grad/Shape*'
_output_shapes
:���������F*
T0*
Tshape0
�
.gradients/Model/Tensordot_6/MatMul_grad/MatMulMatMul(gradients/Model/Tensordot_6_grad/ReshapeModel/Tensordot_6/Reshape_1*
T0*
transpose_a( *(
_output_shapes
:����������
*
transpose_b(
�
0gradients/Model/Tensordot_6/MatMul_grad/MatMul_1MatMulModel/Tensordot_6/Reshape(gradients/Model/Tensordot_6_grad/Reshape*
transpose_b( *
T0*
transpose_a(*'
_output_shapes
:���������F
�
8gradients/Model/Tensordot_6/MatMul_grad/tuple/group_depsNoOp/^gradients/Model/Tensordot_6/MatMul_grad/MatMul1^gradients/Model/Tensordot_6/MatMul_grad/MatMul_1
�
@gradients/Model/Tensordot_6/MatMul_grad/tuple/control_dependencyIdentity.gradients/Model/Tensordot_6/MatMul_grad/MatMul9^gradients/Model/Tensordot_6/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/Model/Tensordot_6/MatMul_grad/MatMul*(
_output_shapes
:����������

�
Bgradients/Model/Tensordot_6/MatMul_grad/tuple/control_dependency_1Identity0gradients/Model/Tensordot_6/MatMul_grad/MatMul_19^gradients/Model/Tensordot_6/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/Model/Tensordot_6/MatMul_grad/MatMul_1*
_output_shapes
:	�
F
z
$gradients/Model/Tensordot_grad/ShapeShapeModel/Tensordot/MatMul*
T0*
out_type0*
_output_shapes
:
�
&gradients/Model/Tensordot_grad/ReshapeReshape1gradients/Model/Add_grad/tuple/control_dependency$gradients/Model/Tensordot_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������F
�
.gradients/Model/Tensordot_3/MatMul_grad/MatMulMatMul(gradients/Model/Tensordot_3_grad/ReshapeModel/Tensordot_3/Reshape_1*
T0*
transpose_a( *(
_output_shapes
:����������
*
transpose_b(
�
0gradients/Model/Tensordot_3/MatMul_grad/MatMul_1MatMulModel/Tensordot_3/Reshape(gradients/Model/Tensordot_3_grad/Reshape*
transpose_b( *
T0*
transpose_a(*'
_output_shapes
:���������F
�
8gradients/Model/Tensordot_3/MatMul_grad/tuple/group_depsNoOp/^gradients/Model/Tensordot_3/MatMul_grad/MatMul1^gradients/Model/Tensordot_3/MatMul_grad/MatMul_1
�
@gradients/Model/Tensordot_3/MatMul_grad/tuple/control_dependencyIdentity.gradients/Model/Tensordot_3/MatMul_grad/MatMul9^gradients/Model/Tensordot_3/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/Model/Tensordot_3/MatMul_grad/MatMul*(
_output_shapes
:����������

�
Bgradients/Model/Tensordot_3/MatMul_grad/tuple/control_dependency_1Identity0gradients/Model/Tensordot_3/MatMul_grad/MatMul_19^gradients/Model/Tensordot_3/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/Model/Tensordot_3/MatMul_grad/MatMul_1*
_output_shapes
:	�
F
�
0gradients/Model/Tensordot_6/Reshape_1_grad/ShapeConst*
valueB"s  F   *
dtype0*
_output_shapes
:
�
2gradients/Model/Tensordot_6/Reshape_1_grad/ReshapeReshapeBgradients/Model/Tensordot_6/MatMul_grad/tuple/control_dependency_10gradients/Model/Tensordot_6/Reshape_1_grad/Shape*
T0*
Tshape0*
_output_shapes
:	�
F
�
,gradients/Model/Tensordot/MatMul_grad/MatMulMatMul&gradients/Model/Tensordot_grad/ReshapeModel/Tensordot/Reshape_1*
T0*
transpose_a( *(
_output_shapes
:����������
*
transpose_b(
�
.gradients/Model/Tensordot/MatMul_grad/MatMul_1MatMulModel/Tensordot/Reshape&gradients/Model/Tensordot_grad/Reshape*
transpose_b( *
T0*
transpose_a(*'
_output_shapes
:���������F
�
6gradients/Model/Tensordot/MatMul_grad/tuple/group_depsNoOp-^gradients/Model/Tensordot/MatMul_grad/MatMul/^gradients/Model/Tensordot/MatMul_grad/MatMul_1
�
>gradients/Model/Tensordot/MatMul_grad/tuple/control_dependencyIdentity,gradients/Model/Tensordot/MatMul_grad/MatMul7^gradients/Model/Tensordot/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Model/Tensordot/MatMul_grad/MatMul*(
_output_shapes
:����������

�
@gradients/Model/Tensordot/MatMul_grad/tuple/control_dependency_1Identity.gradients/Model/Tensordot/MatMul_grad/MatMul_17^gradients/Model/Tensordot/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/Model/Tensordot/MatMul_grad/MatMul_1*
_output_shapes
:	�
F
�
0gradients/Model/Tensordot_3/Reshape_1_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"s  F   
�
2gradients/Model/Tensordot_3/Reshape_1_grad/ReshapeReshapeBgradients/Model/Tensordot_3/MatMul_grad/tuple/control_dependency_10gradients/Model/Tensordot_3/Reshape_1_grad/Shape*
T0*
Tshape0*
_output_shapes
:	�
F
�
>gradients/Model/Tensordot_6/transpose_1_grad/InvertPermutationInvertPermutation"Model/Tensordot_6/transpose_1/perm*
T0*
_output_shapes
:
�
6gradients/Model/Tensordot_6/transpose_1_grad/transpose	Transpose2gradients/Model/Tensordot_6/Reshape_1_grad/Reshape>gradients/Model/Tensordot_6/transpose_1_grad/InvertPermutation*
T0*
_output_shapes
:	�
F*
Tperm0

.gradients/Model/Tensordot/Reshape_1_grad/ShapeConst*
valueB"s  F   *
dtype0*
_output_shapes
:
�
0gradients/Model/Tensordot/Reshape_1_grad/ReshapeReshape@gradients/Model/Tensordot/MatMul_grad/tuple/control_dependency_1.gradients/Model/Tensordot/Reshape_1_grad/Shape*
T0*
Tshape0*
_output_shapes
:	�
F
�
>gradients/Model/Tensordot_3/transpose_1_grad/InvertPermutationInvertPermutation"Model/Tensordot_3/transpose_1/perm*
_output_shapes
:*
T0
�
6gradients/Model/Tensordot_3/transpose_1_grad/transpose	Transpose2gradients/Model/Tensordot_3/Reshape_1_grad/Reshape>gradients/Model/Tensordot_3/transpose_1_grad/InvertPermutation*
T0*
_output_shapes
:	�
F*
Tperm0
�
2gradients/Model/transpose_6_grad/InvertPermutationInvertPermutationModel/transpose_6/sub_1*
T0*
_output_shapes
:
�
*gradients/Model/transpose_6_grad/transpose	Transpose6gradients/Model/Tensordot_6/transpose_1_grad/transpose2gradients/Model/transpose_6_grad/InvertPermutation*
T0*
_output_shapes
:	F�
*
Tperm0
�
<gradients/Model/Tensordot/transpose_1_grad/InvertPermutationInvertPermutation Model/Tensordot/transpose_1/perm*
T0*
_output_shapes
:
�
4gradients/Model/Tensordot/transpose_1_grad/transpose	Transpose0gradients/Model/Tensordot/Reshape_1_grad/Reshape<gradients/Model/Tensordot/transpose_1_grad/InvertPermutation*
Tperm0*
T0*
_output_shapes
:	�
F
�
2gradients/Model/transpose_3_grad/InvertPermutationInvertPermutationModel/transpose_3/sub_1*
T0*
_output_shapes
:
�
*gradients/Model/transpose_3_grad/transpose	Transpose6gradients/Model/Tensordot_3/transpose_1_grad/transpose2gradients/Model/transpose_3_grad/InvertPermutation*
_output_shapes
:	F�
*
Tperm0*
T0
�
gradients/AddN_3AddN"gradients/Cost_func/Abs_4_grad/mul'gradients/Cost_func/Square_5_grad/Mul_1*gradients/Model/transpose_6_grad/transpose*
T0*5
_class+
)'loc:@gradients/Cost_func/Abs_4_grad/mul*
N*
_output_shapes
:	F�

�
0gradients/Model/transpose_grad/InvertPermutationInvertPermutationModel/transpose/sub_1*
T0*
_output_shapes
:
�
(gradients/Model/transpose_grad/transpose	Transpose4gradients/Model/Tensordot/transpose_1_grad/transpose0gradients/Model/transpose_grad/InvertPermutation*
Tperm0*
T0*
_output_shapes
:	F�

�
gradients/AddN_4AddN"gradients/Cost_func/Abs_2_grad/mul'gradients/Cost_func/Square_3_grad/Mul_1*gradients/Model/transpose_3_grad/transpose*
N*
_output_shapes
:	F�
*
T0*5
_class+
)'loc:@gradients/Cost_func/Abs_2_grad/mul
�
gradients/AddN_5AddN gradients/Cost_func/Abs_grad/mul'gradients/Cost_func/Square_1_grad/Mul_1(gradients/Model/transpose_grad/transpose*
N*
_output_shapes
:	F�
*
T0*3
_class)
'%loc:@gradients/Cost_func/Abs_grad/mul
�
beta1_power/initial_valueConst*
valueB
 *fff?*(
_class
loc:@Weights/bias_hidden_1*
dtype0*
_output_shapes
: 
�
beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *(
_class
loc:@Weights/bias_hidden_1*
	container *
shape: 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*(
_class
loc:@Weights/bias_hidden_1*
validate_shape(*
_output_shapes
: *
use_locking(
t
beta1_power/readIdentitybeta1_power*
T0*(
_class
loc:@Weights/bias_hidden_1*
_output_shapes
: 
�
beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *w�?*(
_class
loc:@Weights/bias_hidden_1
�
beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *(
_class
loc:@Weights/bias_hidden_1*
	container *
shape: 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*(
_class
loc:@Weights/bias_hidden_1*
validate_shape(*
_output_shapes
: 
t
beta2_power/readIdentitybeta2_power*
T0*(
_class
loc:@Weights/bias_hidden_1*
_output_shapes
: 
�
8Weights/weight_in/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"F   s  *$
_class
loc:@Weights/weight_in
�
.Weights/weight_in/Adam/Initializer/zeros/ConstConst*
valueB
 *    *$
_class
loc:@Weights/weight_in*
dtype0*
_output_shapes
: 
�
(Weights/weight_in/Adam/Initializer/zerosFill8Weights/weight_in/Adam/Initializer/zeros/shape_as_tensor.Weights/weight_in/Adam/Initializer/zeros/Const*
T0*

index_type0*$
_class
loc:@Weights/weight_in*
_output_shapes
:	F�

�
Weights/weight_in/Adam
VariableV2*
shared_name *$
_class
loc:@Weights/weight_in*
	container *
shape:	F�
*
dtype0*
_output_shapes
:	F�

�
Weights/weight_in/Adam/AssignAssignWeights/weight_in/Adam(Weights/weight_in/Adam/Initializer/zeros*
T0*$
_class
loc:@Weights/weight_in*
validate_shape(*
_output_shapes
:	F�
*
use_locking(
�
Weights/weight_in/Adam/readIdentityWeights/weight_in/Adam*
T0*$
_class
loc:@Weights/weight_in*
_output_shapes
:	F�

�
:Weights/weight_in/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"F   s  *$
_class
loc:@Weights/weight_in*
dtype0*
_output_shapes
:
�
0Weights/weight_in/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *$
_class
loc:@Weights/weight_in
�
*Weights/weight_in/Adam_1/Initializer/zerosFill:Weights/weight_in/Adam_1/Initializer/zeros/shape_as_tensor0Weights/weight_in/Adam_1/Initializer/zeros/Const*
_output_shapes
:	F�
*
T0*

index_type0*$
_class
loc:@Weights/weight_in
�
Weights/weight_in/Adam_1
VariableV2*
shape:	F�
*
dtype0*
_output_shapes
:	F�
*
shared_name *$
_class
loc:@Weights/weight_in*
	container 
�
Weights/weight_in/Adam_1/AssignAssignWeights/weight_in/Adam_1*Weights/weight_in/Adam_1/Initializer/zeros*
T0*$
_class
loc:@Weights/weight_in*
validate_shape(*
_output_shapes
:	F�
*
use_locking(
�
Weights/weight_in/Adam_1/readIdentityWeights/weight_in/Adam_1*
T0*$
_class
loc:@Weights/weight_in*
_output_shapes
:	F�

�
&Weights/bias_in/Adam/Initializer/zerosConst*
valueBF*    *"
_class
loc:@Weights/bias_in*
dtype0*
_output_shapes
:F
�
Weights/bias_in/Adam
VariableV2*"
_class
loc:@Weights/bias_in*
	container *
shape:F*
dtype0*
_output_shapes
:F*
shared_name 
�
Weights/bias_in/Adam/AssignAssignWeights/bias_in/Adam&Weights/bias_in/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:F*
use_locking(*
T0*"
_class
loc:@Weights/bias_in
�
Weights/bias_in/Adam/readIdentityWeights/bias_in/Adam*
T0*"
_class
loc:@Weights/bias_in*
_output_shapes
:F
�
(Weights/bias_in/Adam_1/Initializer/zerosConst*
valueBF*    *"
_class
loc:@Weights/bias_in*
dtype0*
_output_shapes
:F
�
Weights/bias_in/Adam_1
VariableV2*
shared_name *"
_class
loc:@Weights/bias_in*
	container *
shape:F*
dtype0*
_output_shapes
:F
�
Weights/bias_in/Adam_1/AssignAssignWeights/bias_in/Adam_1(Weights/bias_in/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@Weights/bias_in*
validate_shape(*
_output_shapes
:F
�
Weights/bias_in/Adam_1/readIdentityWeights/bias_in/Adam_1*
T0*"
_class
loc:@Weights/bias_in*
_output_shapes
:F
�
>Weights/weight_hidden_1/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"h   F   **
_class 
loc:@Weights/weight_hidden_1*
dtype0*
_output_shapes
:
�
4Weights/weight_hidden_1/Adam/Initializer/zeros/ConstConst*
valueB
 *    **
_class 
loc:@Weights/weight_hidden_1*
dtype0*
_output_shapes
: 
�
.Weights/weight_hidden_1/Adam/Initializer/zerosFill>Weights/weight_hidden_1/Adam/Initializer/zeros/shape_as_tensor4Weights/weight_hidden_1/Adam/Initializer/zeros/Const*
T0*

index_type0**
_class 
loc:@Weights/weight_hidden_1*
_output_shapes

:hF
�
Weights/weight_hidden_1/Adam
VariableV2*
shape
:hF*
dtype0*
_output_shapes

:hF*
shared_name **
_class 
loc:@Weights/weight_hidden_1*
	container 
�
#Weights/weight_hidden_1/Adam/AssignAssignWeights/weight_hidden_1/Adam.Weights/weight_hidden_1/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:hF*
use_locking(*
T0**
_class 
loc:@Weights/weight_hidden_1
�
!Weights/weight_hidden_1/Adam/readIdentityWeights/weight_hidden_1/Adam*
T0**
_class 
loc:@Weights/weight_hidden_1*
_output_shapes

:hF
�
@Weights/weight_hidden_1/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"h   F   **
_class 
loc:@Weights/weight_hidden_1*
dtype0*
_output_shapes
:
�
6Weights/weight_hidden_1/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    **
_class 
loc:@Weights/weight_hidden_1*
dtype0*
_output_shapes
: 
�
0Weights/weight_hidden_1/Adam_1/Initializer/zerosFill@Weights/weight_hidden_1/Adam_1/Initializer/zeros/shape_as_tensor6Weights/weight_hidden_1/Adam_1/Initializer/zeros/Const*
_output_shapes

:hF*
T0*

index_type0**
_class 
loc:@Weights/weight_hidden_1
�
Weights/weight_hidden_1/Adam_1
VariableV2*
shared_name **
_class 
loc:@Weights/weight_hidden_1*
	container *
shape
:hF*
dtype0*
_output_shapes

:hF
�
%Weights/weight_hidden_1/Adam_1/AssignAssignWeights/weight_hidden_1/Adam_10Weights/weight_hidden_1/Adam_1/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Weights/weight_hidden_1*
validate_shape(*
_output_shapes

:hF
�
#Weights/weight_hidden_1/Adam_1/readIdentityWeights/weight_hidden_1/Adam_1*
T0**
_class 
loc:@Weights/weight_hidden_1*
_output_shapes

:hF
�
,Weights/bias_hidden_1/Adam/Initializer/zerosConst*
valueBh*    *(
_class
loc:@Weights/bias_hidden_1*
dtype0*
_output_shapes
:h
�
Weights/bias_hidden_1/Adam
VariableV2*
shared_name *(
_class
loc:@Weights/bias_hidden_1*
	container *
shape:h*
dtype0*
_output_shapes
:h
�
!Weights/bias_hidden_1/Adam/AssignAssignWeights/bias_hidden_1/Adam,Weights/bias_hidden_1/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:h*
use_locking(*
T0*(
_class
loc:@Weights/bias_hidden_1
�
Weights/bias_hidden_1/Adam/readIdentityWeights/bias_hidden_1/Adam*
T0*(
_class
loc:@Weights/bias_hidden_1*
_output_shapes
:h
�
.Weights/bias_hidden_1/Adam_1/Initializer/zerosConst*
valueBh*    *(
_class
loc:@Weights/bias_hidden_1*
dtype0*
_output_shapes
:h
�
Weights/bias_hidden_1/Adam_1
VariableV2*
dtype0*
_output_shapes
:h*
shared_name *(
_class
loc:@Weights/bias_hidden_1*
	container *
shape:h
�
#Weights/bias_hidden_1/Adam_1/AssignAssignWeights/bias_hidden_1/Adam_1.Weights/bias_hidden_1/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:h*
use_locking(*
T0*(
_class
loc:@Weights/bias_hidden_1
�
!Weights/bias_hidden_1/Adam_1/readIdentityWeights/bias_hidden_1/Adam_1*
T0*(
_class
loc:@Weights/bias_hidden_1*
_output_shapes
:h
�
)Weights/weight_out/Adam/Initializer/zerosConst*
valueBh*    *%
_class
loc:@Weights/weight_out*
dtype0*
_output_shapes

:h
�
Weights/weight_out/Adam
VariableV2*
shared_name *%
_class
loc:@Weights/weight_out*
	container *
shape
:h*
dtype0*
_output_shapes

:h
�
Weights/weight_out/Adam/AssignAssignWeights/weight_out/Adam)Weights/weight_out/Adam/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@Weights/weight_out*
validate_shape(*
_output_shapes

:h
�
Weights/weight_out/Adam/readIdentityWeights/weight_out/Adam*
T0*%
_class
loc:@Weights/weight_out*
_output_shapes

:h
�
+Weights/weight_out/Adam_1/Initializer/zerosConst*
valueBh*    *%
_class
loc:@Weights/weight_out*
dtype0*
_output_shapes

:h
�
Weights/weight_out/Adam_1
VariableV2*%
_class
loc:@Weights/weight_out*
	container *
shape
:h*
dtype0*
_output_shapes

:h*
shared_name 
�
 Weights/weight_out/Adam_1/AssignAssignWeights/weight_out/Adam_1+Weights/weight_out/Adam_1/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@Weights/weight_out*
validate_shape(*
_output_shapes

:h
�
Weights/weight_out/Adam_1/readIdentityWeights/weight_out/Adam_1*
T0*%
_class
loc:@Weights/weight_out*
_output_shapes

:h
�
'Weights/bias_out/Adam/Initializer/zerosConst*
valueB*    *#
_class
loc:@Weights/bias_out*
dtype0*
_output_shapes
:
�
Weights/bias_out/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *#
_class
loc:@Weights/bias_out*
	container *
shape:
�
Weights/bias_out/Adam/AssignAssignWeights/bias_out/Adam'Weights/bias_out/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@Weights/bias_out*
validate_shape(*
_output_shapes
:
�
Weights/bias_out/Adam/readIdentityWeights/bias_out/Adam*
_output_shapes
:*
T0*#
_class
loc:@Weights/bias_out
�
)Weights/bias_out/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *#
_class
loc:@Weights/bias_out
�
Weights/bias_out/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *#
_class
loc:@Weights/bias_out*
	container *
shape:
�
Weights/bias_out/Adam_1/AssignAssignWeights/bias_out/Adam_1)Weights/bias_out/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@Weights/bias_out*
validate_shape(*
_output_shapes
:
�
Weights/bias_out/Adam_1/readIdentityWeights/bias_out/Adam_1*
T0*#
_class
loc:@Weights/bias_out*
_output_shapes
:
�
:Weights/weight_in_1/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"F   s  *&
_class
loc:@Weights/weight_in_1*
dtype0*
_output_shapes
:
�
0Weights/weight_in_1/Adam/Initializer/zeros/ConstConst*
valueB
 *    *&
_class
loc:@Weights/weight_in_1*
dtype0*
_output_shapes
: 
�
*Weights/weight_in_1/Adam/Initializer/zerosFill:Weights/weight_in_1/Adam/Initializer/zeros/shape_as_tensor0Weights/weight_in_1/Adam/Initializer/zeros/Const*
_output_shapes
:	F�
*
T0*

index_type0*&
_class
loc:@Weights/weight_in_1
�
Weights/weight_in_1/Adam
VariableV2*
shared_name *&
_class
loc:@Weights/weight_in_1*
	container *
shape:	F�
*
dtype0*
_output_shapes
:	F�

�
Weights/weight_in_1/Adam/AssignAssignWeights/weight_in_1/Adam*Weights/weight_in_1/Adam/Initializer/zeros*
T0*&
_class
loc:@Weights/weight_in_1*
validate_shape(*
_output_shapes
:	F�
*
use_locking(
�
Weights/weight_in_1/Adam/readIdentityWeights/weight_in_1/Adam*
T0*&
_class
loc:@Weights/weight_in_1*
_output_shapes
:	F�

�
<Weights/weight_in_1/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"F   s  *&
_class
loc:@Weights/weight_in_1
�
2Weights/weight_in_1/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *&
_class
loc:@Weights/weight_in_1*
dtype0*
_output_shapes
: 
�
,Weights/weight_in_1/Adam_1/Initializer/zerosFill<Weights/weight_in_1/Adam_1/Initializer/zeros/shape_as_tensor2Weights/weight_in_1/Adam_1/Initializer/zeros/Const*
T0*

index_type0*&
_class
loc:@Weights/weight_in_1*
_output_shapes
:	F�

�
Weights/weight_in_1/Adam_1
VariableV2*
shared_name *&
_class
loc:@Weights/weight_in_1*
	container *
shape:	F�
*
dtype0*
_output_shapes
:	F�

�
!Weights/weight_in_1/Adam_1/AssignAssignWeights/weight_in_1/Adam_1,Weights/weight_in_1/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:	F�
*
use_locking(*
T0*&
_class
loc:@Weights/weight_in_1
�
Weights/weight_in_1/Adam_1/readIdentityWeights/weight_in_1/Adam_1*
T0*&
_class
loc:@Weights/weight_in_1*
_output_shapes
:	F�

�
(Weights/bias_in_1/Adam/Initializer/zerosConst*
valueBF*    *$
_class
loc:@Weights/bias_in_1*
dtype0*
_output_shapes
:F
�
Weights/bias_in_1/Adam
VariableV2*
dtype0*
_output_shapes
:F*
shared_name *$
_class
loc:@Weights/bias_in_1*
	container *
shape:F
�
Weights/bias_in_1/Adam/AssignAssignWeights/bias_in_1/Adam(Weights/bias_in_1/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@Weights/bias_in_1*
validate_shape(*
_output_shapes
:F
�
Weights/bias_in_1/Adam/readIdentityWeights/bias_in_1/Adam*
T0*$
_class
loc:@Weights/bias_in_1*
_output_shapes
:F
�
*Weights/bias_in_1/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:F*
valueBF*    *$
_class
loc:@Weights/bias_in_1
�
Weights/bias_in_1/Adam_1
VariableV2*
shared_name *$
_class
loc:@Weights/bias_in_1*
	container *
shape:F*
dtype0*
_output_shapes
:F
�
Weights/bias_in_1/Adam_1/AssignAssignWeights/bias_in_1/Adam_1*Weights/bias_in_1/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@Weights/bias_in_1*
validate_shape(*
_output_shapes
:F
�
Weights/bias_in_1/Adam_1/readIdentityWeights/bias_in_1/Adam_1*
T0*$
_class
loc:@Weights/bias_in_1*
_output_shapes
:F
�
@Weights/weight_hidden_1_1/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"h   F   *,
_class"
 loc:@Weights/weight_hidden_1_1
�
6Weights/weight_hidden_1_1/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *,
_class"
 loc:@Weights/weight_hidden_1_1
�
0Weights/weight_hidden_1_1/Adam/Initializer/zerosFill@Weights/weight_hidden_1_1/Adam/Initializer/zeros/shape_as_tensor6Weights/weight_hidden_1_1/Adam/Initializer/zeros/Const*
_output_shapes

:hF*
T0*

index_type0*,
_class"
 loc:@Weights/weight_hidden_1_1
�
Weights/weight_hidden_1_1/Adam
VariableV2*
dtype0*
_output_shapes

:hF*
shared_name *,
_class"
 loc:@Weights/weight_hidden_1_1*
	container *
shape
:hF
�
%Weights/weight_hidden_1_1/Adam/AssignAssignWeights/weight_hidden_1_1/Adam0Weights/weight_hidden_1_1/Adam/Initializer/zeros*
T0*,
_class"
 loc:@Weights/weight_hidden_1_1*
validate_shape(*
_output_shapes

:hF*
use_locking(
�
#Weights/weight_hidden_1_1/Adam/readIdentityWeights/weight_hidden_1_1/Adam*
_output_shapes

:hF*
T0*,
_class"
 loc:@Weights/weight_hidden_1_1
�
BWeights/weight_hidden_1_1/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"h   F   *,
_class"
 loc:@Weights/weight_hidden_1_1*
dtype0*
_output_shapes
:
�
8Weights/weight_hidden_1_1/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *,
_class"
 loc:@Weights/weight_hidden_1_1
�
2Weights/weight_hidden_1_1/Adam_1/Initializer/zerosFillBWeights/weight_hidden_1_1/Adam_1/Initializer/zeros/shape_as_tensor8Weights/weight_hidden_1_1/Adam_1/Initializer/zeros/Const*
T0*

index_type0*,
_class"
 loc:@Weights/weight_hidden_1_1*
_output_shapes

:hF
�
 Weights/weight_hidden_1_1/Adam_1
VariableV2*,
_class"
 loc:@Weights/weight_hidden_1_1*
	container *
shape
:hF*
dtype0*
_output_shapes

:hF*
shared_name 
�
'Weights/weight_hidden_1_1/Adam_1/AssignAssign Weights/weight_hidden_1_1/Adam_12Weights/weight_hidden_1_1/Adam_1/Initializer/zeros*
T0*,
_class"
 loc:@Weights/weight_hidden_1_1*
validate_shape(*
_output_shapes

:hF*
use_locking(
�
%Weights/weight_hidden_1_1/Adam_1/readIdentity Weights/weight_hidden_1_1/Adam_1*
T0*,
_class"
 loc:@Weights/weight_hidden_1_1*
_output_shapes

:hF
�
.Weights/bias_hidden_1_1/Adam/Initializer/zerosConst*
valueBh*    **
_class 
loc:@Weights/bias_hidden_1_1*
dtype0*
_output_shapes
:h
�
Weights/bias_hidden_1_1/Adam
VariableV2*
	container *
shape:h*
dtype0*
_output_shapes
:h*
shared_name **
_class 
loc:@Weights/bias_hidden_1_1
�
#Weights/bias_hidden_1_1/Adam/AssignAssignWeights/bias_hidden_1_1/Adam.Weights/bias_hidden_1_1/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:h*
use_locking(*
T0**
_class 
loc:@Weights/bias_hidden_1_1
�
!Weights/bias_hidden_1_1/Adam/readIdentityWeights/bias_hidden_1_1/Adam*
T0**
_class 
loc:@Weights/bias_hidden_1_1*
_output_shapes
:h
�
0Weights/bias_hidden_1_1/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:h*
valueBh*    **
_class 
loc:@Weights/bias_hidden_1_1
�
Weights/bias_hidden_1_1/Adam_1
VariableV2*
shared_name **
_class 
loc:@Weights/bias_hidden_1_1*
	container *
shape:h*
dtype0*
_output_shapes
:h
�
%Weights/bias_hidden_1_1/Adam_1/AssignAssignWeights/bias_hidden_1_1/Adam_10Weights/bias_hidden_1_1/Adam_1/Initializer/zeros*
T0**
_class 
loc:@Weights/bias_hidden_1_1*
validate_shape(*
_output_shapes
:h*
use_locking(
�
#Weights/bias_hidden_1_1/Adam_1/readIdentityWeights/bias_hidden_1_1/Adam_1*
T0**
_class 
loc:@Weights/bias_hidden_1_1*
_output_shapes
:h
�
+Weights/weight_out_1/Adam/Initializer/zerosConst*
valueBh*    *'
_class
loc:@Weights/weight_out_1*
dtype0*
_output_shapes

:h
�
Weights/weight_out_1/Adam
VariableV2*
dtype0*
_output_shapes

:h*
shared_name *'
_class
loc:@Weights/weight_out_1*
	container *
shape
:h
�
 Weights/weight_out_1/Adam/AssignAssignWeights/weight_out_1/Adam+Weights/weight_out_1/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@Weights/weight_out_1*
validate_shape(*
_output_shapes

:h
�
Weights/weight_out_1/Adam/readIdentityWeights/weight_out_1/Adam*
T0*'
_class
loc:@Weights/weight_out_1*
_output_shapes

:h
�
-Weights/weight_out_1/Adam_1/Initializer/zerosConst*
valueBh*    *'
_class
loc:@Weights/weight_out_1*
dtype0*
_output_shapes

:h
�
Weights/weight_out_1/Adam_1
VariableV2*
	container *
shape
:h*
dtype0*
_output_shapes

:h*
shared_name *'
_class
loc:@Weights/weight_out_1
�
"Weights/weight_out_1/Adam_1/AssignAssignWeights/weight_out_1/Adam_1-Weights/weight_out_1/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@Weights/weight_out_1*
validate_shape(*
_output_shapes

:h
�
 Weights/weight_out_1/Adam_1/readIdentityWeights/weight_out_1/Adam_1*
T0*'
_class
loc:@Weights/weight_out_1*
_output_shapes

:h
�
)Weights/bias_out_1/Adam/Initializer/zerosConst*
valueB*    *%
_class
loc:@Weights/bias_out_1*
dtype0*
_output_shapes
:
�
Weights/bias_out_1/Adam
VariableV2*
shared_name *%
_class
loc:@Weights/bias_out_1*
	container *
shape:*
dtype0*
_output_shapes
:
�
Weights/bias_out_1/Adam/AssignAssignWeights/bias_out_1/Adam)Weights/bias_out_1/Adam/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@Weights/bias_out_1*
validate_shape(*
_output_shapes
:
�
Weights/bias_out_1/Adam/readIdentityWeights/bias_out_1/Adam*
_output_shapes
:*
T0*%
_class
loc:@Weights/bias_out_1
�
+Weights/bias_out_1/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *%
_class
loc:@Weights/bias_out_1
�
Weights/bias_out_1/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *%
_class
loc:@Weights/bias_out_1*
	container 
�
 Weights/bias_out_1/Adam_1/AssignAssignWeights/bias_out_1/Adam_1+Weights/bias_out_1/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*%
_class
loc:@Weights/bias_out_1
�
Weights/bias_out_1/Adam_1/readIdentityWeights/bias_out_1/Adam_1*
_output_shapes
:*
T0*%
_class
loc:@Weights/bias_out_1
�
:Weights/weight_in_2/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"F   s  *&
_class
loc:@Weights/weight_in_2*
dtype0*
_output_shapes
:
�
0Weights/weight_in_2/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *&
_class
loc:@Weights/weight_in_2
�
*Weights/weight_in_2/Adam/Initializer/zerosFill:Weights/weight_in_2/Adam/Initializer/zeros/shape_as_tensor0Weights/weight_in_2/Adam/Initializer/zeros/Const*
_output_shapes
:	F�
*
T0*

index_type0*&
_class
loc:@Weights/weight_in_2
�
Weights/weight_in_2/Adam
VariableV2*
	container *
shape:	F�
*
dtype0*
_output_shapes
:	F�
*
shared_name *&
_class
loc:@Weights/weight_in_2
�
Weights/weight_in_2/Adam/AssignAssignWeights/weight_in_2/Adam*Weights/weight_in_2/Adam/Initializer/zeros*
T0*&
_class
loc:@Weights/weight_in_2*
validate_shape(*
_output_shapes
:	F�
*
use_locking(
�
Weights/weight_in_2/Adam/readIdentityWeights/weight_in_2/Adam*
T0*&
_class
loc:@Weights/weight_in_2*
_output_shapes
:	F�

�
<Weights/weight_in_2/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"F   s  *&
_class
loc:@Weights/weight_in_2*
dtype0*
_output_shapes
:
�
2Weights/weight_in_2/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *&
_class
loc:@Weights/weight_in_2*
dtype0*
_output_shapes
: 
�
,Weights/weight_in_2/Adam_1/Initializer/zerosFill<Weights/weight_in_2/Adam_1/Initializer/zeros/shape_as_tensor2Weights/weight_in_2/Adam_1/Initializer/zeros/Const*
T0*

index_type0*&
_class
loc:@Weights/weight_in_2*
_output_shapes
:	F�

�
Weights/weight_in_2/Adam_1
VariableV2*
shared_name *&
_class
loc:@Weights/weight_in_2*
	container *
shape:	F�
*
dtype0*
_output_shapes
:	F�

�
!Weights/weight_in_2/Adam_1/AssignAssignWeights/weight_in_2/Adam_1,Weights/weight_in_2/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:	F�
*
use_locking(*
T0*&
_class
loc:@Weights/weight_in_2
�
Weights/weight_in_2/Adam_1/readIdentityWeights/weight_in_2/Adam_1*
T0*&
_class
loc:@Weights/weight_in_2*
_output_shapes
:	F�

�
(Weights/bias_in_2/Adam/Initializer/zerosConst*
valueBF*    *$
_class
loc:@Weights/bias_in_2*
dtype0*
_output_shapes
:F
�
Weights/bias_in_2/Adam
VariableV2*
dtype0*
_output_shapes
:F*
shared_name *$
_class
loc:@Weights/bias_in_2*
	container *
shape:F
�
Weights/bias_in_2/Adam/AssignAssignWeights/bias_in_2/Adam(Weights/bias_in_2/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@Weights/bias_in_2*
validate_shape(*
_output_shapes
:F
�
Weights/bias_in_2/Adam/readIdentityWeights/bias_in_2/Adam*
T0*$
_class
loc:@Weights/bias_in_2*
_output_shapes
:F
�
*Weights/bias_in_2/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:F*
valueBF*    *$
_class
loc:@Weights/bias_in_2
�
Weights/bias_in_2/Adam_1
VariableV2*
dtype0*
_output_shapes
:F*
shared_name *$
_class
loc:@Weights/bias_in_2*
	container *
shape:F
�
Weights/bias_in_2/Adam_1/AssignAssignWeights/bias_in_2/Adam_1*Weights/bias_in_2/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:F*
use_locking(*
T0*$
_class
loc:@Weights/bias_in_2
�
Weights/bias_in_2/Adam_1/readIdentityWeights/bias_in_2/Adam_1*
_output_shapes
:F*
T0*$
_class
loc:@Weights/bias_in_2
�
@Weights/weight_hidden_1_2/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"h   F   *,
_class"
 loc:@Weights/weight_hidden_1_2*
dtype0*
_output_shapes
:
�
6Weights/weight_hidden_1_2/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *,
_class"
 loc:@Weights/weight_hidden_1_2
�
0Weights/weight_hidden_1_2/Adam/Initializer/zerosFill@Weights/weight_hidden_1_2/Adam/Initializer/zeros/shape_as_tensor6Weights/weight_hidden_1_2/Adam/Initializer/zeros/Const*
T0*

index_type0*,
_class"
 loc:@Weights/weight_hidden_1_2*
_output_shapes

:hF
�
Weights/weight_hidden_1_2/Adam
VariableV2*
dtype0*
_output_shapes

:hF*
shared_name *,
_class"
 loc:@Weights/weight_hidden_1_2*
	container *
shape
:hF
�
%Weights/weight_hidden_1_2/Adam/AssignAssignWeights/weight_hidden_1_2/Adam0Weights/weight_hidden_1_2/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@Weights/weight_hidden_1_2*
validate_shape(*
_output_shapes

:hF
�
#Weights/weight_hidden_1_2/Adam/readIdentityWeights/weight_hidden_1_2/Adam*
T0*,
_class"
 loc:@Weights/weight_hidden_1_2*
_output_shapes

:hF
�
BWeights/weight_hidden_1_2/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"h   F   *,
_class"
 loc:@Weights/weight_hidden_1_2*
dtype0*
_output_shapes
:
�
8Weights/weight_hidden_1_2/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *,
_class"
 loc:@Weights/weight_hidden_1_2
�
2Weights/weight_hidden_1_2/Adam_1/Initializer/zerosFillBWeights/weight_hidden_1_2/Adam_1/Initializer/zeros/shape_as_tensor8Weights/weight_hidden_1_2/Adam_1/Initializer/zeros/Const*
T0*

index_type0*,
_class"
 loc:@Weights/weight_hidden_1_2*
_output_shapes

:hF
�
 Weights/weight_hidden_1_2/Adam_1
VariableV2*
shape
:hF*
dtype0*
_output_shapes

:hF*
shared_name *,
_class"
 loc:@Weights/weight_hidden_1_2*
	container 
�
'Weights/weight_hidden_1_2/Adam_1/AssignAssign Weights/weight_hidden_1_2/Adam_12Weights/weight_hidden_1_2/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@Weights/weight_hidden_1_2*
validate_shape(*
_output_shapes

:hF
�
%Weights/weight_hidden_1_2/Adam_1/readIdentity Weights/weight_hidden_1_2/Adam_1*
T0*,
_class"
 loc:@Weights/weight_hidden_1_2*
_output_shapes

:hF
�
.Weights/bias_hidden_1_2/Adam/Initializer/zerosConst*
valueBh*    **
_class 
loc:@Weights/bias_hidden_1_2*
dtype0*
_output_shapes
:h
�
Weights/bias_hidden_1_2/Adam
VariableV2*
shape:h*
dtype0*
_output_shapes
:h*
shared_name **
_class 
loc:@Weights/bias_hidden_1_2*
	container 
�
#Weights/bias_hidden_1_2/Adam/AssignAssignWeights/bias_hidden_1_2/Adam.Weights/bias_hidden_1_2/Adam/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Weights/bias_hidden_1_2*
validate_shape(*
_output_shapes
:h
�
!Weights/bias_hidden_1_2/Adam/readIdentityWeights/bias_hidden_1_2/Adam*
T0**
_class 
loc:@Weights/bias_hidden_1_2*
_output_shapes
:h
�
0Weights/bias_hidden_1_2/Adam_1/Initializer/zerosConst*
valueBh*    **
_class 
loc:@Weights/bias_hidden_1_2*
dtype0*
_output_shapes
:h
�
Weights/bias_hidden_1_2/Adam_1
VariableV2*
dtype0*
_output_shapes
:h*
shared_name **
_class 
loc:@Weights/bias_hidden_1_2*
	container *
shape:h
�
%Weights/bias_hidden_1_2/Adam_1/AssignAssignWeights/bias_hidden_1_2/Adam_10Weights/bias_hidden_1_2/Adam_1/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Weights/bias_hidden_1_2*
validate_shape(*
_output_shapes
:h
�
#Weights/bias_hidden_1_2/Adam_1/readIdentityWeights/bias_hidden_1_2/Adam_1*
T0**
_class 
loc:@Weights/bias_hidden_1_2*
_output_shapes
:h
�
+Weights/weight_out_2/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:h*
valueBh*    *'
_class
loc:@Weights/weight_out_2
�
Weights/weight_out_2/Adam
VariableV2*
dtype0*
_output_shapes

:h*
shared_name *'
_class
loc:@Weights/weight_out_2*
	container *
shape
:h
�
 Weights/weight_out_2/Adam/AssignAssignWeights/weight_out_2/Adam+Weights/weight_out_2/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:h*
use_locking(*
T0*'
_class
loc:@Weights/weight_out_2
�
Weights/weight_out_2/Adam/readIdentityWeights/weight_out_2/Adam*
T0*'
_class
loc:@Weights/weight_out_2*
_output_shapes

:h
�
-Weights/weight_out_2/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:h*
valueBh*    *'
_class
loc:@Weights/weight_out_2
�
Weights/weight_out_2/Adam_1
VariableV2*'
_class
loc:@Weights/weight_out_2*
	container *
shape
:h*
dtype0*
_output_shapes

:h*
shared_name 
�
"Weights/weight_out_2/Adam_1/AssignAssignWeights/weight_out_2/Adam_1-Weights/weight_out_2/Adam_1/Initializer/zeros*
T0*'
_class
loc:@Weights/weight_out_2*
validate_shape(*
_output_shapes

:h*
use_locking(
�
 Weights/weight_out_2/Adam_1/readIdentityWeights/weight_out_2/Adam_1*
T0*'
_class
loc:@Weights/weight_out_2*
_output_shapes

:h
�
)Weights/bias_out_2/Adam/Initializer/zerosConst*
valueB*    *%
_class
loc:@Weights/bias_out_2*
dtype0*
_output_shapes
:
�
Weights/bias_out_2/Adam
VariableV2*
shared_name *%
_class
loc:@Weights/bias_out_2*
	container *
shape:*
dtype0*
_output_shapes
:
�
Weights/bias_out_2/Adam/AssignAssignWeights/bias_out_2/Adam)Weights/bias_out_2/Adam/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@Weights/bias_out_2*
validate_shape(*
_output_shapes
:
�
Weights/bias_out_2/Adam/readIdentityWeights/bias_out_2/Adam*
T0*%
_class
loc:@Weights/bias_out_2*
_output_shapes
:
�
+Weights/bias_out_2/Adam_1/Initializer/zerosConst*
valueB*    *%
_class
loc:@Weights/bias_out_2*
dtype0*
_output_shapes
:
�
Weights/bias_out_2/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *%
_class
loc:@Weights/bias_out_2*
	container *
shape:
�
 Weights/bias_out_2/Adam_1/AssignAssignWeights/bias_out_2/Adam_1+Weights/bias_out_2/Adam_1/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@Weights/bias_out_2*
validate_shape(*
_output_shapes
:
�
Weights/bias_out_2/Adam_1/readIdentityWeights/bias_out_2/Adam_1*
T0*%
_class
loc:@Weights/bias_out_2*
_output_shapes
:
b
optimisation_op/learning_rateConst*
valueB
 *r[�8*
dtype0*
_output_shapes
: 
Z
optimisation_op/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Z
optimisation_op/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
\
optimisation_op/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
2optimisation_op/update_Weights/weight_in/ApplyAdam	ApplyAdamWeights/weight_inWeights/weight_in/AdamWeights/weight_in/Adam_1beta1_power/readbeta2_power/readoptimisation_op/learning_rateoptimisation_op/beta1optimisation_op/beta2optimisation_op/epsilongradients/AddN_5*
use_locking( *
T0*$
_class
loc:@Weights/weight_in*
use_nesterov( *
_output_shapes
:	F�

�
0optimisation_op/update_Weights/bias_in/ApplyAdam	ApplyAdamWeights/bias_inWeights/bias_in/AdamWeights/bias_in/Adam_1beta1_power/readbeta2_power/readoptimisation_op/learning_rateoptimisation_op/beta1optimisation_op/beta2optimisation_op/epsilon3gradients/Model/Add_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@Weights/bias_in*
use_nesterov( *
_output_shapes
:F
�
8optimisation_op/update_Weights/weight_hidden_1/ApplyAdam	ApplyAdamWeights/weight_hidden_1Weights/weight_hidden_1/AdamWeights/weight_hidden_1/Adam_1beta1_power/readbeta2_power/readoptimisation_op/learning_rateoptimisation_op/beta1optimisation_op/beta2optimisation_op/epsilongradients/AddN_2*
use_locking( *
T0**
_class 
loc:@Weights/weight_hidden_1*
use_nesterov( *
_output_shapes

:hF
�
6optimisation_op/update_Weights/bias_hidden_1/ApplyAdam	ApplyAdamWeights/bias_hidden_1Weights/bias_hidden_1/AdamWeights/bias_hidden_1/Adam_1beta1_power/readbeta2_power/readoptimisation_op/learning_rateoptimisation_op/beta1optimisation_op/beta2optimisation_op/epsilon5gradients/Model/Add_1_grad/tuple/control_dependency_1*
use_locking( *
T0*(
_class
loc:@Weights/bias_hidden_1*
use_nesterov( *
_output_shapes
:h
�
3optimisation_op/update_Weights/weight_out/ApplyAdam	ApplyAdamWeights/weight_outWeights/weight_out/AdamWeights/weight_out/Adam_1beta1_power/readbeta2_power/readoptimisation_op/learning_rateoptimisation_op/beta1optimisation_op/beta2optimisation_op/epsilon*gradients/Model/transpose_2_grad/transpose*
T0*%
_class
loc:@Weights/weight_out*
use_nesterov( *
_output_shapes

:h*
use_locking( 
�
1optimisation_op/update_Weights/bias_out/ApplyAdam	ApplyAdamWeights/bias_outWeights/bias_out/AdamWeights/bias_out/Adam_1beta1_power/readbeta2_power/readoptimisation_op/learning_rateoptimisation_op/beta1optimisation_op/beta2optimisation_op/epsilon5gradients/Model/Add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@Weights/bias_out*
use_nesterov( *
_output_shapes
:
�
4optimisation_op/update_Weights/weight_in_1/ApplyAdam	ApplyAdamWeights/weight_in_1Weights/weight_in_1/AdamWeights/weight_in_1/Adam_1beta1_power/readbeta2_power/readoptimisation_op/learning_rateoptimisation_op/beta1optimisation_op/beta2optimisation_op/epsilongradients/AddN_4*
use_nesterov( *
_output_shapes
:	F�
*
use_locking( *
T0*&
_class
loc:@Weights/weight_in_1
�
2optimisation_op/update_Weights/bias_in_1/ApplyAdam	ApplyAdamWeights/bias_in_1Weights/bias_in_1/AdamWeights/bias_in_1/Adam_1beta1_power/readbeta2_power/readoptimisation_op/learning_rateoptimisation_op/beta1optimisation_op/beta2optimisation_op/epsilon5gradients/Model/Add_4_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@Weights/bias_in_1*
use_nesterov( *
_output_shapes
:F
�
:optimisation_op/update_Weights/weight_hidden_1_1/ApplyAdam	ApplyAdamWeights/weight_hidden_1_1Weights/weight_hidden_1_1/Adam Weights/weight_hidden_1_1/Adam_1beta1_power/readbeta2_power/readoptimisation_op/learning_rateoptimisation_op/beta1optimisation_op/beta2optimisation_op/epsilongradients/AddN_1*
use_locking( *
T0*,
_class"
 loc:@Weights/weight_hidden_1_1*
use_nesterov( *
_output_shapes

:hF
�
8optimisation_op/update_Weights/bias_hidden_1_1/ApplyAdam	ApplyAdamWeights/bias_hidden_1_1Weights/bias_hidden_1_1/AdamWeights/bias_hidden_1_1/Adam_1beta1_power/readbeta2_power/readoptimisation_op/learning_rateoptimisation_op/beta1optimisation_op/beta2optimisation_op/epsilon5gradients/Model/Add_5_grad/tuple/control_dependency_1*
use_locking( *
T0**
_class 
loc:@Weights/bias_hidden_1_1*
use_nesterov( *
_output_shapes
:h
�
5optimisation_op/update_Weights/weight_out_1/ApplyAdam	ApplyAdamWeights/weight_out_1Weights/weight_out_1/AdamWeights/weight_out_1/Adam_1beta1_power/readbeta2_power/readoptimisation_op/learning_rateoptimisation_op/beta1optimisation_op/beta2optimisation_op/epsilon*gradients/Model/transpose_5_grad/transpose*
use_nesterov( *
_output_shapes

:h*
use_locking( *
T0*'
_class
loc:@Weights/weight_out_1
�
3optimisation_op/update_Weights/bias_out_1/ApplyAdam	ApplyAdamWeights/bias_out_1Weights/bias_out_1/AdamWeights/bias_out_1/Adam_1beta1_power/readbeta2_power/readoptimisation_op/learning_rateoptimisation_op/beta1optimisation_op/beta2optimisation_op/epsilon5gradients/Model/Add_6_grad/tuple/control_dependency_1*
use_locking( *
T0*%
_class
loc:@Weights/bias_out_1*
use_nesterov( *
_output_shapes
:
�
4optimisation_op/update_Weights/weight_in_2/ApplyAdam	ApplyAdamWeights/weight_in_2Weights/weight_in_2/AdamWeights/weight_in_2/Adam_1beta1_power/readbeta2_power/readoptimisation_op/learning_rateoptimisation_op/beta1optimisation_op/beta2optimisation_op/epsilongradients/AddN_3*
T0*&
_class
loc:@Weights/weight_in_2*
use_nesterov( *
_output_shapes
:	F�
*
use_locking( 
�
2optimisation_op/update_Weights/bias_in_2/ApplyAdam	ApplyAdamWeights/bias_in_2Weights/bias_in_2/AdamWeights/bias_in_2/Adam_1beta1_power/readbeta2_power/readoptimisation_op/learning_rateoptimisation_op/beta1optimisation_op/beta2optimisation_op/epsilon5gradients/Model/Add_8_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:F*
use_locking( *
T0*$
_class
loc:@Weights/bias_in_2
�
:optimisation_op/update_Weights/weight_hidden_1_2/ApplyAdam	ApplyAdamWeights/weight_hidden_1_2Weights/weight_hidden_1_2/Adam Weights/weight_hidden_1_2/Adam_1beta1_power/readbeta2_power/readoptimisation_op/learning_rateoptimisation_op/beta1optimisation_op/beta2optimisation_op/epsilongradients/AddN*
use_locking( *
T0*,
_class"
 loc:@Weights/weight_hidden_1_2*
use_nesterov( *
_output_shapes

:hF
�
8optimisation_op/update_Weights/bias_hidden_1_2/ApplyAdam	ApplyAdamWeights/bias_hidden_1_2Weights/bias_hidden_1_2/AdamWeights/bias_hidden_1_2/Adam_1beta1_power/readbeta2_power/readoptimisation_op/learning_rateoptimisation_op/beta1optimisation_op/beta2optimisation_op/epsilon5gradients/Model/Add_9_grad/tuple/control_dependency_1*
T0**
_class 
loc:@Weights/bias_hidden_1_2*
use_nesterov( *
_output_shapes
:h*
use_locking( 
�
5optimisation_op/update_Weights/weight_out_2/ApplyAdam	ApplyAdamWeights/weight_out_2Weights/weight_out_2/AdamWeights/weight_out_2/Adam_1beta1_power/readbeta2_power/readoptimisation_op/learning_rateoptimisation_op/beta1optimisation_op/beta2optimisation_op/epsilon*gradients/Model/transpose_8_grad/transpose*
T0*'
_class
loc:@Weights/weight_out_2*
use_nesterov( *
_output_shapes

:h*
use_locking( 
�
3optimisation_op/update_Weights/bias_out_2/ApplyAdam	ApplyAdamWeights/bias_out_2Weights/bias_out_2/AdamWeights/bias_out_2/Adam_1beta1_power/readbeta2_power/readoptimisation_op/learning_rateoptimisation_op/beta1optimisation_op/beta2optimisation_op/epsilon6gradients/Model/Add_10_grad/tuple/control_dependency_1*
use_locking( *
T0*%
_class
loc:@Weights/bias_out_2*
use_nesterov( *
_output_shapes
:
�
optimisation_op/mulMulbeta1_power/readoptimisation_op/beta17^optimisation_op/update_Weights/bias_hidden_1/ApplyAdam9^optimisation_op/update_Weights/bias_hidden_1_1/ApplyAdam9^optimisation_op/update_Weights/bias_hidden_1_2/ApplyAdam1^optimisation_op/update_Weights/bias_in/ApplyAdam3^optimisation_op/update_Weights/bias_in_1/ApplyAdam3^optimisation_op/update_Weights/bias_in_2/ApplyAdam2^optimisation_op/update_Weights/bias_out/ApplyAdam4^optimisation_op/update_Weights/bias_out_1/ApplyAdam4^optimisation_op/update_Weights/bias_out_2/ApplyAdam9^optimisation_op/update_Weights/weight_hidden_1/ApplyAdam;^optimisation_op/update_Weights/weight_hidden_1_1/ApplyAdam;^optimisation_op/update_Weights/weight_hidden_1_2/ApplyAdam3^optimisation_op/update_Weights/weight_in/ApplyAdam5^optimisation_op/update_Weights/weight_in_1/ApplyAdam5^optimisation_op/update_Weights/weight_in_2/ApplyAdam4^optimisation_op/update_Weights/weight_out/ApplyAdam6^optimisation_op/update_Weights/weight_out_1/ApplyAdam6^optimisation_op/update_Weights/weight_out_2/ApplyAdam*
T0*(
_class
loc:@Weights/bias_hidden_1*
_output_shapes
: 
�
optimisation_op/AssignAssignbeta1_poweroptimisation_op/mul*
use_locking( *
T0*(
_class
loc:@Weights/bias_hidden_1*
validate_shape(*
_output_shapes
: 
�
optimisation_op/mul_1Mulbeta2_power/readoptimisation_op/beta27^optimisation_op/update_Weights/bias_hidden_1/ApplyAdam9^optimisation_op/update_Weights/bias_hidden_1_1/ApplyAdam9^optimisation_op/update_Weights/bias_hidden_1_2/ApplyAdam1^optimisation_op/update_Weights/bias_in/ApplyAdam3^optimisation_op/update_Weights/bias_in_1/ApplyAdam3^optimisation_op/update_Weights/bias_in_2/ApplyAdam2^optimisation_op/update_Weights/bias_out/ApplyAdam4^optimisation_op/update_Weights/bias_out_1/ApplyAdam4^optimisation_op/update_Weights/bias_out_2/ApplyAdam9^optimisation_op/update_Weights/weight_hidden_1/ApplyAdam;^optimisation_op/update_Weights/weight_hidden_1_1/ApplyAdam;^optimisation_op/update_Weights/weight_hidden_1_2/ApplyAdam3^optimisation_op/update_Weights/weight_in/ApplyAdam5^optimisation_op/update_Weights/weight_in_1/ApplyAdam5^optimisation_op/update_Weights/weight_in_2/ApplyAdam4^optimisation_op/update_Weights/weight_out/ApplyAdam6^optimisation_op/update_Weights/weight_out_1/ApplyAdam6^optimisation_op/update_Weights/weight_out_2/ApplyAdam*
T0*(
_class
loc:@Weights/bias_hidden_1*
_output_shapes
: 
�
optimisation_op/Assign_1Assignbeta2_poweroptimisation_op/mul_1*
use_locking( *
T0*(
_class
loc:@Weights/bias_hidden_1*
validate_shape(*
_output_shapes
: 
�
optimisation_opNoOp^optimisation_op/Assign^optimisation_op/Assign_17^optimisation_op/update_Weights/bias_hidden_1/ApplyAdam9^optimisation_op/update_Weights/bias_hidden_1_1/ApplyAdam9^optimisation_op/update_Weights/bias_hidden_1_2/ApplyAdam1^optimisation_op/update_Weights/bias_in/ApplyAdam3^optimisation_op/update_Weights/bias_in_1/ApplyAdam3^optimisation_op/update_Weights/bias_in_2/ApplyAdam2^optimisation_op/update_Weights/bias_out/ApplyAdam4^optimisation_op/update_Weights/bias_out_1/ApplyAdam4^optimisation_op/update_Weights/bias_out_2/ApplyAdam9^optimisation_op/update_Weights/weight_hidden_1/ApplyAdam;^optimisation_op/update_Weights/weight_hidden_1_1/ApplyAdam;^optimisation_op/update_Weights/weight_hidden_1_2/ApplyAdam3^optimisation_op/update_Weights/weight_in/ApplyAdam5^optimisation_op/update_Weights/weight_in_1/ApplyAdam5^optimisation_op/update_Weights/weight_in_2/ApplyAdam4^optimisation_op/update_Weights/weight_out/ApplyAdam6^optimisation_op/update_Weights/weight_out_1/ApplyAdam6^optimisation_op/update_Weights/weight_out_2/ApplyAdam
�
initNoOp"^Weights/bias_hidden_1/Adam/Assign$^Weights/bias_hidden_1/Adam_1/Assign^Weights/bias_hidden_1/Assign$^Weights/bias_hidden_1_1/Adam/Assign&^Weights/bias_hidden_1_1/Adam_1/Assign^Weights/bias_hidden_1_1/Assign$^Weights/bias_hidden_1_2/Adam/Assign&^Weights/bias_hidden_1_2/Adam_1/Assign^Weights/bias_hidden_1_2/Assign^Weights/bias_in/Adam/Assign^Weights/bias_in/Adam_1/Assign^Weights/bias_in/Assign^Weights/bias_in_1/Adam/Assign ^Weights/bias_in_1/Adam_1/Assign^Weights/bias_in_1/Assign^Weights/bias_in_2/Adam/Assign ^Weights/bias_in_2/Adam_1/Assign^Weights/bias_in_2/Assign^Weights/bias_out/Adam/Assign^Weights/bias_out/Adam_1/Assign^Weights/bias_out/Assign^Weights/bias_out_1/Adam/Assign!^Weights/bias_out_1/Adam_1/Assign^Weights/bias_out_1/Assign^Weights/bias_out_2/Adam/Assign!^Weights/bias_out_2/Adam_1/Assign^Weights/bias_out_2/Assign$^Weights/weight_hidden_1/Adam/Assign&^Weights/weight_hidden_1/Adam_1/Assign^Weights/weight_hidden_1/Assign&^Weights/weight_hidden_1_1/Adam/Assign(^Weights/weight_hidden_1_1/Adam_1/Assign!^Weights/weight_hidden_1_1/Assign&^Weights/weight_hidden_1_2/Adam/Assign(^Weights/weight_hidden_1_2/Adam_1/Assign!^Weights/weight_hidden_1_2/Assign^Weights/weight_in/Adam/Assign ^Weights/weight_in/Adam_1/Assign^Weights/weight_in/Assign ^Weights/weight_in_1/Adam/Assign"^Weights/weight_in_1/Adam_1/Assign^Weights/weight_in_1/Assign ^Weights/weight_in_2/Adam/Assign"^Weights/weight_in_2/Adam_1/Assign^Weights/weight_in_2/Assign^Weights/weight_out/Adam/Assign!^Weights/weight_out/Adam_1/Assign^Weights/weight_out/Assign!^Weights/weight_out_1/Adam/Assign#^Weights/weight_out_1/Adam_1/Assign^Weights/weight_out_1/Assign!^Weights/weight_out_2/Adam/Assign#^Weights/weight_out_2/Adam_1/Assign^Weights/weight_out_2/Assign^beta1_power/Assign^beta2_power/Assign
�
TensorSliceDatasetTensorSliceDatasetData/DescriptorsData/Atomic-numbersData/Properties**
output_shapes
:	�
::*
Toutput_types
2* 
_class
loc:@Data/Iterator*
_output_shapes
: 
�
ShuffleDatasetShuffleDatasetTensorSliceDatasetData/buffer	Data/seed
Data/seed2**
output_shapes
:	�
::* 
_class
loc:@Data/Iterator*
reshuffle_each_iteration(*
_output_shapes
: *
output_types
2
�
BatchDatasetBatchDatasetShuffleDatasetData/batch_size*Q
output_shapes@
>:����������
:���������:���������* 
_class
loc:@Data/Iterator*
_output_shapes
: *
output_types
2
[
dataset_initMakeIteratorBatchDatasetData/Iterator* 
_class
loc:@Data/Iterator
v
Inputs_pred/ClassesPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
z
Inputs_pred/xyzPlaceholder*
dtype0*+
_output_shapes
:���������* 
shape:���������
X
Inputs_pred/batch_sizeConst*
value	B	 R*
dtype0	*
_output_shapes
: 
�
Inputs_pred/IteratorIterator*
output_types
2*
shared_name *=
output_shapes,
*:���������:���������*
	container *
_output_shapes
: 
j
"Inputs_pred/IteratorToStringHandleIteratorToStringHandleInputs_pred/Iterator*
_output_shapes
: 
�
Inputs_pred/IteratorGetNextIteratorGetNextInputs_pred/Iterator*>
_output_shapes,
*:���������:���������*
output_types
2*=
output_shapes,
*:���������:���������
�
Inputs_pred/TensorSliceDatasetTensorSliceDatasetInputs_pred/xyzInputs_pred/Classes*#
output_shapes
::*
Toutput_types
2*'
_class
loc:@Inputs_pred/Iterator*
_output_shapes
: 
�
Inputs_pred/BatchDatasetBatchDatasetInputs_pred/TensorSliceDatasetInputs_pred/batch_size*=
output_shapes,
*:���������:���������*'
_class
loc:@Inputs_pred/Iterator*
_output_shapes
: *
output_types
2
�
Inputs_pred/dataset_init_predMakeIteratorInputs_pred/BatchDatasetInputs_pred/Iterator*'
_class
loc:@Inputs_pred/Iterator
f
!Descriptor_pred/acsf_params/ConstConst*
valueB
 *33�@*
dtype0*
_output_shapes
: 
h
#Descriptor_pred/acsf_params/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *33�@
�
#Descriptor_pred/acsf_params/Const_2Const*Q
valueHBF"<��L?I��?,��?��?��?�A�?۶@��@��+@��:@�J@�$Y@�:h@uPw@33�@*
dtype0*
_output_shapes
:
�
#Descriptor_pred/acsf_params/Const_3Const*
dtype0*
_output_shapes
:*Q
valueHBF"<��L?I��?,��?��?��?�A�?۶@��@��+@��:@�J@�$Y@�:h@uPw@33�@
�
#Descriptor_pred/acsf_params/Const_4Const*
dtype0*
_output_shapes
:*Q
valueHBF"<    ��e>���>�V,?��e?���?�V�?��?���?A@��@,�@�V,@K�:@�I@
h
#Descriptor_pred/acsf_params/Const_5Const*
valueB
 *� \C*
dtype0*
_output_shapes
: 
h
#Descriptor_pred/acsf_params/Const_6Const*
valueB
 *$�GB*
dtype0*
_output_shapes
: 
v
4Descriptor_pred/Radial_part/Distances/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
0Descriptor_pred/Radial_part/Distances/ExpandDims
ExpandDimsInputs_pred/IteratorGetNext4Descriptor_pred/Radial_part/Distances/ExpandDims/dim*
T0*/
_output_shapes
:���������*

Tdim0
x
6Descriptor_pred/Radial_part/Distances/ExpandDims_1/dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
2Descriptor_pred/Radial_part/Distances/ExpandDims_1
ExpandDimsInputs_pred/IteratorGetNext6Descriptor_pred/Radial_part/Distances/ExpandDims_1/dim*
T0*/
_output_shapes
:���������*

Tdim0
�
)Descriptor_pred/Radial_part/Distances/subSub0Descriptor_pred/Radial_part/Distances/ExpandDims2Descriptor_pred/Radial_part/Distances/ExpandDims_1*
T0*/
_output_shapes
:���������
p
+Descriptor_pred/Radial_part/Distances/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *���$
�
)Descriptor_pred/Radial_part/Distances/addAdd)Descriptor_pred/Radial_part/Distances/sub+Descriptor_pred/Radial_part/Distances/add/y*
T0*/
_output_shapes
:���������
�
.Descriptor_pred/Radial_part/Distances/norm/mulMul)Descriptor_pred/Radial_part/Distances/add)Descriptor_pred/Radial_part/Distances/add*
T0*/
_output_shapes
:���������
�
@Descriptor_pred/Radial_part/Distances/norm/Sum/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
�
.Descriptor_pred/Radial_part/Distances/norm/SumSum.Descriptor_pred/Radial_part/Distances/norm/mul@Descriptor_pred/Radial_part/Distances/norm/Sum/reduction_indices*

Tidx0*
	keep_dims(*
T0*/
_output_shapes
:���������
�
/Descriptor_pred/Radial_part/Distances/norm/SqrtSqrt.Descriptor_pred/Radial_part/Distances/norm/Sum*
T0*/
_output_shapes
:���������
�
2Descriptor_pred/Radial_part/Distances/norm/SqueezeSqueeze/Descriptor_pred/Radial_part/Distances/norm/Sqrt*
T0*+
_output_shapes
:���������*
squeeze_dims

�
!Descriptor_pred/Radial_part/ShapeShape2Descriptor_pred/Radial_part/Distances/norm/Squeeze*
_output_shapes
:*
T0*
out_type0
l
'Descriptor_pred/Radial_part/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
!Descriptor_pred/Radial_part/zerosFill!Descriptor_pred/Radial_part/Shape'Descriptor_pred/Radial_part/zeros/Const*+
_output_shapes
:���������*
T0*

index_type0
�
#Descriptor_pred/Radial_part/Shape_1ShapeInputs_pred/IteratorGetNext:1*
T0*
out_type0*
_output_shapes
:
k
&Descriptor_pred/Radial_part/ones/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
 Descriptor_pred/Radial_part/onesFill#Descriptor_pred/Radial_part/Shape_1&Descriptor_pred/Radial_part/ones/Const*'
_output_shapes
:���������*
T0*

index_type0
�
)Descriptor_pred/Radial_part/MatrixSetDiagMatrixSetDiag!Descriptor_pred/Radial_part/zeros Descriptor_pred/Radial_part/ones*
T0*+
_output_shapes
:���������
�
 Descriptor_pred/Radial_part/CastCast)Descriptor_pred/Radial_part/MatrixSetDiag*

SrcT0*

DstT0
*+
_output_shapes
:���������
}
;Descriptor_pred/Radial_part/Exponential_term/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B : 
�
7Descriptor_pred/Radial_part/Exponential_term/ExpandDims
ExpandDims#Descriptor_pred/acsf_params/Const_2;Descriptor_pred/Radial_part/Exponential_term/ExpandDims/dim*
_output_shapes

:*

Tdim0*
T0

=Descriptor_pred/Radial_part/Exponential_term/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
9Descriptor_pred/Radial_part/Exponential_term/ExpandDims_1
ExpandDims7Descriptor_pred/Radial_part/Exponential_term/ExpandDims=Descriptor_pred/Radial_part/Exponential_term/ExpandDims_1/dim*
T0*"
_output_shapes
:*

Tdim0

=Descriptor_pred/Radial_part/Exponential_term/ExpandDims_2/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
9Descriptor_pred/Radial_part/Exponential_term/ExpandDims_2
ExpandDims9Descriptor_pred/Radial_part/Exponential_term/ExpandDims_1=Descriptor_pred/Radial_part/Exponential_term/ExpandDims_2/dim*

Tdim0*
T0*&
_output_shapes
:
�
=Descriptor_pred/Radial_part/Exponential_term/ExpandDims_3/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
9Descriptor_pred/Radial_part/Exponential_term/ExpandDims_3
ExpandDims2Descriptor_pred/Radial_part/Distances/norm/Squeeze=Descriptor_pred/Radial_part/Exponential_term/ExpandDims_3/dim*

Tdim0*
T0*/
_output_shapes
:���������
}
0Descriptor_pred/Radial_part/Exponential_term/NegNeg#Descriptor_pred/acsf_params/Const_6*
T0*
_output_shapes
: 
�
0Descriptor_pred/Radial_part/Exponential_term/SubSub9Descriptor_pred/Radial_part/Exponential_term/ExpandDims_39Descriptor_pred/Radial_part/Exponential_term/ExpandDims_2*
T0*/
_output_shapes
:���������
�
3Descriptor_pred/Radial_part/Exponential_term/SquareSquare0Descriptor_pred/Radial_part/Exponential_term/Sub*
T0*/
_output_shapes
:���������
�
0Descriptor_pred/Radial_part/Exponential_term/mulMul0Descriptor_pred/Radial_part/Exponential_term/Neg3Descriptor_pred/Radial_part/Exponential_term/Square*
T0*/
_output_shapes
:���������
�
0Descriptor_pred/Radial_part/Exponential_term/ExpExp0Descriptor_pred/Radial_part/Exponential_term/mul*
T0*/
_output_shapes
:���������
�
(Descriptor_pred/Radial_part/fc_term/LessLess2Descriptor_pred/Radial_part/Distances/norm/Squeeze!Descriptor_pred/acsf_params/Const*+
_output_shapes
:���������*
T0
n
)Descriptor_pred/Radial_part/fc_term/mul/xConst*
valueB
 *�I@*
dtype0*
_output_shapes
: 
�
'Descriptor_pred/Radial_part/fc_term/mulMul)Descriptor_pred/Radial_part/fc_term/mul/x2Descriptor_pred/Radial_part/Distances/norm/Squeeze*
T0*+
_output_shapes
:���������
�
+Descriptor_pred/Radial_part/fc_term/truedivRealDiv'Descriptor_pred/Radial_part/fc_term/mul!Descriptor_pred/acsf_params/Const*
T0*+
_output_shapes
:���������
�
'Descriptor_pred/Radial_part/fc_term/CosCos+Descriptor_pred/Radial_part/fc_term/truediv*
T0*+
_output_shapes
:���������
n
)Descriptor_pred/Radial_part/fc_term/add/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
'Descriptor_pred/Radial_part/fc_term/addAdd'Descriptor_pred/Radial_part/fc_term/Cos)Descriptor_pred/Radial_part/fc_term/add/y*
T0*+
_output_shapes
:���������
p
+Descriptor_pred/Radial_part/fc_term/mul_1/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
�
)Descriptor_pred/Radial_part/fc_term/mul_1Mul+Descriptor_pred/Radial_part/fc_term/mul_1/x'Descriptor_pred/Radial_part/fc_term/add*
T0*+
_output_shapes
:���������
�
)Descriptor_pred/Radial_part/fc_term/ShapeShape2Descriptor_pred/Radial_part/Distances/norm/Squeeze*
_output_shapes
:*
T0*
out_type0
t
/Descriptor_pred/Radial_part/fc_term/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
)Descriptor_pred/Radial_part/fc_term/zerosFill)Descriptor_pred/Radial_part/fc_term/Shape/Descriptor_pred/Radial_part/fc_term/zeros/Const*
T0*

index_type0*+
_output_shapes
:���������
�
*Descriptor_pred/Radial_part/fc_term/SelectSelect(Descriptor_pred/Radial_part/fc_term/Less)Descriptor_pred/Radial_part/fc_term/mul_1)Descriptor_pred/Radial_part/fc_term/zeros*+
_output_shapes
:���������*
T0
�
,Descriptor_pred/Radial_part/fc_term/Select_1Select Descriptor_pred/Radial_part/Cast)Descriptor_pred/Radial_part/fc_term/zeros*Descriptor_pred/Radial_part/fc_term/Select*+
_output_shapes
:���������*
T0
k
)Descriptor_pred/Radial_part/fc_term/ConstConst*
value	B : *
dtype0*
_output_shapes
: 
�
)Descriptor_pred/Radial_part/fc_term/EqualEqualInputs_pred/IteratorGetNext:1)Descriptor_pred/Radial_part/fc_term/Const*
T0*'
_output_shapes
:���������
�
.Descriptor_pred/Radial_part/fc_term/LogicalNot
LogicalNot)Descriptor_pred/Radial_part/fc_term/Equal*'
_output_shapes
:���������
t
2Descriptor_pred/Radial_part/fc_term/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
.Descriptor_pred/Radial_part/fc_term/ExpandDims
ExpandDims.Descriptor_pred/Radial_part/fc_term/LogicalNot2Descriptor_pred/Radial_part/fc_term/ExpandDims/dim*+
_output_shapes
:���������*

Tdim0*
T0


4Descriptor_pred/Radial_part/fc_term/ExpandDims_1/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
0Descriptor_pred/Radial_part/fc_term/ExpandDims_1
ExpandDims.Descriptor_pred/Radial_part/fc_term/LogicalNot4Descriptor_pred/Radial_part/fc_term/ExpandDims_1/dim*+
_output_shapes
:���������*

Tdim0*
T0

�
.Descriptor_pred/Radial_part/fc_term/LogicalAnd
LogicalAnd.Descriptor_pred/Radial_part/fc_term/ExpandDims0Descriptor_pred/Radial_part/fc_term/ExpandDims_1*+
_output_shapes
:���������
�
,Descriptor_pred/Radial_part/fc_term/Select_2Select.Descriptor_pred/Radial_part/fc_term/LogicalAnd,Descriptor_pred/Radial_part/fc_term/Select_1)Descriptor_pred/Radial_part/fc_term/zeros*
T0*+
_output_shapes
:���������

4Descriptor_pred/Radial_part/fc_term/ExpandDims_2/dimConst*
dtype0*
_output_shapes
: *
valueB :
���������
�
0Descriptor_pred/Radial_part/fc_term/ExpandDims_2
ExpandDims,Descriptor_pred/Radial_part/fc_term/Select_24Descriptor_pred/Radial_part/fc_term/ExpandDims_2/dim*/
_output_shapes
:���������*

Tdim0*
T0
�
(Descriptor_pred/Radial_part/Rad_term/MulMul0Descriptor_pred/Radial_part/fc_term/ExpandDims_20Descriptor_pred/Radial_part/Exponential_term/Exp*
T0*/
_output_shapes
:���������
h
&Descriptor_pred/Sum_rad/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
"Descriptor_pred/Sum_rad/ExpandDims
ExpandDimsInputs_pred/IteratorGetNext:1&Descriptor_pred/Sum_rad/ExpandDims/dim*
T0*+
_output_shapes
:���������*

Tdim0
s
(Descriptor_pred/Sum_rad/ExpandDims_1/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
$Descriptor_pred/Sum_rad/ExpandDims_1
ExpandDims"Descriptor_pred/Sum_rad/ExpandDims(Descriptor_pred/Sum_rad/ExpandDims_1/dim*

Tdim0*
T0*/
_output_shapes
:���������

&Descriptor_pred/Sum_rad/Tile/multiplesConst*%
valueB"            *
dtype0*
_output_shapes
:
�
Descriptor_pred/Sum_rad/TileTile$Descriptor_pred/Sum_rad/ExpandDims_1&Descriptor_pred/Sum_rad/Tile/multiples*

Tmultiples0*
T0*/
_output_shapes
:���������
�
Descriptor_pred/Sum_rad/ShapeShape(Descriptor_pred/Radial_part/Rad_term/Mul*
_output_shapes
:*
T0*
out_type0
h
#Descriptor_pred/Sum_rad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Descriptor_pred/Sum_rad/zerosFillDescriptor_pred/Sum_rad/Shape#Descriptor_pred/Sum_rad/zeros/Const*
T0*

index_type0*/
_output_shapes
:���������
_
Descriptor_pred/Sum_rad/ConstConst*
dtype0*
_output_shapes
: *
value	B :
�
Descriptor_pred/Sum_rad/EqualEqualDescriptor_pred/Sum_rad/TileDescriptor_pred/Sum_rad/Const*
T0*/
_output_shapes
:���������
�
Descriptor_pred/Sum_rad/SelectSelectDescriptor_pred/Sum_rad/Equal(Descriptor_pred/Radial_part/Rad_term/MulDescriptor_pred/Sum_rad/zeros*
T0*/
_output_shapes
:���������
w
-Descriptor_pred/Sum_rad/Sum/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
�
Descriptor_pred/Sum_rad/SumSumDescriptor_pred/Sum_rad/Select-Descriptor_pred/Sum_rad/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0*+
_output_shapes
:���������
a
Descriptor_pred/Sum_rad/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
�
Descriptor_pred/Sum_rad/Equal_1EqualDescriptor_pred/Sum_rad/TileDescriptor_pred/Sum_rad/Const_1*
T0*/
_output_shapes
:���������
�
 Descriptor_pred/Sum_rad/Select_1SelectDescriptor_pred/Sum_rad/Equal_1(Descriptor_pred/Radial_part/Rad_term/MulDescriptor_pred/Sum_rad/zeros*
T0*/
_output_shapes
:���������
y
/Descriptor_pred/Sum_rad/Sum_1/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
�
Descriptor_pred/Sum_rad/Sum_1Sum Descriptor_pred/Sum_rad/Select_1/Descriptor_pred/Sum_rad/Sum_1/reduction_indices*
T0*+
_output_shapes
:���������*

Tidx0*
	keep_dims( 
a
Descriptor_pred/Sum_rad/Const_2Const*
value	B :*
dtype0*
_output_shapes
: 
�
Descriptor_pred/Sum_rad/Equal_2EqualDescriptor_pred/Sum_rad/TileDescriptor_pred/Sum_rad/Const_2*
T0*/
_output_shapes
:���������
�
 Descriptor_pred/Sum_rad/Select_2SelectDescriptor_pred/Sum_rad/Equal_2(Descriptor_pred/Radial_part/Rad_term/MulDescriptor_pred/Sum_rad/zeros*/
_output_shapes
:���������*
T0
y
/Descriptor_pred/Sum_rad/Sum_2/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
Descriptor_pred/Sum_rad/Sum_2Sum Descriptor_pred/Sum_rad/Select_2/Descriptor_pred/Sum_rad/Sum_2/reduction_indices*
T0*+
_output_shapes
:���������*

Tidx0*
	keep_dims( 
o
$Descriptor_pred/Sum_rad/sum_rad/axisConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Descriptor_pred/Sum_rad/sum_radConcatV2Descriptor_pred/Sum_rad/SumDescriptor_pred/Sum_rad/Sum_1Descriptor_pred/Sum_rad/Sum_2$Descriptor_pred/Sum_rad/sum_rad/axis*
T0*
N*+
_output_shapes
:���������-*

Tidx0
a
Descriptor_pred/Sum_rad/Const_3Const*
dtype0*
_output_shapes
: *
value	B : 
�
Descriptor_pred/Sum_rad/Equal_3EqualInputs_pred/IteratorGetNext:1Descriptor_pred/Sum_rad/Const_3*'
_output_shapes
:���������*
T0
z
"Descriptor_pred/Sum_rad/LogicalNot
LogicalNotDescriptor_pred/Sum_rad/Equal_3*'
_output_shapes
:���������
s
(Descriptor_pred/Sum_rad/ExpandDims_2/dimConst*
dtype0*
_output_shapes
: *
valueB :
���������
�
$Descriptor_pred/Sum_rad/ExpandDims_2
ExpandDims"Descriptor_pred/Sum_rad/LogicalNot(Descriptor_pred/Sum_rad/ExpandDims_2/dim*
T0
*+
_output_shapes
:���������*

Tdim0
}
(Descriptor_pred/Sum_rad/Tile_1/multiplesConst*
dtype0*
_output_shapes
:*!
valueB"      -   
�
Descriptor_pred/Sum_rad/Tile_1Tile$Descriptor_pred/Sum_rad/ExpandDims_2(Descriptor_pred/Sum_rad/Tile_1/multiples*

Tmultiples0*
T0
*+
_output_shapes
:���������-
~
Descriptor_pred/Sum_rad/Shape_1ShapeDescriptor_pred/Sum_rad/sum_rad*
_output_shapes
:*
T0*
out_type0
j
%Descriptor_pred/Sum_rad/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Descriptor_pred/Sum_rad/zeros_1FillDescriptor_pred/Sum_rad/Shape_1%Descriptor_pred/Sum_rad/zeros_1/Const*
T0*

index_type0*+
_output_shapes
:���������-
�
 Descriptor_pred/Sum_rad/Select_3SelectDescriptor_pred/Sum_rad/Tile_1Descriptor_pred/Sum_rad/sum_radDescriptor_pred/Sum_rad/zeros_1*
T0*+
_output_shapes
:���������-
{
9Descriptor_pred/Angular_part/Sum_distances/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B :
�
5Descriptor_pred/Angular_part/Sum_distances/ExpandDims
ExpandDimsInputs_pred/IteratorGetNext9Descriptor_pred/Angular_part/Sum_distances/ExpandDims/dim*

Tdim0*
T0*/
_output_shapes
:���������
}
;Descriptor_pred/Angular_part/Sum_distances/ExpandDims_1/dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
7Descriptor_pred/Angular_part/Sum_distances/ExpandDims_1
ExpandDimsInputs_pred/IteratorGetNext;Descriptor_pred/Angular_part/Sum_distances/ExpandDims_1/dim*
T0*/
_output_shapes
:���������*

Tdim0
�
.Descriptor_pred/Angular_part/Sum_distances/subSub5Descriptor_pred/Angular_part/Sum_distances/ExpandDims7Descriptor_pred/Angular_part/Sum_distances/ExpandDims_1*
T0*/
_output_shapes
:���������
u
0Descriptor_pred/Angular_part/Sum_distances/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *���$
�
.Descriptor_pred/Angular_part/Sum_distances/addAdd.Descriptor_pred/Angular_part/Sum_distances/sub0Descriptor_pred/Angular_part/Sum_distances/add/y*
T0*/
_output_shapes
:���������
�
3Descriptor_pred/Angular_part/Sum_distances/norm/mulMul.Descriptor_pred/Angular_part/Sum_distances/add.Descriptor_pred/Angular_part/Sum_distances/add*
T0*/
_output_shapes
:���������
�
EDescriptor_pred/Angular_part/Sum_distances/norm/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
3Descriptor_pred/Angular_part/Sum_distances/norm/SumSum3Descriptor_pred/Angular_part/Sum_distances/norm/mulEDescriptor_pred/Angular_part/Sum_distances/norm/Sum/reduction_indices*
T0*/
_output_shapes
:���������*

Tidx0*
	keep_dims(
�
4Descriptor_pred/Angular_part/Sum_distances/norm/SqrtSqrt3Descriptor_pred/Angular_part/Sum_distances/norm/Sum*
T0*/
_output_shapes
:���������
�
7Descriptor_pred/Angular_part/Sum_distances/norm/SqueezeSqueeze4Descriptor_pred/Angular_part/Sum_distances/norm/Sqrt*
squeeze_dims
*
T0*+
_output_shapes
:���������
}
;Descriptor_pred/Angular_part/Sum_distances/ExpandDims_2/dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
7Descriptor_pred/Angular_part/Sum_distances/ExpandDims_2
ExpandDims7Descriptor_pred/Angular_part/Sum_distances/norm/Squeeze;Descriptor_pred/Angular_part/Sum_distances/ExpandDims_2/dim*

Tdim0*
T0*/
_output_shapes
:���������
}
;Descriptor_pred/Angular_part/Sum_distances/ExpandDims_3/dimConst*
dtype0*
_output_shapes
: *
value	B :
�
7Descriptor_pred/Angular_part/Sum_distances/ExpandDims_3
ExpandDims7Descriptor_pred/Angular_part/Sum_distances/norm/Squeeze;Descriptor_pred/Angular_part/Sum_distances/ExpandDims_3/dim*
T0*/
_output_shapes
:���������*

Tdim0
�
0Descriptor_pred/Angular_part/Sum_distances/add_1Add7Descriptor_pred/Angular_part/Sum_distances/ExpandDims_27Descriptor_pred/Angular_part/Sum_distances/ExpandDims_3*
T0*/
_output_shapes
:���������
�
"Descriptor_pred/Angular_part/ConstConst*
dtype0
*"
_output_shapes
:*�
value�B�
Z�                                                                                                                                                                                                                  
m
+Descriptor_pred/Angular_part/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
'Descriptor_pred/Angular_part/ExpandDims
ExpandDims"Descriptor_pred/Angular_part/Const+Descriptor_pred/Angular_part/ExpandDims/dim*

Tdim0*
T0
*&
_output_shapes
:
�
"Descriptor_pred/Angular_part/ShapeShape0Descriptor_pred/Angular_part/Sum_distances/add_1*
_output_shapes
:*
T0*
out_type0
z
0Descriptor_pred/Angular_part/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
|
2Descriptor_pred/Angular_part/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
|
2Descriptor_pred/Angular_part/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
*Descriptor_pred/Angular_part/strided_sliceStridedSlice"Descriptor_pred/Angular_part/Shape0Descriptor_pred/Angular_part/strided_slice/stack2Descriptor_pred/Angular_part/strided_slice/stack_12Descriptor_pred/Angular_part/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
o
-Descriptor_pred/Angular_part/Tile/multiples/1Const*
value	B :*
dtype0*
_output_shapes
: 
o
-Descriptor_pred/Angular_part/Tile/multiples/2Const*
value	B :*
dtype0*
_output_shapes
: 
o
-Descriptor_pred/Angular_part/Tile/multiples/3Const*
value	B :*
dtype0*
_output_shapes
: 
�
+Descriptor_pred/Angular_part/Tile/multiplesPack*Descriptor_pred/Angular_part/strided_slice-Descriptor_pred/Angular_part/Tile/multiples/1-Descriptor_pred/Angular_part/Tile/multiples/2-Descriptor_pred/Angular_part/Tile/multiples/3*
N*
_output_shapes
:*
T0*

axis 
�
!Descriptor_pred/Angular_part/TileTile'Descriptor_pred/Angular_part/ExpandDims+Descriptor_pred/Angular_part/Tile/multiples*/
_output_shapes
:���������*

Tmultiples0*
T0

�
$Descriptor_pred/Angular_part/Shape_1Shape0Descriptor_pred/Angular_part/Sum_distances/add_1*
T0*
out_type0*
_output_shapes
:
m
(Descriptor_pred/Angular_part/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
"Descriptor_pred/Angular_part/zerosFill$Descriptor_pred/Angular_part/Shape_1(Descriptor_pred/Angular_part/zeros/Const*
T0*

index_type0*/
_output_shapes
:���������
�
)Descriptor_pred/Angular_part/Fc_term/LessLess7Descriptor_pred/Angular_part/Sum_distances/norm/Squeeze#Descriptor_pred/acsf_params/Const_1*
T0*+
_output_shapes
:���������
o
*Descriptor_pred/Angular_part/Fc_term/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *�I@
�
(Descriptor_pred/Angular_part/Fc_term/mulMul*Descriptor_pred/Angular_part/Fc_term/mul/x7Descriptor_pred/Angular_part/Sum_distances/norm/Squeeze*+
_output_shapes
:���������*
T0
�
,Descriptor_pred/Angular_part/Fc_term/truedivRealDiv(Descriptor_pred/Angular_part/Fc_term/mul#Descriptor_pred/acsf_params/Const_1*
T0*+
_output_shapes
:���������
�
(Descriptor_pred/Angular_part/Fc_term/CosCos,Descriptor_pred/Angular_part/Fc_term/truediv*
T0*+
_output_shapes
:���������
o
*Descriptor_pred/Angular_part/Fc_term/add/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
(Descriptor_pred/Angular_part/Fc_term/addAdd(Descriptor_pred/Angular_part/Fc_term/Cos*Descriptor_pred/Angular_part/Fc_term/add/y*
T0*+
_output_shapes
:���������
q
,Descriptor_pred/Angular_part/Fc_term/mul_1/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
�
*Descriptor_pred/Angular_part/Fc_term/mul_1Mul,Descriptor_pred/Angular_part/Fc_term/mul_1/x(Descriptor_pred/Angular_part/Fc_term/add*
T0*+
_output_shapes
:���������
�
*Descriptor_pred/Angular_part/Fc_term/ShapeShape7Descriptor_pred/Angular_part/Sum_distances/norm/Squeeze*
T0*
out_type0*
_output_shapes
:
u
0Descriptor_pred/Angular_part/Fc_term/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
*Descriptor_pred/Angular_part/Fc_term/zerosFill*Descriptor_pred/Angular_part/Fc_term/Shape0Descriptor_pred/Angular_part/Fc_term/zeros/Const*
T0*

index_type0*+
_output_shapes
:���������
�
+Descriptor_pred/Angular_part/Fc_term/SelectSelect)Descriptor_pred/Angular_part/Fc_term/Less*Descriptor_pred/Angular_part/Fc_term/mul_1*Descriptor_pred/Angular_part/Fc_term/zeros*
T0*+
_output_shapes
:���������
u
3Descriptor_pred/Angular_part/Fc_term/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
/Descriptor_pred/Angular_part/Fc_term/ExpandDims
ExpandDims+Descriptor_pred/Angular_part/Fc_term/Select3Descriptor_pred/Angular_part/Fc_term/ExpandDims/dim*

Tdim0*
T0*/
_output_shapes
:���������
w
5Descriptor_pred/Angular_part/Fc_term/ExpandDims_1/dimConst*
dtype0*
_output_shapes
: *
value	B :
�
1Descriptor_pred/Angular_part/Fc_term/ExpandDims_1
ExpandDims+Descriptor_pred/Angular_part/Fc_term/Select5Descriptor_pred/Angular_part/Fc_term/ExpandDims_1/dim*

Tdim0*
T0*/
_output_shapes
:���������
�
*Descriptor_pred/Angular_part/Fc_term/Mul_2Mul/Descriptor_pred/Angular_part/Fc_term/ExpandDims1Descriptor_pred/Angular_part/Fc_term/ExpandDims_1*
T0*/
_output_shapes
:���������
�
-Descriptor_pred/Angular_part/Fc_term/Select_1Select!Descriptor_pred/Angular_part/Tile"Descriptor_pred/Angular_part/zeros*Descriptor_pred/Angular_part/Fc_term/Mul_2*
T0*/
_output_shapes
:���������
l
*Descriptor_pred/Angular_part/Fc_term/ConstConst*
value	B : *
dtype0*
_output_shapes
: 
�
*Descriptor_pred/Angular_part/Fc_term/EqualEqualInputs_pred/IteratorGetNext:1*Descriptor_pred/Angular_part/Fc_term/Const*'
_output_shapes
:���������*
T0
�
/Descriptor_pred/Angular_part/Fc_term/LogicalNot
LogicalNot*Descriptor_pred/Angular_part/Fc_term/Equal*'
_output_shapes
:���������
w
5Descriptor_pred/Angular_part/Fc_term/ExpandDims_2/dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
1Descriptor_pred/Angular_part/Fc_term/ExpandDims_2
ExpandDims/Descriptor_pred/Angular_part/Fc_term/LogicalNot5Descriptor_pred/Angular_part/Fc_term/ExpandDims_2/dim*
T0
*+
_output_shapes
:���������*

Tdim0
�
5Descriptor_pred/Angular_part/Fc_term/ExpandDims_3/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
1Descriptor_pred/Angular_part/Fc_term/ExpandDims_3
ExpandDims/Descriptor_pred/Angular_part/Fc_term/LogicalNot5Descriptor_pred/Angular_part/Fc_term/ExpandDims_3/dim*

Tdim0*
T0
*+
_output_shapes
:���������
�
/Descriptor_pred/Angular_part/Fc_term/LogicalAnd
LogicalAnd1Descriptor_pred/Angular_part/Fc_term/ExpandDims_21Descriptor_pred/Angular_part/Fc_term/ExpandDims_3*+
_output_shapes
:���������
w
5Descriptor_pred/Angular_part/Fc_term/ExpandDims_4/dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
1Descriptor_pred/Angular_part/Fc_term/ExpandDims_4
ExpandDims/Descriptor_pred/Angular_part/Fc_term/LogicalAnd5Descriptor_pred/Angular_part/Fc_term/ExpandDims_4/dim*/
_output_shapes
:���������*

Tdim0*
T0

�
5Descriptor_pred/Angular_part/Fc_term/ExpandDims_5/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
1Descriptor_pred/Angular_part/Fc_term/ExpandDims_5
ExpandDims/Descriptor_pred/Angular_part/Fc_term/LogicalNot5Descriptor_pred/Angular_part/Fc_term/ExpandDims_5/dim*

Tdim0*
T0
*+
_output_shapes
:���������
�
5Descriptor_pred/Angular_part/Fc_term/ExpandDims_6/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
1Descriptor_pred/Angular_part/Fc_term/ExpandDims_6
ExpandDims1Descriptor_pred/Angular_part/Fc_term/ExpandDims_55Descriptor_pred/Angular_part/Fc_term/ExpandDims_6/dim*
T0
*/
_output_shapes
:���������*

Tdim0
�
1Descriptor_pred/Angular_part/Fc_term/LogicalAnd_1
LogicalAnd1Descriptor_pred/Angular_part/Fc_term/ExpandDims_41Descriptor_pred/Angular_part/Fc_term/ExpandDims_6*/
_output_shapes
:���������
�
-Descriptor_pred/Angular_part/Fc_term/Select_2Select1Descriptor_pred/Angular_part/Fc_term/LogicalAnd_1-Descriptor_pred/Angular_part/Fc_term/Select_1"Descriptor_pred/Angular_part/zeros*
T0*/
_output_shapes
:���������
s
1Descriptor_pred/Angular_part/Theta/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B :
�
-Descriptor_pred/Angular_part/Theta/ExpandDims
ExpandDims.Descriptor_pred/Angular_part/Sum_distances/sub1Descriptor_pred/Angular_part/Theta/ExpandDims/dim*
T0*3
_output_shapes!
:���������*

Tdim0
u
3Descriptor_pred/Angular_part/Theta/ExpandDims_1/dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
/Descriptor_pred/Angular_part/Theta/ExpandDims_1
ExpandDims.Descriptor_pred/Angular_part/Sum_distances/sub3Descriptor_pred/Angular_part/Theta/ExpandDims_1/dim*
T0*3
_output_shapes!
:���������*

Tdim0
�
&Descriptor_pred/Angular_part/Theta/MulMul-Descriptor_pred/Angular_part/Theta/ExpandDims/Descriptor_pred/Angular_part/Theta/ExpandDims_1*
T0*3
_output_shapes!
:���������
z
8Descriptor_pred/Angular_part/Theta/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
�
&Descriptor_pred/Angular_part/Theta/SumSum&Descriptor_pred/Angular_part/Theta/Mul8Descriptor_pred/Angular_part/Theta/Sum/reduction_indices*
T0*/
_output_shapes
:���������*

Tidx0*
	keep_dims( 
u
3Descriptor_pred/Angular_part/Theta/ExpandDims_2/dimConst*
dtype0*
_output_shapes
: *
value	B :
�
/Descriptor_pred/Angular_part/Theta/ExpandDims_2
ExpandDims7Descriptor_pred/Angular_part/Sum_distances/norm/Squeeze3Descriptor_pred/Angular_part/Theta/ExpandDims_2/dim*

Tdim0*
T0*/
_output_shapes
:���������
u
3Descriptor_pred/Angular_part/Theta/ExpandDims_3/dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
/Descriptor_pred/Angular_part/Theta/ExpandDims_3
ExpandDims7Descriptor_pred/Angular_part/Sum_distances/norm/Squeeze3Descriptor_pred/Angular_part/Theta/ExpandDims_3/dim*

Tdim0*
T0*/
_output_shapes
:���������
�
(Descriptor_pred/Angular_part/Theta/Mul_1Mul/Descriptor_pred/Angular_part/Theta/ExpandDims_2/Descriptor_pred/Angular_part/Theta/ExpandDims_3*
T0*/
_output_shapes
:���������
�
*Descriptor_pred/Angular_part/Theta/truedivRealDiv&Descriptor_pred/Angular_part/Theta/Sum(Descriptor_pred/Angular_part/Theta/Mul_1*
T0*/
_output_shapes
:���������
m
(Descriptor_pred/Angular_part/Theta/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *���
o
*Descriptor_pred/Angular_part/Theta/Const_1Const*
valueB
 *��?*
dtype0*
_output_shapes
: 
�
8Descriptor_pred/Angular_part/Theta/clip_by_value/MinimumMinimum*Descriptor_pred/Angular_part/Theta/truediv*Descriptor_pred/Angular_part/Theta/Const_1*
T0*/
_output_shapes
:���������
�
0Descriptor_pred/Angular_part/Theta/clip_by_valueMaximum8Descriptor_pred/Angular_part/Theta/clip_by_value/Minimum(Descriptor_pred/Angular_part/Theta/Const*
T0*/
_output_shapes
:���������
�
'Descriptor_pred/Angular_part/Theta/AcosAcos0Descriptor_pred/Angular_part/Theta/clip_by_value*
T0*/
_output_shapes
:���������
�
)Descriptor_pred/Angular_part/Theta/SelectSelect!Descriptor_pred/Angular_part/Tile"Descriptor_pred/Angular_part/zeros'Descriptor_pred/Angular_part/Theta/Acos*
T0*/
_output_shapes
:���������
l
*Descriptor_pred/Angular_part/Theta/Const_2Const*
value	B : *
dtype0*
_output_shapes
: 
�
(Descriptor_pred/Angular_part/Theta/EqualEqualInputs_pred/IteratorGetNext:1*Descriptor_pred/Angular_part/Theta/Const_2*'
_output_shapes
:���������*
T0
�
-Descriptor_pred/Angular_part/Theta/LogicalNot
LogicalNot(Descriptor_pred/Angular_part/Theta/Equal*'
_output_shapes
:���������
u
3Descriptor_pred/Angular_part/Theta/ExpandDims_4/dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
/Descriptor_pred/Angular_part/Theta/ExpandDims_4
ExpandDims-Descriptor_pred/Angular_part/Theta/LogicalNot3Descriptor_pred/Angular_part/Theta/ExpandDims_4/dim*

Tdim0*
T0
*+
_output_shapes
:���������
~
3Descriptor_pred/Angular_part/Theta/ExpandDims_5/dimConst*
dtype0*
_output_shapes
: *
valueB :
���������
�
/Descriptor_pred/Angular_part/Theta/ExpandDims_5
ExpandDims-Descriptor_pred/Angular_part/Theta/LogicalNot3Descriptor_pred/Angular_part/Theta/ExpandDims_5/dim*

Tdim0*
T0
*+
_output_shapes
:���������
�
-Descriptor_pred/Angular_part/Theta/LogicalAnd
LogicalAnd/Descriptor_pred/Angular_part/Theta/ExpandDims_4/Descriptor_pred/Angular_part/Theta/ExpandDims_5*+
_output_shapes
:���������
u
3Descriptor_pred/Angular_part/Theta/ExpandDims_6/dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
/Descriptor_pred/Angular_part/Theta/ExpandDims_6
ExpandDims-Descriptor_pred/Angular_part/Theta/LogicalAnd3Descriptor_pred/Angular_part/Theta/ExpandDims_6/dim*

Tdim0*
T0
*/
_output_shapes
:���������
~
3Descriptor_pred/Angular_part/Theta/ExpandDims_7/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
/Descriptor_pred/Angular_part/Theta/ExpandDims_7
ExpandDims-Descriptor_pred/Angular_part/Theta/LogicalNot3Descriptor_pred/Angular_part/Theta/ExpandDims_7/dim*

Tdim0*
T0
*+
_output_shapes
:���������
~
3Descriptor_pred/Angular_part/Theta/ExpandDims_8/dimConst*
dtype0*
_output_shapes
: *
valueB :
���������
�
/Descriptor_pred/Angular_part/Theta/ExpandDims_8
ExpandDims/Descriptor_pred/Angular_part/Theta/ExpandDims_73Descriptor_pred/Angular_part/Theta/ExpandDims_8/dim*
T0
*/
_output_shapes
:���������*

Tdim0
�
/Descriptor_pred/Angular_part/Theta/LogicalAnd_1
LogicalAnd/Descriptor_pred/Angular_part/Theta/ExpandDims_6/Descriptor_pred/Angular_part/Theta/ExpandDims_8*/
_output_shapes
:���������
�
+Descriptor_pred/Angular_part/Theta/Select_1Select/Descriptor_pred/Angular_part/Theta/LogicalAnd_1)Descriptor_pred/Angular_part/Theta/Select"Descriptor_pred/Angular_part/zeros*
T0*/
_output_shapes
:���������
v
4Descriptor_pred/Angular_part/Exp_term/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B : 
�
0Descriptor_pred/Angular_part/Exp_term/ExpandDims
ExpandDims#Descriptor_pred/acsf_params/Const_34Descriptor_pred/Angular_part/Exp_term/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:
x
6Descriptor_pred/Angular_part/Exp_term/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
2Descriptor_pred/Angular_part/Exp_term/ExpandDims_1
ExpandDims0Descriptor_pred/Angular_part/Exp_term/ExpandDims6Descriptor_pred/Angular_part/Exp_term/ExpandDims_1/dim*"
_output_shapes
:*

Tdim0*
T0
x
6Descriptor_pred/Angular_part/Exp_term/ExpandDims_2/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
2Descriptor_pred/Angular_part/Exp_term/ExpandDims_2
ExpandDims2Descriptor_pred/Angular_part/Exp_term/ExpandDims_16Descriptor_pred/Angular_part/Exp_term/ExpandDims_2/dim*

Tdim0*
T0*&
_output_shapes
:
x
6Descriptor_pred/Angular_part/Exp_term/ExpandDims_3/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
2Descriptor_pred/Angular_part/Exp_term/ExpandDims_3
ExpandDims2Descriptor_pred/Angular_part/Exp_term/ExpandDims_26Descriptor_pred/Angular_part/Exp_term/ExpandDims_3/dim*

Tdim0*
T0**
_output_shapes
:
p
+Descriptor_pred/Angular_part/Exp_term/mul/yConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
�
)Descriptor_pred/Angular_part/Exp_term/mulMul0Descriptor_pred/Angular_part/Sum_distances/add_1+Descriptor_pred/Angular_part/Exp_term/mul/y*
T0*/
_output_shapes
:���������
�
6Descriptor_pred/Angular_part/Exp_term/ExpandDims_4/dimConst*
dtype0*
_output_shapes
: *
valueB :
���������
�
2Descriptor_pred/Angular_part/Exp_term/ExpandDims_4
ExpandDims)Descriptor_pred/Angular_part/Exp_term/mul6Descriptor_pred/Angular_part/Exp_term/ExpandDims_4/dim*

Tdim0*
T0*3
_output_shapes!
:���������
�
)Descriptor_pred/Angular_part/Exp_term/SubSub2Descriptor_pred/Angular_part/Exp_term/ExpandDims_42Descriptor_pred/Angular_part/Exp_term/ExpandDims_3*3
_output_shapes!
:���������*
T0
v
)Descriptor_pred/Angular_part/Exp_term/NegNeg#Descriptor_pred/acsf_params/Const_6*
_output_shapes
: *
T0
�
,Descriptor_pred/Angular_part/Exp_term/SquareSquare)Descriptor_pred/Angular_part/Exp_term/Sub*
T0*3
_output_shapes!
:���������
�
+Descriptor_pred/Angular_part/Exp_term/mul_1Mul)Descriptor_pred/Angular_part/Exp_term/Neg,Descriptor_pred/Angular_part/Exp_term/Square*
T0*3
_output_shapes!
:���������
�
)Descriptor_pred/Angular_part/Exp_term/ExpExp+Descriptor_pred/Angular_part/Exp_term/mul_1*
T0*3
_output_shapes!
:���������
v
4Descriptor_pred/Angular_part/Cos_term/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
0Descriptor_pred/Angular_part/Cos_term/ExpandDims
ExpandDims#Descriptor_pred/acsf_params/Const_44Descriptor_pred/Angular_part/Cos_term/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:
x
6Descriptor_pred/Angular_part/Cos_term/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
2Descriptor_pred/Angular_part/Cos_term/ExpandDims_1
ExpandDims0Descriptor_pred/Angular_part/Cos_term/ExpandDims6Descriptor_pred/Angular_part/Cos_term/ExpandDims_1/dim*

Tdim0*
T0*"
_output_shapes
:
x
6Descriptor_pred/Angular_part/Cos_term/ExpandDims_2/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
2Descriptor_pred/Angular_part/Cos_term/ExpandDims_2
ExpandDims2Descriptor_pred/Angular_part/Cos_term/ExpandDims_16Descriptor_pred/Angular_part/Cos_term/ExpandDims_2/dim*

Tdim0*
T0*&
_output_shapes
:
x
6Descriptor_pred/Angular_part/Cos_term/ExpandDims_3/dimConst*
dtype0*
_output_shapes
: *
value	B : 
�
2Descriptor_pred/Angular_part/Cos_term/ExpandDims_3
ExpandDims2Descriptor_pred/Angular_part/Cos_term/ExpandDims_26Descriptor_pred/Angular_part/Cos_term/ExpandDims_3/dim**
_output_shapes
:*

Tdim0*
T0
�
6Descriptor_pred/Angular_part/Cos_term/ExpandDims_4/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
2Descriptor_pred/Angular_part/Cos_term/ExpandDims_4
ExpandDims+Descriptor_pred/Angular_part/Theta/Select_16Descriptor_pred/Angular_part/Cos_term/ExpandDims_4/dim*3
_output_shapes!
:���������*

Tdim0*
T0
�
)Descriptor_pred/Angular_part/Cos_term/SubSub2Descriptor_pred/Angular_part/Cos_term/ExpandDims_42Descriptor_pred/Angular_part/Cos_term/ExpandDims_3*
T0*3
_output_shapes!
:���������
�
)Descriptor_pred/Angular_part/Cos_term/CosCos)Descriptor_pred/Angular_part/Cos_term/Sub*3
_output_shapes!
:���������*
T0
�
+Descriptor_pred/Angular_part/Cos_term/ShapeShape)Descriptor_pred/Angular_part/Cos_term/Cos*
T0*
out_type0*
_output_shapes
:
u
0Descriptor_pred/Angular_part/Cos_term/ones/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
*Descriptor_pred/Angular_part/Cos_term/onesFill+Descriptor_pred/Angular_part/Cos_term/Shape0Descriptor_pred/Angular_part/Cos_term/ones/Const*3
_output_shapes!
:���������*
T0*

index_type0
�
)Descriptor_pred/Angular_part/Cos_term/AddAdd*Descriptor_pred/Angular_part/Cos_term/ones)Descriptor_pred/Angular_part/Cos_term/Cos*
T0*3
_output_shapes!
:���������
p
+Descriptor_pred/Angular_part/Cos_term/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *   @
�
-Descriptor_pred/Angular_part/Cos_term/truedivRealDiv)Descriptor_pred/Angular_part/Cos_term/Add+Descriptor_pred/Angular_part/Cos_term/Const*
T0*3
_output_shapes!
:���������
�
)Descriptor_pred/Angular_part/Cos_term/PowPow-Descriptor_pred/Angular_part/Cos_term/truediv#Descriptor_pred/acsf_params/Const_5*
T0*3
_output_shapes!
:���������
x
-Descriptor_pred/Angular_part/ExpandDims_1/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
)Descriptor_pred/Angular_part/ExpandDims_1
ExpandDims-Descriptor_pred/Angular_part/Fc_term/Select_2-Descriptor_pred/Angular_part/ExpandDims_1/dim*3
_output_shapes!
:���������*

Tdim0*
T0
w
,Descriptor_pred/Angular_part/Expanded_fc/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
(Descriptor_pred/Angular_part/Expanded_fc
ExpandDims)Descriptor_pred/Angular_part/ExpandDims_1,Descriptor_pred/Angular_part/Expanded_fc/dim*

Tdim0*
T0*7
_output_shapes%
#:!���������
x
-Descriptor_pred/Angular_part/Expanded_cos/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
)Descriptor_pred/Angular_part/Expanded_cos
ExpandDims)Descriptor_pred/Angular_part/Cos_term/Pow-Descriptor_pred/Angular_part/Expanded_cos/dim*7
_output_shapes%
#:!���������*

Tdim0*
T0
x
-Descriptor_pred/Angular_part/Expanded_exp/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
)Descriptor_pred/Angular_part/Expanded_exp
ExpandDims)Descriptor_pred/Angular_part/Exp_term/Exp-Descriptor_pred/Angular_part/Expanded_exp/dim*
T0*7
_output_shapes%
#:!���������*

Tdim0
i
$Descriptor_pred/Angular_part/Const_1Const*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
)Descriptor_pred/Angular_part/Ang_term/MulMul)Descriptor_pred/Angular_part/Expanded_cos)Descriptor_pred/Angular_part/Expanded_exp*7
_output_shapes%
#:!���������*
T0
�
+Descriptor_pred/Angular_part/Ang_term/Mul_1Mul)Descriptor_pred/Angular_part/Ang_term/Mul(Descriptor_pred/Angular_part/Expanded_fc*
T0*7
_output_shapes%
#:!���������
�
+Descriptor_pred/Angular_part/Ang_term/mul_2Mul$Descriptor_pred/Angular_part/Const_1+Descriptor_pred/Angular_part/Ang_term/Mul_1*
T0*7
_output_shapes%
#:!���������
�
+Descriptor_pred/Angular_part/Ang_term/ShapeShape+Descriptor_pred/Angular_part/Ang_term/mul_2*
_output_shapes
:*
T0*
out_type0
�
9Descriptor_pred/Angular_part/Ang_term/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
�
;Descriptor_pred/Angular_part/Ang_term/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
;Descriptor_pred/Angular_part/Ang_term/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
3Descriptor_pred/Angular_part/Ang_term/strided_sliceStridedSlice+Descriptor_pred/Angular_part/Ang_term/Shape9Descriptor_pred/Angular_part/Ang_term/strided_slice/stack;Descriptor_pred/Angular_part/Ang_term/strided_slice/stack_1;Descriptor_pred/Angular_part/Ang_term/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
w
5Descriptor_pred/Angular_part/Ang_term/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
w
5Descriptor_pred/Angular_part/Ang_term/Reshape/shape/2Const*
dtype0*
_output_shapes
: *
value	B :
w
5Descriptor_pred/Angular_part/Ang_term/Reshape/shape/3Const*
dtype0*
_output_shapes
: *
value	B :
x
5Descriptor_pred/Angular_part/Ang_term/Reshape/shape/4Const*
value
B :�*
dtype0*
_output_shapes
: 
�
3Descriptor_pred/Angular_part/Ang_term/Reshape/shapePack3Descriptor_pred/Angular_part/Ang_term/strided_slice5Descriptor_pred/Angular_part/Ang_term/Reshape/shape/15Descriptor_pred/Angular_part/Ang_term/Reshape/shape/25Descriptor_pred/Angular_part/Ang_term/Reshape/shape/35Descriptor_pred/Angular_part/Ang_term/Reshape/shape/4*
T0*

axis *
N*
_output_shapes
:
�
-Descriptor_pred/Angular_part/Ang_term/ReshapeReshape+Descriptor_pred/Angular_part/Ang_term/mul_23Descriptor_pred/Angular_part/Ang_term/Reshape/shape*
T0*
Tshape0*4
_output_shapes"
 :����������
h
&Descriptor_pred/Sum_ang/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B :
�
"Descriptor_pred/Sum_ang/ExpandDims
ExpandDimsInputs_pred/IteratorGetNext:1&Descriptor_pred/Sum_ang/ExpandDims/dim*+
_output_shapes
:���������*

Tdim0*
T0
{
&Descriptor_pred/Sum_ang/Tile/multiplesConst*!
valueB"         *
dtype0*
_output_shapes
:
�
Descriptor_pred/Sum_ang/TileTile"Descriptor_pred/Sum_ang/ExpandDims&Descriptor_pred/Sum_ang/Tile/multiples*

Tmultiples0*
T0*+
_output_shapes
:���������
s
(Descriptor_pred/Sum_ang/ExpandDims_1/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
$Descriptor_pred/Sum_ang/ExpandDims_1
ExpandDimsDescriptor_pred/Sum_ang/Tile(Descriptor_pred/Sum_ang/ExpandDims_1/dim*
T0*/
_output_shapes
:���������*

Tdim0
s
(Descriptor_pred/Sum_ang/ExpandDims_2/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
$Descriptor_pred/Sum_ang/ExpandDims_2
ExpandDimsInputs_pred/IteratorGetNext:1(Descriptor_pred/Sum_ang/ExpandDims_2/dim*
T0*+
_output_shapes
:���������*

Tdim0
}
(Descriptor_pred/Sum_ang/Tile_1/multiplesConst*!
valueB"         *
dtype0*
_output_shapes
:
�
Descriptor_pred/Sum_ang/Tile_1Tile$Descriptor_pred/Sum_ang/ExpandDims_2(Descriptor_pred/Sum_ang/Tile_1/multiples*

Tmultiples0*
T0*+
_output_shapes
:���������
s
(Descriptor_pred/Sum_ang/ExpandDims_3/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
$Descriptor_pred/Sum_ang/ExpandDims_3
ExpandDimsDescriptor_pred/Sum_ang/Tile_1(Descriptor_pred/Sum_ang/ExpandDims_3/dim*

Tdim0*
T0*/
_output_shapes
:���������
n
#Descriptor_pred/Sum_ang/concat/axisConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Descriptor_pred/Sum_ang/concatConcatV2$Descriptor_pred/Sum_ang/ExpandDims_1$Descriptor_pred/Sum_ang/ExpandDims_3#Descriptor_pred/Sum_ang/concat/axis*

Tidx0*
T0*
N*/
_output_shapes
:���������
�
Descriptor_pred/Sum_ang/ConstConst*
valuevBt
Zb                                                                                    *
dtype0
*"
_output_shapes
:
j
(Descriptor_pred/Sum_ang/ExpandDims_4/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
$Descriptor_pred/Sum_ang/ExpandDims_4
ExpandDimsDescriptor_pred/Sum_ang/Const(Descriptor_pred/Sum_ang/ExpandDims_4/dim*&
_output_shapes
:*

Tdim0*
T0

z
Descriptor_pred/Sum_ang/ShapeShapeInputs_pred/IteratorGetNext:1*
T0*
out_type0*
_output_shapes
:
u
+Descriptor_pred/Sum_ang/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
w
-Descriptor_pred/Sum_ang/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
w
-Descriptor_pred/Sum_ang/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
%Descriptor_pred/Sum_ang/strided_sliceStridedSliceDescriptor_pred/Sum_ang/Shape+Descriptor_pred/Sum_ang/strided_slice/stack-Descriptor_pred/Sum_ang/strided_slice/stack_1-Descriptor_pred/Sum_ang/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
l
*Descriptor_pred/Sum_ang/Tile_2/multiples/1Const*
value	B :*
dtype0*
_output_shapes
: 
l
*Descriptor_pred/Sum_ang/Tile_2/multiples/2Const*
dtype0*
_output_shapes
: *
value	B :
l
*Descriptor_pred/Sum_ang/Tile_2/multiples/3Const*
value	B :*
dtype0*
_output_shapes
: 
�
(Descriptor_pred/Sum_ang/Tile_2/multiplesPack%Descriptor_pred/Sum_ang/strided_slice*Descriptor_pred/Sum_ang/Tile_2/multiples/1*Descriptor_pred/Sum_ang/Tile_2/multiples/2*Descriptor_pred/Sum_ang/Tile_2/multiples/3*
T0*

axis *
N*
_output_shapes
:
�
Descriptor_pred/Sum_ang/Tile_2Tile$Descriptor_pred/Sum_ang/ExpandDims_4(Descriptor_pred/Sum_ang/Tile_2/multiples*
T0
*/
_output_shapes
:���������*

Tmultiples0
}
Descriptor_pred/Sum_ang/Shape_1ShapeDescriptor_pred/Sum_ang/concat*
T0*
out_type0*
_output_shapes
:
e
#Descriptor_pred/Sum_ang/zeros/ConstConst*
dtype0*
_output_shapes
: *
value	B : 
�
Descriptor_pred/Sum_ang/zerosFillDescriptor_pred/Sum_ang/Shape_1#Descriptor_pred/Sum_ang/zeros/Const*/
_output_shapes
:���������*
T0*

index_type0
�
Descriptor_pred/Sum_ang/SelectSelectDescriptor_pred/Sum_ang/Tile_2Descriptor_pred/Sum_ang/zerosDescriptor_pred/Sum_ang/concat*/
_output_shapes
:���������*
T0
b
 Descriptor_pred/Sum_ang/TopKV2/kConst*
value	B :*
dtype0*
_output_shapes
: 
�
Descriptor_pred/Sum_ang/TopKV2TopKV2Descriptor_pred/Sum_ang/Select Descriptor_pred/Sum_ang/TopKV2/k*
sorted(*
T0*J
_output_shapes8
6:���������:���������
�
Descriptor_pred/Sum_ang/Const_1Const*
dtype0
*"
_output_shapes
:*�
value�B�
Z�                                                                                                                                     
j
(Descriptor_pred/Sum_ang/ExpandDims_5/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
$Descriptor_pred/Sum_ang/ExpandDims_5
ExpandDimsDescriptor_pred/Sum_ang/Const_1(Descriptor_pred/Sum_ang/ExpandDims_5/dim*
T0
*&
_output_shapes
:*

Tdim0
|
Descriptor_pred/Sum_ang/Shape_2ShapeInputs_pred/IteratorGetNext:1*
T0*
out_type0*
_output_shapes
:
w
-Descriptor_pred/Sum_ang/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
y
/Descriptor_pred/Sum_ang/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/Descriptor_pred/Sum_ang/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
'Descriptor_pred/Sum_ang/strided_slice_1StridedSliceDescriptor_pred/Sum_ang/Shape_2-Descriptor_pred/Sum_ang/strided_slice_1/stack/Descriptor_pred/Sum_ang/strided_slice_1/stack_1/Descriptor_pred/Sum_ang/strided_slice_1/stack_2*
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
l
*Descriptor_pred/Sum_ang/Tile_3/multiples/1Const*
dtype0*
_output_shapes
: *
value	B :
l
*Descriptor_pred/Sum_ang/Tile_3/multiples/2Const*
value	B :*
dtype0*
_output_shapes
: 
l
*Descriptor_pred/Sum_ang/Tile_3/multiples/3Const*
value	B :*
dtype0*
_output_shapes
: 
�
(Descriptor_pred/Sum_ang/Tile_3/multiplesPack'Descriptor_pred/Sum_ang/strided_slice_1*Descriptor_pred/Sum_ang/Tile_3/multiples/1*Descriptor_pred/Sum_ang/Tile_3/multiples/2*Descriptor_pred/Sum_ang/Tile_3/multiples/3*
N*
_output_shapes
:*
T0*

axis 
�
Descriptor_pred/Sum_ang/Tile_3Tile$Descriptor_pred/Sum_ang/ExpandDims_5(Descriptor_pred/Sum_ang/Tile_3/multiples*/
_output_shapes
:���������*

Tmultiples0*
T0

s
(Descriptor_pred/Sum_ang/ExpandDims_6/dimConst*
dtype0*
_output_shapes
: *
valueB :
���������
�
$Descriptor_pred/Sum_ang/ExpandDims_6
ExpandDimsDescriptor_pred/Sum_ang/Tile_3(Descriptor_pred/Sum_ang/ExpandDims_6/dim*

Tdim0*
T0
*3
_output_shapes!
:���������
�
Descriptor_pred/Sum_ang/Shape_3Shape-Descriptor_pred/Angular_part/Ang_term/Reshape*
T0*
out_type0*
_output_shapes
:
m
(Descriptor_pred/Sum_ang/zero_large/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
"Descriptor_pred/Sum_ang/zero_largeFillDescriptor_pred/Sum_ang/Shape_3(Descriptor_pred/Sum_ang/zero_large/Const*
T0*

index_type0*4
_output_shapes"
 :����������
v
%Descriptor_pred/Sum_ang/Extract/ConstConst*
valueB"      *
dtype0*
_output_shapes
:
p
.Descriptor_pred/Sum_ang/Extract/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
*Descriptor_pred/Sum_ang/Extract/ExpandDims
ExpandDims%Descriptor_pred/Sum_ang/Extract/Const.Descriptor_pred/Sum_ang/Extract/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:
r
0Descriptor_pred/Sum_ang/Extract/ExpandDims_1/dimConst*
dtype0*
_output_shapes
: *
value	B : 
�
,Descriptor_pred/Sum_ang/Extract/ExpandDims_1
ExpandDims*Descriptor_pred/Sum_ang/Extract/ExpandDims0Descriptor_pred/Sum_ang/Extract/ExpandDims_1/dim*
T0*"
_output_shapes
:*

Tdim0
r
0Descriptor_pred/Sum_ang/Extract/ExpandDims_2/dimConst*
dtype0*
_output_shapes
: *
value	B : 
�
,Descriptor_pred/Sum_ang/Extract/ExpandDims_2
ExpandDims,Descriptor_pred/Sum_ang/Extract/ExpandDims_10Descriptor_pred/Sum_ang/Extract/ExpandDims_2/dim*
T0*&
_output_shapes
:*

Tdim0
�
%Descriptor_pred/Sum_ang/Extract/ShapeShapeInputs_pred/IteratorGetNext:1*
T0*
out_type0*
_output_shapes
:
}
3Descriptor_pred/Sum_ang/Extract/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

5Descriptor_pred/Sum_ang/Extract/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

5Descriptor_pred/Sum_ang/Extract/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
-Descriptor_pred/Sum_ang/Extract/strided_sliceStridedSlice%Descriptor_pred/Sum_ang/Extract/Shape3Descriptor_pred/Sum_ang/Extract/strided_slice/stack5Descriptor_pred/Sum_ang/Extract/strided_slice/stack_15Descriptor_pred/Sum_ang/Extract/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
y
7Descriptor_pred/Sum_ang/Extract/expand_pair/multiples/1Const*
value	B :*
dtype0*
_output_shapes
: 
y
7Descriptor_pred/Sum_ang/Extract/expand_pair/multiples/2Const*
value	B :*
dtype0*
_output_shapes
: 
y
7Descriptor_pred/Sum_ang/Extract/expand_pair/multiples/3Const*
value	B :*
dtype0*
_output_shapes
: 
�
5Descriptor_pred/Sum_ang/Extract/expand_pair/multiplesPack-Descriptor_pred/Sum_ang/Extract/strided_slice7Descriptor_pred/Sum_ang/Extract/expand_pair/multiples/17Descriptor_pred/Sum_ang/Extract/expand_pair/multiples/27Descriptor_pred/Sum_ang/Extract/expand_pair/multiples/3*
N*
_output_shapes
:*
T0*

axis 
�
+Descriptor_pred/Sum_ang/Extract/expand_pairTile,Descriptor_pred/Sum_ang/Extract/ExpandDims_25Descriptor_pred/Sum_ang/Extract/expand_pair/multiples*
T0*/
_output_shapes
:���������*

Tmultiples0
�
%Descriptor_pred/Sum_ang/Extract/EqualEqual+Descriptor_pred/Sum_ang/Extract/expand_pairDescriptor_pred/Sum_ang/TopKV2*/
_output_shapes
:���������*
T0
i
'Descriptor_pred/Sum_ang/Extract/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
z
/Descriptor_pred/Sum_ang/Extract/split/split_dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
%Descriptor_pred/Sum_ang/Extract/splitSplit/Descriptor_pred/Sum_ang/Extract/split/split_dim%Descriptor_pred/Sum_ang/Extract/Equal*
T0
*
	num_split*J
_output_shapes8
6:���������:���������
�
*Descriptor_pred/Sum_ang/Extract/LogicalAnd
LogicalAnd%Descriptor_pred/Sum_ang/Extract/split'Descriptor_pred/Sum_ang/Extract/split:1*/
_output_shapes
:���������
z
0Descriptor_pred/Sum_ang/Extract/ExpandDims_3/dimConst*
valueB:*
dtype0*
_output_shapes
:
�
,Descriptor_pred/Sum_ang/Extract/ExpandDims_3
ExpandDims*Descriptor_pred/Sum_ang/Extract/LogicalAnd0Descriptor_pred/Sum_ang/Extract/ExpandDims_3/dim*
T0
*3
_output_shapes!
:���������*

Tdim0
�
.Descriptor_pred/Sum_ang/Extract/Tile/multiplesConst*)
value B"               *
dtype0*
_output_shapes
:
�
$Descriptor_pred/Sum_ang/Extract/TileTile,Descriptor_pred/Sum_ang/Extract/ExpandDims_3.Descriptor_pred/Sum_ang/Extract/Tile/multiples*

Tmultiples0*
T0
*3
_output_shapes!
:���������
�
,Descriptor_pred/Sum_ang/Extract/LogicalAnd_1
LogicalAnd$Descriptor_pred/Sum_ang/Extract/Tile$Descriptor_pred/Sum_ang/ExpandDims_6*3
_output_shapes!
:���������
�
0Descriptor_pred/Sum_ang/Extract/Tile_1/multiplesConst*)
value B"            �   *
dtype0*
_output_shapes
:
�
&Descriptor_pred/Sum_ang/Extract/Tile_1Tile,Descriptor_pred/Sum_ang/Extract/LogicalAnd_10Descriptor_pred/Sum_ang/Extract/Tile_1/multiples*

Tmultiples0*
T0
*4
_output_shapes"
 :����������
�
'Descriptor_pred/Sum_ang/Extract/sl_pr_sSelect&Descriptor_pred/Sum_ang/Extract/Tile_1-Descriptor_pred/Angular_part/Ang_term/Reshape"Descriptor_pred/Sum_ang/zero_large*4
_output_shapes"
 :����������*
T0
�
9Descriptor_pred/Sum_ang/Extract/sum_ang/reduction_indicesConst*
valueB"      *
dtype0*
_output_shapes
:
�
'Descriptor_pred/Sum_ang/Extract/sum_angSum'Descriptor_pred/Sum_ang/Extract/sl_pr_s9Descriptor_pred/Sum_ang/Extract/sum_ang/reduction_indices*
T0*,
_output_shapes
:����������*

Tidx0*
	keep_dims( 
j
%Descriptor_pred/Sum_ang/Extract/mul/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
�
#Descriptor_pred/Sum_ang/Extract/mulMul%Descriptor_pred/Sum_ang/Extract/mul/x'Descriptor_pred/Sum_ang/Extract/sum_ang*
T0*,
_output_shapes
:����������
x
'Descriptor_pred/Sum_ang/Extract/Const_2Const*
valueB"      *
dtype0*
_output_shapes
:
r
0Descriptor_pred/Sum_ang/Extract/ExpandDims_4/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
,Descriptor_pred/Sum_ang/Extract/ExpandDims_4
ExpandDims'Descriptor_pred/Sum_ang/Extract/Const_20Descriptor_pred/Sum_ang/Extract/ExpandDims_4/dim*

Tdim0*
T0*
_output_shapes

:
r
0Descriptor_pred/Sum_ang/Extract/ExpandDims_5/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
,Descriptor_pred/Sum_ang/Extract/ExpandDims_5
ExpandDims,Descriptor_pred/Sum_ang/Extract/ExpandDims_40Descriptor_pred/Sum_ang/Extract/ExpandDims_5/dim*
T0*"
_output_shapes
:*

Tdim0
r
0Descriptor_pred/Sum_ang/Extract/ExpandDims_6/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
,Descriptor_pred/Sum_ang/Extract/ExpandDims_6
ExpandDims,Descriptor_pred/Sum_ang/Extract/ExpandDims_50Descriptor_pred/Sum_ang/Extract/ExpandDims_6/dim*

Tdim0*
T0*&
_output_shapes
:
�
'Descriptor_pred/Sum_ang/Extract/Shape_1ShapeInputs_pred/IteratorGetNext:1*
T0*
out_type0*
_output_shapes
:

5Descriptor_pred/Sum_ang/Extract/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
7Descriptor_pred/Sum_ang/Extract/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
7Descriptor_pred/Sum_ang/Extract/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
/Descriptor_pred/Sum_ang/Extract/strided_slice_1StridedSlice'Descriptor_pred/Sum_ang/Extract/Shape_15Descriptor_pred/Sum_ang/Extract/strided_slice_1/stack7Descriptor_pred/Sum_ang/Extract/strided_slice_1/stack_17Descriptor_pred/Sum_ang/Extract/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
{
9Descriptor_pred/Sum_ang/Extract/expand_pair_1/multiples/1Const*
value	B :*
dtype0*
_output_shapes
: 
{
9Descriptor_pred/Sum_ang/Extract/expand_pair_1/multiples/2Const*
value	B :*
dtype0*
_output_shapes
: 
{
9Descriptor_pred/Sum_ang/Extract/expand_pair_1/multiples/3Const*
value	B :*
dtype0*
_output_shapes
: 
�
7Descriptor_pred/Sum_ang/Extract/expand_pair_1/multiplesPack/Descriptor_pred/Sum_ang/Extract/strided_slice_19Descriptor_pred/Sum_ang/Extract/expand_pair_1/multiples/19Descriptor_pred/Sum_ang/Extract/expand_pair_1/multiples/29Descriptor_pred/Sum_ang/Extract/expand_pair_1/multiples/3*
N*
_output_shapes
:*
T0*

axis 
�
-Descriptor_pred/Sum_ang/Extract/expand_pair_1Tile,Descriptor_pred/Sum_ang/Extract/ExpandDims_67Descriptor_pred/Sum_ang/Extract/expand_pair_1/multiples*/
_output_shapes
:���������*

Tmultiples0*
T0
�
'Descriptor_pred/Sum_ang/Extract/Equal_1Equal-Descriptor_pred/Sum_ang/Extract/expand_pair_1Descriptor_pred/Sum_ang/TopKV2*
T0*/
_output_shapes
:���������
i
'Descriptor_pred/Sum_ang/Extract/Const_3Const*
value	B :*
dtype0*
_output_shapes
: 
|
1Descriptor_pred/Sum_ang/Extract/split_1/split_dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
'Descriptor_pred/Sum_ang/Extract/split_1Split1Descriptor_pred/Sum_ang/Extract/split_1/split_dim'Descriptor_pred/Sum_ang/Extract/Equal_1*
T0
*
	num_split*J
_output_shapes8
6:���������:���������
�
,Descriptor_pred/Sum_ang/Extract/LogicalAnd_2
LogicalAnd'Descriptor_pred/Sum_ang/Extract/split_1)Descriptor_pred/Sum_ang/Extract/split_1:1*/
_output_shapes
:���������
z
0Descriptor_pred/Sum_ang/Extract/ExpandDims_7/dimConst*
dtype0*
_output_shapes
:*
valueB:
�
,Descriptor_pred/Sum_ang/Extract/ExpandDims_7
ExpandDims,Descriptor_pred/Sum_ang/Extract/LogicalAnd_20Descriptor_pred/Sum_ang/Extract/ExpandDims_7/dim*
T0
*3
_output_shapes!
:���������*

Tdim0
�
0Descriptor_pred/Sum_ang/Extract/Tile_2/multiplesConst*)
value B"               *
dtype0*
_output_shapes
:
�
&Descriptor_pred/Sum_ang/Extract/Tile_2Tile,Descriptor_pred/Sum_ang/Extract/ExpandDims_70Descriptor_pred/Sum_ang/Extract/Tile_2/multiples*3
_output_shapes!
:���������*

Tmultiples0*
T0

�
,Descriptor_pred/Sum_ang/Extract/LogicalAnd_3
LogicalAnd&Descriptor_pred/Sum_ang/Extract/Tile_2$Descriptor_pred/Sum_ang/ExpandDims_6*3
_output_shapes!
:���������
�
0Descriptor_pred/Sum_ang/Extract/Tile_3/multiplesConst*)
value B"            �   *
dtype0*
_output_shapes
:
�
&Descriptor_pred/Sum_ang/Extract/Tile_3Tile,Descriptor_pred/Sum_ang/Extract/LogicalAnd_30Descriptor_pred/Sum_ang/Extract/Tile_3/multiples*
T0
*4
_output_shapes"
 :����������*

Tmultiples0
�
)Descriptor_pred/Sum_ang/Extract/sl_pr_s_1Select&Descriptor_pred/Sum_ang/Extract/Tile_3-Descriptor_pred/Angular_part/Ang_term/Reshape"Descriptor_pred/Sum_ang/zero_large*
T0*4
_output_shapes"
 :����������
�
;Descriptor_pred/Sum_ang/Extract/sum_ang_1/reduction_indicesConst*
valueB"      *
dtype0*
_output_shapes
:
�
)Descriptor_pred/Sum_ang/Extract/sum_ang_1Sum)Descriptor_pred/Sum_ang/Extract/sl_pr_s_1;Descriptor_pred/Sum_ang/Extract/sum_ang_1/reduction_indices*,
_output_shapes
:����������*

Tidx0*
	keep_dims( *
T0
l
'Descriptor_pred/Sum_ang/Extract/mul_1/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
�
%Descriptor_pred/Sum_ang/Extract/mul_1Mul'Descriptor_pred/Sum_ang/Extract/mul_1/x)Descriptor_pred/Sum_ang/Extract/sum_ang_1*
T0*,
_output_shapes
:����������
x
'Descriptor_pred/Sum_ang/Extract/Const_4Const*
dtype0*
_output_shapes
:*
valueB"      
r
0Descriptor_pred/Sum_ang/Extract/ExpandDims_8/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
,Descriptor_pred/Sum_ang/Extract/ExpandDims_8
ExpandDims'Descriptor_pred/Sum_ang/Extract/Const_40Descriptor_pred/Sum_ang/Extract/ExpandDims_8/dim*

Tdim0*
T0*
_output_shapes

:
r
0Descriptor_pred/Sum_ang/Extract/ExpandDims_9/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
,Descriptor_pred/Sum_ang/Extract/ExpandDims_9
ExpandDims,Descriptor_pred/Sum_ang/Extract/ExpandDims_80Descriptor_pred/Sum_ang/Extract/ExpandDims_9/dim*

Tdim0*
T0*"
_output_shapes
:
s
1Descriptor_pred/Sum_ang/Extract/ExpandDims_10/dimConst*
dtype0*
_output_shapes
: *
value	B : 
�
-Descriptor_pred/Sum_ang/Extract/ExpandDims_10
ExpandDims,Descriptor_pred/Sum_ang/Extract/ExpandDims_91Descriptor_pred/Sum_ang/Extract/ExpandDims_10/dim*
T0*&
_output_shapes
:*

Tdim0
�
'Descriptor_pred/Sum_ang/Extract/Shape_2ShapeInputs_pred/IteratorGetNext:1*
T0*
out_type0*
_output_shapes
:

5Descriptor_pred/Sum_ang/Extract/strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
7Descriptor_pred/Sum_ang/Extract/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
7Descriptor_pred/Sum_ang/Extract/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
/Descriptor_pred/Sum_ang/Extract/strided_slice_2StridedSlice'Descriptor_pred/Sum_ang/Extract/Shape_25Descriptor_pred/Sum_ang/Extract/strided_slice_2/stack7Descriptor_pred/Sum_ang/Extract/strided_slice_2/stack_17Descriptor_pred/Sum_ang/Extract/strided_slice_2/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
{
9Descriptor_pred/Sum_ang/Extract/expand_pair_2/multiples/1Const*
value	B :*
dtype0*
_output_shapes
: 
{
9Descriptor_pred/Sum_ang/Extract/expand_pair_2/multiples/2Const*
value	B :*
dtype0*
_output_shapes
: 
{
9Descriptor_pred/Sum_ang/Extract/expand_pair_2/multiples/3Const*
dtype0*
_output_shapes
: *
value	B :
�
7Descriptor_pred/Sum_ang/Extract/expand_pair_2/multiplesPack/Descriptor_pred/Sum_ang/Extract/strided_slice_29Descriptor_pred/Sum_ang/Extract/expand_pair_2/multiples/19Descriptor_pred/Sum_ang/Extract/expand_pair_2/multiples/29Descriptor_pred/Sum_ang/Extract/expand_pair_2/multiples/3*
T0*

axis *
N*
_output_shapes
:
�
-Descriptor_pred/Sum_ang/Extract/expand_pair_2Tile-Descriptor_pred/Sum_ang/Extract/ExpandDims_107Descriptor_pred/Sum_ang/Extract/expand_pair_2/multiples*
T0*/
_output_shapes
:���������*

Tmultiples0
�
'Descriptor_pred/Sum_ang/Extract/Equal_2Equal-Descriptor_pred/Sum_ang/Extract/expand_pair_2Descriptor_pred/Sum_ang/TopKV2*/
_output_shapes
:���������*
T0
i
'Descriptor_pred/Sum_ang/Extract/Const_5Const*
value	B :*
dtype0*
_output_shapes
: 
|
1Descriptor_pred/Sum_ang/Extract/split_2/split_dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
'Descriptor_pred/Sum_ang/Extract/split_2Split1Descriptor_pred/Sum_ang/Extract/split_2/split_dim'Descriptor_pred/Sum_ang/Extract/Equal_2*
T0
*
	num_split*J
_output_shapes8
6:���������:���������
�
,Descriptor_pred/Sum_ang/Extract/LogicalAnd_4
LogicalAnd'Descriptor_pred/Sum_ang/Extract/split_2)Descriptor_pred/Sum_ang/Extract/split_2:1*/
_output_shapes
:���������
{
1Descriptor_pred/Sum_ang/Extract/ExpandDims_11/dimConst*
valueB:*
dtype0*
_output_shapes
:
�
-Descriptor_pred/Sum_ang/Extract/ExpandDims_11
ExpandDims,Descriptor_pred/Sum_ang/Extract/LogicalAnd_41Descriptor_pred/Sum_ang/Extract/ExpandDims_11/dim*3
_output_shapes!
:���������*

Tdim0*
T0

�
0Descriptor_pred/Sum_ang/Extract/Tile_4/multiplesConst*)
value B"               *
dtype0*
_output_shapes
:
�
&Descriptor_pred/Sum_ang/Extract/Tile_4Tile-Descriptor_pred/Sum_ang/Extract/ExpandDims_110Descriptor_pred/Sum_ang/Extract/Tile_4/multiples*

Tmultiples0*
T0
*3
_output_shapes!
:���������
�
,Descriptor_pred/Sum_ang/Extract/LogicalAnd_5
LogicalAnd&Descriptor_pred/Sum_ang/Extract/Tile_4$Descriptor_pred/Sum_ang/ExpandDims_6*3
_output_shapes!
:���������
�
0Descriptor_pred/Sum_ang/Extract/Tile_5/multiplesConst*
dtype0*
_output_shapes
:*)
value B"            �   
�
&Descriptor_pred/Sum_ang/Extract/Tile_5Tile,Descriptor_pred/Sum_ang/Extract/LogicalAnd_50Descriptor_pred/Sum_ang/Extract/Tile_5/multiples*
T0
*4
_output_shapes"
 :����������*

Tmultiples0
�
)Descriptor_pred/Sum_ang/Extract/sl_pr_s_2Select&Descriptor_pred/Sum_ang/Extract/Tile_5-Descriptor_pred/Angular_part/Ang_term/Reshape"Descriptor_pred/Sum_ang/zero_large*
T0*4
_output_shapes"
 :����������
�
;Descriptor_pred/Sum_ang/Extract/sum_ang_2/reduction_indicesConst*
valueB"      *
dtype0*
_output_shapes
:
�
)Descriptor_pred/Sum_ang/Extract/sum_ang_2Sum)Descriptor_pred/Sum_ang/Extract/sl_pr_s_2;Descriptor_pred/Sum_ang/Extract/sum_ang_2/reduction_indices*
T0*,
_output_shapes
:����������*

Tidx0*
	keep_dims( 
l
'Descriptor_pred/Sum_ang/Extract/mul_2/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
�
%Descriptor_pred/Sum_ang/Extract/mul_2Mul'Descriptor_pred/Sum_ang/Extract/mul_2/x)Descriptor_pred/Sum_ang/Extract/sum_ang_2*
T0*,
_output_shapes
:����������
x
'Descriptor_pred/Sum_ang/Extract/Const_6Const*
valueB"      *
dtype0*
_output_shapes
:
s
1Descriptor_pred/Sum_ang/Extract/ExpandDims_12/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
-Descriptor_pred/Sum_ang/Extract/ExpandDims_12
ExpandDims'Descriptor_pred/Sum_ang/Extract/Const_61Descriptor_pred/Sum_ang/Extract/ExpandDims_12/dim*

Tdim0*
T0*
_output_shapes

:
s
1Descriptor_pred/Sum_ang/Extract/ExpandDims_13/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
-Descriptor_pred/Sum_ang/Extract/ExpandDims_13
ExpandDims-Descriptor_pred/Sum_ang/Extract/ExpandDims_121Descriptor_pred/Sum_ang/Extract/ExpandDims_13/dim*

Tdim0*
T0*"
_output_shapes
:
s
1Descriptor_pred/Sum_ang/Extract/ExpandDims_14/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
-Descriptor_pred/Sum_ang/Extract/ExpandDims_14
ExpandDims-Descriptor_pred/Sum_ang/Extract/ExpandDims_131Descriptor_pred/Sum_ang/Extract/ExpandDims_14/dim*&
_output_shapes
:*

Tdim0*
T0
�
'Descriptor_pred/Sum_ang/Extract/Shape_3ShapeInputs_pred/IteratorGetNext:1*
T0*
out_type0*
_output_shapes
:

5Descriptor_pred/Sum_ang/Extract/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
7Descriptor_pred/Sum_ang/Extract/strided_slice_3/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
7Descriptor_pred/Sum_ang/Extract/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
/Descriptor_pred/Sum_ang/Extract/strided_slice_3StridedSlice'Descriptor_pred/Sum_ang/Extract/Shape_35Descriptor_pred/Sum_ang/Extract/strided_slice_3/stack7Descriptor_pred/Sum_ang/Extract/strided_slice_3/stack_17Descriptor_pred/Sum_ang/Extract/strided_slice_3/stack_2*
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
{
9Descriptor_pred/Sum_ang/Extract/expand_pair_3/multiples/1Const*
value	B :*
dtype0*
_output_shapes
: 
{
9Descriptor_pred/Sum_ang/Extract/expand_pair_3/multiples/2Const*
value	B :*
dtype0*
_output_shapes
: 
{
9Descriptor_pred/Sum_ang/Extract/expand_pair_3/multiples/3Const*
value	B :*
dtype0*
_output_shapes
: 
�
7Descriptor_pred/Sum_ang/Extract/expand_pair_3/multiplesPack/Descriptor_pred/Sum_ang/Extract/strided_slice_39Descriptor_pred/Sum_ang/Extract/expand_pair_3/multiples/19Descriptor_pred/Sum_ang/Extract/expand_pair_3/multiples/29Descriptor_pred/Sum_ang/Extract/expand_pair_3/multiples/3*
T0*

axis *
N*
_output_shapes
:
�
-Descriptor_pred/Sum_ang/Extract/expand_pair_3Tile-Descriptor_pred/Sum_ang/Extract/ExpandDims_147Descriptor_pred/Sum_ang/Extract/expand_pair_3/multiples*

Tmultiples0*
T0*/
_output_shapes
:���������
�
'Descriptor_pred/Sum_ang/Extract/Equal_3Equal-Descriptor_pred/Sum_ang/Extract/expand_pair_3Descriptor_pred/Sum_ang/TopKV2*
T0*/
_output_shapes
:���������
i
'Descriptor_pred/Sum_ang/Extract/Const_7Const*
dtype0*
_output_shapes
: *
value	B :
|
1Descriptor_pred/Sum_ang/Extract/split_3/split_dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
'Descriptor_pred/Sum_ang/Extract/split_3Split1Descriptor_pred/Sum_ang/Extract/split_3/split_dim'Descriptor_pred/Sum_ang/Extract/Equal_3*
T0
*
	num_split*J
_output_shapes8
6:���������:���������
�
,Descriptor_pred/Sum_ang/Extract/LogicalAnd_6
LogicalAnd'Descriptor_pred/Sum_ang/Extract/split_3)Descriptor_pred/Sum_ang/Extract/split_3:1*/
_output_shapes
:���������
{
1Descriptor_pred/Sum_ang/Extract/ExpandDims_15/dimConst*
valueB:*
dtype0*
_output_shapes
:
�
-Descriptor_pred/Sum_ang/Extract/ExpandDims_15
ExpandDims,Descriptor_pred/Sum_ang/Extract/LogicalAnd_61Descriptor_pred/Sum_ang/Extract/ExpandDims_15/dim*
T0
*3
_output_shapes!
:���������*

Tdim0
�
0Descriptor_pred/Sum_ang/Extract/Tile_6/multiplesConst*)
value B"               *
dtype0*
_output_shapes
:
�
&Descriptor_pred/Sum_ang/Extract/Tile_6Tile-Descriptor_pred/Sum_ang/Extract/ExpandDims_150Descriptor_pred/Sum_ang/Extract/Tile_6/multiples*
T0
*3
_output_shapes!
:���������*

Tmultiples0
�
,Descriptor_pred/Sum_ang/Extract/LogicalAnd_7
LogicalAnd&Descriptor_pred/Sum_ang/Extract/Tile_6$Descriptor_pred/Sum_ang/ExpandDims_6*3
_output_shapes!
:���������
�
0Descriptor_pred/Sum_ang/Extract/Tile_7/multiplesConst*)
value B"            �   *
dtype0*
_output_shapes
:
�
&Descriptor_pred/Sum_ang/Extract/Tile_7Tile,Descriptor_pred/Sum_ang/Extract/LogicalAnd_70Descriptor_pred/Sum_ang/Extract/Tile_7/multiples*4
_output_shapes"
 :����������*

Tmultiples0*
T0

�
)Descriptor_pred/Sum_ang/Extract/sl_pr_s_3Select&Descriptor_pred/Sum_ang/Extract/Tile_7-Descriptor_pred/Angular_part/Ang_term/Reshape"Descriptor_pred/Sum_ang/zero_large*4
_output_shapes"
 :����������*
T0
�
;Descriptor_pred/Sum_ang/Extract/sum_ang_3/reduction_indicesConst*
valueB"      *
dtype0*
_output_shapes
:
�
)Descriptor_pred/Sum_ang/Extract/sum_ang_3Sum)Descriptor_pred/Sum_ang/Extract/sl_pr_s_3;Descriptor_pred/Sum_ang/Extract/sum_ang_3/reduction_indices*

Tidx0*
	keep_dims( *
T0*,
_output_shapes
:����������
l
'Descriptor_pred/Sum_ang/Extract/mul_3/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
�
%Descriptor_pred/Sum_ang/Extract/mul_3Mul'Descriptor_pred/Sum_ang/Extract/mul_3/x)Descriptor_pred/Sum_ang/Extract/sum_ang_3*
T0*,
_output_shapes
:����������
x
'Descriptor_pred/Sum_ang/Extract/Const_8Const*
valueB"      *
dtype0*
_output_shapes
:
s
1Descriptor_pred/Sum_ang/Extract/ExpandDims_16/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
-Descriptor_pred/Sum_ang/Extract/ExpandDims_16
ExpandDims'Descriptor_pred/Sum_ang/Extract/Const_81Descriptor_pred/Sum_ang/Extract/ExpandDims_16/dim*
T0*
_output_shapes

:*

Tdim0
s
1Descriptor_pred/Sum_ang/Extract/ExpandDims_17/dimConst*
dtype0*
_output_shapes
: *
value	B : 
�
-Descriptor_pred/Sum_ang/Extract/ExpandDims_17
ExpandDims-Descriptor_pred/Sum_ang/Extract/ExpandDims_161Descriptor_pred/Sum_ang/Extract/ExpandDims_17/dim*

Tdim0*
T0*"
_output_shapes
:
s
1Descriptor_pred/Sum_ang/Extract/ExpandDims_18/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
-Descriptor_pred/Sum_ang/Extract/ExpandDims_18
ExpandDims-Descriptor_pred/Sum_ang/Extract/ExpandDims_171Descriptor_pred/Sum_ang/Extract/ExpandDims_18/dim*
T0*&
_output_shapes
:*

Tdim0
�
'Descriptor_pred/Sum_ang/Extract/Shape_4ShapeInputs_pred/IteratorGetNext:1*
T0*
out_type0*
_output_shapes
:

5Descriptor_pred/Sum_ang/Extract/strided_slice_4/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
7Descriptor_pred/Sum_ang/Extract/strided_slice_4/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
7Descriptor_pred/Sum_ang/Extract/strided_slice_4/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
/Descriptor_pred/Sum_ang/Extract/strided_slice_4StridedSlice'Descriptor_pred/Sum_ang/Extract/Shape_45Descriptor_pred/Sum_ang/Extract/strided_slice_4/stack7Descriptor_pred/Sum_ang/Extract/strided_slice_4/stack_17Descriptor_pred/Sum_ang/Extract/strided_slice_4/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
{
9Descriptor_pred/Sum_ang/Extract/expand_pair_4/multiples/1Const*
value	B :*
dtype0*
_output_shapes
: 
{
9Descriptor_pred/Sum_ang/Extract/expand_pair_4/multiples/2Const*
value	B :*
dtype0*
_output_shapes
: 
{
9Descriptor_pred/Sum_ang/Extract/expand_pair_4/multiples/3Const*
value	B :*
dtype0*
_output_shapes
: 
�
7Descriptor_pred/Sum_ang/Extract/expand_pair_4/multiplesPack/Descriptor_pred/Sum_ang/Extract/strided_slice_49Descriptor_pred/Sum_ang/Extract/expand_pair_4/multiples/19Descriptor_pred/Sum_ang/Extract/expand_pair_4/multiples/29Descriptor_pred/Sum_ang/Extract/expand_pair_4/multiples/3*
T0*

axis *
N*
_output_shapes
:
�
-Descriptor_pred/Sum_ang/Extract/expand_pair_4Tile-Descriptor_pred/Sum_ang/Extract/ExpandDims_187Descriptor_pred/Sum_ang/Extract/expand_pair_4/multiples*

Tmultiples0*
T0*/
_output_shapes
:���������
�
'Descriptor_pred/Sum_ang/Extract/Equal_4Equal-Descriptor_pred/Sum_ang/Extract/expand_pair_4Descriptor_pred/Sum_ang/TopKV2*/
_output_shapes
:���������*
T0
i
'Descriptor_pred/Sum_ang/Extract/Const_9Const*
dtype0*
_output_shapes
: *
value	B :
|
1Descriptor_pred/Sum_ang/Extract/split_4/split_dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
'Descriptor_pred/Sum_ang/Extract/split_4Split1Descriptor_pred/Sum_ang/Extract/split_4/split_dim'Descriptor_pred/Sum_ang/Extract/Equal_4*
T0
*
	num_split*J
_output_shapes8
6:���������:���������
�
,Descriptor_pred/Sum_ang/Extract/LogicalAnd_8
LogicalAnd'Descriptor_pred/Sum_ang/Extract/split_4)Descriptor_pred/Sum_ang/Extract/split_4:1*/
_output_shapes
:���������
{
1Descriptor_pred/Sum_ang/Extract/ExpandDims_19/dimConst*
valueB:*
dtype0*
_output_shapes
:
�
-Descriptor_pred/Sum_ang/Extract/ExpandDims_19
ExpandDims,Descriptor_pred/Sum_ang/Extract/LogicalAnd_81Descriptor_pred/Sum_ang/Extract/ExpandDims_19/dim*
T0
*3
_output_shapes!
:���������*

Tdim0
�
0Descriptor_pred/Sum_ang/Extract/Tile_8/multiplesConst*)
value B"               *
dtype0*
_output_shapes
:
�
&Descriptor_pred/Sum_ang/Extract/Tile_8Tile-Descriptor_pred/Sum_ang/Extract/ExpandDims_190Descriptor_pred/Sum_ang/Extract/Tile_8/multiples*3
_output_shapes!
:���������*

Tmultiples0*
T0

�
,Descriptor_pred/Sum_ang/Extract/LogicalAnd_9
LogicalAnd&Descriptor_pred/Sum_ang/Extract/Tile_8$Descriptor_pred/Sum_ang/ExpandDims_6*3
_output_shapes!
:���������
�
0Descriptor_pred/Sum_ang/Extract/Tile_9/multiplesConst*)
value B"            �   *
dtype0*
_output_shapes
:
�
&Descriptor_pred/Sum_ang/Extract/Tile_9Tile,Descriptor_pred/Sum_ang/Extract/LogicalAnd_90Descriptor_pred/Sum_ang/Extract/Tile_9/multiples*

Tmultiples0*
T0
*4
_output_shapes"
 :����������
�
)Descriptor_pred/Sum_ang/Extract/sl_pr_s_4Select&Descriptor_pred/Sum_ang/Extract/Tile_9-Descriptor_pred/Angular_part/Ang_term/Reshape"Descriptor_pred/Sum_ang/zero_large*
T0*4
_output_shapes"
 :����������
�
;Descriptor_pred/Sum_ang/Extract/sum_ang_4/reduction_indicesConst*
valueB"      *
dtype0*
_output_shapes
:
�
)Descriptor_pred/Sum_ang/Extract/sum_ang_4Sum)Descriptor_pred/Sum_ang/Extract/sl_pr_s_4;Descriptor_pred/Sum_ang/Extract/sum_ang_4/reduction_indices*,
_output_shapes
:����������*

Tidx0*
	keep_dims( *
T0
l
'Descriptor_pred/Sum_ang/Extract/mul_4/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
�
%Descriptor_pred/Sum_ang/Extract/mul_4Mul'Descriptor_pred/Sum_ang/Extract/mul_4/x)Descriptor_pred/Sum_ang/Extract/sum_ang_4*,
_output_shapes
:����������*
T0
y
(Descriptor_pred/Sum_ang/Extract/Const_10Const*
dtype0*
_output_shapes
:*
valueB"      
s
1Descriptor_pred/Sum_ang/Extract/ExpandDims_20/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
-Descriptor_pred/Sum_ang/Extract/ExpandDims_20
ExpandDims(Descriptor_pred/Sum_ang/Extract/Const_101Descriptor_pred/Sum_ang/Extract/ExpandDims_20/dim*
T0*
_output_shapes

:*

Tdim0
s
1Descriptor_pred/Sum_ang/Extract/ExpandDims_21/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
-Descriptor_pred/Sum_ang/Extract/ExpandDims_21
ExpandDims-Descriptor_pred/Sum_ang/Extract/ExpandDims_201Descriptor_pred/Sum_ang/Extract/ExpandDims_21/dim*
T0*"
_output_shapes
:*

Tdim0
s
1Descriptor_pred/Sum_ang/Extract/ExpandDims_22/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
-Descriptor_pred/Sum_ang/Extract/ExpandDims_22
ExpandDims-Descriptor_pred/Sum_ang/Extract/ExpandDims_211Descriptor_pred/Sum_ang/Extract/ExpandDims_22/dim*

Tdim0*
T0*&
_output_shapes
:
�
'Descriptor_pred/Sum_ang/Extract/Shape_5ShapeInputs_pred/IteratorGetNext:1*
T0*
out_type0*
_output_shapes
:

5Descriptor_pred/Sum_ang/Extract/strided_slice_5/stackConst*
dtype0*
_output_shapes
:*
valueB: 
�
7Descriptor_pred/Sum_ang/Extract/strided_slice_5/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
�
7Descriptor_pred/Sum_ang/Extract/strided_slice_5/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
/Descriptor_pred/Sum_ang/Extract/strided_slice_5StridedSlice'Descriptor_pred/Sum_ang/Extract/Shape_55Descriptor_pred/Sum_ang/Extract/strided_slice_5/stack7Descriptor_pred/Sum_ang/Extract/strided_slice_5/stack_17Descriptor_pred/Sum_ang/Extract/strided_slice_5/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
{
9Descriptor_pred/Sum_ang/Extract/expand_pair_5/multiples/1Const*
value	B :*
dtype0*
_output_shapes
: 
{
9Descriptor_pred/Sum_ang/Extract/expand_pair_5/multiples/2Const*
dtype0*
_output_shapes
: *
value	B :
{
9Descriptor_pred/Sum_ang/Extract/expand_pair_5/multiples/3Const*
value	B :*
dtype0*
_output_shapes
: 
�
7Descriptor_pred/Sum_ang/Extract/expand_pair_5/multiplesPack/Descriptor_pred/Sum_ang/Extract/strided_slice_59Descriptor_pred/Sum_ang/Extract/expand_pair_5/multiples/19Descriptor_pred/Sum_ang/Extract/expand_pair_5/multiples/29Descriptor_pred/Sum_ang/Extract/expand_pair_5/multiples/3*
T0*

axis *
N*
_output_shapes
:
�
-Descriptor_pred/Sum_ang/Extract/expand_pair_5Tile-Descriptor_pred/Sum_ang/Extract/ExpandDims_227Descriptor_pred/Sum_ang/Extract/expand_pair_5/multiples*
T0*/
_output_shapes
:���������*

Tmultiples0
�
'Descriptor_pred/Sum_ang/Extract/Equal_5Equal-Descriptor_pred/Sum_ang/Extract/expand_pair_5Descriptor_pred/Sum_ang/TopKV2*
T0*/
_output_shapes
:���������
j
(Descriptor_pred/Sum_ang/Extract/Const_11Const*
value	B :*
dtype0*
_output_shapes
: 
|
1Descriptor_pred/Sum_ang/Extract/split_5/split_dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
'Descriptor_pred/Sum_ang/Extract/split_5Split1Descriptor_pred/Sum_ang/Extract/split_5/split_dim'Descriptor_pred/Sum_ang/Extract/Equal_5*
T0
*
	num_split*J
_output_shapes8
6:���������:���������
�
-Descriptor_pred/Sum_ang/Extract/LogicalAnd_10
LogicalAnd'Descriptor_pred/Sum_ang/Extract/split_5)Descriptor_pred/Sum_ang/Extract/split_5:1*/
_output_shapes
:���������
{
1Descriptor_pred/Sum_ang/Extract/ExpandDims_23/dimConst*
valueB:*
dtype0*
_output_shapes
:
�
-Descriptor_pred/Sum_ang/Extract/ExpandDims_23
ExpandDims-Descriptor_pred/Sum_ang/Extract/LogicalAnd_101Descriptor_pred/Sum_ang/Extract/ExpandDims_23/dim*

Tdim0*
T0
*3
_output_shapes!
:���������
�
1Descriptor_pred/Sum_ang/Extract/Tile_10/multiplesConst*
dtype0*
_output_shapes
:*)
value B"               
�
'Descriptor_pred/Sum_ang/Extract/Tile_10Tile-Descriptor_pred/Sum_ang/Extract/ExpandDims_231Descriptor_pred/Sum_ang/Extract/Tile_10/multiples*

Tmultiples0*
T0
*3
_output_shapes!
:���������
�
-Descriptor_pred/Sum_ang/Extract/LogicalAnd_11
LogicalAnd'Descriptor_pred/Sum_ang/Extract/Tile_10$Descriptor_pred/Sum_ang/ExpandDims_6*3
_output_shapes!
:���������
�
1Descriptor_pred/Sum_ang/Extract/Tile_11/multiplesConst*)
value B"            �   *
dtype0*
_output_shapes
:
�
'Descriptor_pred/Sum_ang/Extract/Tile_11Tile-Descriptor_pred/Sum_ang/Extract/LogicalAnd_111Descriptor_pred/Sum_ang/Extract/Tile_11/multiples*

Tmultiples0*
T0
*4
_output_shapes"
 :����������
�
)Descriptor_pred/Sum_ang/Extract/sl_pr_s_5Select'Descriptor_pred/Sum_ang/Extract/Tile_11-Descriptor_pred/Angular_part/Ang_term/Reshape"Descriptor_pred/Sum_ang/zero_large*
T0*4
_output_shapes"
 :����������
�
;Descriptor_pred/Sum_ang/Extract/sum_ang_5/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB"      
�
)Descriptor_pred/Sum_ang/Extract/sum_ang_5Sum)Descriptor_pred/Sum_ang/Extract/sl_pr_s_5;Descriptor_pred/Sum_ang/Extract/sum_ang_5/reduction_indices*
T0*,
_output_shapes
:����������*

Tidx0*
	keep_dims( 
l
'Descriptor_pred/Sum_ang/Extract/mul_5/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
�
%Descriptor_pred/Sum_ang/Extract/mul_5Mul'Descriptor_pred/Sum_ang/Extract/mul_5/x)Descriptor_pred/Sum_ang/Extract/sum_ang_5*
T0*,
_output_shapes
:����������
u
*Descriptor_pred/Sum_ang/concat_presum/axisConst*
dtype0*
_output_shapes
: *
valueB :
���������
�
%Descriptor_pred/Sum_ang/concat_presumConcatV2#Descriptor_pred/Sum_ang/Extract/mul%Descriptor_pred/Sum_ang/Extract/mul_1%Descriptor_pred/Sum_ang/Extract/mul_2%Descriptor_pred/Sum_ang/Extract/mul_3%Descriptor_pred/Sum_ang/Extract/mul_4%Descriptor_pred/Sum_ang/Extract/mul_5*Descriptor_pred/Sum_ang/concat_presum/axis*
N*,
_output_shapes
:����������
*

Tidx0*
T0
a
Descriptor_pred/Sum_ang/Const_2Const*
value	B : *
dtype0*
_output_shapes
: 
�
Descriptor_pred/Sum_ang/EqualEqualInputs_pred/IteratorGetNext:1Descriptor_pred/Sum_ang/Const_2*'
_output_shapes
:���������*
T0
x
"Descriptor_pred/Sum_ang/LogicalNot
LogicalNotDescriptor_pred/Sum_ang/Equal*'
_output_shapes
:���������
s
(Descriptor_pred/Sum_ang/ExpandDims_7/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
$Descriptor_pred/Sum_ang/ExpandDims_7
ExpandDims"Descriptor_pred/Sum_ang/LogicalNot(Descriptor_pred/Sum_ang/ExpandDims_7/dim*

Tdim0*
T0
*+
_output_shapes
:���������
}
(Descriptor_pred/Sum_ang/Tile_4/multiplesConst*!
valueB"      F  *
dtype0*
_output_shapes
:
�
Descriptor_pred/Sum_ang/Tile_4Tile$Descriptor_pred/Sum_ang/ExpandDims_7(Descriptor_pred/Sum_ang/Tile_4/multiples*
T0
*,
_output_shapes
:����������
*

Tmultiples0
�
Descriptor_pred/Sum_ang/Shape_4Shape%Descriptor_pred/Sum_ang/concat_presum*
T0*
out_type0*
_output_shapes
:
j
%Descriptor_pred/Sum_ang/zeros_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
Descriptor_pred/Sum_ang/zeros_1FillDescriptor_pred/Sum_ang/Shape_4%Descriptor_pred/Sum_ang/zeros_1/Const*
T0*

index_type0*,
_output_shapes
:����������

�
 Descriptor_pred/Sum_ang/Select_1SelectDescriptor_pred/Sum_ang/Tile_4%Descriptor_pred/Sum_ang/concat_presumDescriptor_pred/Sum_ang/zeros_1*
T0*,
_output_shapes
:����������

i
Descriptor_pred/ACSF/acsf/axisConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Descriptor_pred/ACSF/acsfConcatV2 Descriptor_pred/Sum_rad/Select_3 Descriptor_pred/Sum_ang/Select_1Descriptor_pred/ACSF/acsf/axis*
T0*
N*,
_output_shapes
:����������
*

Tidx0
x
Model_pred/zeros_like/ShapeShapeInputs_pred/IteratorGetNext:1*
T0*
out_type0*
_output_shapes
:
`
Model_pred/zeros_like/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Model_pred/zeros_likeFillModel_pred/zeros_like/ShapeModel_pred/zeros_like/Const*
T0*

index_type0*'
_output_shapes
:���������
R
Model_pred/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
[
Model_pred/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/ExpandDims
ExpandDimsModel_pred/ConstModel_pred/ExpandDims/dim*
_output_shapes
:*

Tdim0*
T0
�
Model_pred/EqualEqualInputs_pred/IteratorGetNext:1Model_pred/ExpandDims*
T0*'
_output_shapes
:���������
]
Model_pred/WhereWhereModel_pred/Equal*
T0
*'
_output_shapes
:���������
j
Model_pred/CastCastModel_pred/Where*

DstT0*'
_output_shapes
:���������*

SrcT0	
�
Model_pred/GatherNdGatherNdDescriptor_pred/ACSF/acsfModel_pred/Cast*(
_output_shapes
:����������
*
Tindices0*
Tparams0
Z
Model_pred/transpose/RankRankWeights/weight_in/read*
T0*
_output_shapes
: 
\
Model_pred/transpose/sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
w
Model_pred/transpose/subSubModel_pred/transpose/RankModel_pred/transpose/sub/y*
T0*
_output_shapes
: 
b
 Model_pred/transpose/Range/startConst*
value	B : *
dtype0*
_output_shapes
: 
b
 Model_pred/transpose/Range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Model_pred/transpose/RangeRange Model_pred/transpose/Range/startModel_pred/transpose/Rank Model_pred/transpose/Range/delta*
_output_shapes
:*

Tidx0
|
Model_pred/transpose/sub_1SubModel_pred/transpose/subModel_pred/transpose/Range*
T0*
_output_shapes
:
�
Model_pred/transpose	TransposeWeights/weight_in/readModel_pred/transpose/sub_1*
_output_shapes
:	�
F*
Tperm0*
T0
m
Model_pred/Tensordot/ShapeShapeModel_pred/GatherNd*
T0*
out_type0*
_output_shapes
:
[
Model_pred/Tensordot/RankConst*
value	B :*
dtype0*
_output_shapes
: 
c
Model_pred/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:
e
#Model_pred/Tensordot/GreaterEqual/yConst*
value	B : *
dtype0*
_output_shapes
: 
�
!Model_pred/Tensordot/GreaterEqualGreaterEqualModel_pred/Tensordot/axes#Model_pred/Tensordot/GreaterEqual/y*
T0*
_output_shapes
:
x
Model_pred/Tensordot/CastCast!Model_pred/Tensordot/GreaterEqual*

SrcT0
*

DstT0*
_output_shapes
:
z
Model_pred/Tensordot/mulMulModel_pred/Tensordot/CastModel_pred/Tensordot/axes*
_output_shapes
:*
T0
]
Model_pred/Tensordot/Less/yConst*
value	B : *
dtype0*
_output_shapes
: 
~
Model_pred/Tensordot/LessLessModel_pred/Tensordot/axesModel_pred/Tensordot/Less/y*
T0*
_output_shapes
:
r
Model_pred/Tensordot/Cast_1CastModel_pred/Tensordot/Less*

DstT0*
_output_shapes
:*

SrcT0

z
Model_pred/Tensordot/addAddModel_pred/Tensordot/axesModel_pred/Tensordot/Rank*
T0*
_output_shapes
:
}
Model_pred/Tensordot/mul_1MulModel_pred/Tensordot/Cast_1Model_pred/Tensordot/add*
T0*
_output_shapes
:
|
Model_pred/Tensordot/add_1AddModel_pred/Tensordot/mulModel_pred/Tensordot/mul_1*
T0*
_output_shapes
:
b
 Model_pred/Tensordot/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
b
 Model_pred/Tensordot/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot/rangeRange Model_pred/Tensordot/range/startModel_pred/Tensordot/Rank Model_pred/Tensordot/range/delta*
_output_shapes
:*

Tidx0
�
Model_pred/Tensordot/ListDiffListDiffModel_pred/Tensordot/rangeModel_pred/Tensordot/add_1*
T0*
out_idx0*2
_output_shapes 
:���������:���������
d
"Model_pred/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot/GatherV2GatherV2Model_pred/Tensordot/ShapeModel_pred/Tensordot/ListDiff"Model_pred/Tensordot/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0*#
_output_shapes
:���������
f
$Model_pred/Tensordot/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
Model_pred/Tensordot/GatherV2_1GatherV2Model_pred/Tensordot/ShapeModel_pred/Tensordot/add_1$Model_pred/Tensordot/GatherV2_1/axis*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0
d
Model_pred/Tensordot/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
Model_pred/Tensordot/ProdProdModel_pred/Tensordot/GatherV2Model_pred/Tensordot/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
f
Model_pred/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
Model_pred/Tensordot/Prod_1ProdModel_pred/Tensordot/GatherV2_1Model_pred/Tensordot/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
b
 Model_pred/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot/concatConcatV2Model_pred/Tensordot/GatherV2_1Model_pred/Tensordot/GatherV2 Model_pred/Tensordot/concat/axis*
N*#
_output_shapes
:���������*

Tidx0*
T0
d
"Model_pred/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot/concat_1ConcatV2Model_pred/Tensordot/ListDiffModel_pred/Tensordot/add_1"Model_pred/Tensordot/concat_1/axis*
N*#
_output_shapes
:���������*

Tidx0*
T0
�
Model_pred/Tensordot/stackPackModel_pred/Tensordot/ProdModel_pred/Tensordot/Prod_1*
T0*

axis *
N*
_output_shapes
:
�
Model_pred/Tensordot/transpose	TransposeModel_pred/GatherNdModel_pred/Tensordot/concat_1*0
_output_shapes
:������������������*
Tperm0*
T0
�
Model_pred/Tensordot/ReshapeReshapeModel_pred/Tensordot/transposeModel_pred/Tensordot/stack*0
_output_shapes
:������������������*
T0*
Tshape0
v
%Model_pred/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
�
 Model_pred/Tensordot/transpose_1	TransposeModel_pred/transpose%Model_pred/Tensordot/transpose_1/perm*
_output_shapes
:	�
F*
Tperm0*
T0
u
$Model_pred/Tensordot/Reshape_1/shapeConst*
valueB"s  F   *
dtype0*
_output_shapes
:
�
Model_pred/Tensordot/Reshape_1Reshape Model_pred/Tensordot/transpose_1$Model_pred/Tensordot/Reshape_1/shape*
_output_shapes
:	�
F*
T0*
Tshape0
�
Model_pred/Tensordot/MatMulMatMulModel_pred/Tensordot/ReshapeModel_pred/Tensordot/Reshape_1*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:���������F
f
Model_pred/Tensordot/Const_2Const*
valueB:F*
dtype0*
_output_shapes
:
d
"Model_pred/Tensordot/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot/concat_2ConcatV2Model_pred/Tensordot/GatherV2Model_pred/Tensordot/Const_2"Model_pred/Tensordot/concat_2/axis*
N*#
_output_shapes
:���������*

Tidx0*
T0
�
Model_pred/TensordotReshapeModel_pred/Tensordot/MatMulModel_pred/Tensordot/concat_2*
T0*
Tshape0*'
_output_shapes
:���������F
s
Model_pred/AddAddModel_pred/TensordotWeights/bias_in/read*'
_output_shapes
:���������F*
T0
_
Model_pred/SigmoidSigmoidModel_pred/Add*
T0*'
_output_shapes
:���������F
b
Model_pred/transpose_1/RankRankWeights/weight_hidden_1/read*
T0*
_output_shapes
: 
^
Model_pred/transpose_1/sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
}
Model_pred/transpose_1/subSubModel_pred/transpose_1/RankModel_pred/transpose_1/sub/y*
_output_shapes
: *
T0
d
"Model_pred/transpose_1/Range/startConst*
value	B : *
dtype0*
_output_shapes
: 
d
"Model_pred/transpose_1/Range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Model_pred/transpose_1/RangeRange"Model_pred/transpose_1/Range/startModel_pred/transpose_1/Rank"Model_pred/transpose_1/Range/delta*

Tidx0*
_output_shapes
:
�
Model_pred/transpose_1/sub_1SubModel_pred/transpose_1/subModel_pred/transpose_1/Range*
T0*
_output_shapes
:
�
Model_pred/transpose_1	TransposeWeights/weight_hidden_1/readModel_pred/transpose_1/sub_1*
T0*
_output_shapes

:Fh*
Tperm0
n
Model_pred/Tensordot_1/ShapeShapeModel_pred/Sigmoid*
T0*
out_type0*
_output_shapes
:
]
Model_pred/Tensordot_1/RankConst*
value	B :*
dtype0*
_output_shapes
: 
e
Model_pred/Tensordot_1/axesConst*
valueB:*
dtype0*
_output_shapes
:
g
%Model_pred/Tensordot_1/GreaterEqual/yConst*
value	B : *
dtype0*
_output_shapes
: 
�
#Model_pred/Tensordot_1/GreaterEqualGreaterEqualModel_pred/Tensordot_1/axes%Model_pred/Tensordot_1/GreaterEqual/y*
T0*
_output_shapes
:
|
Model_pred/Tensordot_1/CastCast#Model_pred/Tensordot_1/GreaterEqual*

DstT0*
_output_shapes
:*

SrcT0

�
Model_pred/Tensordot_1/mulMulModel_pred/Tensordot_1/CastModel_pred/Tensordot_1/axes*
T0*
_output_shapes
:
_
Model_pred/Tensordot_1/Less/yConst*
dtype0*
_output_shapes
: *
value	B : 
�
Model_pred/Tensordot_1/LessLessModel_pred/Tensordot_1/axesModel_pred/Tensordot_1/Less/y*
T0*
_output_shapes
:
v
Model_pred/Tensordot_1/Cast_1CastModel_pred/Tensordot_1/Less*

SrcT0
*

DstT0*
_output_shapes
:
�
Model_pred/Tensordot_1/addAddModel_pred/Tensordot_1/axesModel_pred/Tensordot_1/Rank*
T0*
_output_shapes
:
�
Model_pred/Tensordot_1/mul_1MulModel_pred/Tensordot_1/Cast_1Model_pred/Tensordot_1/add*
T0*
_output_shapes
:
�
Model_pred/Tensordot_1/add_1AddModel_pred/Tensordot_1/mulModel_pred/Tensordot_1/mul_1*
T0*
_output_shapes
:
d
"Model_pred/Tensordot_1/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
d
"Model_pred/Tensordot_1/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
�
Model_pred/Tensordot_1/rangeRange"Model_pred/Tensordot_1/range/startModel_pred/Tensordot_1/Rank"Model_pred/Tensordot_1/range/delta*

Tidx0*
_output_shapes
:
�
Model_pred/Tensordot_1/ListDiffListDiffModel_pred/Tensordot_1/rangeModel_pred/Tensordot_1/add_1*2
_output_shapes 
:���������:���������*
T0*
out_idx0
f
$Model_pred/Tensordot_1/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot_1/GatherV2GatherV2Model_pred/Tensordot_1/ShapeModel_pred/Tensordot_1/ListDiff$Model_pred/Tensordot_1/GatherV2/axis*#
_output_shapes
:���������*
Taxis0*
Tindices0*
Tparams0
h
&Model_pred/Tensordot_1/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
!Model_pred/Tensordot_1/GatherV2_1GatherV2Model_pred/Tensordot_1/ShapeModel_pred/Tensordot_1/add_1&Model_pred/Tensordot_1/GatherV2_1/axis*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0
f
Model_pred/Tensordot_1/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
Model_pred/Tensordot_1/ProdProdModel_pred/Tensordot_1/GatherV2Model_pred/Tensordot_1/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
h
Model_pred/Tensordot_1/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
Model_pred/Tensordot_1/Prod_1Prod!Model_pred/Tensordot_1/GatherV2_1Model_pred/Tensordot_1/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
d
"Model_pred/Tensordot_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot_1/concatConcatV2!Model_pred/Tensordot_1/GatherV2_1Model_pred/Tensordot_1/GatherV2"Model_pred/Tensordot_1/concat/axis*
T0*
N*#
_output_shapes
:���������*

Tidx0
f
$Model_pred/Tensordot_1/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot_1/concat_1ConcatV2Model_pred/Tensordot_1/ListDiffModel_pred/Tensordot_1/add_1$Model_pred/Tensordot_1/concat_1/axis*
N*#
_output_shapes
:���������*

Tidx0*
T0
�
Model_pred/Tensordot_1/stackPackModel_pred/Tensordot_1/ProdModel_pred/Tensordot_1/Prod_1*
T0*

axis *
N*
_output_shapes
:
�
 Model_pred/Tensordot_1/transpose	TransposeModel_pred/SigmoidModel_pred/Tensordot_1/concat_1*
T0*0
_output_shapes
:������������������*
Tperm0
�
Model_pred/Tensordot_1/ReshapeReshape Model_pred/Tensordot_1/transposeModel_pred/Tensordot_1/stack*
T0*
Tshape0*0
_output_shapes
:������������������
x
'Model_pred/Tensordot_1/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
�
"Model_pred/Tensordot_1/transpose_1	TransposeModel_pred/transpose_1'Model_pred/Tensordot_1/transpose_1/perm*
T0*
_output_shapes

:Fh*
Tperm0
w
&Model_pred/Tensordot_1/Reshape_1/shapeConst*
valueB"F   h   *
dtype0*
_output_shapes
:
�
 Model_pred/Tensordot_1/Reshape_1Reshape"Model_pred/Tensordot_1/transpose_1&Model_pred/Tensordot_1/Reshape_1/shape*
_output_shapes

:Fh*
T0*
Tshape0
�
Model_pred/Tensordot_1/MatMulMatMulModel_pred/Tensordot_1/Reshape Model_pred/Tensordot_1/Reshape_1*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:���������h
h
Model_pred/Tensordot_1/Const_2Const*
valueB:h*
dtype0*
_output_shapes
:
f
$Model_pred/Tensordot_1/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot_1/concat_2ConcatV2Model_pred/Tensordot_1/GatherV2Model_pred/Tensordot_1/Const_2$Model_pred/Tensordot_1/concat_2/axis*
N*#
_output_shapes
:���������*

Tidx0*
T0
�
Model_pred/Tensordot_1ReshapeModel_pred/Tensordot_1/MatMulModel_pred/Tensordot_1/concat_2*
T0*
Tshape0*'
_output_shapes
:���������h
}
Model_pred/Add_1AddModel_pred/Tensordot_1Weights/bias_hidden_1/read*
T0*'
_output_shapes
:���������h
c
Model_pred/Sigmoid_1SigmoidModel_pred/Add_1*
T0*'
_output_shapes
:���������h
]
Model_pred/transpose_2/RankRankWeights/weight_out/read*
T0*
_output_shapes
: 
^
Model_pred/transpose_2/sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
}
Model_pred/transpose_2/subSubModel_pred/transpose_2/RankModel_pred/transpose_2/sub/y*
T0*
_output_shapes
: 
d
"Model_pred/transpose_2/Range/startConst*
dtype0*
_output_shapes
: *
value	B : 
d
"Model_pred/transpose_2/Range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Model_pred/transpose_2/RangeRange"Model_pred/transpose_2/Range/startModel_pred/transpose_2/Rank"Model_pred/transpose_2/Range/delta*
_output_shapes
:*

Tidx0
�
Model_pred/transpose_2/sub_1SubModel_pred/transpose_2/subModel_pred/transpose_2/Range*
T0*
_output_shapes
:
�
Model_pred/transpose_2	TransposeWeights/weight_out/readModel_pred/transpose_2/sub_1*
_output_shapes

:h*
Tperm0*
T0
p
Model_pred/Tensordot_2/ShapeShapeModel_pred/Sigmoid_1*
T0*
out_type0*
_output_shapes
:
]
Model_pred/Tensordot_2/RankConst*
value	B :*
dtype0*
_output_shapes
: 
e
Model_pred/Tensordot_2/axesConst*
valueB:*
dtype0*
_output_shapes
:
g
%Model_pred/Tensordot_2/GreaterEqual/yConst*
dtype0*
_output_shapes
: *
value	B : 
�
#Model_pred/Tensordot_2/GreaterEqualGreaterEqualModel_pred/Tensordot_2/axes%Model_pred/Tensordot_2/GreaterEqual/y*
T0*
_output_shapes
:
|
Model_pred/Tensordot_2/CastCast#Model_pred/Tensordot_2/GreaterEqual*

SrcT0
*

DstT0*
_output_shapes
:
�
Model_pred/Tensordot_2/mulMulModel_pred/Tensordot_2/CastModel_pred/Tensordot_2/axes*
T0*
_output_shapes
:
_
Model_pred/Tensordot_2/Less/yConst*
dtype0*
_output_shapes
: *
value	B : 
�
Model_pred/Tensordot_2/LessLessModel_pred/Tensordot_2/axesModel_pred/Tensordot_2/Less/y*
T0*
_output_shapes
:
v
Model_pred/Tensordot_2/Cast_1CastModel_pred/Tensordot_2/Less*

SrcT0
*

DstT0*
_output_shapes
:
�
Model_pred/Tensordot_2/addAddModel_pred/Tensordot_2/axesModel_pred/Tensordot_2/Rank*
_output_shapes
:*
T0
�
Model_pred/Tensordot_2/mul_1MulModel_pred/Tensordot_2/Cast_1Model_pred/Tensordot_2/add*
T0*
_output_shapes
:
�
Model_pred/Tensordot_2/add_1AddModel_pred/Tensordot_2/mulModel_pred/Tensordot_2/mul_1*
T0*
_output_shapes
:
d
"Model_pred/Tensordot_2/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
d
"Model_pred/Tensordot_2/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot_2/rangeRange"Model_pred/Tensordot_2/range/startModel_pred/Tensordot_2/Rank"Model_pred/Tensordot_2/range/delta*
_output_shapes
:*

Tidx0
�
Model_pred/Tensordot_2/ListDiffListDiffModel_pred/Tensordot_2/rangeModel_pred/Tensordot_2/add_1*
T0*
out_idx0*2
_output_shapes 
:���������:���������
f
$Model_pred/Tensordot_2/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot_2/GatherV2GatherV2Model_pred/Tensordot_2/ShapeModel_pred/Tensordot_2/ListDiff$Model_pred/Tensordot_2/GatherV2/axis*
Tindices0*
Tparams0*#
_output_shapes
:���������*
Taxis0
h
&Model_pred/Tensordot_2/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
!Model_pred/Tensordot_2/GatherV2_1GatherV2Model_pred/Tensordot_2/ShapeModel_pred/Tensordot_2/add_1&Model_pred/Tensordot_2/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
f
Model_pred/Tensordot_2/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
Model_pred/Tensordot_2/ProdProdModel_pred/Tensordot_2/GatherV2Model_pred/Tensordot_2/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
h
Model_pred/Tensordot_2/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
Model_pred/Tensordot_2/Prod_1Prod!Model_pred/Tensordot_2/GatherV2_1Model_pred/Tensordot_2/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
d
"Model_pred/Tensordot_2/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
Model_pred/Tensordot_2/concatConcatV2!Model_pred/Tensordot_2/GatherV2_1Model_pred/Tensordot_2/GatherV2"Model_pred/Tensordot_2/concat/axis*
T0*
N*#
_output_shapes
:���������*

Tidx0
f
$Model_pred/Tensordot_2/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot_2/concat_1ConcatV2Model_pred/Tensordot_2/ListDiffModel_pred/Tensordot_2/add_1$Model_pred/Tensordot_2/concat_1/axis*
T0*
N*#
_output_shapes
:���������*

Tidx0
�
Model_pred/Tensordot_2/stackPackModel_pred/Tensordot_2/ProdModel_pred/Tensordot_2/Prod_1*
N*
_output_shapes
:*
T0*

axis 
�
 Model_pred/Tensordot_2/transpose	TransposeModel_pred/Sigmoid_1Model_pred/Tensordot_2/concat_1*
T0*0
_output_shapes
:������������������*
Tperm0
�
Model_pred/Tensordot_2/ReshapeReshape Model_pred/Tensordot_2/transposeModel_pred/Tensordot_2/stack*0
_output_shapes
:������������������*
T0*
Tshape0
x
'Model_pred/Tensordot_2/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
�
"Model_pred/Tensordot_2/transpose_1	TransposeModel_pred/transpose_2'Model_pred/Tensordot_2/transpose_1/perm*
T0*
_output_shapes

:h*
Tperm0
w
&Model_pred/Tensordot_2/Reshape_1/shapeConst*
valueB"h      *
dtype0*
_output_shapes
:
�
 Model_pred/Tensordot_2/Reshape_1Reshape"Model_pred/Tensordot_2/transpose_1&Model_pred/Tensordot_2/Reshape_1/shape*
_output_shapes

:h*
T0*
Tshape0
�
Model_pred/Tensordot_2/MatMulMatMulModel_pred/Tensordot_2/Reshape Model_pred/Tensordot_2/Reshape_1*
T0*
transpose_a( *'
_output_shapes
:���������*
transpose_b( 
h
Model_pred/Tensordot_2/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
f
$Model_pred/Tensordot_2/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot_2/concat_2ConcatV2Model_pred/Tensordot_2/GatherV2Model_pred/Tensordot_2/Const_2$Model_pred/Tensordot_2/concat_2/axis*

Tidx0*
T0*
N*#
_output_shapes
:���������
�
Model_pred/Tensordot_2ReshapeModel_pred/Tensordot_2/MatMulModel_pred/Tensordot_2/concat_2*
T0*
Tshape0*'
_output_shapes
:���������
x
Model_pred/Add_2AddModel_pred/Tensordot_2Weights/bias_out/read*
T0*'
_output_shapes
:���������
}
Model_pred/SqueezeSqueezeModel_pred/Add_2*
squeeze_dims

���������*
T0*#
_output_shapes
:���������
m
Model_pred/ShapeShapeInputs_pred/IteratorGetNext:1*
T0*
out_type0*
_output_shapes
:
�
Model_pred/ScatterNd	ScatterNdModel_pred/CastModel_pred/SqueezeModel_pred/Shape*'
_output_shapes
:���������*
Tindices0*
T0
v
Model_pred/Add_3AddModel_pred/zeros_likeModel_pred/ScatterNd*
T0*'
_output_shapes
:���������
T
Model_pred/Const_1Const*
dtype0*
_output_shapes
: *
value	B :
]
Model_pred/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/ExpandDims_1
ExpandDimsModel_pred/Const_1Model_pred/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes
:
�
Model_pred/Equal_1EqualInputs_pred/IteratorGetNext:1Model_pred/ExpandDims_1*
T0*'
_output_shapes
:���������
a
Model_pred/Where_1WhereModel_pred/Equal_1*
T0
*'
_output_shapes
:���������
n
Model_pred/Cast_1CastModel_pred/Where_1*

SrcT0	*

DstT0*'
_output_shapes
:���������
�
Model_pred/GatherNd_1GatherNdDescriptor_pred/ACSF/acsfModel_pred/Cast_1*
Tindices0*
Tparams0*(
_output_shapes
:����������

^
Model_pred/transpose_3/RankRankWeights/weight_in_1/read*
T0*
_output_shapes
: 
^
Model_pred/transpose_3/sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
}
Model_pred/transpose_3/subSubModel_pred/transpose_3/RankModel_pred/transpose_3/sub/y*
_output_shapes
: *
T0
d
"Model_pred/transpose_3/Range/startConst*
value	B : *
dtype0*
_output_shapes
: 
d
"Model_pred/transpose_3/Range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Model_pred/transpose_3/RangeRange"Model_pred/transpose_3/Range/startModel_pred/transpose_3/Rank"Model_pred/transpose_3/Range/delta*
_output_shapes
:*

Tidx0
�
Model_pred/transpose_3/sub_1SubModel_pred/transpose_3/subModel_pred/transpose_3/Range*
T0*
_output_shapes
:
�
Model_pred/transpose_3	TransposeWeights/weight_in_1/readModel_pred/transpose_3/sub_1*
T0*
_output_shapes
:	�
F*
Tperm0
q
Model_pred/Tensordot_3/ShapeShapeModel_pred/GatherNd_1*
T0*
out_type0*
_output_shapes
:
]
Model_pred/Tensordot_3/RankConst*
value	B :*
dtype0*
_output_shapes
: 
e
Model_pred/Tensordot_3/axesConst*
valueB:*
dtype0*
_output_shapes
:
g
%Model_pred/Tensordot_3/GreaterEqual/yConst*
value	B : *
dtype0*
_output_shapes
: 
�
#Model_pred/Tensordot_3/GreaterEqualGreaterEqualModel_pred/Tensordot_3/axes%Model_pred/Tensordot_3/GreaterEqual/y*
T0*
_output_shapes
:
|
Model_pred/Tensordot_3/CastCast#Model_pred/Tensordot_3/GreaterEqual*

SrcT0
*

DstT0*
_output_shapes
:
�
Model_pred/Tensordot_3/mulMulModel_pred/Tensordot_3/CastModel_pred/Tensordot_3/axes*
T0*
_output_shapes
:
_
Model_pred/Tensordot_3/Less/yConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot_3/LessLessModel_pred/Tensordot_3/axesModel_pred/Tensordot_3/Less/y*
T0*
_output_shapes
:
v
Model_pred/Tensordot_3/Cast_1CastModel_pred/Tensordot_3/Less*

SrcT0
*

DstT0*
_output_shapes
:
�
Model_pred/Tensordot_3/addAddModel_pred/Tensordot_3/axesModel_pred/Tensordot_3/Rank*
T0*
_output_shapes
:
�
Model_pred/Tensordot_3/mul_1MulModel_pred/Tensordot_3/Cast_1Model_pred/Tensordot_3/add*
T0*
_output_shapes
:
�
Model_pred/Tensordot_3/add_1AddModel_pred/Tensordot_3/mulModel_pred/Tensordot_3/mul_1*
_output_shapes
:*
T0
d
"Model_pred/Tensordot_3/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
d
"Model_pred/Tensordot_3/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot_3/rangeRange"Model_pred/Tensordot_3/range/startModel_pred/Tensordot_3/Rank"Model_pred/Tensordot_3/range/delta*
_output_shapes
:*

Tidx0
�
Model_pred/Tensordot_3/ListDiffListDiffModel_pred/Tensordot_3/rangeModel_pred/Tensordot_3/add_1*
T0*
out_idx0*2
_output_shapes 
:���������:���������
f
$Model_pred/Tensordot_3/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot_3/GatherV2GatherV2Model_pred/Tensordot_3/ShapeModel_pred/Tensordot_3/ListDiff$Model_pred/Tensordot_3/GatherV2/axis*
Tindices0*
Tparams0*#
_output_shapes
:���������*
Taxis0
h
&Model_pred/Tensordot_3/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
!Model_pred/Tensordot_3/GatherV2_1GatherV2Model_pred/Tensordot_3/ShapeModel_pred/Tensordot_3/add_1&Model_pred/Tensordot_3/GatherV2_1/axis*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0
f
Model_pred/Tensordot_3/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
Model_pred/Tensordot_3/ProdProdModel_pred/Tensordot_3/GatherV2Model_pred/Tensordot_3/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
h
Model_pred/Tensordot_3/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
Model_pred/Tensordot_3/Prod_1Prod!Model_pred/Tensordot_3/GatherV2_1Model_pred/Tensordot_3/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
d
"Model_pred/Tensordot_3/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot_3/concatConcatV2!Model_pred/Tensordot_3/GatherV2_1Model_pred/Tensordot_3/GatherV2"Model_pred/Tensordot_3/concat/axis*

Tidx0*
T0*
N*#
_output_shapes
:���������
f
$Model_pred/Tensordot_3/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
Model_pred/Tensordot_3/concat_1ConcatV2Model_pred/Tensordot_3/ListDiffModel_pred/Tensordot_3/add_1$Model_pred/Tensordot_3/concat_1/axis*

Tidx0*
T0*
N*#
_output_shapes
:���������
�
Model_pred/Tensordot_3/stackPackModel_pred/Tensordot_3/ProdModel_pred/Tensordot_3/Prod_1*
N*
_output_shapes
:*
T0*

axis 
�
 Model_pred/Tensordot_3/transpose	TransposeModel_pred/GatherNd_1Model_pred/Tensordot_3/concat_1*0
_output_shapes
:������������������*
Tperm0*
T0
�
Model_pred/Tensordot_3/ReshapeReshape Model_pred/Tensordot_3/transposeModel_pred/Tensordot_3/stack*
T0*
Tshape0*0
_output_shapes
:������������������
x
'Model_pred/Tensordot_3/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
�
"Model_pred/Tensordot_3/transpose_1	TransposeModel_pred/transpose_3'Model_pred/Tensordot_3/transpose_1/perm*
T0*
_output_shapes
:	�
F*
Tperm0
w
&Model_pred/Tensordot_3/Reshape_1/shapeConst*
valueB"s  F   *
dtype0*
_output_shapes
:
�
 Model_pred/Tensordot_3/Reshape_1Reshape"Model_pred/Tensordot_3/transpose_1&Model_pred/Tensordot_3/Reshape_1/shape*
T0*
Tshape0*
_output_shapes
:	�
F
�
Model_pred/Tensordot_3/MatMulMatMulModel_pred/Tensordot_3/Reshape Model_pred/Tensordot_3/Reshape_1*
transpose_a( *'
_output_shapes
:���������F*
transpose_b( *
T0
h
Model_pred/Tensordot_3/Const_2Const*
valueB:F*
dtype0*
_output_shapes
:
f
$Model_pred/Tensordot_3/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot_3/concat_2ConcatV2Model_pred/Tensordot_3/GatherV2Model_pred/Tensordot_3/Const_2$Model_pred/Tensordot_3/concat_2/axis*
N*#
_output_shapes
:���������*

Tidx0*
T0
�
Model_pred/Tensordot_3ReshapeModel_pred/Tensordot_3/MatMulModel_pred/Tensordot_3/concat_2*
T0*
Tshape0*'
_output_shapes
:���������F
y
Model_pred/Add_4AddModel_pred/Tensordot_3Weights/bias_in_1/read*'
_output_shapes
:���������F*
T0
c
Model_pred/Sigmoid_2SigmoidModel_pred/Add_4*'
_output_shapes
:���������F*
T0
d
Model_pred/transpose_4/RankRankWeights/weight_hidden_1_1/read*
T0*
_output_shapes
: 
^
Model_pred/transpose_4/sub/yConst*
dtype0*
_output_shapes
: *
value	B :
}
Model_pred/transpose_4/subSubModel_pred/transpose_4/RankModel_pred/transpose_4/sub/y*
_output_shapes
: *
T0
d
"Model_pred/transpose_4/Range/startConst*
dtype0*
_output_shapes
: *
value	B : 
d
"Model_pred/transpose_4/Range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Model_pred/transpose_4/RangeRange"Model_pred/transpose_4/Range/startModel_pred/transpose_4/Rank"Model_pred/transpose_4/Range/delta*

Tidx0*
_output_shapes
:
�
Model_pred/transpose_4/sub_1SubModel_pred/transpose_4/subModel_pred/transpose_4/Range*
T0*
_output_shapes
:
�
Model_pred/transpose_4	TransposeWeights/weight_hidden_1_1/readModel_pred/transpose_4/sub_1*
T0*
_output_shapes

:Fh*
Tperm0
p
Model_pred/Tensordot_4/ShapeShapeModel_pred/Sigmoid_2*
T0*
out_type0*
_output_shapes
:
]
Model_pred/Tensordot_4/RankConst*
dtype0*
_output_shapes
: *
value	B :
e
Model_pred/Tensordot_4/axesConst*
dtype0*
_output_shapes
:*
valueB:
g
%Model_pred/Tensordot_4/GreaterEqual/yConst*
value	B : *
dtype0*
_output_shapes
: 
�
#Model_pred/Tensordot_4/GreaterEqualGreaterEqualModel_pred/Tensordot_4/axes%Model_pred/Tensordot_4/GreaterEqual/y*
_output_shapes
:*
T0
|
Model_pred/Tensordot_4/CastCast#Model_pred/Tensordot_4/GreaterEqual*

SrcT0
*

DstT0*
_output_shapes
:
�
Model_pred/Tensordot_4/mulMulModel_pred/Tensordot_4/CastModel_pred/Tensordot_4/axes*
T0*
_output_shapes
:
_
Model_pred/Tensordot_4/Less/yConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot_4/LessLessModel_pred/Tensordot_4/axesModel_pred/Tensordot_4/Less/y*
T0*
_output_shapes
:
v
Model_pred/Tensordot_4/Cast_1CastModel_pred/Tensordot_4/Less*

SrcT0
*

DstT0*
_output_shapes
:
�
Model_pred/Tensordot_4/addAddModel_pred/Tensordot_4/axesModel_pred/Tensordot_4/Rank*
T0*
_output_shapes
:
�
Model_pred/Tensordot_4/mul_1MulModel_pred/Tensordot_4/Cast_1Model_pred/Tensordot_4/add*
_output_shapes
:*
T0
�
Model_pred/Tensordot_4/add_1AddModel_pred/Tensordot_4/mulModel_pred/Tensordot_4/mul_1*
T0*
_output_shapes
:
d
"Model_pred/Tensordot_4/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
d
"Model_pred/Tensordot_4/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot_4/rangeRange"Model_pred/Tensordot_4/range/startModel_pred/Tensordot_4/Rank"Model_pred/Tensordot_4/range/delta*
_output_shapes
:*

Tidx0
�
Model_pred/Tensordot_4/ListDiffListDiffModel_pred/Tensordot_4/rangeModel_pred/Tensordot_4/add_1*
T0*
out_idx0*2
_output_shapes 
:���������:���������
f
$Model_pred/Tensordot_4/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot_4/GatherV2GatherV2Model_pred/Tensordot_4/ShapeModel_pred/Tensordot_4/ListDiff$Model_pred/Tensordot_4/GatherV2/axis*
Tparams0*#
_output_shapes
:���������*
Taxis0*
Tindices0
h
&Model_pred/Tensordot_4/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
!Model_pred/Tensordot_4/GatherV2_1GatherV2Model_pred/Tensordot_4/ShapeModel_pred/Tensordot_4/add_1&Model_pred/Tensordot_4/GatherV2_1/axis*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0
f
Model_pred/Tensordot_4/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
Model_pred/Tensordot_4/ProdProdModel_pred/Tensordot_4/GatherV2Model_pred/Tensordot_4/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
h
Model_pred/Tensordot_4/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
Model_pred/Tensordot_4/Prod_1Prod!Model_pred/Tensordot_4/GatherV2_1Model_pred/Tensordot_4/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
d
"Model_pred/Tensordot_4/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot_4/concatConcatV2!Model_pred/Tensordot_4/GatherV2_1Model_pred/Tensordot_4/GatherV2"Model_pred/Tensordot_4/concat/axis*
T0*
N*#
_output_shapes
:���������*

Tidx0
f
$Model_pred/Tensordot_4/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot_4/concat_1ConcatV2Model_pred/Tensordot_4/ListDiffModel_pred/Tensordot_4/add_1$Model_pred/Tensordot_4/concat_1/axis*
N*#
_output_shapes
:���������*

Tidx0*
T0
�
Model_pred/Tensordot_4/stackPackModel_pred/Tensordot_4/ProdModel_pred/Tensordot_4/Prod_1*
T0*

axis *
N*
_output_shapes
:
�
 Model_pred/Tensordot_4/transpose	TransposeModel_pred/Sigmoid_2Model_pred/Tensordot_4/concat_1*
T0*0
_output_shapes
:������������������*
Tperm0
�
Model_pred/Tensordot_4/ReshapeReshape Model_pred/Tensordot_4/transposeModel_pred/Tensordot_4/stack*
T0*
Tshape0*0
_output_shapes
:������������������
x
'Model_pred/Tensordot_4/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
�
"Model_pred/Tensordot_4/transpose_1	TransposeModel_pred/transpose_4'Model_pred/Tensordot_4/transpose_1/perm*
_output_shapes

:Fh*
Tperm0*
T0
w
&Model_pred/Tensordot_4/Reshape_1/shapeConst*
valueB"F   h   *
dtype0*
_output_shapes
:
�
 Model_pred/Tensordot_4/Reshape_1Reshape"Model_pred/Tensordot_4/transpose_1&Model_pred/Tensordot_4/Reshape_1/shape*
T0*
Tshape0*
_output_shapes

:Fh
�
Model_pred/Tensordot_4/MatMulMatMulModel_pred/Tensordot_4/Reshape Model_pred/Tensordot_4/Reshape_1*
T0*
transpose_a( *'
_output_shapes
:���������h*
transpose_b( 
h
Model_pred/Tensordot_4/Const_2Const*
dtype0*
_output_shapes
:*
valueB:h
f
$Model_pred/Tensordot_4/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot_4/concat_2ConcatV2Model_pred/Tensordot_4/GatherV2Model_pred/Tensordot_4/Const_2$Model_pred/Tensordot_4/concat_2/axis*
T0*
N*#
_output_shapes
:���������*

Tidx0
�
Model_pred/Tensordot_4ReshapeModel_pred/Tensordot_4/MatMulModel_pred/Tensordot_4/concat_2*
T0*
Tshape0*'
_output_shapes
:���������h

Model_pred/Add_5AddModel_pred/Tensordot_4Weights/bias_hidden_1_1/read*'
_output_shapes
:���������h*
T0
c
Model_pred/Sigmoid_3SigmoidModel_pred/Add_5*
T0*'
_output_shapes
:���������h
_
Model_pred/transpose_5/RankRankWeights/weight_out_1/read*
T0*
_output_shapes
: 
^
Model_pred/transpose_5/sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
}
Model_pred/transpose_5/subSubModel_pred/transpose_5/RankModel_pred/transpose_5/sub/y*
T0*
_output_shapes
: 
d
"Model_pred/transpose_5/Range/startConst*
value	B : *
dtype0*
_output_shapes
: 
d
"Model_pred/transpose_5/Range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Model_pred/transpose_5/RangeRange"Model_pred/transpose_5/Range/startModel_pred/transpose_5/Rank"Model_pred/transpose_5/Range/delta*
_output_shapes
:*

Tidx0
�
Model_pred/transpose_5/sub_1SubModel_pred/transpose_5/subModel_pred/transpose_5/Range*
T0*
_output_shapes
:
�
Model_pred/transpose_5	TransposeWeights/weight_out_1/readModel_pred/transpose_5/sub_1*
T0*
_output_shapes

:h*
Tperm0
p
Model_pred/Tensordot_5/ShapeShapeModel_pred/Sigmoid_3*
T0*
out_type0*
_output_shapes
:
]
Model_pred/Tensordot_5/RankConst*
value	B :*
dtype0*
_output_shapes
: 
e
Model_pred/Tensordot_5/axesConst*
valueB:*
dtype0*
_output_shapes
:
g
%Model_pred/Tensordot_5/GreaterEqual/yConst*
value	B : *
dtype0*
_output_shapes
: 
�
#Model_pred/Tensordot_5/GreaterEqualGreaterEqualModel_pred/Tensordot_5/axes%Model_pred/Tensordot_5/GreaterEqual/y*
T0*
_output_shapes
:
|
Model_pred/Tensordot_5/CastCast#Model_pred/Tensordot_5/GreaterEqual*

SrcT0
*

DstT0*
_output_shapes
:
�
Model_pred/Tensordot_5/mulMulModel_pred/Tensordot_5/CastModel_pred/Tensordot_5/axes*
T0*
_output_shapes
:
_
Model_pred/Tensordot_5/Less/yConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot_5/LessLessModel_pred/Tensordot_5/axesModel_pred/Tensordot_5/Less/y*
T0*
_output_shapes
:
v
Model_pred/Tensordot_5/Cast_1CastModel_pred/Tensordot_5/Less*

SrcT0
*

DstT0*
_output_shapes
:
�
Model_pred/Tensordot_5/addAddModel_pred/Tensordot_5/axesModel_pred/Tensordot_5/Rank*
T0*
_output_shapes
:
�
Model_pred/Tensordot_5/mul_1MulModel_pred/Tensordot_5/Cast_1Model_pred/Tensordot_5/add*
T0*
_output_shapes
:
�
Model_pred/Tensordot_5/add_1AddModel_pred/Tensordot_5/mulModel_pred/Tensordot_5/mul_1*
_output_shapes
:*
T0
d
"Model_pred/Tensordot_5/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
d
"Model_pred/Tensordot_5/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot_5/rangeRange"Model_pred/Tensordot_5/range/startModel_pred/Tensordot_5/Rank"Model_pred/Tensordot_5/range/delta*

Tidx0*
_output_shapes
:
�
Model_pred/Tensordot_5/ListDiffListDiffModel_pred/Tensordot_5/rangeModel_pred/Tensordot_5/add_1*2
_output_shapes 
:���������:���������*
T0*
out_idx0
f
$Model_pred/Tensordot_5/GatherV2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
Model_pred/Tensordot_5/GatherV2GatherV2Model_pred/Tensordot_5/ShapeModel_pred/Tensordot_5/ListDiff$Model_pred/Tensordot_5/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0*#
_output_shapes
:���������
h
&Model_pred/Tensordot_5/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
!Model_pred/Tensordot_5/GatherV2_1GatherV2Model_pred/Tensordot_5/ShapeModel_pred/Tensordot_5/add_1&Model_pred/Tensordot_5/GatherV2_1/axis*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0
f
Model_pred/Tensordot_5/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
Model_pred/Tensordot_5/ProdProdModel_pred/Tensordot_5/GatherV2Model_pred/Tensordot_5/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
h
Model_pred/Tensordot_5/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
Model_pred/Tensordot_5/Prod_1Prod!Model_pred/Tensordot_5/GatherV2_1Model_pred/Tensordot_5/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
d
"Model_pred/Tensordot_5/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
Model_pred/Tensordot_5/concatConcatV2!Model_pred/Tensordot_5/GatherV2_1Model_pred/Tensordot_5/GatherV2"Model_pred/Tensordot_5/concat/axis*
T0*
N*#
_output_shapes
:���������*

Tidx0
f
$Model_pred/Tensordot_5/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
Model_pred/Tensordot_5/concat_1ConcatV2Model_pred/Tensordot_5/ListDiffModel_pred/Tensordot_5/add_1$Model_pred/Tensordot_5/concat_1/axis*

Tidx0*
T0*
N*#
_output_shapes
:���������
�
Model_pred/Tensordot_5/stackPackModel_pred/Tensordot_5/ProdModel_pred/Tensordot_5/Prod_1*
T0*

axis *
N*
_output_shapes
:
�
 Model_pred/Tensordot_5/transpose	TransposeModel_pred/Sigmoid_3Model_pred/Tensordot_5/concat_1*
T0*0
_output_shapes
:������������������*
Tperm0
�
Model_pred/Tensordot_5/ReshapeReshape Model_pred/Tensordot_5/transposeModel_pred/Tensordot_5/stack*
T0*
Tshape0*0
_output_shapes
:������������������
x
'Model_pred/Tensordot_5/transpose_1/permConst*
dtype0*
_output_shapes
:*
valueB"       
�
"Model_pred/Tensordot_5/transpose_1	TransposeModel_pred/transpose_5'Model_pred/Tensordot_5/transpose_1/perm*
T0*
_output_shapes

:h*
Tperm0
w
&Model_pred/Tensordot_5/Reshape_1/shapeConst*
valueB"h      *
dtype0*
_output_shapes
:
�
 Model_pred/Tensordot_5/Reshape_1Reshape"Model_pred/Tensordot_5/transpose_1&Model_pred/Tensordot_5/Reshape_1/shape*
_output_shapes

:h*
T0*
Tshape0
�
Model_pred/Tensordot_5/MatMulMatMulModel_pred/Tensordot_5/Reshape Model_pred/Tensordot_5/Reshape_1*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:���������
h
Model_pred/Tensordot_5/Const_2Const*
dtype0*
_output_shapes
:*
valueB:
f
$Model_pred/Tensordot_5/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot_5/concat_2ConcatV2Model_pred/Tensordot_5/GatherV2Model_pred/Tensordot_5/Const_2$Model_pred/Tensordot_5/concat_2/axis*
N*#
_output_shapes
:���������*

Tidx0*
T0
�
Model_pred/Tensordot_5ReshapeModel_pred/Tensordot_5/MatMulModel_pred/Tensordot_5/concat_2*
T0*
Tshape0*'
_output_shapes
:���������
z
Model_pred/Add_6AddModel_pred/Tensordot_5Weights/bias_out_1/read*
T0*'
_output_shapes
:���������

Model_pred/Squeeze_1SqueezeModel_pred/Add_6*
T0*#
_output_shapes
:���������*
squeeze_dims

���������
o
Model_pred/Shape_1ShapeInputs_pred/IteratorGetNext:1*
T0*
out_type0*
_output_shapes
:
�
Model_pred/ScatterNd_1	ScatterNdModel_pred/Cast_1Model_pred/Squeeze_1Model_pred/Shape_1*'
_output_shapes
:���������*
Tindices0*
T0
s
Model_pred/Add_7AddModel_pred/Add_3Model_pred/ScatterNd_1*
T0*'
_output_shapes
:���������
T
Model_pred/Const_2Const*
dtype0*
_output_shapes
: *
value	B :
]
Model_pred/ExpandDims_2/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/ExpandDims_2
ExpandDimsModel_pred/Const_2Model_pred/ExpandDims_2/dim*

Tdim0*
T0*
_output_shapes
:
�
Model_pred/Equal_2EqualInputs_pred/IteratorGetNext:1Model_pred/ExpandDims_2*'
_output_shapes
:���������*
T0
a
Model_pred/Where_2WhereModel_pred/Equal_2*'
_output_shapes
:���������*
T0

n
Model_pred/Cast_2CastModel_pred/Where_2*

SrcT0	*

DstT0*'
_output_shapes
:���������
�
Model_pred/GatherNd_2GatherNdDescriptor_pred/ACSF/acsfModel_pred/Cast_2*(
_output_shapes
:����������
*
Tindices0*
Tparams0
^
Model_pred/transpose_6/RankRankWeights/weight_in_2/read*
T0*
_output_shapes
: 
^
Model_pred/transpose_6/sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
}
Model_pred/transpose_6/subSubModel_pred/transpose_6/RankModel_pred/transpose_6/sub/y*
T0*
_output_shapes
: 
d
"Model_pred/transpose_6/Range/startConst*
value	B : *
dtype0*
_output_shapes
: 
d
"Model_pred/transpose_6/Range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
�
Model_pred/transpose_6/RangeRange"Model_pred/transpose_6/Range/startModel_pred/transpose_6/Rank"Model_pred/transpose_6/Range/delta*
_output_shapes
:*

Tidx0
�
Model_pred/transpose_6/sub_1SubModel_pred/transpose_6/subModel_pred/transpose_6/Range*
_output_shapes
:*
T0
�
Model_pred/transpose_6	TransposeWeights/weight_in_2/readModel_pred/transpose_6/sub_1*
T0*
_output_shapes
:	�
F*
Tperm0
q
Model_pred/Tensordot_6/ShapeShapeModel_pred/GatherNd_2*
T0*
out_type0*
_output_shapes
:
]
Model_pred/Tensordot_6/RankConst*
dtype0*
_output_shapes
: *
value	B :
e
Model_pred/Tensordot_6/axesConst*
valueB:*
dtype0*
_output_shapes
:
g
%Model_pred/Tensordot_6/GreaterEqual/yConst*
value	B : *
dtype0*
_output_shapes
: 
�
#Model_pred/Tensordot_6/GreaterEqualGreaterEqualModel_pred/Tensordot_6/axes%Model_pred/Tensordot_6/GreaterEqual/y*
T0*
_output_shapes
:
|
Model_pred/Tensordot_6/CastCast#Model_pred/Tensordot_6/GreaterEqual*

SrcT0
*

DstT0*
_output_shapes
:
�
Model_pred/Tensordot_6/mulMulModel_pred/Tensordot_6/CastModel_pred/Tensordot_6/axes*
T0*
_output_shapes
:
_
Model_pred/Tensordot_6/Less/yConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot_6/LessLessModel_pred/Tensordot_6/axesModel_pred/Tensordot_6/Less/y*
_output_shapes
:*
T0
v
Model_pred/Tensordot_6/Cast_1CastModel_pred/Tensordot_6/Less*

SrcT0
*

DstT0*
_output_shapes
:
�
Model_pred/Tensordot_6/addAddModel_pred/Tensordot_6/axesModel_pred/Tensordot_6/Rank*
T0*
_output_shapes
:
�
Model_pred/Tensordot_6/mul_1MulModel_pred/Tensordot_6/Cast_1Model_pred/Tensordot_6/add*
T0*
_output_shapes
:
�
Model_pred/Tensordot_6/add_1AddModel_pred/Tensordot_6/mulModel_pred/Tensordot_6/mul_1*
_output_shapes
:*
T0
d
"Model_pred/Tensordot_6/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
d
"Model_pred/Tensordot_6/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
�
Model_pred/Tensordot_6/rangeRange"Model_pred/Tensordot_6/range/startModel_pred/Tensordot_6/Rank"Model_pred/Tensordot_6/range/delta*
_output_shapes
:*

Tidx0
�
Model_pred/Tensordot_6/ListDiffListDiffModel_pred/Tensordot_6/rangeModel_pred/Tensordot_6/add_1*2
_output_shapes 
:���������:���������*
T0*
out_idx0
f
$Model_pred/Tensordot_6/GatherV2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
Model_pred/Tensordot_6/GatherV2GatherV2Model_pred/Tensordot_6/ShapeModel_pred/Tensordot_6/ListDiff$Model_pred/Tensordot_6/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0*#
_output_shapes
:���������
h
&Model_pred/Tensordot_6/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
!Model_pred/Tensordot_6/GatherV2_1GatherV2Model_pred/Tensordot_6/ShapeModel_pred/Tensordot_6/add_1&Model_pred/Tensordot_6/GatherV2_1/axis*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0
f
Model_pred/Tensordot_6/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
Model_pred/Tensordot_6/ProdProdModel_pred/Tensordot_6/GatherV2Model_pred/Tensordot_6/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
h
Model_pred/Tensordot_6/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
Model_pred/Tensordot_6/Prod_1Prod!Model_pred/Tensordot_6/GatherV2_1Model_pred/Tensordot_6/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
d
"Model_pred/Tensordot_6/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot_6/concatConcatV2!Model_pred/Tensordot_6/GatherV2_1Model_pred/Tensordot_6/GatherV2"Model_pred/Tensordot_6/concat/axis*
T0*
N*#
_output_shapes
:���������*

Tidx0
f
$Model_pred/Tensordot_6/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot_6/concat_1ConcatV2Model_pred/Tensordot_6/ListDiffModel_pred/Tensordot_6/add_1$Model_pred/Tensordot_6/concat_1/axis*
T0*
N*#
_output_shapes
:���������*

Tidx0
�
Model_pred/Tensordot_6/stackPackModel_pred/Tensordot_6/ProdModel_pred/Tensordot_6/Prod_1*
N*
_output_shapes
:*
T0*

axis 
�
 Model_pred/Tensordot_6/transpose	TransposeModel_pred/GatherNd_2Model_pred/Tensordot_6/concat_1*
T0*0
_output_shapes
:������������������*
Tperm0
�
Model_pred/Tensordot_6/ReshapeReshape Model_pred/Tensordot_6/transposeModel_pred/Tensordot_6/stack*0
_output_shapes
:������������������*
T0*
Tshape0
x
'Model_pred/Tensordot_6/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
�
"Model_pred/Tensordot_6/transpose_1	TransposeModel_pred/transpose_6'Model_pred/Tensordot_6/transpose_1/perm*
Tperm0*
T0*
_output_shapes
:	�
F
w
&Model_pred/Tensordot_6/Reshape_1/shapeConst*
valueB"s  F   *
dtype0*
_output_shapes
:
�
 Model_pred/Tensordot_6/Reshape_1Reshape"Model_pred/Tensordot_6/transpose_1&Model_pred/Tensordot_6/Reshape_1/shape*
T0*
Tshape0*
_output_shapes
:	�
F
�
Model_pred/Tensordot_6/MatMulMatMulModel_pred/Tensordot_6/Reshape Model_pred/Tensordot_6/Reshape_1*
T0*
transpose_a( *'
_output_shapes
:���������F*
transpose_b( 
h
Model_pred/Tensordot_6/Const_2Const*
valueB:F*
dtype0*
_output_shapes
:
f
$Model_pred/Tensordot_6/concat_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
Model_pred/Tensordot_6/concat_2ConcatV2Model_pred/Tensordot_6/GatherV2Model_pred/Tensordot_6/Const_2$Model_pred/Tensordot_6/concat_2/axis*

Tidx0*
T0*
N*#
_output_shapes
:���������
�
Model_pred/Tensordot_6ReshapeModel_pred/Tensordot_6/MatMulModel_pred/Tensordot_6/concat_2*
T0*
Tshape0*'
_output_shapes
:���������F
y
Model_pred/Add_8AddModel_pred/Tensordot_6Weights/bias_in_2/read*
T0*'
_output_shapes
:���������F
c
Model_pred/Sigmoid_4SigmoidModel_pred/Add_8*
T0*'
_output_shapes
:���������F
d
Model_pred/transpose_7/RankRankWeights/weight_hidden_1_2/read*
T0*
_output_shapes
: 
^
Model_pred/transpose_7/sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
}
Model_pred/transpose_7/subSubModel_pred/transpose_7/RankModel_pred/transpose_7/sub/y*
T0*
_output_shapes
: 
d
"Model_pred/transpose_7/Range/startConst*
value	B : *
dtype0*
_output_shapes
: 
d
"Model_pred/transpose_7/Range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Model_pred/transpose_7/RangeRange"Model_pred/transpose_7/Range/startModel_pred/transpose_7/Rank"Model_pred/transpose_7/Range/delta*
_output_shapes
:*

Tidx0
�
Model_pred/transpose_7/sub_1SubModel_pred/transpose_7/subModel_pred/transpose_7/Range*
_output_shapes
:*
T0
�
Model_pred/transpose_7	TransposeWeights/weight_hidden_1_2/readModel_pred/transpose_7/sub_1*
T0*
_output_shapes

:Fh*
Tperm0
p
Model_pred/Tensordot_7/ShapeShapeModel_pred/Sigmoid_4*
_output_shapes
:*
T0*
out_type0
]
Model_pred/Tensordot_7/RankConst*
value	B :*
dtype0*
_output_shapes
: 
e
Model_pred/Tensordot_7/axesConst*
valueB:*
dtype0*
_output_shapes
:
g
%Model_pred/Tensordot_7/GreaterEqual/yConst*
dtype0*
_output_shapes
: *
value	B : 
�
#Model_pred/Tensordot_7/GreaterEqualGreaterEqualModel_pred/Tensordot_7/axes%Model_pred/Tensordot_7/GreaterEqual/y*
T0*
_output_shapes
:
|
Model_pred/Tensordot_7/CastCast#Model_pred/Tensordot_7/GreaterEqual*

DstT0*
_output_shapes
:*

SrcT0

�
Model_pred/Tensordot_7/mulMulModel_pred/Tensordot_7/CastModel_pred/Tensordot_7/axes*
T0*
_output_shapes
:
_
Model_pred/Tensordot_7/Less/yConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot_7/LessLessModel_pred/Tensordot_7/axesModel_pred/Tensordot_7/Less/y*
T0*
_output_shapes
:
v
Model_pred/Tensordot_7/Cast_1CastModel_pred/Tensordot_7/Less*

SrcT0
*

DstT0*
_output_shapes
:
�
Model_pred/Tensordot_7/addAddModel_pred/Tensordot_7/axesModel_pred/Tensordot_7/Rank*
T0*
_output_shapes
:
�
Model_pred/Tensordot_7/mul_1MulModel_pred/Tensordot_7/Cast_1Model_pred/Tensordot_7/add*
T0*
_output_shapes
:
�
Model_pred/Tensordot_7/add_1AddModel_pred/Tensordot_7/mulModel_pred/Tensordot_7/mul_1*
T0*
_output_shapes
:
d
"Model_pred/Tensordot_7/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
d
"Model_pred/Tensordot_7/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
�
Model_pred/Tensordot_7/rangeRange"Model_pred/Tensordot_7/range/startModel_pred/Tensordot_7/Rank"Model_pred/Tensordot_7/range/delta*
_output_shapes
:*

Tidx0
�
Model_pred/Tensordot_7/ListDiffListDiffModel_pred/Tensordot_7/rangeModel_pred/Tensordot_7/add_1*
T0*
out_idx0*2
_output_shapes 
:���������:���������
f
$Model_pred/Tensordot_7/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot_7/GatherV2GatherV2Model_pred/Tensordot_7/ShapeModel_pred/Tensordot_7/ListDiff$Model_pred/Tensordot_7/GatherV2/axis*
Tparams0*#
_output_shapes
:���������*
Taxis0*
Tindices0
h
&Model_pred/Tensordot_7/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
!Model_pred/Tensordot_7/GatherV2_1GatherV2Model_pred/Tensordot_7/ShapeModel_pred/Tensordot_7/add_1&Model_pred/Tensordot_7/GatherV2_1/axis*
_output_shapes
:*
Taxis0*
Tindices0*
Tparams0
f
Model_pred/Tensordot_7/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
Model_pred/Tensordot_7/ProdProdModel_pred/Tensordot_7/GatherV2Model_pred/Tensordot_7/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
h
Model_pred/Tensordot_7/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
Model_pred/Tensordot_7/Prod_1Prod!Model_pred/Tensordot_7/GatherV2_1Model_pred/Tensordot_7/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
d
"Model_pred/Tensordot_7/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
Model_pred/Tensordot_7/concatConcatV2!Model_pred/Tensordot_7/GatherV2_1Model_pred/Tensordot_7/GatherV2"Model_pred/Tensordot_7/concat/axis*

Tidx0*
T0*
N*#
_output_shapes
:���������
f
$Model_pred/Tensordot_7/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot_7/concat_1ConcatV2Model_pred/Tensordot_7/ListDiffModel_pred/Tensordot_7/add_1$Model_pred/Tensordot_7/concat_1/axis*
T0*
N*#
_output_shapes
:���������*

Tidx0
�
Model_pred/Tensordot_7/stackPackModel_pred/Tensordot_7/ProdModel_pred/Tensordot_7/Prod_1*
T0*

axis *
N*
_output_shapes
:
�
 Model_pred/Tensordot_7/transpose	TransposeModel_pred/Sigmoid_4Model_pred/Tensordot_7/concat_1*
T0*0
_output_shapes
:������������������*
Tperm0
�
Model_pred/Tensordot_7/ReshapeReshape Model_pred/Tensordot_7/transposeModel_pred/Tensordot_7/stack*0
_output_shapes
:������������������*
T0*
Tshape0
x
'Model_pred/Tensordot_7/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
�
"Model_pred/Tensordot_7/transpose_1	TransposeModel_pred/transpose_7'Model_pred/Tensordot_7/transpose_1/perm*
T0*
_output_shapes

:Fh*
Tperm0
w
&Model_pred/Tensordot_7/Reshape_1/shapeConst*
valueB"F   h   *
dtype0*
_output_shapes
:
�
 Model_pred/Tensordot_7/Reshape_1Reshape"Model_pred/Tensordot_7/transpose_1&Model_pred/Tensordot_7/Reshape_1/shape*
T0*
Tshape0*
_output_shapes

:Fh
�
Model_pred/Tensordot_7/MatMulMatMulModel_pred/Tensordot_7/Reshape Model_pred/Tensordot_7/Reshape_1*
transpose_a( *'
_output_shapes
:���������h*
transpose_b( *
T0
h
Model_pred/Tensordot_7/Const_2Const*
dtype0*
_output_shapes
:*
valueB:h
f
$Model_pred/Tensordot_7/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot_7/concat_2ConcatV2Model_pred/Tensordot_7/GatherV2Model_pred/Tensordot_7/Const_2$Model_pred/Tensordot_7/concat_2/axis*
T0*
N*#
_output_shapes
:���������*

Tidx0
�
Model_pred/Tensordot_7ReshapeModel_pred/Tensordot_7/MatMulModel_pred/Tensordot_7/concat_2*
T0*
Tshape0*'
_output_shapes
:���������h

Model_pred/Add_9AddModel_pred/Tensordot_7Weights/bias_hidden_1_2/read*
T0*'
_output_shapes
:���������h
c
Model_pred/Sigmoid_5SigmoidModel_pred/Add_9*
T0*'
_output_shapes
:���������h
_
Model_pred/transpose_8/RankRankWeights/weight_out_2/read*
T0*
_output_shapes
: 
^
Model_pred/transpose_8/sub/yConst*
dtype0*
_output_shapes
: *
value	B :
}
Model_pred/transpose_8/subSubModel_pred/transpose_8/RankModel_pred/transpose_8/sub/y*
T0*
_output_shapes
: 
d
"Model_pred/transpose_8/Range/startConst*
dtype0*
_output_shapes
: *
value	B : 
d
"Model_pred/transpose_8/Range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Model_pred/transpose_8/RangeRange"Model_pred/transpose_8/Range/startModel_pred/transpose_8/Rank"Model_pred/transpose_8/Range/delta*
_output_shapes
:*

Tidx0
�
Model_pred/transpose_8/sub_1SubModel_pred/transpose_8/subModel_pred/transpose_8/Range*
T0*
_output_shapes
:
�
Model_pred/transpose_8	TransposeWeights/weight_out_2/readModel_pred/transpose_8/sub_1*
T0*
_output_shapes

:h*
Tperm0
p
Model_pred/Tensordot_8/ShapeShapeModel_pred/Sigmoid_5*
T0*
out_type0*
_output_shapes
:
]
Model_pred/Tensordot_8/RankConst*
dtype0*
_output_shapes
: *
value	B :
e
Model_pred/Tensordot_8/axesConst*
valueB:*
dtype0*
_output_shapes
:
g
%Model_pred/Tensordot_8/GreaterEqual/yConst*
value	B : *
dtype0*
_output_shapes
: 
�
#Model_pred/Tensordot_8/GreaterEqualGreaterEqualModel_pred/Tensordot_8/axes%Model_pred/Tensordot_8/GreaterEqual/y*
T0*
_output_shapes
:
|
Model_pred/Tensordot_8/CastCast#Model_pred/Tensordot_8/GreaterEqual*

SrcT0
*

DstT0*
_output_shapes
:
�
Model_pred/Tensordot_8/mulMulModel_pred/Tensordot_8/CastModel_pred/Tensordot_8/axes*
T0*
_output_shapes
:
_
Model_pred/Tensordot_8/Less/yConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot_8/LessLessModel_pred/Tensordot_8/axesModel_pred/Tensordot_8/Less/y*
T0*
_output_shapes
:
v
Model_pred/Tensordot_8/Cast_1CastModel_pred/Tensordot_8/Less*

SrcT0
*

DstT0*
_output_shapes
:
�
Model_pred/Tensordot_8/addAddModel_pred/Tensordot_8/axesModel_pred/Tensordot_8/Rank*
T0*
_output_shapes
:
�
Model_pred/Tensordot_8/mul_1MulModel_pred/Tensordot_8/Cast_1Model_pred/Tensordot_8/add*
T0*
_output_shapes
:
�
Model_pred/Tensordot_8/add_1AddModel_pred/Tensordot_8/mulModel_pred/Tensordot_8/mul_1*
T0*
_output_shapes
:
d
"Model_pred/Tensordot_8/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
d
"Model_pred/Tensordot_8/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot_8/rangeRange"Model_pred/Tensordot_8/range/startModel_pred/Tensordot_8/Rank"Model_pred/Tensordot_8/range/delta*
_output_shapes
:*

Tidx0
�
Model_pred/Tensordot_8/ListDiffListDiffModel_pred/Tensordot_8/rangeModel_pred/Tensordot_8/add_1*
T0*
out_idx0*2
_output_shapes 
:���������:���������
f
$Model_pred/Tensordot_8/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot_8/GatherV2GatherV2Model_pred/Tensordot_8/ShapeModel_pred/Tensordot_8/ListDiff$Model_pred/Tensordot_8/GatherV2/axis*
Tparams0*#
_output_shapes
:���������*
Taxis0*
Tindices0
h
&Model_pred/Tensordot_8/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
!Model_pred/Tensordot_8/GatherV2_1GatherV2Model_pred/Tensordot_8/ShapeModel_pred/Tensordot_8/add_1&Model_pred/Tensordot_8/GatherV2_1/axis*
Tparams0*
_output_shapes
:*
Taxis0*
Tindices0
f
Model_pred/Tensordot_8/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
Model_pred/Tensordot_8/ProdProdModel_pred/Tensordot_8/GatherV2Model_pred/Tensordot_8/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
h
Model_pred/Tensordot_8/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
Model_pred/Tensordot_8/Prod_1Prod!Model_pred/Tensordot_8/GatherV2_1Model_pred/Tensordot_8/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
d
"Model_pred/Tensordot_8/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
Model_pred/Tensordot_8/concatConcatV2!Model_pred/Tensordot_8/GatherV2_1Model_pred/Tensordot_8/GatherV2"Model_pred/Tensordot_8/concat/axis*

Tidx0*
T0*
N*#
_output_shapes
:���������
f
$Model_pred/Tensordot_8/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Model_pred/Tensordot_8/concat_1ConcatV2Model_pred/Tensordot_8/ListDiffModel_pred/Tensordot_8/add_1$Model_pred/Tensordot_8/concat_1/axis*
T0*
N*#
_output_shapes
:���������*

Tidx0
�
Model_pred/Tensordot_8/stackPackModel_pred/Tensordot_8/ProdModel_pred/Tensordot_8/Prod_1*
N*
_output_shapes
:*
T0*

axis 
�
 Model_pred/Tensordot_8/transpose	TransposeModel_pred/Sigmoid_5Model_pred/Tensordot_8/concat_1*
T0*0
_output_shapes
:������������������*
Tperm0
�
Model_pred/Tensordot_8/ReshapeReshape Model_pred/Tensordot_8/transposeModel_pred/Tensordot_8/stack*
T0*
Tshape0*0
_output_shapes
:������������������
x
'Model_pred/Tensordot_8/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
�
"Model_pred/Tensordot_8/transpose_1	TransposeModel_pred/transpose_8'Model_pred/Tensordot_8/transpose_1/perm*
T0*
_output_shapes

:h*
Tperm0
w
&Model_pred/Tensordot_8/Reshape_1/shapeConst*
valueB"h      *
dtype0*
_output_shapes
:
�
 Model_pred/Tensordot_8/Reshape_1Reshape"Model_pred/Tensordot_8/transpose_1&Model_pred/Tensordot_8/Reshape_1/shape*
T0*
Tshape0*
_output_shapes

:h
�
Model_pred/Tensordot_8/MatMulMatMulModel_pred/Tensordot_8/Reshape Model_pred/Tensordot_8/Reshape_1*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:���������
h
Model_pred/Tensordot_8/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
f
$Model_pred/Tensordot_8/concat_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
Model_pred/Tensordot_8/concat_2ConcatV2Model_pred/Tensordot_8/GatherV2Model_pred/Tensordot_8/Const_2$Model_pred/Tensordot_8/concat_2/axis*
T0*
N*#
_output_shapes
:���������*

Tidx0
�
Model_pred/Tensordot_8ReshapeModel_pred/Tensordot_8/MatMulModel_pred/Tensordot_8/concat_2*
T0*
Tshape0*'
_output_shapes
:���������
{
Model_pred/Add_10AddModel_pred/Tensordot_8Weights/bias_out_2/read*
T0*'
_output_shapes
:���������
�
Model_pred/Squeeze_2SqueezeModel_pred/Add_10*
squeeze_dims

���������*
T0*#
_output_shapes
:���������
o
Model_pred/Shape_2ShapeInputs_pred/IteratorGetNext:1*
_output_shapes
:*
T0*
out_type0
�
Model_pred/ScatterNd_2	ScatterNdModel_pred/Cast_2Model_pred/Squeeze_2Model_pred/Shape_2*
Tindices0*
T0*'
_output_shapes
:���������
t
Model_pred/Add_11AddModel_pred/Add_7Model_pred/ScatterNd_2*
T0*'
_output_shapes
:���������
n
#Model_pred/output/reduction_indicesConst*
dtype0*
_output_shapes
: *
valueB :
���������
�
Model_pred/outputSumModel_pred/Add_11#Model_pred/output/reduction_indices*
T0*'
_output_shapes
:���������*

Tidx0*
	keep_dims(
P

save/ConstConst*
dtype0*
_output_shapes
: *
valueB Bmodel
�
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_f679a6cc3b4443078e4182fcb0cadfb6/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*�
value�B�8BWeights/bias_hidden_1BWeights/bias_hidden_1/AdamBWeights/bias_hidden_1/Adam_1BWeights/bias_hidden_1_1BWeights/bias_hidden_1_1/AdamBWeights/bias_hidden_1_1/Adam_1BWeights/bias_hidden_1_2BWeights/bias_hidden_1_2/AdamBWeights/bias_hidden_1_2/Adam_1BWeights/bias_inBWeights/bias_in/AdamBWeights/bias_in/Adam_1BWeights/bias_in_1BWeights/bias_in_1/AdamBWeights/bias_in_1/Adam_1BWeights/bias_in_2BWeights/bias_in_2/AdamBWeights/bias_in_2/Adam_1BWeights/bias_outBWeights/bias_out/AdamBWeights/bias_out/Adam_1BWeights/bias_out_1BWeights/bias_out_1/AdamBWeights/bias_out_1/Adam_1BWeights/bias_out_2BWeights/bias_out_2/AdamBWeights/bias_out_2/Adam_1BWeights/weight_hidden_1BWeights/weight_hidden_1/AdamBWeights/weight_hidden_1/Adam_1BWeights/weight_hidden_1_1BWeights/weight_hidden_1_1/AdamB Weights/weight_hidden_1_1/Adam_1BWeights/weight_hidden_1_2BWeights/weight_hidden_1_2/AdamB Weights/weight_hidden_1_2/Adam_1BWeights/weight_inBWeights/weight_in/AdamBWeights/weight_in/Adam_1BWeights/weight_in_1BWeights/weight_in_1/AdamBWeights/weight_in_1/Adam_1BWeights/weight_in_2BWeights/weight_in_2/AdamBWeights/weight_in_2/Adam_1BWeights/weight_outBWeights/weight_out/AdamBWeights/weight_out/Adam_1BWeights/weight_out_1BWeights/weight_out_1/AdamBWeights/weight_out_1/Adam_1BWeights/weight_out_2BWeights/weight_out_2/AdamBWeights/weight_out_2/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:8
�
save/SaveV2/shape_and_slicesConst*�
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:8
�
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesWeights/bias_hidden_1Weights/bias_hidden_1/AdamWeights/bias_hidden_1/Adam_1Weights/bias_hidden_1_1Weights/bias_hidden_1_1/AdamWeights/bias_hidden_1_1/Adam_1Weights/bias_hidden_1_2Weights/bias_hidden_1_2/AdamWeights/bias_hidden_1_2/Adam_1Weights/bias_inWeights/bias_in/AdamWeights/bias_in/Adam_1Weights/bias_in_1Weights/bias_in_1/AdamWeights/bias_in_1/Adam_1Weights/bias_in_2Weights/bias_in_2/AdamWeights/bias_in_2/Adam_1Weights/bias_outWeights/bias_out/AdamWeights/bias_out/Adam_1Weights/bias_out_1Weights/bias_out_1/AdamWeights/bias_out_1/Adam_1Weights/bias_out_2Weights/bias_out_2/AdamWeights/bias_out_2/Adam_1Weights/weight_hidden_1Weights/weight_hidden_1/AdamWeights/weight_hidden_1/Adam_1Weights/weight_hidden_1_1Weights/weight_hidden_1_1/Adam Weights/weight_hidden_1_1/Adam_1Weights/weight_hidden_1_2Weights/weight_hidden_1_2/Adam Weights/weight_hidden_1_2/Adam_1Weights/weight_inWeights/weight_in/AdamWeights/weight_in/Adam_1Weights/weight_in_1Weights/weight_in_1/AdamWeights/weight_in_1/Adam_1Weights/weight_in_2Weights/weight_in_2/AdamWeights/weight_in_2/Adam_1Weights/weight_outWeights/weight_out/AdamWeights/weight_out/Adam_1Weights/weight_out_1Weights/weight_out_1/AdamWeights/weight_out_1/Adam_1Weights/weight_out_2Weights/weight_out_2/AdamWeights/weight_out_2/Adam_1beta1_powerbeta2_power*F
dtypes<
:28
�
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
�
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
N*
_output_shapes
:*
T0*

axis 
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
T0*
_output_shapes
: 
�
save/RestoreV2/tensor_namesConst*�
value�B�8BWeights/bias_hidden_1BWeights/bias_hidden_1/AdamBWeights/bias_hidden_1/Adam_1BWeights/bias_hidden_1_1BWeights/bias_hidden_1_1/AdamBWeights/bias_hidden_1_1/Adam_1BWeights/bias_hidden_1_2BWeights/bias_hidden_1_2/AdamBWeights/bias_hidden_1_2/Adam_1BWeights/bias_inBWeights/bias_in/AdamBWeights/bias_in/Adam_1BWeights/bias_in_1BWeights/bias_in_1/AdamBWeights/bias_in_1/Adam_1BWeights/bias_in_2BWeights/bias_in_2/AdamBWeights/bias_in_2/Adam_1BWeights/bias_outBWeights/bias_out/AdamBWeights/bias_out/Adam_1BWeights/bias_out_1BWeights/bias_out_1/AdamBWeights/bias_out_1/Adam_1BWeights/bias_out_2BWeights/bias_out_2/AdamBWeights/bias_out_2/Adam_1BWeights/weight_hidden_1BWeights/weight_hidden_1/AdamBWeights/weight_hidden_1/Adam_1BWeights/weight_hidden_1_1BWeights/weight_hidden_1_1/AdamB Weights/weight_hidden_1_1/Adam_1BWeights/weight_hidden_1_2BWeights/weight_hidden_1_2/AdamB Weights/weight_hidden_1_2/Adam_1BWeights/weight_inBWeights/weight_in/AdamBWeights/weight_in/Adam_1BWeights/weight_in_1BWeights/weight_in_1/AdamBWeights/weight_in_1/Adam_1BWeights/weight_in_2BWeights/weight_in_2/AdamBWeights/weight_in_2/Adam_1BWeights/weight_outBWeights/weight_out/AdamBWeights/weight_out/Adam_1BWeights/weight_out_1BWeights/weight_out_1/AdamBWeights/weight_out_1/Adam_1BWeights/weight_out_2BWeights/weight_out_2/AdamBWeights/weight_out_2/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:8
�
save/RestoreV2/shape_and_slicesConst*�
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:8
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::*F
dtypes<
:28
�
save/AssignAssignWeights/bias_hidden_1save/RestoreV2*
T0*(
_class
loc:@Weights/bias_hidden_1*
validate_shape(*
_output_shapes
:h*
use_locking(
�
save/Assign_1AssignWeights/bias_hidden_1/Adamsave/RestoreV2:1*
use_locking(*
T0*(
_class
loc:@Weights/bias_hidden_1*
validate_shape(*
_output_shapes
:h
�
save/Assign_2AssignWeights/bias_hidden_1/Adam_1save/RestoreV2:2*
use_locking(*
T0*(
_class
loc:@Weights/bias_hidden_1*
validate_shape(*
_output_shapes
:h
�
save/Assign_3AssignWeights/bias_hidden_1_1save/RestoreV2:3*
T0**
_class 
loc:@Weights/bias_hidden_1_1*
validate_shape(*
_output_shapes
:h*
use_locking(
�
save/Assign_4AssignWeights/bias_hidden_1_1/Adamsave/RestoreV2:4*
use_locking(*
T0**
_class 
loc:@Weights/bias_hidden_1_1*
validate_shape(*
_output_shapes
:h
�
save/Assign_5AssignWeights/bias_hidden_1_1/Adam_1save/RestoreV2:5*
use_locking(*
T0**
_class 
loc:@Weights/bias_hidden_1_1*
validate_shape(*
_output_shapes
:h
�
save/Assign_6AssignWeights/bias_hidden_1_2save/RestoreV2:6*
T0**
_class 
loc:@Weights/bias_hidden_1_2*
validate_shape(*
_output_shapes
:h*
use_locking(
�
save/Assign_7AssignWeights/bias_hidden_1_2/Adamsave/RestoreV2:7*
validate_shape(*
_output_shapes
:h*
use_locking(*
T0**
_class 
loc:@Weights/bias_hidden_1_2
�
save/Assign_8AssignWeights/bias_hidden_1_2/Adam_1save/RestoreV2:8*
use_locking(*
T0**
_class 
loc:@Weights/bias_hidden_1_2*
validate_shape(*
_output_shapes
:h
�
save/Assign_9AssignWeights/bias_insave/RestoreV2:9*
T0*"
_class
loc:@Weights/bias_in*
validate_shape(*
_output_shapes
:F*
use_locking(
�
save/Assign_10AssignWeights/bias_in/Adamsave/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@Weights/bias_in*
validate_shape(*
_output_shapes
:F
�
save/Assign_11AssignWeights/bias_in/Adam_1save/RestoreV2:11*
use_locking(*
T0*"
_class
loc:@Weights/bias_in*
validate_shape(*
_output_shapes
:F
�
save/Assign_12AssignWeights/bias_in_1save/RestoreV2:12*
T0*$
_class
loc:@Weights/bias_in_1*
validate_shape(*
_output_shapes
:F*
use_locking(
�
save/Assign_13AssignWeights/bias_in_1/Adamsave/RestoreV2:13*
use_locking(*
T0*$
_class
loc:@Weights/bias_in_1*
validate_shape(*
_output_shapes
:F
�
save/Assign_14AssignWeights/bias_in_1/Adam_1save/RestoreV2:14*
use_locking(*
T0*$
_class
loc:@Weights/bias_in_1*
validate_shape(*
_output_shapes
:F
�
save/Assign_15AssignWeights/bias_in_2save/RestoreV2:15*
validate_shape(*
_output_shapes
:F*
use_locking(*
T0*$
_class
loc:@Weights/bias_in_2
�
save/Assign_16AssignWeights/bias_in_2/Adamsave/RestoreV2:16*
use_locking(*
T0*$
_class
loc:@Weights/bias_in_2*
validate_shape(*
_output_shapes
:F
�
save/Assign_17AssignWeights/bias_in_2/Adam_1save/RestoreV2:17*
T0*$
_class
loc:@Weights/bias_in_2*
validate_shape(*
_output_shapes
:F*
use_locking(
�
save/Assign_18AssignWeights/bias_outsave/RestoreV2:18*
T0*#
_class
loc:@Weights/bias_out*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_19AssignWeights/bias_out/Adamsave/RestoreV2:19*
T0*#
_class
loc:@Weights/bias_out*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_20AssignWeights/bias_out/Adam_1save/RestoreV2:20*
T0*#
_class
loc:@Weights/bias_out*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_21AssignWeights/bias_out_1save/RestoreV2:21*
T0*%
_class
loc:@Weights/bias_out_1*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_22AssignWeights/bias_out_1/Adamsave/RestoreV2:22*
use_locking(*
T0*%
_class
loc:@Weights/bias_out_1*
validate_shape(*
_output_shapes
:
�
save/Assign_23AssignWeights/bias_out_1/Adam_1save/RestoreV2:23*
use_locking(*
T0*%
_class
loc:@Weights/bias_out_1*
validate_shape(*
_output_shapes
:
�
save/Assign_24AssignWeights/bias_out_2save/RestoreV2:24*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*%
_class
loc:@Weights/bias_out_2
�
save/Assign_25AssignWeights/bias_out_2/Adamsave/RestoreV2:25*
use_locking(*
T0*%
_class
loc:@Weights/bias_out_2*
validate_shape(*
_output_shapes
:
�
save/Assign_26AssignWeights/bias_out_2/Adam_1save/RestoreV2:26*
T0*%
_class
loc:@Weights/bias_out_2*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_27AssignWeights/weight_hidden_1save/RestoreV2:27*
use_locking(*
T0**
_class 
loc:@Weights/weight_hidden_1*
validate_shape(*
_output_shapes

:hF
�
save/Assign_28AssignWeights/weight_hidden_1/Adamsave/RestoreV2:28*
T0**
_class 
loc:@Weights/weight_hidden_1*
validate_shape(*
_output_shapes

:hF*
use_locking(
�
save/Assign_29AssignWeights/weight_hidden_1/Adam_1save/RestoreV2:29*
T0**
_class 
loc:@Weights/weight_hidden_1*
validate_shape(*
_output_shapes

:hF*
use_locking(
�
save/Assign_30AssignWeights/weight_hidden_1_1save/RestoreV2:30*
use_locking(*
T0*,
_class"
 loc:@Weights/weight_hidden_1_1*
validate_shape(*
_output_shapes

:hF
�
save/Assign_31AssignWeights/weight_hidden_1_1/Adamsave/RestoreV2:31*
use_locking(*
T0*,
_class"
 loc:@Weights/weight_hidden_1_1*
validate_shape(*
_output_shapes

:hF
�
save/Assign_32Assign Weights/weight_hidden_1_1/Adam_1save/RestoreV2:32*
use_locking(*
T0*,
_class"
 loc:@Weights/weight_hidden_1_1*
validate_shape(*
_output_shapes

:hF
�
save/Assign_33AssignWeights/weight_hidden_1_2save/RestoreV2:33*
validate_shape(*
_output_shapes

:hF*
use_locking(*
T0*,
_class"
 loc:@Weights/weight_hidden_1_2
�
save/Assign_34AssignWeights/weight_hidden_1_2/Adamsave/RestoreV2:34*
use_locking(*
T0*,
_class"
 loc:@Weights/weight_hidden_1_2*
validate_shape(*
_output_shapes

:hF
�
save/Assign_35Assign Weights/weight_hidden_1_2/Adam_1save/RestoreV2:35*
T0*,
_class"
 loc:@Weights/weight_hidden_1_2*
validate_shape(*
_output_shapes

:hF*
use_locking(
�
save/Assign_36AssignWeights/weight_insave/RestoreV2:36*
use_locking(*
T0*$
_class
loc:@Weights/weight_in*
validate_shape(*
_output_shapes
:	F�

�
save/Assign_37AssignWeights/weight_in/Adamsave/RestoreV2:37*
validate_shape(*
_output_shapes
:	F�
*
use_locking(*
T0*$
_class
loc:@Weights/weight_in
�
save/Assign_38AssignWeights/weight_in/Adam_1save/RestoreV2:38*
use_locking(*
T0*$
_class
loc:@Weights/weight_in*
validate_shape(*
_output_shapes
:	F�

�
save/Assign_39AssignWeights/weight_in_1save/RestoreV2:39*
validate_shape(*
_output_shapes
:	F�
*
use_locking(*
T0*&
_class
loc:@Weights/weight_in_1
�
save/Assign_40AssignWeights/weight_in_1/Adamsave/RestoreV2:40*
T0*&
_class
loc:@Weights/weight_in_1*
validate_shape(*
_output_shapes
:	F�
*
use_locking(
�
save/Assign_41AssignWeights/weight_in_1/Adam_1save/RestoreV2:41*
use_locking(*
T0*&
_class
loc:@Weights/weight_in_1*
validate_shape(*
_output_shapes
:	F�

�
save/Assign_42AssignWeights/weight_in_2save/RestoreV2:42*
use_locking(*
T0*&
_class
loc:@Weights/weight_in_2*
validate_shape(*
_output_shapes
:	F�

�
save/Assign_43AssignWeights/weight_in_2/Adamsave/RestoreV2:43*
T0*&
_class
loc:@Weights/weight_in_2*
validate_shape(*
_output_shapes
:	F�
*
use_locking(
�
save/Assign_44AssignWeights/weight_in_2/Adam_1save/RestoreV2:44*
use_locking(*
T0*&
_class
loc:@Weights/weight_in_2*
validate_shape(*
_output_shapes
:	F�

�
save/Assign_45AssignWeights/weight_outsave/RestoreV2:45*
validate_shape(*
_output_shapes

:h*
use_locking(*
T0*%
_class
loc:@Weights/weight_out
�
save/Assign_46AssignWeights/weight_out/Adamsave/RestoreV2:46*
use_locking(*
T0*%
_class
loc:@Weights/weight_out*
validate_shape(*
_output_shapes

:h
�
save/Assign_47AssignWeights/weight_out/Adam_1save/RestoreV2:47*
use_locking(*
T0*%
_class
loc:@Weights/weight_out*
validate_shape(*
_output_shapes

:h
�
save/Assign_48AssignWeights/weight_out_1save/RestoreV2:48*
T0*'
_class
loc:@Weights/weight_out_1*
validate_shape(*
_output_shapes

:h*
use_locking(
�
save/Assign_49AssignWeights/weight_out_1/Adamsave/RestoreV2:49*
T0*'
_class
loc:@Weights/weight_out_1*
validate_shape(*
_output_shapes

:h*
use_locking(
�
save/Assign_50AssignWeights/weight_out_1/Adam_1save/RestoreV2:50*
validate_shape(*
_output_shapes

:h*
use_locking(*
T0*'
_class
loc:@Weights/weight_out_1
�
save/Assign_51AssignWeights/weight_out_2save/RestoreV2:51*
validate_shape(*
_output_shapes

:h*
use_locking(*
T0*'
_class
loc:@Weights/weight_out_2
�
save/Assign_52AssignWeights/weight_out_2/Adamsave/RestoreV2:52*
use_locking(*
T0*'
_class
loc:@Weights/weight_out_2*
validate_shape(*
_output_shapes

:h
�
save/Assign_53AssignWeights/weight_out_2/Adam_1save/RestoreV2:53*
use_locking(*
T0*'
_class
loc:@Weights/weight_out_2*
validate_shape(*
_output_shapes

:h
�
save/Assign_54Assignbeta1_powersave/RestoreV2:54*
T0*(
_class
loc:@Weights/bias_hidden_1*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_55Assignbeta2_powersave/RestoreV2:55*
use_locking(*
T0*(
_class
loc:@Weights/bias_hidden_1*
validate_shape(*
_output_shapes
: 
�
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard "<
save/Const:0save/Identity:0save/restore_all (5 @F8"�
trainable_variables��
g
Weights/weight_in:0Weights/weight_in/AssignWeights/weight_in/read:02Weights/truncated_normal:08
V
Weights/bias_in:0Weights/bias_in/AssignWeights/bias_in/read:02Weights/zeros:08
{
Weights/weight_hidden_1:0Weights/weight_hidden_1/AssignWeights/weight_hidden_1/read:02Weights/truncated_normal_1:08
j
Weights/bias_hidden_1:0Weights/bias_hidden_1/AssignWeights/bias_hidden_1/read:02Weights/zeros_1:08
l
Weights/weight_out:0Weights/weight_out/AssignWeights/weight_out/read:02Weights/truncated_normal_2:08
[
Weights/bias_out:0Weights/bias_out/AssignWeights/bias_out/read:02Weights/zeros_2:08
o
Weights/weight_in_1:0Weights/weight_in_1/AssignWeights/weight_in_1/read:02Weights/truncated_normal_3:08
^
Weights/bias_in_1:0Weights/bias_in_1/AssignWeights/bias_in_1/read:02Weights/zeros_3:08
�
Weights/weight_hidden_1_1:0 Weights/weight_hidden_1_1/Assign Weights/weight_hidden_1_1/read:02Weights/truncated_normal_4:08
p
Weights/bias_hidden_1_1:0Weights/bias_hidden_1_1/AssignWeights/bias_hidden_1_1/read:02Weights/zeros_4:08
r
Weights/weight_out_1:0Weights/weight_out_1/AssignWeights/weight_out_1/read:02Weights/truncated_normal_5:08
a
Weights/bias_out_1:0Weights/bias_out_1/AssignWeights/bias_out_1/read:02Weights/zeros_5:08
o
Weights/weight_in_2:0Weights/weight_in_2/AssignWeights/weight_in_2/read:02Weights/truncated_normal_6:08
^
Weights/bias_in_2:0Weights/bias_in_2/AssignWeights/bias_in_2/read:02Weights/zeros_6:08
�
Weights/weight_hidden_1_2:0 Weights/weight_hidden_1_2/Assign Weights/weight_hidden_1_2/read:02Weights/truncated_normal_7:08
p
Weights/bias_hidden_1_2:0Weights/bias_hidden_1_2/AssignWeights/bias_hidden_1_2/read:02Weights/zeros_7:08
r
Weights/weight_out_2:0Weights/weight_out_2/AssignWeights/weight_out_2/read:02Weights/truncated_normal_8:08
a
Weights/bias_out_2:0Weights/bias_out_2/AssignWeights/bias_out_2/read:02Weights/zeros_8:08"
train_op

optimisation_op"�:
	variables�:�:
g
Weights/weight_in:0Weights/weight_in/AssignWeights/weight_in/read:02Weights/truncated_normal:08
V
Weights/bias_in:0Weights/bias_in/AssignWeights/bias_in/read:02Weights/zeros:08
{
Weights/weight_hidden_1:0Weights/weight_hidden_1/AssignWeights/weight_hidden_1/read:02Weights/truncated_normal_1:08
j
Weights/bias_hidden_1:0Weights/bias_hidden_1/AssignWeights/bias_hidden_1/read:02Weights/zeros_1:08
l
Weights/weight_out:0Weights/weight_out/AssignWeights/weight_out/read:02Weights/truncated_normal_2:08
[
Weights/bias_out:0Weights/bias_out/AssignWeights/bias_out/read:02Weights/zeros_2:08
o
Weights/weight_in_1:0Weights/weight_in_1/AssignWeights/weight_in_1/read:02Weights/truncated_normal_3:08
^
Weights/bias_in_1:0Weights/bias_in_1/AssignWeights/bias_in_1/read:02Weights/zeros_3:08
�
Weights/weight_hidden_1_1:0 Weights/weight_hidden_1_1/Assign Weights/weight_hidden_1_1/read:02Weights/truncated_normal_4:08
p
Weights/bias_hidden_1_1:0Weights/bias_hidden_1_1/AssignWeights/bias_hidden_1_1/read:02Weights/zeros_4:08
r
Weights/weight_out_1:0Weights/weight_out_1/AssignWeights/weight_out_1/read:02Weights/truncated_normal_5:08
a
Weights/bias_out_1:0Weights/bias_out_1/AssignWeights/bias_out_1/read:02Weights/zeros_5:08
o
Weights/weight_in_2:0Weights/weight_in_2/AssignWeights/weight_in_2/read:02Weights/truncated_normal_6:08
^
Weights/bias_in_2:0Weights/bias_in_2/AssignWeights/bias_in_2/read:02Weights/zeros_6:08
�
Weights/weight_hidden_1_2:0 Weights/weight_hidden_1_2/Assign Weights/weight_hidden_1_2/read:02Weights/truncated_normal_7:08
p
Weights/bias_hidden_1_2:0Weights/bias_hidden_1_2/AssignWeights/bias_hidden_1_2/read:02Weights/zeros_7:08
r
Weights/weight_out_2:0Weights/weight_out_2/AssignWeights/weight_out_2/read:02Weights/truncated_normal_8:08
a
Weights/bias_out_2:0Weights/bias_out_2/AssignWeights/bias_out_2/read:02Weights/zeros_8:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
�
Weights/weight_in/Adam:0Weights/weight_in/Adam/AssignWeights/weight_in/Adam/read:02*Weights/weight_in/Adam/Initializer/zeros:0
�
Weights/weight_in/Adam_1:0Weights/weight_in/Adam_1/AssignWeights/weight_in/Adam_1/read:02,Weights/weight_in/Adam_1/Initializer/zeros:0
|
Weights/bias_in/Adam:0Weights/bias_in/Adam/AssignWeights/bias_in/Adam/read:02(Weights/bias_in/Adam/Initializer/zeros:0
�
Weights/bias_in/Adam_1:0Weights/bias_in/Adam_1/AssignWeights/bias_in/Adam_1/read:02*Weights/bias_in/Adam_1/Initializer/zeros:0
�
Weights/weight_hidden_1/Adam:0#Weights/weight_hidden_1/Adam/Assign#Weights/weight_hidden_1/Adam/read:020Weights/weight_hidden_1/Adam/Initializer/zeros:0
�
 Weights/weight_hidden_1/Adam_1:0%Weights/weight_hidden_1/Adam_1/Assign%Weights/weight_hidden_1/Adam_1/read:022Weights/weight_hidden_1/Adam_1/Initializer/zeros:0
�
Weights/bias_hidden_1/Adam:0!Weights/bias_hidden_1/Adam/Assign!Weights/bias_hidden_1/Adam/read:02.Weights/bias_hidden_1/Adam/Initializer/zeros:0
�
Weights/bias_hidden_1/Adam_1:0#Weights/bias_hidden_1/Adam_1/Assign#Weights/bias_hidden_1/Adam_1/read:020Weights/bias_hidden_1/Adam_1/Initializer/zeros:0
�
Weights/weight_out/Adam:0Weights/weight_out/Adam/AssignWeights/weight_out/Adam/read:02+Weights/weight_out/Adam/Initializer/zeros:0
�
Weights/weight_out/Adam_1:0 Weights/weight_out/Adam_1/Assign Weights/weight_out/Adam_1/read:02-Weights/weight_out/Adam_1/Initializer/zeros:0
�
Weights/bias_out/Adam:0Weights/bias_out/Adam/AssignWeights/bias_out/Adam/read:02)Weights/bias_out/Adam/Initializer/zeros:0
�
Weights/bias_out/Adam_1:0Weights/bias_out/Adam_1/AssignWeights/bias_out/Adam_1/read:02+Weights/bias_out/Adam_1/Initializer/zeros:0
�
Weights/weight_in_1/Adam:0Weights/weight_in_1/Adam/AssignWeights/weight_in_1/Adam/read:02,Weights/weight_in_1/Adam/Initializer/zeros:0
�
Weights/weight_in_1/Adam_1:0!Weights/weight_in_1/Adam_1/Assign!Weights/weight_in_1/Adam_1/read:02.Weights/weight_in_1/Adam_1/Initializer/zeros:0
�
Weights/bias_in_1/Adam:0Weights/bias_in_1/Adam/AssignWeights/bias_in_1/Adam/read:02*Weights/bias_in_1/Adam/Initializer/zeros:0
�
Weights/bias_in_1/Adam_1:0Weights/bias_in_1/Adam_1/AssignWeights/bias_in_1/Adam_1/read:02,Weights/bias_in_1/Adam_1/Initializer/zeros:0
�
 Weights/weight_hidden_1_1/Adam:0%Weights/weight_hidden_1_1/Adam/Assign%Weights/weight_hidden_1_1/Adam/read:022Weights/weight_hidden_1_1/Adam/Initializer/zeros:0
�
"Weights/weight_hidden_1_1/Adam_1:0'Weights/weight_hidden_1_1/Adam_1/Assign'Weights/weight_hidden_1_1/Adam_1/read:024Weights/weight_hidden_1_1/Adam_1/Initializer/zeros:0
�
Weights/bias_hidden_1_1/Adam:0#Weights/bias_hidden_1_1/Adam/Assign#Weights/bias_hidden_1_1/Adam/read:020Weights/bias_hidden_1_1/Adam/Initializer/zeros:0
�
 Weights/bias_hidden_1_1/Adam_1:0%Weights/bias_hidden_1_1/Adam_1/Assign%Weights/bias_hidden_1_1/Adam_1/read:022Weights/bias_hidden_1_1/Adam_1/Initializer/zeros:0
�
Weights/weight_out_1/Adam:0 Weights/weight_out_1/Adam/Assign Weights/weight_out_1/Adam/read:02-Weights/weight_out_1/Adam/Initializer/zeros:0
�
Weights/weight_out_1/Adam_1:0"Weights/weight_out_1/Adam_1/Assign"Weights/weight_out_1/Adam_1/read:02/Weights/weight_out_1/Adam_1/Initializer/zeros:0
�
Weights/bias_out_1/Adam:0Weights/bias_out_1/Adam/AssignWeights/bias_out_1/Adam/read:02+Weights/bias_out_1/Adam/Initializer/zeros:0
�
Weights/bias_out_1/Adam_1:0 Weights/bias_out_1/Adam_1/Assign Weights/bias_out_1/Adam_1/read:02-Weights/bias_out_1/Adam_1/Initializer/zeros:0
�
Weights/weight_in_2/Adam:0Weights/weight_in_2/Adam/AssignWeights/weight_in_2/Adam/read:02,Weights/weight_in_2/Adam/Initializer/zeros:0
�
Weights/weight_in_2/Adam_1:0!Weights/weight_in_2/Adam_1/Assign!Weights/weight_in_2/Adam_1/read:02.Weights/weight_in_2/Adam_1/Initializer/zeros:0
�
Weights/bias_in_2/Adam:0Weights/bias_in_2/Adam/AssignWeights/bias_in_2/Adam/read:02*Weights/bias_in_2/Adam/Initializer/zeros:0
�
Weights/bias_in_2/Adam_1:0Weights/bias_in_2/Adam_1/AssignWeights/bias_in_2/Adam_1/read:02,Weights/bias_in_2/Adam_1/Initializer/zeros:0
�
 Weights/weight_hidden_1_2/Adam:0%Weights/weight_hidden_1_2/Adam/Assign%Weights/weight_hidden_1_2/Adam/read:022Weights/weight_hidden_1_2/Adam/Initializer/zeros:0
�
"Weights/weight_hidden_1_2/Adam_1:0'Weights/weight_hidden_1_2/Adam_1/Assign'Weights/weight_hidden_1_2/Adam_1/read:024Weights/weight_hidden_1_2/Adam_1/Initializer/zeros:0
�
Weights/bias_hidden_1_2/Adam:0#Weights/bias_hidden_1_2/Adam/Assign#Weights/bias_hidden_1_2/Adam/read:020Weights/bias_hidden_1_2/Adam/Initializer/zeros:0
�
 Weights/bias_hidden_1_2/Adam_1:0%Weights/bias_hidden_1_2/Adam_1/Assign%Weights/bias_hidden_1_2/Adam_1/read:022Weights/bias_hidden_1_2/Adam_1/Initializer/zeros:0
�
Weights/weight_out_2/Adam:0 Weights/weight_out_2/Adam/Assign Weights/weight_out_2/Adam/read:02-Weights/weight_out_2/Adam/Initializer/zeros:0
�
Weights/weight_out_2/Adam_1:0"Weights/weight_out_2/Adam_1/Assign"Weights/weight_out_2/Adam_1/read:02/Weights/weight_out_2/Adam_1/Initializer/zeros:0
�
Weights/bias_out_2/Adam:0Weights/bias_out_2/Adam/AssignWeights/bias_out_2/Adam/read:02+Weights/bias_out_2/Adam/Initializer/zeros:0
�
Weights/bias_out_2/Adam_1:0 Weights/bias_out_2/Adam_1/Assign Weights/bias_out_2/Adam_1/read:02-Weights/bias_out_2/Adam_1/Initializer/zeros:0"8
	iterators+
)
Data/Iterator:0
Inputs_pred/Iterator:0*�
serving_default�
=
Data/Properties:0(
Data/Properties:0���������
E
Data/Atomic-numbers:0,
Data/Atomic-numbers:0���������
D
Data/Descriptors:0.
Data/Descriptors:0����������
7
Model/output:0%
Model/output:0���������tensorflow/serving/predict