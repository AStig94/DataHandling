Ņ:
Øł
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

Ą
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

;
Elu
features"T
activations"T"
Ttype:
2
ś
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%·Ń8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
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
@
ReadVariableOp
resource
value"dtype"
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
Į
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ø
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.7.02v2.7.0-rc1-69-gc256c071bb28®ś4

batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma

-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0

batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta

,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0

batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean

3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0

#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance

7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:@*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:@*
dtype0

batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_1/gamma

/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:@*
dtype0

batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_1/beta

.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:@*
dtype0

!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_1/moving_mean

5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:@*
dtype0
¢
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_1/moving_variance

9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:@*
dtype0

conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_1/kernel
|
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*'
_output_shapes
:@*
dtype0
s
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
l
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes	
:*
dtype0

batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_2/gamma

/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes	
:*
dtype0

batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_2/beta

.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes	
:*
dtype0

!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_2/moving_mean

5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes	
:*
dtype0
£
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_2/moving_variance

9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes	
:*
dtype0

conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_2/kernel
}
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*(
_output_shapes
:*
dtype0
s
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_2/bias
l
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes	
:*
dtype0

batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_3/gamma

/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes	
:*
dtype0

batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_3/beta

.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes	
:*
dtype0

!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_3/moving_mean

5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes	
:*
dtype0
£
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_3/moving_variance

9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes	
:*
dtype0

conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_3/kernel
}
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:*
dtype0
s
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_3/bias
l
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes	
:*
dtype0

batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_4/gamma

/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes	
:*
dtype0

batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_4/beta

.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes	
:*
dtype0

!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_4/moving_mean

5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes	
:*
dtype0
£
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_4/moving_variance

9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes	
:*
dtype0

conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose/kernel

+conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose/kernel*(
_output_shapes
:*
dtype0

conv2d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameconv2d_transpose/bias
|
)conv2d_transpose/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose/bias*
_output_shapes	
:*
dtype0

batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_5/gamma

/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes	
:*
dtype0

batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_5/beta

.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes	
:*
dtype0

!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_5/moving_mean

5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes	
:*
dtype0
£
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_5/moving_variance

9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes	
:*
dtype0

conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2d_transpose_1/kernel

-conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/kernel*(
_output_shapes
:*
dtype0

conv2d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_1/bias

+conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/bias*
_output_shapes	
:*
dtype0

batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_6/gamma

/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes	
:*
dtype0

batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_6/beta

.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes	
:*
dtype0

!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_6/moving_mean

5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes	
:*
dtype0
£
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_6/moving_variance

9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes	
:*
dtype0

conv2d_transpose_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2d_transpose_2/kernel

-conv2d_transpose_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/kernel*(
_output_shapes
:*
dtype0

conv2d_transpose_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_2/bias

+conv2d_transpose_2/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/bias*
_output_shapes	
:*
dtype0

batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_7/gamma

/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes	
:*
dtype0

batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_7/beta

.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes	
:*
dtype0

!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_7/moving_mean

5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes	
:*
dtype0
£
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_7/moving_variance

9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes	
:*
dtype0

conv2d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@Ą**
shared_nameconv2d_transpose_3/kernel

-conv2d_transpose_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/kernel*'
_output_shapes
:@Ą*
dtype0

conv2d_transpose_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameconv2d_transpose_3/bias

+conv2d_transpose_3/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/bias*
_output_shapes
:@*
dtype0

batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_8/gamma

/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes
:@*
dtype0

batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_8/beta

.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes
:@*
dtype0

!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_8/moving_mean

5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes
:@*
dtype0
¢
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_8/moving_variance

9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes
:@*
dtype0

conv2d_transpose_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:D**
shared_nameconv2d_transpose_4/kernel

-conv2d_transpose_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_4/kernel*&
_output_shapes
:D*
dtype0

conv2d_transpose_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_4/bias

+conv2d_transpose_4/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_4/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/batch_normalization/gamma/m

4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes
:*
dtype0

Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/batch_normalization/beta/m

3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes
:*
dtype0

Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d/kernel/m

(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:@*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:@*
dtype0

"Adam/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_1/gamma/m

6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/m*
_output_shapes
:@*
dtype0

!Adam/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_1/beta/m

5Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/m*
_output_shapes
:@*
dtype0

Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_1/kernel/m

*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*'
_output_shapes
:@*
dtype0

Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_1/bias/m
z
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_2/gamma/m

6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/m*
_output_shapes	
:*
dtype0

!Adam/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_2/beta/m

5Adam/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/m*
_output_shapes	
:*
dtype0

Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_2/kernel/m

*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_2/bias/m
z
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_3/gamma/m

6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/m*
_output_shapes	
:*
dtype0

!Adam/batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_3/beta/m

5Adam/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/m*
_output_shapes	
:*
dtype0

Adam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_3/kernel/m

*Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_3/bias/m
z
(Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/m*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_4/gamma/m

6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/m*
_output_shapes	
:*
dtype0

!Adam/batch_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_4/beta/m

5Adam/batch_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/m*
_output_shapes	
:*
dtype0
¢
Adam/conv2d_transpose/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose/kernel/m

2Adam/conv2d_transpose/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_transpose/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/conv2d_transpose/bias/m

0Adam/conv2d_transpose/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/bias/m*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_5/gamma/m

6Adam/batch_normalization_5/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_5/gamma/m*
_output_shapes	
:*
dtype0

!Adam/batch_normalization_5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_5/beta/m

5Adam/batch_normalization_5/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_5/beta/m*
_output_shapes	
:*
dtype0
¦
 Adam/conv2d_transpose_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2d_transpose_1/kernel/m

4Adam/conv2d_transpose_1/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_1/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_transpose_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_1/bias/m

2Adam/conv2d_transpose_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_1/bias/m*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_6/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_6/gamma/m

6Adam/batch_normalization_6/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_6/gamma/m*
_output_shapes	
:*
dtype0

!Adam/batch_normalization_6/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_6/beta/m

5Adam/batch_normalization_6/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_6/beta/m*
_output_shapes	
:*
dtype0
¦
 Adam/conv2d_transpose_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2d_transpose_2/kernel/m

4Adam/conv2d_transpose_2/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_2/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_transpose_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_2/bias/m

2Adam/conv2d_transpose_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_2/bias/m*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_7/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_7/gamma/m

6Adam/batch_normalization_7/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/m*
_output_shapes	
:*
dtype0

!Adam/batch_normalization_7/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_7/beta/m

5Adam/batch_normalization_7/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/m*
_output_shapes	
:*
dtype0
„
 Adam/conv2d_transpose_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@Ą*1
shared_name" Adam/conv2d_transpose_3/kernel/m

4Adam/conv2d_transpose_3/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_3/kernel/m*'
_output_shapes
:@Ą*
dtype0

Adam/conv2d_transpose_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/conv2d_transpose_3/bias/m

2Adam/conv2d_transpose_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_3/bias/m*
_output_shapes
:@*
dtype0

"Adam/batch_normalization_8/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_8/gamma/m

6Adam/batch_normalization_8/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_8/gamma/m*
_output_shapes
:@*
dtype0

!Adam/batch_normalization_8/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_8/beta/m

5Adam/batch_normalization_8/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_8/beta/m*
_output_shapes
:@*
dtype0
¤
 Adam/conv2d_transpose_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:D*1
shared_name" Adam/conv2d_transpose_4/kernel/m

4Adam/conv2d_transpose_4/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_4/kernel/m*&
_output_shapes
:D*
dtype0

Adam/conv2d_transpose_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_4/bias/m

2Adam/conv2d_transpose_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_4/bias/m*
_output_shapes
:*
dtype0

 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/batch_normalization/gamma/v

4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes
:*
dtype0

Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/batch_normalization/beta/v

3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes
:*
dtype0

Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d/kernel/v

(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:@*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:@*
dtype0

"Adam/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_1/gamma/v

6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/v*
_output_shapes
:@*
dtype0

!Adam/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_1/beta/v

5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/v*
_output_shapes
:@*
dtype0

Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_1/kernel/v

*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*'
_output_shapes
:@*
dtype0

Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_1/bias/v
z
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_2/gamma/v

6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/v*
_output_shapes	
:*
dtype0

!Adam/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_2/beta/v

5Adam/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/v*
_output_shapes	
:*
dtype0

Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_2/kernel/v

*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_2/bias/v
z
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_3/gamma/v

6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/v*
_output_shapes	
:*
dtype0

!Adam/batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_3/beta/v

5Adam/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/v*
_output_shapes	
:*
dtype0

Adam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_3/kernel/v

*Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_3/bias/v
z
(Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/v*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_4/gamma/v

6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/v*
_output_shapes	
:*
dtype0

!Adam/batch_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_4/beta/v

5Adam/batch_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/v*
_output_shapes	
:*
dtype0
¢
Adam/conv2d_transpose/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose/kernel/v

2Adam/conv2d_transpose/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_transpose/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/conv2d_transpose/bias/v

0Adam/conv2d_transpose/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/bias/v*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_5/gamma/v

6Adam/batch_normalization_5/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_5/gamma/v*
_output_shapes	
:*
dtype0

!Adam/batch_normalization_5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_5/beta/v

5Adam/batch_normalization_5/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_5/beta/v*
_output_shapes	
:*
dtype0
¦
 Adam/conv2d_transpose_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2d_transpose_1/kernel/v

4Adam/conv2d_transpose_1/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_1/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_transpose_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_1/bias/v

2Adam/conv2d_transpose_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_1/bias/v*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_6/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_6/gamma/v

6Adam/batch_normalization_6/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_6/gamma/v*
_output_shapes	
:*
dtype0

!Adam/batch_normalization_6/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_6/beta/v

5Adam/batch_normalization_6/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_6/beta/v*
_output_shapes	
:*
dtype0
¦
 Adam/conv2d_transpose_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2d_transpose_2/kernel/v

4Adam/conv2d_transpose_2/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_2/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_transpose_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_2/bias/v

2Adam/conv2d_transpose_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_2/bias/v*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_7/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_7/gamma/v

6Adam/batch_normalization_7/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/v*
_output_shapes	
:*
dtype0

!Adam/batch_normalization_7/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_7/beta/v

5Adam/batch_normalization_7/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/v*
_output_shapes	
:*
dtype0
„
 Adam/conv2d_transpose_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@Ą*1
shared_name" Adam/conv2d_transpose_3/kernel/v

4Adam/conv2d_transpose_3/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_3/kernel/v*'
_output_shapes
:@Ą*
dtype0

Adam/conv2d_transpose_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/conv2d_transpose_3/bias/v

2Adam/conv2d_transpose_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_3/bias/v*
_output_shapes
:@*
dtype0

"Adam/batch_normalization_8/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_8/gamma/v

6Adam/batch_normalization_8/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_8/gamma/v*
_output_shapes
:@*
dtype0

!Adam/batch_normalization_8/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_8/beta/v

5Adam/batch_normalization_8/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_8/beta/v*
_output_shapes
:@*
dtype0
¤
 Adam/conv2d_transpose_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:D*1
shared_name" Adam/conv2d_transpose_4/kernel/v

4Adam/conv2d_transpose_4/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_4/kernel/v*&
_output_shapes
:D*
dtype0

Adam/conv2d_transpose_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_4/bias/v

2Adam/conv2d_transpose_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_4/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
¾š
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ųļ
valueķļBéļ Bįļ
É
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer_with_weights-0
layer-13
layer_with_weights-1
layer-14
layer_with_weights-2
layer-15
layer_with_weights-3
layer-16
layer_with_weights-4
layer-17
layer_with_weights-5
layer-18
layer_with_weights-6
layer-19
layer_with_weights-7
layer-20
layer_with_weights-8
layer-21
layer-22
layer_with_weights-9
layer-23
layer_with_weights-10
layer-24
layer-25
layer_with_weights-11
layer-26
layer_with_weights-12
layer-27
layer-28
layer_with_weights-13
layer-29
layer_with_weights-14
layer-30
 layer-31
!layer_with_weights-15
!layer-32
"layer_with_weights-16
"layer-33
#layer-34
$layer_with_weights-17
$layer-35
%layer-36
&	optimizer
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+
signatures
 
 
 
 
R
,	variables
-trainable_variables
.regularization_losses
/	keras_api
R
0	variables
1trainable_variables
2regularization_losses
3	keras_api
R
4	variables
5trainable_variables
6regularization_losses
7	keras_api
R
8	variables
9trainable_variables
:regularization_losses
;	keras_api
R
<	variables
=trainable_variables
>regularization_losses
?	keras_api
R
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
R
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
R
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
R
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api

Paxis
	Qgamma
Rbeta
Smoving_mean
Tmoving_variance
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
h

Ykernel
Zbias
[	variables
\trainable_variables
]regularization_losses
^	keras_api

_axis
	`gamma
abeta
bmoving_mean
cmoving_variance
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h

hkernel
ibias
j	variables
ktrainable_variables
lregularization_losses
m	keras_api

naxis
	ogamma
pbeta
qmoving_mean
rmoving_variance
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
h

wkernel
xbias
y	variables
ztrainable_variables
{regularization_losses
|	keras_api

}axis
	~gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
 
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
 
	axis

 gamma
	”beta
¢moving_mean
£moving_variance
¤	variables
„trainable_variables
¦regularization_losses
§	keras_api
V
Ø	variables
©trainable_variables
Ŗregularization_losses
«	keras_api
n
¬kernel
	­bias
®	variables
Ætrainable_variables
°regularization_losses
±	keras_api
 
	²axis

³gamma
	“beta
µmoving_mean
¶moving_variance
·	variables
øtrainable_variables
¹regularization_losses
ŗ	keras_api
V
»	variables
¼trainable_variables
½regularization_losses
¾	keras_api
n
ækernel
	Ąbias
Į	variables
Ātrainable_variables
Ćregularization_losses
Ä	keras_api
 
	Åaxis

Ęgamma
	Ēbeta
Čmoving_mean
Émoving_variance
Ź	variables
Ėtrainable_variables
Ģregularization_losses
Ķ	keras_api
V
Ī	variables
Ļtrainable_variables
Šregularization_losses
Ń	keras_api
n
Ņkernel
	Óbias
Ō	variables
Õtrainable_variables
Öregularization_losses
×	keras_api
 
	Ųaxis

Łgamma
	Śbeta
Ūmoving_mean
Ümoving_variance
Ż	variables
Žtrainable_variables
ßregularization_losses
ą	keras_api
V
į	variables
ātrainable_variables
ćregularization_losses
ä	keras_api
n
åkernel
	ębias
ē	variables
čtrainable_variables
éregularization_losses
ź	keras_api
V
ė	variables
ģtrainable_variables
ķregularization_losses
ī	keras_api
Į
	ļiter
šbeta_1
ńbeta_2

ņdecay
ólearning_rateQm£Rm¤Ym„Zm¦`m§amØhm©imŖom«pm¬wm­xm®~mÆm°	m±	m²	m³	m“	mµ	m¶	 m·	”mø	¬m¹	­mŗ	³m»	“m¼	æm½	Ąm¾	Ęmæ	ĒmĄ	ŅmĮ	ÓmĀ	ŁmĆ	ŚmÄ	åmÅ	ęmĘQvĒRvČYvÉZvŹ`vĖavĢhvĶivĪovĻpvŠwvŃxvŅ~vÓvŌ	vÕ	vÖ	v×	vŲ	vŁ	vŚ	 vŪ	”vÜ	¬vŻ	­vŽ	³vß	“vą	ævį	Ąvā	Ęvć	Ēvä	Ņvå	Óvę	Łvē	Śvč	åvé	ęvź
Č
Q0
R1
S2
T3
Y4
Z5
`6
a7
b8
c9
h10
i11
o12
p13
q14
r15
w16
x17
~18
19
20
21
22
23
24
25
26
27
28
29
 30
”31
¢32
£33
¬34
­35
³36
“37
µ38
¶39
æ40
Ą41
Ę42
Ē43
Č44
É45
Ņ46
Ó47
Ł48
Ś49
Ū50
Ü51
å52
ę53
¬
Q0
R1
Y2
Z3
`4
a5
h6
i7
o8
p9
w10
x11
~12
13
14
15
16
17
18
19
 20
”21
¬22
­23
³24
“25
æ26
Ą27
Ę28
Ē29
Ņ30
Ó31
Ł32
Ś33
å34
ę35
 
²
ōnon_trainable_variables
õlayers
ömetrics
 ÷layer_regularization_losses
ųlayer_metrics
'	variables
(trainable_variables
)regularization_losses
 
 
 
 
²
łnon_trainable_variables
ślayers
ūmetrics
 ülayer_regularization_losses
żlayer_metrics
,	variables
-trainable_variables
.regularization_losses
 
 
 
²
žnon_trainable_variables
’layers
metrics
 layer_regularization_losses
layer_metrics
0	variables
1trainable_variables
2regularization_losses
 
 
 
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
4	variables
5trainable_variables
6regularization_losses
 
 
 
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
8	variables
9trainable_variables
:regularization_losses
 
 
 
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
<	variables
=trainable_variables
>regularization_losses
 
 
 
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
 
 
 
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
 
 
 
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
 
 
 
²
”non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
„layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

Q0
R1
S2
T3

Q0
R1
 
²
¦non_trainable_variables
§layers
Ømetrics
 ©layer_regularization_losses
Ŗlayer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

Y0
Z1

Y0
Z1
 
²
«non_trainable_variables
¬layers
­metrics
 ®layer_regularization_losses
Ælayer_metrics
[	variables
\trainable_variables
]regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

`0
a1
b2
c3

`0
a1
 
²
°non_trainable_variables
±layers
²metrics
 ³layer_regularization_losses
“layer_metrics
d	variables
etrainable_variables
fregularization_losses
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

h0
i1

h0
i1
 
²
µnon_trainable_variables
¶layers
·metrics
 ølayer_regularization_losses
¹layer_metrics
j	variables
ktrainable_variables
lregularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

o0
p1
q2
r3

o0
p1
 
²
ŗnon_trainable_variables
»layers
¼metrics
 ½layer_regularization_losses
¾layer_metrics
s	variables
ttrainable_variables
uregularization_losses
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

w0
x1

w0
x1
 
²
ænon_trainable_variables
Ąlayers
Įmetrics
 Ālayer_regularization_losses
Ćlayer_metrics
y	variables
ztrainable_variables
{regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

~0
1
2
3

~0
1
 
µ
Änon_trainable_variables
Ålayers
Ęmetrics
 Ēlayer_regularization_losses
Člayer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
µ
Énon_trainable_variables
Źlayers
Ėmetrics
 Ģlayer_regularization_losses
Ķlayer_metrics
	variables
trainable_variables
regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
0
1
2
3

0
1
 
µ
Īnon_trainable_variables
Ļlayers
Šmetrics
 Ńlayer_regularization_losses
Ņlayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
µ
Ónon_trainable_variables
Ōlayers
Õmetrics
 Ölayer_regularization_losses
×layer_metrics
	variables
trainable_variables
regularization_losses
ca
VARIABLE_VALUEconv2d_transpose/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_transpose/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
µ
Ųnon_trainable_variables
Łlayers
Śmetrics
 Ūlayer_regularization_losses
Ülayer_metrics
	variables
trainable_variables
regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_5/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_5/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_5/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_5/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 0
”1
¢2
£3

 0
”1
 
µ
Żnon_trainable_variables
Žlayers
ßmetrics
 ąlayer_regularization_losses
įlayer_metrics
¤	variables
„trainable_variables
¦regularization_losses
 
 
 
µ
ānon_trainable_variables
ćlayers
ämetrics
 ålayer_regularization_losses
ęlayer_metrics
Ø	variables
©trainable_variables
Ŗregularization_losses
fd
VARIABLE_VALUEconv2d_transpose_1/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_1/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

¬0
­1

¬0
­1
 
µ
ēnon_trainable_variables
člayers
émetrics
 źlayer_regularization_losses
ėlayer_metrics
®	variables
Ætrainable_variables
°regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_6/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_6/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_6/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_6/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
³0
“1
µ2
¶3

³0
“1
 
µ
ģnon_trainable_variables
ķlayers
īmetrics
 ļlayer_regularization_losses
šlayer_metrics
·	variables
øtrainable_variables
¹regularization_losses
 
 
 
µ
ńnon_trainable_variables
ņlayers
ómetrics
 ōlayer_regularization_losses
õlayer_metrics
»	variables
¼trainable_variables
½regularization_losses
fd
VARIABLE_VALUEconv2d_transpose_2/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_2/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE

æ0
Ą1

æ0
Ą1
 
µ
önon_trainable_variables
÷layers
ųmetrics
 łlayer_regularization_losses
ślayer_metrics
Į	variables
Ātrainable_variables
Ćregularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_7/gamma6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_7/beta5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_7/moving_mean<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_7/moving_variance@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
Ę0
Ē1
Č2
É3

Ę0
Ē1
 
µ
ūnon_trainable_variables
ülayers
żmetrics
 žlayer_regularization_losses
’layer_metrics
Ź	variables
Ėtrainable_variables
Ģregularization_losses
 
 
 
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ī	variables
Ļtrainable_variables
Šregularization_losses
fd
VARIABLE_VALUEconv2d_transpose_3/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_3/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE

Ņ0
Ó1

Ņ0
Ó1
 
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ō	variables
Õtrainable_variables
Öregularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_8/gamma6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_8/beta5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_8/moving_mean<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_8/moving_variance@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
Ł0
Ś1
Ū2
Ü3

Ł0
Ś1
 
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ż	variables
Žtrainable_variables
ßregularization_losses
 
 
 
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
į	variables
ātrainable_variables
ćregularization_losses
fd
VARIABLE_VALUEconv2d_transpose_4/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_4/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE

å0
ę1

å0
ę1
 
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ē	variables
čtrainable_variables
éregularization_losses
 
 
 
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ė	variables
ģtrainable_variables
ķregularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

S0
T1
b2
c3
q4
r5
6
7
8
9
¢10
£11
µ12
¶13
Č14
É15
Ū16
Ü17

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

S0
T1
 
 
 
 
 
 
 
 
 

b0
c1
 
 
 
 
 
 
 
 
 

q0
r1
 
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

¢0
£1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

µ0
¶1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

Č0
É1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

Ū0
Ü1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

total

 count
”	variables
¢	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
 1

”	variables

VARIABLE_VALUE Adam/batch_normalization/gamma/mQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/batch_normalization/beta/mPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_1/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_2/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_2/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_3/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_3/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_4/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_4/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/conv2d_transpose/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/conv2d_transpose/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_5/gamma/mRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_5/beta/mQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/conv2d_transpose_1/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/conv2d_transpose_1/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_6/gamma/mRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_6/beta/mQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/conv2d_transpose_2/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/conv2d_transpose_2/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_7/gamma/mRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_7/beta/mQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/conv2d_transpose_3/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/conv2d_transpose_3/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_8/gamma/mRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_8/beta/mQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/conv2d_transpose_4/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/conv2d_transpose_4/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/batch_normalization/gamma/vQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/batch_normalization/beta/vPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_1/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_2/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_2/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_3/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_3/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_4/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_4/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/conv2d_transpose/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/conv2d_transpose/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_5/gamma/vRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_5/beta/vQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/conv2d_transpose_1/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/conv2d_transpose_1/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_6/gamma/vRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_6/beta/vQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/conv2d_transpose_2/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/conv2d_transpose_2/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_7/gamma/vRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_7/beta/vQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/conv2d_transpose_3/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/conv2d_transpose_3/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_8/gamma/vRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_8/beta/vQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/conv2d_transpose_4/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/conv2d_transpose_4/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_pr0.71Placeholder*-
_output_shapes
:’’’’’’’’’*
dtype0*"
shape:’’’’’’’’’

serving_default_u_velPlaceholder*-
_output_shapes
:’’’’’’’’’*
dtype0*"
shape:’’’’’’’’’

serving_default_v_velPlaceholder*-
_output_shapes
:’’’’’’’’’*
dtype0*"
shape:’’’’’’’’’

serving_default_w_velPlaceholder*-
_output_shapes
:’’’’’’’’’*
dtype0*"
shape:’’’’’’’’’

StatefulPartitionedCallStatefulPartitionedCallserving_default_pr0.71serving_default_u_velserving_default_v_velserving_default_w_velbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d/kernelconv2d/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_3/kernelconv2d_3/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceconv2d_transpose/kernelconv2d_transpose/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_transpose_1/kernelconv2d_transpose_1/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv2d_transpose_2/kernelconv2d_transpose_2/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_transpose_3/kernelconv2d_transpose_3/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv2d_transpose_4/kernelconv2d_transpose_4/bias*E
Tin>
<2:*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456789*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_1288018
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ų6
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp+conv2d_transpose/kernel/Read/ReadVariableOp)conv2d_transpose/bias/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp-conv2d_transpose_1/kernel/Read/ReadVariableOp+conv2d_transpose_1/bias/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp-conv2d_transpose_2/kernel/Read/ReadVariableOp+conv2d_transpose_2/bias/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp-conv2d_transpose_3/kernel/Read/ReadVariableOp+conv2d_transpose_3/bias/Read/ReadVariableOp/batch_normalization_8/gamma/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOp-conv2d_transpose_4/kernel/Read/ReadVariableOp+conv2d_transpose_4/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_1/beta/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_2/beta/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_3/beta/m/Read/ReadVariableOp*Adam/conv2d_3/kernel/m/Read/ReadVariableOp(Adam/conv2d_3/bias/m/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_4/beta/m/Read/ReadVariableOp2Adam/conv2d_transpose/kernel/m/Read/ReadVariableOp0Adam/conv2d_transpose/bias/m/Read/ReadVariableOp6Adam/batch_normalization_5/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_5/beta/m/Read/ReadVariableOp4Adam/conv2d_transpose_1/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose_1/bias/m/Read/ReadVariableOp6Adam/batch_normalization_6/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_6/beta/m/Read/ReadVariableOp4Adam/conv2d_transpose_2/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose_2/bias/m/Read/ReadVariableOp6Adam/batch_normalization_7/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_7/beta/m/Read/ReadVariableOp4Adam/conv2d_transpose_3/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose_3/bias/m/Read/ReadVariableOp6Adam/batch_normalization_8/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_8/beta/m/Read/ReadVariableOp4Adam/conv2d_transpose_4/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose_4/bias/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_1/beta/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_2/beta/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_3/beta/v/Read/ReadVariableOp*Adam/conv2d_3/kernel/v/Read/ReadVariableOp(Adam/conv2d_3/bias/v/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_4/beta/v/Read/ReadVariableOp2Adam/conv2d_transpose/kernel/v/Read/ReadVariableOp0Adam/conv2d_transpose/bias/v/Read/ReadVariableOp6Adam/batch_normalization_5/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_5/beta/v/Read/ReadVariableOp4Adam/conv2d_transpose_1/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose_1/bias/v/Read/ReadVariableOp6Adam/batch_normalization_6/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_6/beta/v/Read/ReadVariableOp4Adam/conv2d_transpose_2/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose_2/bias/v/Read/ReadVariableOp6Adam/batch_normalization_7/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_7/beta/v/Read/ReadVariableOp4Adam/conv2d_transpose_3/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose_3/bias/v/Read/ReadVariableOp6Adam/batch_normalization_8/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_8/beta/v/Read/ReadVariableOp4Adam/conv2d_transpose_4/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose_4/bias/v/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__traced_save_1291361
ļ!
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d/kernelconv2d/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_3/kernelconv2d_3/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceconv2d_transpose/kernelconv2d_transpose/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_transpose_1/kernelconv2d_transpose_1/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv2d_transpose_2/kernelconv2d_transpose_2/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_transpose_3/kernelconv2d_transpose_3/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv2d_transpose_4/kernelconv2d_transpose_4/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/mAdam/conv2d/kernel/mAdam/conv2d/bias/m"Adam/batch_normalization_1/gamma/m!Adam/batch_normalization_1/beta/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/m"Adam/batch_normalization_2/gamma/m!Adam/batch_normalization_2/beta/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/m"Adam/batch_normalization_3/gamma/m!Adam/batch_normalization_3/beta/mAdam/conv2d_3/kernel/mAdam/conv2d_3/bias/m"Adam/batch_normalization_4/gamma/m!Adam/batch_normalization_4/beta/mAdam/conv2d_transpose/kernel/mAdam/conv2d_transpose/bias/m"Adam/batch_normalization_5/gamma/m!Adam/batch_normalization_5/beta/m Adam/conv2d_transpose_1/kernel/mAdam/conv2d_transpose_1/bias/m"Adam/batch_normalization_6/gamma/m!Adam/batch_normalization_6/beta/m Adam/conv2d_transpose_2/kernel/mAdam/conv2d_transpose_2/bias/m"Adam/batch_normalization_7/gamma/m!Adam/batch_normalization_7/beta/m Adam/conv2d_transpose_3/kernel/mAdam/conv2d_transpose_3/bias/m"Adam/batch_normalization_8/gamma/m!Adam/batch_normalization_8/beta/m Adam/conv2d_transpose_4/kernel/mAdam/conv2d_transpose_4/bias/m Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/vAdam/conv2d/kernel/vAdam/conv2d/bias/v"Adam/batch_normalization_1/gamma/v!Adam/batch_normalization_1/beta/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/v"Adam/batch_normalization_2/gamma/v!Adam/batch_normalization_2/beta/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/v"Adam/batch_normalization_3/gamma/v!Adam/batch_normalization_3/beta/vAdam/conv2d_3/kernel/vAdam/conv2d_3/bias/v"Adam/batch_normalization_4/gamma/v!Adam/batch_normalization_4/beta/vAdam/conv2d_transpose/kernel/vAdam/conv2d_transpose/bias/v"Adam/batch_normalization_5/gamma/v!Adam/batch_normalization_5/beta/v Adam/conv2d_transpose_1/kernel/vAdam/conv2d_transpose_1/bias/v"Adam/batch_normalization_6/gamma/v!Adam/batch_normalization_6/beta/v Adam/conv2d_transpose_2/kernel/vAdam/conv2d_transpose_2/bias/v"Adam/batch_normalization_7/gamma/v!Adam/batch_normalization_7/beta/v Adam/conv2d_transpose_3/kernel/vAdam/conv2d_transpose_3/bias/v"Adam/batch_normalization_8/gamma/v!Adam/batch_normalization_8/beta/v Adam/conv2d_transpose_4/kernel/vAdam/conv2d_transpose_4/bias/v*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__traced_restore_1291770¬ī/
«
Å
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1289807

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ė
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
ä
c
G__inference_cropping2d_layer_call_and_return_conditional_losses_1290936

inputs
identityl
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"              n
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"    ų’’’ų’’’    n
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ī
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:’’’’’’’’’*

begin_mask	*
end_mask	h
IdentityIdentitystrided_slice:output:0*
T0*1
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:’’’’’’’’’:Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ū
Į
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1289483

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs

”
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1290367

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0½
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
is_training( n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs

¬
4__inference_conv2d_transpose_1_layer_call_fn_1290199

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallņ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1286203z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
ś#

O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1290454

inputsD
(conv2d_transpose_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B :F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype0Ž
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’i
EluEluBiasAdd:output:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’{
IdentityIdentityElu:activations:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ņ
R
6__inference_periodic_padding2d_2_layer_call_fn_1289105

inputs
identityÅ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_periodic_padding2d_2_layer_call_and_return_conditional_losses_1285774f
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:’’’’’’’’’:U Q
-
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
«
Å
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1289951

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ė
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
ė
Å
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1285121

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ū
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
«
Å
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1286638

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ė
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
Ż
”
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1290114

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ķ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
«
Å
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1290168

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ė
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs


R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1285964

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
is_training( m
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
±ų
­2
B__inference_model_layer_call_and_return_conditional_losses_1288646
inputs_0
inputs_1
inputs_2
inputs_39
+batch_normalization_readvariableop_resource:;
-batch_normalization_readvariableop_1_resource:J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:?
%conv2d_conv2d_readvariableop_resource:@4
&conv2d_biasadd_readvariableop_resource:@;
-batch_normalization_1_readvariableop_resource:@=
/batch_normalization_1_readvariableop_1_resource:@L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@B
'conv2d_1_conv2d_readvariableop_resource:@7
(conv2d_1_biasadd_readvariableop_resource:	<
-batch_normalization_2_readvariableop_resource:	>
/batch_normalization_2_readvariableop_1_resource:	M
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	C
'conv2d_2_conv2d_readvariableop_resource:7
(conv2d_2_biasadd_readvariableop_resource:	<
-batch_normalization_3_readvariableop_resource:	>
/batch_normalization_3_readvariableop_1_resource:	M
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	C
'conv2d_3_conv2d_readvariableop_resource:7
(conv2d_3_biasadd_readvariableop_resource:	<
-batch_normalization_4_readvariableop_resource:	>
/batch_normalization_4_readvariableop_1_resource:	M
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	U
9conv2d_transpose_conv2d_transpose_readvariableop_resource:?
0conv2d_transpose_biasadd_readvariableop_resource:	<
-batch_normalization_5_readvariableop_resource:	>
/batch_normalization_5_readvariableop_1_resource:	M
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:	W
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:A
2conv2d_transpose_1_biasadd_readvariableop_resource:	<
-batch_normalization_6_readvariableop_resource:	>
/batch_normalization_6_readvariableop_1_resource:	M
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:	W
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource:A
2conv2d_transpose_2_biasadd_readvariableop_resource:	<
-batch_normalization_7_readvariableop_resource:	>
/batch_normalization_7_readvariableop_1_resource:	M
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:	V
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource:@Ą@
2conv2d_transpose_3_biasadd_readvariableop_resource:@;
-batch_normalization_8_readvariableop_resource:@=
/batch_normalization_8_readvariableop_1_resource:@L
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:@U
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource:D@
2conv2d_transpose_4_biasadd_readvariableop_resource:
identity¢3batch_normalization/FusedBatchNormV3/ReadVariableOp¢5batch_normalization/FusedBatchNormV3/ReadVariableOp_1¢"batch_normalization/ReadVariableOp¢$batch_normalization/ReadVariableOp_1¢5batch_normalization_1/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_1/ReadVariableOp¢&batch_normalization_1/ReadVariableOp_1¢5batch_normalization_2/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_2/ReadVariableOp¢&batch_normalization_2/ReadVariableOp_1¢5batch_normalization_3/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_3/ReadVariableOp¢&batch_normalization_3/ReadVariableOp_1¢5batch_normalization_4/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_4/ReadVariableOp¢&batch_normalization_4/ReadVariableOp_1¢5batch_normalization_5/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_5/ReadVariableOp¢&batch_normalization_5/ReadVariableOp_1¢5batch_normalization_6/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_6/ReadVariableOp¢&batch_normalization_6/ReadVariableOp_1¢5batch_normalization_7/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_7/ReadVariableOp¢&batch_normalization_7/ReadVariableOp_1¢5batch_normalization_8/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_8/ReadVariableOp¢&batch_normalization_8/ReadVariableOp_1¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOp¢conv2d_2/BiasAdd/ReadVariableOp¢conv2d_2/Conv2D/ReadVariableOp¢conv2d_3/BiasAdd/ReadVariableOp¢conv2d_3/Conv2D/ReadVariableOp¢'conv2d_transpose/BiasAdd/ReadVariableOp¢0conv2d_transpose/conv2d_transpose/ReadVariableOp¢)conv2d_transpose_1/BiasAdd/ReadVariableOp¢2conv2d_transpose_1/conv2d_transpose/ReadVariableOp¢)conv2d_transpose_2/BiasAdd/ReadVariableOp¢2conv2d_transpose_2/conv2d_transpose/ReadVariableOp¢)conv2d_transpose_3/BiasAdd/ReadVariableOp¢2conv2d_transpose_3/conv2d_transpose/ReadVariableOp¢)conv2d_transpose_4/BiasAdd/ReadVariableOp¢2conv2d_transpose_4/conv2d_transpose/ReadVariableOp}
(periodic_padding2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"        ų’’’
*periodic_padding2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            
*periodic_padding2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         æ
"periodic_padding2d_3/strided_sliceStridedSliceinputs_31periodic_padding2d_3/strided_slice/stack:output:03periodic_padding2d_3/strided_slice/stack_1:output:03periodic_padding2d_3/strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_mask
*periodic_padding2d_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
,periodic_padding2d_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
,periodic_padding2d_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ē
$periodic_padding2d_3/strided_slice_1StridedSliceinputs_33periodic_padding2d_3/strided_slice_1/stack:output:05periodic_padding2d_3/strided_slice_1/stack_1:output:05periodic_padding2d_3/strided_slice_1/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskb
 periodic_padding2d_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ł
periodic_padding2d_3/concatConcatV2+periodic_padding2d_3/strided_slice:output:0inputs_3-periodic_padding2d_3/strided_slice_1:output:0)periodic_padding2d_3/concat/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’
*periodic_padding2d_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ų’’’    
,periodic_padding2d_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            
,periodic_padding2d_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ć
$periodic_padding2d_3/strided_slice_2StridedSlice$periodic_padding2d_3/concat:output:03periodic_padding2d_3/strided_slice_2/stack:output:05periodic_padding2d_3/strided_slice_2/stack_1:output:05periodic_padding2d_3/strided_slice_2/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_mask
*periodic_padding2d_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
,periodic_padding2d_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
,periodic_padding2d_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ć
$periodic_padding2d_3/strided_slice_3StridedSlice$periodic_padding2d_3/concat:output:03periodic_padding2d_3/strided_slice_3/stack:output:05periodic_padding2d_3/strided_slice_3/stack_1:output:05periodic_padding2d_3/strided_slice_3/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskd
"periodic_padding2d_3/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :
periodic_padding2d_3/concat_1ConcatV2-periodic_padding2d_3/strided_slice_2:output:0$periodic_padding2d_3/concat:output:0-periodic_padding2d_3/strided_slice_3:output:0+periodic_padding2d_3/concat_1/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’}
(periodic_padding2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"        ų’’’
*periodic_padding2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            
*periodic_padding2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         æ
"periodic_padding2d_2/strided_sliceStridedSliceinputs_21periodic_padding2d_2/strided_slice/stack:output:03periodic_padding2d_2/strided_slice/stack_1:output:03periodic_padding2d_2/strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_mask
*periodic_padding2d_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
,periodic_padding2d_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
,periodic_padding2d_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ē
$periodic_padding2d_2/strided_slice_1StridedSliceinputs_23periodic_padding2d_2/strided_slice_1/stack:output:05periodic_padding2d_2/strided_slice_1/stack_1:output:05periodic_padding2d_2/strided_slice_1/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskb
 periodic_padding2d_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ł
periodic_padding2d_2/concatConcatV2+periodic_padding2d_2/strided_slice:output:0inputs_2-periodic_padding2d_2/strided_slice_1:output:0)periodic_padding2d_2/concat/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’
*periodic_padding2d_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ų’’’    
,periodic_padding2d_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            
,periodic_padding2d_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ć
$periodic_padding2d_2/strided_slice_2StridedSlice$periodic_padding2d_2/concat:output:03periodic_padding2d_2/strided_slice_2/stack:output:05periodic_padding2d_2/strided_slice_2/stack_1:output:05periodic_padding2d_2/strided_slice_2/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_mask
*periodic_padding2d_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
,periodic_padding2d_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
,periodic_padding2d_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ć
$periodic_padding2d_2/strided_slice_3StridedSlice$periodic_padding2d_2/concat:output:03periodic_padding2d_2/strided_slice_3/stack:output:05periodic_padding2d_2/strided_slice_3/stack_1:output:05periodic_padding2d_2/strided_slice_3/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskd
"periodic_padding2d_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :
periodic_padding2d_2/concat_1ConcatV2-periodic_padding2d_2/strided_slice_2:output:0$periodic_padding2d_2/concat:output:0-periodic_padding2d_2/strided_slice_3:output:0+periodic_padding2d_2/concat_1/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’}
(periodic_padding2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"        ų’’’
*periodic_padding2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            
*periodic_padding2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         æ
"periodic_padding2d_1/strided_sliceStridedSliceinputs_11periodic_padding2d_1/strided_slice/stack:output:03periodic_padding2d_1/strided_slice/stack_1:output:03periodic_padding2d_1/strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_mask
*periodic_padding2d_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
,periodic_padding2d_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
,periodic_padding2d_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ē
$periodic_padding2d_1/strided_slice_1StridedSliceinputs_13periodic_padding2d_1/strided_slice_1/stack:output:05periodic_padding2d_1/strided_slice_1/stack_1:output:05periodic_padding2d_1/strided_slice_1/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskb
 periodic_padding2d_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ł
periodic_padding2d_1/concatConcatV2+periodic_padding2d_1/strided_slice:output:0inputs_1-periodic_padding2d_1/strided_slice_1:output:0)periodic_padding2d_1/concat/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’
*periodic_padding2d_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ų’’’    
,periodic_padding2d_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            
,periodic_padding2d_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ć
$periodic_padding2d_1/strided_slice_2StridedSlice$periodic_padding2d_1/concat:output:03periodic_padding2d_1/strided_slice_2/stack:output:05periodic_padding2d_1/strided_slice_2/stack_1:output:05periodic_padding2d_1/strided_slice_2/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_mask
*periodic_padding2d_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
,periodic_padding2d_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
,periodic_padding2d_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ć
$periodic_padding2d_1/strided_slice_3StridedSlice$periodic_padding2d_1/concat:output:03periodic_padding2d_1/strided_slice_3/stack:output:05periodic_padding2d_1/strided_slice_3/stack_1:output:05periodic_padding2d_1/strided_slice_3/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskd
"periodic_padding2d_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :
periodic_padding2d_1/concat_1ConcatV2-periodic_padding2d_1/strided_slice_2:output:0$periodic_padding2d_1/concat:output:0-periodic_padding2d_1/strided_slice_3:output:0+periodic_padding2d_1/concat_1/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’{
&periodic_padding2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"        ų’’’}
(periodic_padding2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            }
(periodic_padding2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ·
 periodic_padding2d/strided_sliceStridedSliceinputs_0/periodic_padding2d/strided_slice/stack:output:01periodic_padding2d/strided_slice/stack_1:output:01periodic_padding2d/strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_mask}
(periodic_padding2d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
*periodic_padding2d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
*periodic_padding2d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         æ
"periodic_padding2d/strided_slice_1StridedSliceinputs_01periodic_padding2d/strided_slice_1/stack:output:03periodic_padding2d/strided_slice_1/stack_1:output:03periodic_padding2d/strided_slice_1/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_mask`
periodic_padding2d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ń
periodic_padding2d/concatConcatV2)periodic_padding2d/strided_slice:output:0inputs_0+periodic_padding2d/strided_slice_1:output:0'periodic_padding2d/concat/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’}
(periodic_padding2d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ų’’’    
*periodic_padding2d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            
*periodic_padding2d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ł
"periodic_padding2d/strided_slice_2StridedSlice"periodic_padding2d/concat:output:01periodic_padding2d/strided_slice_2/stack:output:03periodic_padding2d/strided_slice_2/stack_1:output:03periodic_padding2d/strided_slice_2/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_mask}
(periodic_padding2d/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
*periodic_padding2d/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
*periodic_padding2d/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ł
"periodic_padding2d/strided_slice_3StridedSlice"periodic_padding2d/concat:output:01periodic_padding2d/strided_slice_3/stack:output:03periodic_padding2d/strided_slice_3/stack_1:output:03periodic_padding2d/strided_slice_3/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskb
 periodic_padding2d/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :
periodic_padding2d/concat_1ConcatV2+periodic_padding2d/strided_slice_2:output:0"periodic_padding2d/concat:output:0+periodic_padding2d/strided_slice_3:output:0)periodic_padding2d/concat_1/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’a
reshape/ShapeShape$periodic_padding2d/concat_1:output:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ł
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :Z
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :Y
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Ń
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
reshape/ReshapeReshape$periodic_padding2d/concat_1:output:0reshape/Reshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’e
reshape_1/ShapeShape&periodic_padding2d_1/concat_1:output:0*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :\
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :[
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Ū
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:¢
reshape_1/ReshapeReshape&periodic_padding2d_1/concat_1:output:0 reshape_1/Reshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’e
reshape_2/ShapeShape&periodic_padding2d_2/concat_1:output:0*
T0*
_output_shapes
:g
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :\
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :[
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Ū
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:¢
reshape_2/ReshapeReshape&periodic_padding2d_2/concat_1:output:0 reshape_2/Reshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’e
reshape_3/ShapeShape&periodic_padding2d_3/concat_1:output:0*
T0*
_output_shapes
:g
reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_3/strided_sliceStridedSlicereshape_3/Shape:output:0&reshape_3/strided_slice/stack:output:0(reshape_3/strided_slice/stack_1:output:0(reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :\
reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :[
reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Ū
reshape_3/Reshape/shapePack reshape_3/strided_slice:output:0"reshape_3/Reshape/shape/1:output:0"reshape_3/Reshape/shape/2:output:0"reshape_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:¢
reshape_3/ReshapeReshape&periodic_padding2d_3/concat_1:output:0 reshape_3/Reshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ó
concatenate/concatConcatV2reshape/Reshape:output:0reshape_1/Reshape:output:0reshape_2/Reshape:output:0reshape_3/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:’’’’’’’’’
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype0
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype0¬
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0°
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0±
$batch_normalization/FusedBatchNormV3FusedBatchNormV3concatenate/concat:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:’’’’’’’’’:::::*
epsilon%o:*
is_training( 
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ģ
conv2d/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’@*
paddingVALID*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’@f

conv2d/EluEluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:’’’’’’’’’@
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0°
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0“
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ø
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d/Elu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
is_training( 
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ó
conv2d_1/Conv2DConv2D*batch_normalization_1/FusedBatchNormV3:y:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’*
paddingVALID*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’k
conv2d_1/EluEluconv2d_1/BiasAdd:output:0*
T0*2
_output_shapes 
:’’’’’’’’’
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0æ
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_1/Elu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
is_training( 
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ó
conv2d_2/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’*
paddingVALID*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’k
conv2d_2/EluEluconv2d_2/BiasAdd:output:0*
T0*2
_output_shapes 
:’’’’’’’’’
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0æ
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_2/Elu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
is_training( 
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ó
conv2d_3/Conv2DConv2D*batch_normalization_3/FusedBatchNormV3:y:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’*
paddingVALID*
strides

conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’k
conv2d_3/EluEluconv2d_3/BiasAdd:output:0*
T0*2
_output_shapes 
:’’’’’’’’’
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0æ
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_3/Elu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
is_training( [
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ņ
concatenate_1/concatConcatV2conv2d_3/Elu:activations:0*batch_normalization_4/FusedBatchNormV3:y:0"concatenate_1/concat/axis:output:0*
N*
T0*2
_output_shapes 
:’’’’’’’’’c
conv2d_transpose/ShapeShapeconcatenate_1/concat:output:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¦
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value
B :[
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :[
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value
B :Ž
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:®
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask“
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype0
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0concatenate_1/concat:output:0*
T0*2
_output_shapes 
:’’’’’’’’’*
paddingVALID*
strides

'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0½
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’{
conv2d_transpose/EluElu!conv2d_transpose/BiasAdd:output:0*
T0*2
_output_shapes 
:’’’’’’’’’
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ē
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3"conv2d_transpose/Elu:activations:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
is_training( [
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ņ
concatenate_2/concatConcatV2conv2d_2/Elu:activations:0*batch_normalization_5/FusedBatchNormV3:y:0"concatenate_2/concat/axis:output:0*
N*
T0*2
_output_shapes 
:’’’’’’’’’e
conv2d_transpose_1/ShapeShapeconcatenate_2/concat:output:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value
B :]
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :]
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :č
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ø
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskø
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype0
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0concatenate_2/concat:output:0*
T0*2
_output_shapes 
:’’’’’’’’’*
paddingVALID*
strides

)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ć
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’
conv2d_transpose_1/EluElu#conv2d_transpose_1/BiasAdd:output:0*
T0*2
_output_shapes 
:’’’’’’’’’
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0É
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_1/Elu:activations:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
is_training( [
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ņ
concatenate_3/concatConcatV2conv2d_1/Elu:activations:0*batch_normalization_6/FusedBatchNormV3:y:0"concatenate_3/concat/axis:output:0*
N*
T0*2
_output_shapes 
:’’’’’’’’’e
conv2d_transpose_2/ShapeShapeconcatenate_3/concat:output:0*
T0*
_output_shapes
:p
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value
B :]
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value
B :]
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value
B :č
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ø
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskø
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype0
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0concatenate_3/concat:output:0*
T0*2
_output_shapes 
:’’’’’’’’’*
paddingVALID*
strides

)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ć
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’
conv2d_transpose_2/EluElu#conv2d_transpose_2/BiasAdd:output:0*
T0*2
_output_shapes 
:’’’’’’’’’
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0É
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_2/Elu:activations:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
is_training( [
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Š
concatenate_4/concatConcatV2conv2d/Elu:activations:0*batch_normalization_7/FusedBatchNormV3:y:0"concatenate_4/concat/axis:output:0*
N*
T0*2
_output_shapes 
:’’’’’’’’’Ąe
conv2d_transpose_3/ShapeShapeconcatenate_4/concat:output:0*
T0*
_output_shapes
:p
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value
B :]
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value
B :\
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@č
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ø
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask·
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@Ą*
dtype0
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0concatenate_4/concat:output:0*
T0*1
_output_shapes
:’’’’’’’’’@*
paddingVALID*
strides

)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ā
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’@~
conv2d_transpose_3/EluElu#conv2d_transpose_3/BiasAdd:output:0*
T0*1
_output_shapes
:’’’’’’’’’@
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:@*
dtype0
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:@*
dtype0°
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0“
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ä
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_3/Elu:activations:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
is_training( [
concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ņ
concatenate_5/concatConcatV2concatenate/concat:output:0*batch_normalization_8/FusedBatchNormV3:y:0"concatenate_5/concat/axis:output:0*
N*
T0*1
_output_shapes
:’’’’’’’’’De
conv2d_transpose_4/ShapeShapeconcatenate_5/concat:output:0*
T0*
_output_shapes
:p
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value
B :]
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value
B :\
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :č
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0#conv2d_transpose_4/stack/1:output:0#conv2d_transpose_4/stack/2:output:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ø
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¶
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:D*
dtype0
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0concatenate_5/concat:output:0*
T0*1
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides

)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ā
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’w
cropping2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"              y
 cropping2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"    ų’’’ų’’’    y
 cropping2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ·
cropping2d/strided_sliceStridedSlice#conv2d_transpose_4/BiasAdd:output:0'cropping2d/strided_slice/stack:output:0)cropping2d/strided_slice/stack_1:output:0)cropping2d/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:’’’’’’’’’*

begin_mask	*
end_mask	z
IdentityIdentity!cropping2d/strided_slice:output:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’ł
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*å
_input_shapesÓ
Š:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp:W S
-
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/0:WS
-
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/1:WS
-
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/2:WS
-
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/3
ų#

M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1285238

inputsD
(conv2d_transpose_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B :F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype0Ž
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’i
EluEluBiasAdd:output:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’{
IdentityIdentityElu:activations:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ķ

O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1290261

inputsD
(conv2d_transpose_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskJ
stack/1Const*
_output_shapes
: *
dtype0*
value
B :J
stack/2Const*
_output_shapes
: *
dtype0*
value
B :J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype0Ī
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*2
_output_shapes 
:’’’’’’’’’*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’Y
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:’’’’’’’’’k
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
Ķ

R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1290765

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Č
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs


E__inference_conv2d_3_layer_call_and_return_conditional_losses_1289827

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’Y
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:’’’’’’’’’k
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
Ļ
Ŗ
2__inference_conv2d_transpose_layer_call_fn_1289973

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1285238
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
	
Ö
7__inference_batch_normalization_6_layer_call_fn_1290274

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1285380
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ż
Ö
7__inference_batch_normalization_6_layer_call_fn_1290300

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1286226z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs

v
J__inference_concatenate_5_layer_call_and_return_conditional_losses_1290832
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:’’’’’’’’’Da
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:’’’’’’’’’D"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::’’’’’’’’’:’’’’’’’’’@:[ W
1
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:’’’’’’’’’@
"
_user_specified_name
inputs/1
Ė

P__inference_batch_normalization_layer_call_and_return_conditional_losses_1284898

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Č
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ä
G
+__inference_reshape_1_layer_call_fn_1289182

inputs
identity¾
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_reshape_1_layer_call_and_return_conditional_losses_1285858j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:’’’’’’’’’:U Q
-
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

ü
C__inference_conv2d_layer_call_and_return_conditional_losses_1289395

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’@X
EluEluBiasAdd:output:0*
T0*1
_output_shapes
:’’’’’’’’’@j
IdentityIdentityElu:activations:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ė
Å
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1290349

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ū
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
å

O__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1290910

inputsB
(conv2d_transpose_readvariableop_resource:D-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskJ
stack/1Const*
_output_shapes
: *
dtype0*
value
B :J
stack/2Const*
_output_shapes
: *
dtype0*
value
B :I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:D*
dtype0Ķ
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:’’’’’’’’’D: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:’’’’’’’’’D
 
_user_specified_nameinputs
	
Ņ
7__inference_batch_normalization_1_layer_call_fn_1289421

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1284993
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs
ś#

O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1285464

inputsD
(conv2d_transpose_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B :F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype0Ž
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’i
EluEluBiasAdd:output:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’{
IdentityIdentityElu:activations:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
ß
b
F__inference_reshape_1_layer_call_and_return_conditional_losses_1285858

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :R
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :©
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’b
IdentityIdentityReshape:output:0*
T0*1
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:’’’’’’’’’:U Q
-
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ū
Į
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1284993

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs
	
Ö
7__inference_batch_normalization_7_layer_call_fn_1290504

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1285524
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ę
Ō?
 __inference__traced_save_1291361
file_prefix8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop6
2savev2_conv2d_transpose_kernel_read_readvariableop4
0savev2_conv2d_transpose_bias_read_readvariableop:
6savev2_batch_normalization_5_gamma_read_readvariableop9
5savev2_batch_normalization_5_beta_read_readvariableop@
<savev2_batch_normalization_5_moving_mean_read_readvariableopD
@savev2_batch_normalization_5_moving_variance_read_readvariableop8
4savev2_conv2d_transpose_1_kernel_read_readvariableop6
2savev2_conv2d_transpose_1_bias_read_readvariableop:
6savev2_batch_normalization_6_gamma_read_readvariableop9
5savev2_batch_normalization_6_beta_read_readvariableop@
<savev2_batch_normalization_6_moving_mean_read_readvariableopD
@savev2_batch_normalization_6_moving_variance_read_readvariableop8
4savev2_conv2d_transpose_2_kernel_read_readvariableop6
2savev2_conv2d_transpose_2_bias_read_readvariableop:
6savev2_batch_normalization_7_gamma_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableop8
4savev2_conv2d_transpose_3_kernel_read_readvariableop6
2savev2_conv2d_transpose_3_bias_read_readvariableop:
6savev2_batch_normalization_8_gamma_read_readvariableop9
5savev2_batch_normalization_8_beta_read_readvariableop@
<savev2_batch_normalization_8_moving_mean_read_readvariableopD
@savev2_batch_normalization_8_moving_variance_read_readvariableop8
4savev2_conv2d_transpose_4_kernel_read_readvariableop6
2savev2_conv2d_transpose_4_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop?
;savev2_adam_batch_normalization_gamma_m_read_readvariableop>
:savev2_adam_batch_normalization_beta_m_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_m_read_readvariableop5
1savev2_adam_conv2d_3_kernel_m_read_readvariableop3
/savev2_adam_conv2d_3_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_m_read_readvariableop=
9savev2_adam_conv2d_transpose_kernel_m_read_readvariableop;
7savev2_adam_conv2d_transpose_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_5_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_5_beta_m_read_readvariableop?
;savev2_adam_conv2d_transpose_1_kernel_m_read_readvariableop=
9savev2_adam_conv2d_transpose_1_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_6_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_6_beta_m_read_readvariableop?
;savev2_adam_conv2d_transpose_2_kernel_m_read_readvariableop=
9savev2_adam_conv2d_transpose_2_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_7_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_7_beta_m_read_readvariableop?
;savev2_adam_conv2d_transpose_3_kernel_m_read_readvariableop=
9savev2_adam_conv2d_transpose_3_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_8_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_8_beta_m_read_readvariableop?
;savev2_adam_conv2d_transpose_4_kernel_m_read_readvariableop=
9savev2_adam_conv2d_transpose_4_bias_m_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_v_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop3
/savev2_adam_conv2d_3_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_v_read_readvariableop=
9savev2_adam_conv2d_transpose_kernel_v_read_readvariableop;
7savev2_adam_conv2d_transpose_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_5_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_5_beta_v_read_readvariableop?
;savev2_adam_conv2d_transpose_1_kernel_v_read_readvariableop=
9savev2_adam_conv2d_transpose_1_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_6_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_6_beta_v_read_readvariableop?
;savev2_adam_conv2d_transpose_2_kernel_v_read_readvariableop=
9savev2_adam_conv2d_transpose_2_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_7_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_7_beta_v_read_readvariableop?
;savev2_adam_conv2d_transpose_3_kernel_v_read_readvariableop=
9savev2_adam_conv2d_transpose_3_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_8_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_8_beta_v_read_readvariableop?
;savev2_adam_conv2d_transpose_4_kernel_v_read_readvariableop=
9savev2_adam_conv2d_transpose_4_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: üJ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*¤J
valueJBJB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHž
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*¢
valueBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B =
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop2savev2_conv2d_transpose_kernel_read_readvariableop0savev2_conv2d_transpose_bias_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop4savev2_conv2d_transpose_1_kernel_read_readvariableop2savev2_conv2d_transpose_1_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop4savev2_conv2d_transpose_2_kernel_read_readvariableop2savev2_conv2d_transpose_2_bias_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop4savev2_conv2d_transpose_3_kernel_read_readvariableop2savev2_conv2d_transpose_3_bias_read_readvariableop6savev2_batch_normalization_8_gamma_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableop4savev2_conv2d_transpose_4_kernel_read_readvariableop2savev2_conv2d_transpose_4_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop<savev2_adam_batch_normalization_2_beta_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop<savev2_adam_batch_normalization_3_beta_m_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_3_bias_m_read_readvariableop=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop<savev2_adam_batch_normalization_4_beta_m_read_readvariableop9savev2_adam_conv2d_transpose_kernel_m_read_readvariableop7savev2_adam_conv2d_transpose_bias_m_read_readvariableop=savev2_adam_batch_normalization_5_gamma_m_read_readvariableop<savev2_adam_batch_normalization_5_beta_m_read_readvariableop;savev2_adam_conv2d_transpose_1_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_1_bias_m_read_readvariableop=savev2_adam_batch_normalization_6_gamma_m_read_readvariableop<savev2_adam_batch_normalization_6_beta_m_read_readvariableop;savev2_adam_conv2d_transpose_2_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_2_bias_m_read_readvariableop=savev2_adam_batch_normalization_7_gamma_m_read_readvariableop<savev2_adam_batch_normalization_7_beta_m_read_readvariableop;savev2_adam_conv2d_transpose_3_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_3_bias_m_read_readvariableop=savev2_adam_batch_normalization_8_gamma_m_read_readvariableop<savev2_adam_batch_normalization_8_beta_m_read_readvariableop;savev2_adam_conv2d_transpose_4_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_4_bias_m_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop<savev2_adam_batch_normalization_2_beta_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop<savev2_adam_batch_normalization_3_beta_v_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop/savev2_adam_conv2d_3_bias_v_read_readvariableop=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop<savev2_adam_batch_normalization_4_beta_v_read_readvariableop9savev2_adam_conv2d_transpose_kernel_v_read_readvariableop7savev2_adam_conv2d_transpose_bias_v_read_readvariableop=savev2_adam_batch_normalization_5_gamma_v_read_readvariableop<savev2_adam_batch_normalization_5_beta_v_read_readvariableop;savev2_adam_conv2d_transpose_1_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_1_bias_v_read_readvariableop=savev2_adam_batch_normalization_6_gamma_v_read_readvariableop<savev2_adam_batch_normalization_6_beta_v_read_readvariableop;savev2_adam_conv2d_transpose_2_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_2_bias_v_read_readvariableop=savev2_adam_batch_normalization_7_gamma_v_read_readvariableop<savev2_adam_batch_normalization_7_beta_v_read_readvariableop;savev2_adam_conv2d_transpose_3_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_3_bias_v_read_readvariableop=savev2_adam_batch_normalization_8_gamma_v_read_readvariableop<savev2_adam_batch_normalization_8_beta_v_read_readvariableop;savev2_adam_conv2d_transpose_4_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_4_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Å	
_input_shapes³	
°	: :::::@:@:@:@:@:@:@::::::::::::::::::::::::::::::::::::@Ą:@:@:@:@:@:D:: : : : : : : :::@:@:@:@:@::::::::::::::::::::::::@Ą:@:@:@:D::::@:@:@:@:@::::::::::::::::::::::::@Ą:@:@:@:D:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::! 

_output_shapes	
::!!

_output_shapes	
::!"

_output_shapes	
::.#*
(
_output_shapes
::!$

_output_shapes	
::!%

_output_shapes	
::!&

_output_shapes	
::!'

_output_shapes	
::!(

_output_shapes	
::.)*
(
_output_shapes
::!*

_output_shapes	
::!+

_output_shapes	
::!,

_output_shapes	
::!-

_output_shapes	
::!.

_output_shapes	
::-/)
'
_output_shapes
:@Ą: 0

_output_shapes
:@: 1

_output_shapes
:@: 2

_output_shapes
:@: 3

_output_shapes
:@: 4

_output_shapes
:@:,5(
&
_output_shapes
:D: 6

_output_shapes
::7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: ::

_output_shapes
: :;

_output_shapes
: :<

_output_shapes
: :=

_output_shapes
: : >

_output_shapes
:: ?

_output_shapes
::,@(
&
_output_shapes
:@: A

_output_shapes
:@: B

_output_shapes
:@: C

_output_shapes
:@:-D)
'
_output_shapes
:@:!E

_output_shapes	
::!F

_output_shapes	
::!G

_output_shapes	
::.H*
(
_output_shapes
::!I

_output_shapes	
::!J

_output_shapes	
::!K

_output_shapes	
::.L*
(
_output_shapes
::!M

_output_shapes	
::!N

_output_shapes	
::!O

_output_shapes	
::.P*
(
_output_shapes
::!Q

_output_shapes	
::!R

_output_shapes	
::!S

_output_shapes	
::.T*
(
_output_shapes
::!U

_output_shapes	
::!V

_output_shapes	
::!W

_output_shapes	
::.X*
(
_output_shapes
::!Y

_output_shapes	
::!Z

_output_shapes	
::![

_output_shapes	
::-\)
'
_output_shapes
:@Ą: ]

_output_shapes
:@: ^

_output_shapes
:@: _

_output_shapes
:@:,`(
&
_output_shapes
:D: a

_output_shapes
:: b

_output_shapes
:: c

_output_shapes
::,d(
&
_output_shapes
:@: e

_output_shapes
:@: f

_output_shapes
:@: g

_output_shapes
:@:-h)
'
_output_shapes
:@:!i

_output_shapes	
::!j

_output_shapes	
::!k

_output_shapes	
::.l*
(
_output_shapes
::!m

_output_shapes	
::!n

_output_shapes	
::!o

_output_shapes	
::.p*
(
_output_shapes
::!q

_output_shapes	
::!r

_output_shapes	
::!s

_output_shapes	
::.t*
(
_output_shapes
::!u

_output_shapes	
::!v

_output_shapes	
::!w

_output_shapes	
::.x*
(
_output_shapes
::!y

_output_shapes	
::!z

_output_shapes	
::!{

_output_shapes	
::.|*
(
_output_shapes
::!}

_output_shapes	
::!~

_output_shapes	
::!

_output_shapes	
::.)
'
_output_shapes
:@Ą:!

_output_shapes
:@:!

_output_shapes
:@:!

_output_shapes
:@:-(
&
_output_shapes
:D:!

_output_shapes
::

_output_shapes
: 
	
Ö
7__inference_batch_normalization_4_layer_call_fn_1289853

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1285185
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
ķ
[
/__inference_concatenate_5_layer_call_fn_1290825
inputs_0
inputs_1
identityĻ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’D* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_5_layer_call_and_return_conditional_losses_1286373j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:’’’’’’’’’D"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::’’’’’’’’’:’’’’’’’’’@:[ W
1
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:’’’’’’’’’@
"
_user_specified_name
inputs/1

m
Q__inference_periodic_padding2d_3_layer_call_and_return_conditional_losses_1285748

inputs
identityh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"        ų’’’j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         é
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskj
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ń
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :£
concatConcatV2strided_slice:output:0inputsstrided_slice_1:output:0concat/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’j
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ų’’’    l
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ś
strided_slice_2StridedSliceconcat:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskj
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ś
strided_slice_3StridedSliceconcat:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :²
concat_1ConcatV2strided_slice_2:output:0concat:output:0strided_slice_3:output:0concat_1/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’_
IdentityIdentityconcat_1:output:0*
T0*-
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:’’’’’’’’’:U Q
-
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ä
c
G__inference_cropping2d_layer_call_and_return_conditional_losses_1286410

inputs
identityl
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"              n
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"    ų’’’ų’’’    n
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ī
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:’’’’’’’’’*

begin_mask	*
end_mask	h
IdentityIdentitystrided_slice:output:0*
T0*1
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:’’’’’’’’’:Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
	
Ö
7__inference_batch_normalization_3_layer_call_fn_1289696

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1285090
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
ė
Å
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1285411

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ū
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ż
”
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1285267

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ķ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ū
Ö
7__inference_batch_normalization_3_layer_call_fn_1289735

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1286872z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
ó
[
/__inference_concatenate_3_layer_call_fn_1290391
inputs_0
inputs_1
identityŠ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_3_layer_call_and_return_conditional_losses_1286243k
IdentityIdentityPartitionedCall:output:0*
T0*2
_output_shapes 
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:’’’’’’’’’:’’’’’’’’’:\ X
2
_output_shapes 
:’’’’’’’’’
"
_user_specified_name
inputs/0:\X
2
_output_shapes 
:’’’’’’’’’
"
_user_specified_name
inputs/1
	
Ņ
7__inference_batch_normalization_8_layer_call_fn_1290721

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1285637
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs

”
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1289933

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0½
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
is_training( n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
ó
[
/__inference_concatenate_2_layer_call_fn_1290174
inputs_0
inputs_1
identityŠ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_1286178k
IdentityIdentityPartitionedCall:output:0*
T0*2
_output_shapes 
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:’’’’’’’’’:’’’’’’’’’:\ X
2
_output_shapes 
:’’’’’’’’’
"
_user_specified_name
inputs/0:\X
2
_output_shapes 
:’’’’’’’’’
"
_user_specified_name
inputs/1

”
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1290150

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0½
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
is_training( n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
Ķ

O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1286203

inputsD
(conv2d_transpose_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskJ
stack/1Const*
_output_shapes
: *
dtype0*
value
B :J
stack/2Const*
_output_shapes
: *
dtype0*
value
B :J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype0Ī
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*2
_output_shapes 
:’’’’’’’’’*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’Y
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:’’’’’’’’’k
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
	
Š
5__inference_batch_normalization_layer_call_fn_1289277

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1284929
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ł
æ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1289339

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
ė
Å
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1285524

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ū
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs

Į
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1286578

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ę
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0m
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’@Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
Ż
”
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1285090

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ķ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ż
Ö
7__inference_batch_normalization_3_layer_call_fn_1289722

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1286052z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
ß
b
F__inference_reshape_3_layer_call_and_return_conditional_losses_1285890

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :R
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :©
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’b
IdentityIdentityReshape:output:0*
T0*1
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:’’’’’’’’’:U Q
-
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ß
b
F__inference_reshape_2_layer_call_and_return_conditional_losses_1289215

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :R
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :©
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’b
IdentityIdentityReshape:output:0*
T0*1
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:’’’’’’’’’:U Q
-
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

v
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1289964
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*2
_output_shapes 
:’’’’’’’’’b
IdentityIdentityconcat:output:0*
T0*2
_output_shapes 
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:’’’’’’’’’:’’’’’’’’’:\ X
2
_output_shapes 
:’’’’’’’’’
"
_user_specified_name
inputs/0:\X
2
_output_shapes 
:’’’’’’’’’
"
_user_specified_name
inputs/1
Ż
”
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1285154

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ķ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ė

P__inference_batch_normalization_layer_call_and_return_conditional_losses_1289321

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Č
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ż
”
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1285380

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ķ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ż
”
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1285026

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ķ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ż
Ö
7__inference_batch_normalization_4_layer_call_fn_1289866

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1286096z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
Ó
¬
4__inference_conv2d_transpose_2_layer_call_fn_1290407

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1285464
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ż
`
D__inference_reshape_layer_call_and_return_conditional_losses_1289177

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :R
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :©
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’b
IdentityIdentityReshape:output:0*
T0*1
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:’’’’’’’’’:U Q
-
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ė
Å
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1290566

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ū
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs


H__inference_concatenate_layer_call_and_return_conditional_losses_1285901

inputs
inputs_1
inputs_2
inputs_3
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputsinputs_1inputs_2inputs_3concat/axis:output:0*
N*
T0*1
_output_shapes
:’’’’’’’’’a
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesv
t:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:YU
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:YU
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:YU
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ū
Į
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1285637

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs

m
Q__inference_periodic_padding2d_1_layer_call_and_return_conditional_losses_1289100

inputs
identityh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"        ų’’’j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         é
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskj
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ń
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :£
concatConcatV2strided_slice:output:0inputsstrided_slice_1:output:0concat/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’j
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ų’’’    l
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ś
strided_slice_2StridedSliceconcat:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskj
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ś
strided_slice_3StridedSliceconcat:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :²
concat_1ConcatV2strided_slice_2:output:0concat:output:0strided_slice_3:output:0concat_1/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’_
IdentityIdentityconcat_1:output:0*
T0*-
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:’’’’’’’’’:U Q
-
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
	
Ö
7__inference_batch_normalization_2_layer_call_fn_1289552

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1285026
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
÷
t
J__inference_concatenate_5_layer_call_and_return_conditional_losses_1286373

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:’’’’’’’’’Da
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:’’’’’’’’’D"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::’’’’’’’’’:’’’’’’’’’@:Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:YU
1
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
ś¦
 
B__inference_model_layer_call_and_return_conditional_losses_1286413

inputs
inputs_1
inputs_2
inputs_3)
batch_normalization_1285921:)
batch_normalization_1285923:)
batch_normalization_1285925:)
batch_normalization_1285927:(
conv2d_1285942:@
conv2d_1285944:@+
batch_normalization_1_1285965:@+
batch_normalization_1_1285967:@+
batch_normalization_1_1285969:@+
batch_normalization_1_1285971:@+
conv2d_1_1285986:@
conv2d_1_1285988:	,
batch_normalization_2_1286009:	,
batch_normalization_2_1286011:	,
batch_normalization_2_1286013:	,
batch_normalization_2_1286015:	,
conv2d_2_1286030:
conv2d_2_1286032:	,
batch_normalization_3_1286053:	,
batch_normalization_3_1286055:	,
batch_normalization_3_1286057:	,
batch_normalization_3_1286059:	,
conv2d_3_1286074:
conv2d_3_1286076:	,
batch_normalization_4_1286097:	,
batch_normalization_4_1286099:	,
batch_normalization_4_1286101:	,
batch_normalization_4_1286103:	4
conv2d_transpose_1286139:'
conv2d_transpose_1286141:	,
batch_normalization_5_1286162:	,
batch_normalization_5_1286164:	,
batch_normalization_5_1286166:	,
batch_normalization_5_1286168:	6
conv2d_transpose_1_1286204:)
conv2d_transpose_1_1286206:	,
batch_normalization_6_1286227:	,
batch_normalization_6_1286229:	,
batch_normalization_6_1286231:	,
batch_normalization_6_1286233:	6
conv2d_transpose_2_1286269:)
conv2d_transpose_2_1286271:	,
batch_normalization_7_1286292:	,
batch_normalization_7_1286294:	,
batch_normalization_7_1286296:	,
batch_normalization_7_1286298:	5
conv2d_transpose_3_1286334:@Ą(
conv2d_transpose_3_1286336:@+
batch_normalization_8_1286357:@+
batch_normalization_8_1286359:@+
batch_normalization_8_1286361:@+
batch_normalization_8_1286363:@4
conv2d_transpose_4_1286398:D(
conv2d_transpose_4_1286400:
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢-batch_normalization_6/StatefulPartitionedCall¢-batch_normalization_7/StatefulPartitionedCall¢-batch_normalization_8/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¢(conv2d_transpose/StatefulPartitionedCall¢*conv2d_transpose_1/StatefulPartitionedCall¢*conv2d_transpose_2/StatefulPartitionedCall¢*conv2d_transpose_3/StatefulPartitionedCall¢*conv2d_transpose_4/StatefulPartitionedCallÜ
$periodic_padding2d_3/PartitionedCallPartitionedCallinputs_3*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_periodic_padding2d_3_layer_call_and_return_conditional_losses_1285748Ü
$periodic_padding2d_2/PartitionedCallPartitionedCallinputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_periodic_padding2d_2_layer_call_and_return_conditional_losses_1285774Ü
$periodic_padding2d_1/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_periodic_padding2d_1_layer_call_and_return_conditional_losses_1285800Ö
"periodic_padding2d/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_periodic_padding2d_layer_call_and_return_conditional_losses_1285826é
reshape/PartitionedCallPartitionedCall+periodic_padding2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_1285842ļ
reshape_1/PartitionedCallPartitionedCall-periodic_padding2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_reshape_1_layer_call_and_return_conditional_losses_1285858ļ
reshape_2/PartitionedCallPartitionedCall-periodic_padding2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_1285874ļ
reshape_3/PartitionedCallPartitionedCall-periodic_padding2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_reshape_3_layer_call_and_return_conditional_losses_1285890Õ
concatenate/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0"reshape_1/PartitionedCall:output:0"reshape_2/PartitionedCall:output:0"reshape_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_1285901
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0batch_normalization_1285921batch_normalization_1285923batch_normalization_1285925batch_normalization_1285927*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1285920¦
conv2d/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_1285942conv2d_1285944*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_1285941
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_1_1285965batch_normalization_1_1285967batch_normalization_1_1285969batch_normalization_1_1285971*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1285964±
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_1_1285986conv2d_1_1285988*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1285985
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_2_1286009batch_normalization_2_1286011batch_normalization_2_1286013batch_normalization_2_1286015*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1286008±
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_2_1286030conv2d_2_1286032*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1286029
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_3_1286053batch_normalization_3_1286055batch_normalization_3_1286057batch_normalization_3_1286059*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1286052±
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_3_1286074conv2d_3_1286076*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1286073
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_4_1286097batch_normalization_4_1286099batch_normalization_4_1286101batch_normalization_4_1286103*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1286096­
concatenate_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:06batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1286113Į
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0conv2d_transpose_1286139conv2d_transpose_1286141*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1286138¢
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0batch_normalization_5_1286162batch_normalization_5_1286164batch_normalization_5_1286166batch_normalization_5_1286168*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1286161­
concatenate_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:06batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_1286178É
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0conv2d_transpose_1_1286204conv2d_transpose_1_1286206*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1286203¤
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0batch_normalization_6_1286227batch_normalization_6_1286229batch_normalization_6_1286231batch_normalization_6_1286233*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1286226­
concatenate_3/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:06batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_3_layer_call_and_return_conditional_losses_1286243É
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0conv2d_transpose_2_1286269conv2d_transpose_2_1286271*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1286268¤
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0batch_normalization_7_1286292batch_normalization_7_1286294batch_normalization_7_1286296batch_normalization_7_1286298*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1286291«
concatenate_4/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:06batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’Ą* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_4_layer_call_and_return_conditional_losses_1286308Č
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv2d_transpose_3_1286334conv2d_transpose_3_1286336*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1286333£
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0batch_normalization_8_1286357batch_normalization_8_1286359batch_normalization_8_1286361batch_normalization_8_1286363*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1286356§
concatenate_5/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:06batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’D* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_5_layer_call_and_return_conditional_losses_1286373Č
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0conv2d_transpose_4_1286398conv2d_transpose_4_1286400*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1286397÷
cropping2d/PartitionedCallPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_cropping2d_layer_call_and_return_conditional_losses_1286410|
IdentityIdentity#cropping2d/PartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’Ż
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*å
_input_shapesÓ
Š:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall:U Q
-
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:UQ
-
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:UQ
-
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:UQ
-
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

”
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1286096

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0½
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
is_training( n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
#

O__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1285689

inputsB
(conv2d_transpose_readvariableop_resource:D-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B : F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:D*
dtype0Ż
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+’’’’’’’’’’’’’’’’’’’’’’’’’’’D: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’D
 
_user_specified_nameinputs
Õ
Ņ
7__inference_batch_normalization_8_layer_call_fn_1290734

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1286356y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs

”
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1286226

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0½
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
is_training( n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
Ū
Į
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1290783

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs
	
Ö
7__inference_batch_normalization_4_layer_call_fn_1289840

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1285154
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ż
`
D__inference_reshape_layer_call_and_return_conditional_losses_1285842

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :R
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :©
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’b
IdentityIdentityReshape:output:0*
T0*1
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:’’’’’’’’’:U Q
-
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ž
¢
*__inference_conv2d_2_layer_call_fn_1289672

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallč
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1286029z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
ž
¢
*__inference_conv2d_3_layer_call_fn_1289816

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallč
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1286073z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
ė
Å
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1290132

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ū
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
ß
b
F__inference_reshape_3_layer_call_and_return_conditional_losses_1289234

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :R
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :©
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’b
IdentityIdentityReshape:output:0*
T0*1
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:’’’’’’’’’:U Q
-
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ń#

O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1290671

inputsC
(conv2d_transpose_readvariableop_resource:@Ą-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B :F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@Ą*
dtype0Ż
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@h
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@z
IdentityIdentityElu:activations:0^NoOp*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ą: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ą
 
_user_specified_nameinputs

Į
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1286980

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ę
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0m
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’@Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
Ū
Ö
7__inference_batch_normalization_2_layer_call_fn_1289591

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1286926z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs

¬
4__inference_conv2d_transpose_2_layer_call_fn_1290416

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallņ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1286268z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
Ņ
R
6__inference_periodic_padding2d_1_layer_call_fn_1289076

inputs
identityÅ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_periodic_padding2d_1_layer_call_and_return_conditional_losses_1285800f
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:’’’’’’’’’:U Q
-
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ļ
Š
5__inference_batch_normalization_layer_call_fn_1289303

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1287034y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ż
”
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1285493

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ķ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
#

O__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1290887

inputsB
(conv2d_transpose_readvariableop_resource:D-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B : F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:D*
dtype0Ż
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+’’’’’’’’’’’’’’’’’’’’’’’’’’’D: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’D
 
_user_specified_nameinputs
	
Ö
7__inference_batch_normalization_2_layer_call_fn_1289565

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1285057
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ó
Ņ
7__inference_batch_normalization_1_layer_call_fn_1289447

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1286980y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
«
Å
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1286818

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ė
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs

”
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1289645

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0½
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
is_training( n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs

k
O__inference_periodic_padding2d_layer_call_and_return_conditional_losses_1285826

inputs
identityh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"        ų’’’j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         é
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskj
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ń
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :£
concatConcatV2strided_slice:output:0inputsstrided_slice_1:output:0concat/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’j
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ų’’’    l
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ś
strided_slice_2StridedSliceconcat:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskj
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ś
strided_slice_3StridedSliceconcat:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :²
concat_1ConcatV2strided_slice_2:output:0concat:output:0strided_slice_3:output:0concat_1/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’_
IdentityIdentityconcat_1:output:0*
T0*-
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:’’’’’’’’’:U Q
-
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ķ

O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1290478

inputsD
(conv2d_transpose_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskJ
stack/1Const*
_output_shapes
: *
dtype0*
value
B :J
stack/2Const*
_output_shapes
: *
dtype0*
value
B :J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype0Ī
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*2
_output_shapes 
:’’’’’’’’’*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’Y
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:’’’’’’’’’k
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs

m
Q__inference_periodic_padding2d_2_layer_call_and_return_conditional_losses_1289129

inputs
identityh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"        ų’’’j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         é
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskj
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ń
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :£
concatConcatV2strided_slice:output:0inputsstrided_slice_1:output:0concat/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’j
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ų’’’    l
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ś
strided_slice_2StridedSliceconcat:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskj
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ś
strided_slice_3StridedSliceconcat:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :²
concat_1ConcatV2strided_slice_2:output:0concat:output:0strided_slice_3:output:0concat_1/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’_
IdentityIdentityconcat_1:output:0*
T0*-
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:’’’’’’’’’:U Q
-
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ī
P
4__inference_periodic_padding2d_layer_call_fn_1289047

inputs
identityĆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_periodic_padding2d_layer_call_and_return_conditional_losses_1285826f
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:’’’’’’’’’:U Q
-
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
	
Ö
7__inference_batch_normalization_5_layer_call_fn_1290070

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1285298
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ż
Ö
7__inference_batch_normalization_5_layer_call_fn_1290083

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1286161z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
ß
b
F__inference_reshape_2_layer_call_and_return_conditional_losses_1285874

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :R
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :©
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’b
IdentityIdentityReshape:output:0*
T0*1
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:’’’’’’’’’:U Q
-
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

Į
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1289519

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ę
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0m
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’@Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
å¦

B__inference_model_layer_call_and_return_conditional_losses_1287746	
u_vel	
v_vel	
w_vel

pr0_71)
batch_normalization_1287613:)
batch_normalization_1287615:)
batch_normalization_1287617:)
batch_normalization_1287619:(
conv2d_1287622:@
conv2d_1287624:@+
batch_normalization_1_1287627:@+
batch_normalization_1_1287629:@+
batch_normalization_1_1287631:@+
batch_normalization_1_1287633:@+
conv2d_1_1287636:@
conv2d_1_1287638:	,
batch_normalization_2_1287641:	,
batch_normalization_2_1287643:	,
batch_normalization_2_1287645:	,
batch_normalization_2_1287647:	,
conv2d_2_1287650:
conv2d_2_1287652:	,
batch_normalization_3_1287655:	,
batch_normalization_3_1287657:	,
batch_normalization_3_1287659:	,
batch_normalization_3_1287661:	,
conv2d_3_1287664:
conv2d_3_1287666:	,
batch_normalization_4_1287669:	,
batch_normalization_4_1287671:	,
batch_normalization_4_1287673:	,
batch_normalization_4_1287675:	4
conv2d_transpose_1287679:'
conv2d_transpose_1287681:	,
batch_normalization_5_1287684:	,
batch_normalization_5_1287686:	,
batch_normalization_5_1287688:	,
batch_normalization_5_1287690:	6
conv2d_transpose_1_1287694:)
conv2d_transpose_1_1287696:	,
batch_normalization_6_1287699:	,
batch_normalization_6_1287701:	,
batch_normalization_6_1287703:	,
batch_normalization_6_1287705:	6
conv2d_transpose_2_1287709:)
conv2d_transpose_2_1287711:	,
batch_normalization_7_1287714:	,
batch_normalization_7_1287716:	,
batch_normalization_7_1287718:	,
batch_normalization_7_1287720:	5
conv2d_transpose_3_1287724:@Ą(
conv2d_transpose_3_1287726:@+
batch_normalization_8_1287729:@+
batch_normalization_8_1287731:@+
batch_normalization_8_1287733:@+
batch_normalization_8_1287735:@4
conv2d_transpose_4_1287739:D(
conv2d_transpose_4_1287741:
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢-batch_normalization_6/StatefulPartitionedCall¢-batch_normalization_7/StatefulPartitionedCall¢-batch_normalization_8/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¢(conv2d_transpose/StatefulPartitionedCall¢*conv2d_transpose_1/StatefulPartitionedCall¢*conv2d_transpose_2/StatefulPartitionedCall¢*conv2d_transpose_3/StatefulPartitionedCall¢*conv2d_transpose_4/StatefulPartitionedCallŚ
$periodic_padding2d_3/PartitionedCallPartitionedCallpr0_71*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_periodic_padding2d_3_layer_call_and_return_conditional_losses_1285748Ł
$periodic_padding2d_2/PartitionedCallPartitionedCallw_vel*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_periodic_padding2d_2_layer_call_and_return_conditional_losses_1285774Ł
$periodic_padding2d_1/PartitionedCallPartitionedCallv_vel*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_periodic_padding2d_1_layer_call_and_return_conditional_losses_1285800Õ
"periodic_padding2d/PartitionedCallPartitionedCallu_vel*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_periodic_padding2d_layer_call_and_return_conditional_losses_1285826é
reshape/PartitionedCallPartitionedCall+periodic_padding2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_1285842ļ
reshape_1/PartitionedCallPartitionedCall-periodic_padding2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_reshape_1_layer_call_and_return_conditional_losses_1285858ļ
reshape_2/PartitionedCallPartitionedCall-periodic_padding2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_1285874ļ
reshape_3/PartitionedCallPartitionedCall-periodic_padding2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_reshape_3_layer_call_and_return_conditional_losses_1285890Õ
concatenate/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0"reshape_1/PartitionedCall:output:0"reshape_2/PartitionedCall:output:0"reshape_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_1285901
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0batch_normalization_1287613batch_normalization_1287615batch_normalization_1287617batch_normalization_1287619*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1285920¦
conv2d/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_1287622conv2d_1287624*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_1285941
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_1_1287627batch_normalization_1_1287629batch_normalization_1_1287631batch_normalization_1_1287633*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1285964±
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_1_1287636conv2d_1_1287638*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1285985
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_2_1287641batch_normalization_2_1287643batch_normalization_2_1287645batch_normalization_2_1287647*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1286008±
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_2_1287650conv2d_2_1287652*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1286029
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_3_1287655batch_normalization_3_1287657batch_normalization_3_1287659batch_normalization_3_1287661*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1286052±
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_3_1287664conv2d_3_1287666*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1286073
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_4_1287669batch_normalization_4_1287671batch_normalization_4_1287673batch_normalization_4_1287675*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1286096­
concatenate_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:06batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1286113Į
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0conv2d_transpose_1287679conv2d_transpose_1287681*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1286138¢
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0batch_normalization_5_1287684batch_normalization_5_1287686batch_normalization_5_1287688batch_normalization_5_1287690*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1286161­
concatenate_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:06batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_1286178É
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0conv2d_transpose_1_1287694conv2d_transpose_1_1287696*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1286203¤
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0batch_normalization_6_1287699batch_normalization_6_1287701batch_normalization_6_1287703batch_normalization_6_1287705*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1286226­
concatenate_3/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:06batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_3_layer_call_and_return_conditional_losses_1286243É
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0conv2d_transpose_2_1287709conv2d_transpose_2_1287711*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1286268¤
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0batch_normalization_7_1287714batch_normalization_7_1287716batch_normalization_7_1287718batch_normalization_7_1287720*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1286291«
concatenate_4/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:06batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’Ą* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_4_layer_call_and_return_conditional_losses_1286308Č
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv2d_transpose_3_1287724conv2d_transpose_3_1287726*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1286333£
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0batch_normalization_8_1287729batch_normalization_8_1287731batch_normalization_8_1287733batch_normalization_8_1287735*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1286356§
concatenate_5/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:06batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’D* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_5_layer_call_and_return_conditional_losses_1286373Č
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0conv2d_transpose_4_1287739conv2d_transpose_4_1287741*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1286397÷
cropping2d/PartitionedCallPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_cropping2d_layer_call_and_return_conditional_losses_1286410|
IdentityIdentity#cropping2d/PartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’Ż
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*å
_input_shapesÓ
Š:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall:T P
-
_output_shapes
:’’’’’’’’’

_user_specified_nameu_vel:TP
-
_output_shapes
:’’’’’’’’’

_user_specified_namev_vel:TP
-
_output_shapes
:’’’’’’’’’

_user_specified_namew_vel:UQ
-
_output_shapes
:’’’’’’’’’
 
_user_specified_namepr0.71
Ó
Ņ
7__inference_batch_normalization_8_layer_call_fn_1290747

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1286578y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
ń
[
/__inference_concatenate_4_layer_call_fn_1290608
inputs_0
inputs_1
identityŠ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’Ą* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_4_layer_call_and_return_conditional_losses_1286308k
IdentityIdentityPartitionedCall:output:0*
T0*2
_output_shapes 
:’’’’’’’’’Ą"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:’’’’’’’’’@:’’’’’’’’’:[ W
1
_output_shapes
:’’’’’’’’’@
"
_user_specified_name
inputs/0:\X
2
_output_shapes 
:’’’’’’’’’
"
_user_specified_name
inputs/1
	
Ö
7__inference_batch_normalization_5_layer_call_fn_1290057

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1285267
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs

”
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1286008

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0½
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
is_training( n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs


E__inference_conv2d_1_layer_call_and_return_conditional_losses_1285985

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’Y
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:’’’’’’’’’k
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:’’’’’’’’’@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
ś#

O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1290237

inputsD
(conv2d_transpose_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B :F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype0Ž
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’i
EluEluBiasAdd:output:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’{
IdentityIdentityElu:activations:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs

Ŗ
4__inference_conv2d_transpose_3_layer_call_fn_1290633

inputs"
unknown:@Ą
	unknown_0:@
identity¢StatefulPartitionedCallń
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1286333y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":’’’’’’’’’Ą: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:’’’’’’’’’Ą
 
_user_specified_nameinputs

”
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1286161

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0½
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
is_training( n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
ž
t
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1286113

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*2
_output_shapes 
:’’’’’’’’’b
IdentityIdentityconcat:output:0*
T0*2
_output_shapes 
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:’’’’’’’’’:’’’’’’’’’:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs:ZV
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs


R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1290801

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
is_training( m
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs


P__inference_batch_normalization_layer_call_and_return_conditional_losses_1289357

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:’’’’’’’’’:::::*
epsilon%o:*
is_training( m
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

æ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1289375

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ę
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0m
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ž
t
J__inference_concatenate_3_layer_call_and_return_conditional_losses_1286243

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*2
_output_shapes 
:’’’’’’’’’b
IdentityIdentityconcat:output:0*
T0*2
_output_shapes 
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:’’’’’’’’’:’’’’’’’’’:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs:ZV
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs

Ŗ
2__inference_conv2d_transpose_layer_call_fn_1289982

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallš
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1286138z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
Ż
Ö
7__inference_batch_normalization_7_layer_call_fn_1290517

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1286291z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
«
Å
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1290602

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ė
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
Ż
”
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1290331

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ķ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
ó
[
/__inference_concatenate_1_layer_call_fn_1289957
inputs_0
inputs_1
identityŠ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1286113k
IdentityIdentityPartitionedCall:output:0*
T0*2
_output_shapes 
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:’’’’’’’’’:’’’’’’’’’:\ X
2
_output_shapes 
:’’’’’’’’’
"
_user_specified_name
inputs/0:\X
2
_output_shapes 
:’’’’’’’’’
"
_user_specified_name
inputs/1
Ä

O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1290695

inputsC
(conv2d_transpose_readvariableop_resource:@Ą-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskJ
stack/1Const*
_output_shapes
: *
dtype0*
value
B :J
stack/2Const*
_output_shapes
: *
dtype0*
value
B :I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@Ą*
dtype0Ķ
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:’’’’’’’’’@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’@X
EluEluBiasAdd:output:0*
T0*1
_output_shapes
:’’’’’’’’’@j
IdentityIdentityElu:activations:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’@
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":’’’’’’’’’Ą: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Z V
2
_output_shapes 
:’’’’’’’’’Ą
 
_user_specified_nameinputs
Ė

M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1290044

inputsD
(conv2d_transpose_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskJ
stack/1Const*
_output_shapes
: *
dtype0*
value
B :J
stack/2Const*
_output_shapes
: *
dtype0*
value
B :J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype0Ī
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*2
_output_shapes 
:’’’’’’’’’*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’Y
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:’’’’’’’’’k
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs

”
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1286291

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0½
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
is_training( n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs

”
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1286052

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0½
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
is_training( n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
Ķ

R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1284962

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Č
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs
ņ

'__inference_model_layer_call_fn_1286524	
u_vel	
v_vel	
w_vel

pr0_71
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@$
	unknown_9:@

unknown_10:	

unknown_11:	

unknown_12:	

unknown_13:	

unknown_14:	&

unknown_15:

unknown_16:	

unknown_17:	

unknown_18:	

unknown_19:	

unknown_20:	&

unknown_21:

unknown_22:	

unknown_23:	

unknown_24:	

unknown_25:	

unknown_26:	&

unknown_27:

unknown_28:	

unknown_29:	

unknown_30:	

unknown_31:	

unknown_32:	&

unknown_33:

unknown_34:	

unknown_35:	

unknown_36:	

unknown_37:	

unknown_38:	&

unknown_39:

unknown_40:	

unknown_41:	

unknown_42:	

unknown_43:	

unknown_44:	%

unknown_45:@Ą

unknown_46:@

unknown_47:@

unknown_48:@

unknown_49:@

unknown_50:@$

unknown_51:D

unknown_52:
identity¢StatefulPartitionedCallĖ
StatefulPartitionedCallStatefulPartitionedCallu_velv_velw_velpr0_71unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*E
Tin>
<2:*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456789*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1286413y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*å
_input_shapesÓ
Š:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
-
_output_shapes
:’’’’’’’’’

_user_specified_nameu_vel:TP
-
_output_shapes
:’’’’’’’’’

_user_specified_namev_vel:TP
-
_output_shapes
:’’’’’’’’’

_user_specified_namew_vel:UQ
-
_output_shapes
:’’’’’’’’’
 
_user_specified_namepr0.71
Ń
Š
5__inference_batch_normalization_layer_call_fn_1289290

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1285920y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ķ

O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1286268

inputsD
(conv2d_transpose_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskJ
stack/1Const*
_output_shapes
: *
dtype0*
value
B :J
stack/2Const*
_output_shapes
: *
dtype0*
value
B :J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype0Ī
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*2
_output_shapes 
:’’’’’’’’’*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’Y
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:’’’’’’’’’k
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
Š

%__inference_signature_wrapper_1288018

pr0_71	
u_vel	
v_vel	
w_vel
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@$
	unknown_9:@

unknown_10:	

unknown_11:	

unknown_12:	

unknown_13:	

unknown_14:	&

unknown_15:

unknown_16:	

unknown_17:	

unknown_18:	

unknown_19:	

unknown_20:	&

unknown_21:

unknown_22:	

unknown_23:	

unknown_24:	

unknown_25:	

unknown_26:	&

unknown_27:

unknown_28:	

unknown_29:	

unknown_30:	

unknown_31:	

unknown_32:	&

unknown_33:

unknown_34:	

unknown_35:	

unknown_36:	

unknown_37:	

unknown_38:	&

unknown_39:

unknown_40:	

unknown_41:	

unknown_42:	

unknown_43:	

unknown_44:	%

unknown_45:@Ą

unknown_46:@

unknown_47:@

unknown_48:@

unknown_49:@

unknown_50:@$

unknown_51:D

unknown_52:
identity¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallu_velv_velw_velpr0_71unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*E
Tin>
<2:*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456789*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_1284876y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*å
_input_shapesÓ
Š:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:’’’’’’’’’
 
_user_specified_namepr0.71:TP
-
_output_shapes
:’’’’’’’’’

_user_specified_nameu_vel:TP
-
_output_shapes
:’’’’’’’’’

_user_specified_namev_vel:TP
-
_output_shapes
:’’’’’’’’’

_user_specified_namew_vel

Į
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1290819

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ę
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0m
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’@Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs


'__inference_model_layer_call_fn_1288134
inputs_0
inputs_1
inputs_2
inputs_3
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@$
	unknown_9:@

unknown_10:	

unknown_11:	

unknown_12:	

unknown_13:	

unknown_14:	&

unknown_15:

unknown_16:	

unknown_17:	

unknown_18:	

unknown_19:	

unknown_20:	&

unknown_21:

unknown_22:	

unknown_23:	

unknown_24:	

unknown_25:	

unknown_26:	&

unknown_27:

unknown_28:	

unknown_29:	

unknown_30:	

unknown_31:	

unknown_32:	&

unknown_33:

unknown_34:	

unknown_35:	

unknown_36:	

unknown_37:	

unknown_38:	&

unknown_39:

unknown_40:	

unknown_41:	

unknown_42:	

unknown_43:	

unknown_44:	%

unknown_45:@Ą

unknown_46:@

unknown_47:@

unknown_48:@

unknown_49:@

unknown_50:@$

unknown_51:D

unknown_52:
identity¢StatefulPartitionedCallÖ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*E
Tin>
<2:*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456789*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1286413y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*å
_input_shapesÓ
Š:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
-
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/0:WS
-
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/1:WS
-
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/2:WS
-
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/3
Ä
G
+__inference_reshape_2_layer_call_fn_1289201

inputs
identity¾
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_1285874j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:’’’’’’’’’:U Q
-
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
	
Ņ
7__inference_batch_normalization_1_layer_call_fn_1289408

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1284962
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs

m
Q__inference_periodic_padding2d_1_layer_call_and_return_conditional_losses_1285800

inputs
identityh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"        ų’’’j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         é
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskj
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ń
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :£
concatConcatV2strided_slice:output:0inputsstrided_slice_1:output:0concat/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’j
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ų’’’    l
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ś
strided_slice_2StridedSliceconcat:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskj
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ś
strided_slice_3StridedSliceconcat:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :²
concat_1ConcatV2strided_slice_2:output:0concat:output:0strided_slice_3:output:0concat_1/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’_
IdentityIdentityconcat_1:output:0*
T0*-
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:’’’’’’’’’:U Q
-
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ū
”
*__inference_conv2d_1_layer_call_fn_1289528

inputs"
unknown:@
	unknown_0:	
identity¢StatefulPartitionedCallč
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1285985z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:’’’’’’’’’@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
ė
Å
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1289627

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ū
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
«
Å
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1286698

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ė
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
«
Å
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1286758

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ė
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs

v
J__inference_concatenate_3_layer_call_and_return_conditional_losses_1290398
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*2
_output_shapes 
:’’’’’’’’’b
IdentityIdentityconcat:output:0*
T0*2
_output_shapes 
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:’’’’’’’’’:’’’’’’’’’:\ X
2
_output_shapes 
:’’’’’’’’’
"
_user_specified_name
inputs/0:\X
2
_output_shapes 
:’’’’’’’’’
"
_user_specified_name
inputs/1

”
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1289789

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0½
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
is_training( n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
Ū
Ö
7__inference_batch_normalization_4_layer_call_fn_1289879

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1286818z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
Ż
”
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1289897

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ķ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ó
¬
4__inference_conv2d_transpose_1_layer_call_fn_1290190

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1285351
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
	
Ö
7__inference_batch_normalization_3_layer_call_fn_1289709

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1285121
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ķ

R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1289465

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Č
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs

”
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1290584

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0½
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
is_training( n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs

ü
C__inference_conv2d_layer_call_and_return_conditional_losses_1285941

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’@X
EluEluBiasAdd:output:0*
T0*1
_output_shapes
:’’’’’’’’’@j
IdentityIdentityElu:activations:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs


R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1286356

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
is_training( m
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
ė
Å
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1285298

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ū
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs


P__inference_batch_normalization_layer_call_and_return_conditional_losses_1285920

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:’’’’’’’’’:::::*
epsilon%o:*
is_training( m
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs


E__inference_conv2d_2_layer_call_and_return_conditional_losses_1289683

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’Y
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:’’’’’’’’’k
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
	
Ö
7__inference_batch_normalization_6_layer_call_fn_1290287

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1285411
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs


E__inference_conv2d_3_layer_call_and_return_conditional_losses_1286073

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’Y
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:’’’’’’’’’k
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs

v
J__inference_concatenate_2_layer_call_and_return_conditional_losses_1290181
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*2
_output_shapes 
:’’’’’’’’’b
IdentityIdentityconcat:output:0*
T0*2
_output_shapes 
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:’’’’’’’’’:’’’’’’’’’:\ X
2
_output_shapes 
:’’’’’’’’’
"
_user_specified_name
inputs/0:\X
2
_output_shapes 
:’’’’’’’’’
"
_user_specified_name
inputs/1
	
Ņ
7__inference_batch_normalization_8_layer_call_fn_1290708

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1285606
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs


E__inference_conv2d_2_layer_call_and_return_conditional_losses_1286029

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’Y
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:’’’’’’’’’k
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
Ä¢
ē\
#__inference__traced_restore_1291770
file_prefix8
*assignvariableop_batch_normalization_gamma:9
+assignvariableop_1_batch_normalization_beta:@
2assignvariableop_2_batch_normalization_moving_mean:D
6assignvariableop_3_batch_normalization_moving_variance::
 assignvariableop_4_conv2d_kernel:@,
assignvariableop_5_conv2d_bias:@<
.assignvariableop_6_batch_normalization_1_gamma:@;
-assignvariableop_7_batch_normalization_1_beta:@B
4assignvariableop_8_batch_normalization_1_moving_mean:@F
8assignvariableop_9_batch_normalization_1_moving_variance:@>
#assignvariableop_10_conv2d_1_kernel:@0
!assignvariableop_11_conv2d_1_bias:	>
/assignvariableop_12_batch_normalization_2_gamma:	=
.assignvariableop_13_batch_normalization_2_beta:	D
5assignvariableop_14_batch_normalization_2_moving_mean:	H
9assignvariableop_15_batch_normalization_2_moving_variance:	?
#assignvariableop_16_conv2d_2_kernel:0
!assignvariableop_17_conv2d_2_bias:	>
/assignvariableop_18_batch_normalization_3_gamma:	=
.assignvariableop_19_batch_normalization_3_beta:	D
5assignvariableop_20_batch_normalization_3_moving_mean:	H
9assignvariableop_21_batch_normalization_3_moving_variance:	?
#assignvariableop_22_conv2d_3_kernel:0
!assignvariableop_23_conv2d_3_bias:	>
/assignvariableop_24_batch_normalization_4_gamma:	=
.assignvariableop_25_batch_normalization_4_beta:	D
5assignvariableop_26_batch_normalization_4_moving_mean:	H
9assignvariableop_27_batch_normalization_4_moving_variance:	G
+assignvariableop_28_conv2d_transpose_kernel:8
)assignvariableop_29_conv2d_transpose_bias:	>
/assignvariableop_30_batch_normalization_5_gamma:	=
.assignvariableop_31_batch_normalization_5_beta:	D
5assignvariableop_32_batch_normalization_5_moving_mean:	H
9assignvariableop_33_batch_normalization_5_moving_variance:	I
-assignvariableop_34_conv2d_transpose_1_kernel::
+assignvariableop_35_conv2d_transpose_1_bias:	>
/assignvariableop_36_batch_normalization_6_gamma:	=
.assignvariableop_37_batch_normalization_6_beta:	D
5assignvariableop_38_batch_normalization_6_moving_mean:	H
9assignvariableop_39_batch_normalization_6_moving_variance:	I
-assignvariableop_40_conv2d_transpose_2_kernel::
+assignvariableop_41_conv2d_transpose_2_bias:	>
/assignvariableop_42_batch_normalization_7_gamma:	=
.assignvariableop_43_batch_normalization_7_beta:	D
5assignvariableop_44_batch_normalization_7_moving_mean:	H
9assignvariableop_45_batch_normalization_7_moving_variance:	H
-assignvariableop_46_conv2d_transpose_3_kernel:@Ą9
+assignvariableop_47_conv2d_transpose_3_bias:@=
/assignvariableop_48_batch_normalization_8_gamma:@<
.assignvariableop_49_batch_normalization_8_beta:@C
5assignvariableop_50_batch_normalization_8_moving_mean:@G
9assignvariableop_51_batch_normalization_8_moving_variance:@G
-assignvariableop_52_conv2d_transpose_4_kernel:D9
+assignvariableop_53_conv2d_transpose_4_bias:'
assignvariableop_54_adam_iter:	 )
assignvariableop_55_adam_beta_1: )
assignvariableop_56_adam_beta_2: (
assignvariableop_57_adam_decay: 0
&assignvariableop_58_adam_learning_rate: #
assignvariableop_59_total: #
assignvariableop_60_count: B
4assignvariableop_61_adam_batch_normalization_gamma_m:A
3assignvariableop_62_adam_batch_normalization_beta_m:B
(assignvariableop_63_adam_conv2d_kernel_m:@4
&assignvariableop_64_adam_conv2d_bias_m:@D
6assignvariableop_65_adam_batch_normalization_1_gamma_m:@C
5assignvariableop_66_adam_batch_normalization_1_beta_m:@E
*assignvariableop_67_adam_conv2d_1_kernel_m:@7
(assignvariableop_68_adam_conv2d_1_bias_m:	E
6assignvariableop_69_adam_batch_normalization_2_gamma_m:	D
5assignvariableop_70_adam_batch_normalization_2_beta_m:	F
*assignvariableop_71_adam_conv2d_2_kernel_m:7
(assignvariableop_72_adam_conv2d_2_bias_m:	E
6assignvariableop_73_adam_batch_normalization_3_gamma_m:	D
5assignvariableop_74_adam_batch_normalization_3_beta_m:	F
*assignvariableop_75_adam_conv2d_3_kernel_m:7
(assignvariableop_76_adam_conv2d_3_bias_m:	E
6assignvariableop_77_adam_batch_normalization_4_gamma_m:	D
5assignvariableop_78_adam_batch_normalization_4_beta_m:	N
2assignvariableop_79_adam_conv2d_transpose_kernel_m:?
0assignvariableop_80_adam_conv2d_transpose_bias_m:	E
6assignvariableop_81_adam_batch_normalization_5_gamma_m:	D
5assignvariableop_82_adam_batch_normalization_5_beta_m:	P
4assignvariableop_83_adam_conv2d_transpose_1_kernel_m:A
2assignvariableop_84_adam_conv2d_transpose_1_bias_m:	E
6assignvariableop_85_adam_batch_normalization_6_gamma_m:	D
5assignvariableop_86_adam_batch_normalization_6_beta_m:	P
4assignvariableop_87_adam_conv2d_transpose_2_kernel_m:A
2assignvariableop_88_adam_conv2d_transpose_2_bias_m:	E
6assignvariableop_89_adam_batch_normalization_7_gamma_m:	D
5assignvariableop_90_adam_batch_normalization_7_beta_m:	O
4assignvariableop_91_adam_conv2d_transpose_3_kernel_m:@Ą@
2assignvariableop_92_adam_conv2d_transpose_3_bias_m:@D
6assignvariableop_93_adam_batch_normalization_8_gamma_m:@C
5assignvariableop_94_adam_batch_normalization_8_beta_m:@N
4assignvariableop_95_adam_conv2d_transpose_4_kernel_m:D@
2assignvariableop_96_adam_conv2d_transpose_4_bias_m:B
4assignvariableop_97_adam_batch_normalization_gamma_v:A
3assignvariableop_98_adam_batch_normalization_beta_v:B
(assignvariableop_99_adam_conv2d_kernel_v:@5
'assignvariableop_100_adam_conv2d_bias_v:@E
7assignvariableop_101_adam_batch_normalization_1_gamma_v:@D
6assignvariableop_102_adam_batch_normalization_1_beta_v:@F
+assignvariableop_103_adam_conv2d_1_kernel_v:@8
)assignvariableop_104_adam_conv2d_1_bias_v:	F
7assignvariableop_105_adam_batch_normalization_2_gamma_v:	E
6assignvariableop_106_adam_batch_normalization_2_beta_v:	G
+assignvariableop_107_adam_conv2d_2_kernel_v:8
)assignvariableop_108_adam_conv2d_2_bias_v:	F
7assignvariableop_109_adam_batch_normalization_3_gamma_v:	E
6assignvariableop_110_adam_batch_normalization_3_beta_v:	G
+assignvariableop_111_adam_conv2d_3_kernel_v:8
)assignvariableop_112_adam_conv2d_3_bias_v:	F
7assignvariableop_113_adam_batch_normalization_4_gamma_v:	E
6assignvariableop_114_adam_batch_normalization_4_beta_v:	O
3assignvariableop_115_adam_conv2d_transpose_kernel_v:@
1assignvariableop_116_adam_conv2d_transpose_bias_v:	F
7assignvariableop_117_adam_batch_normalization_5_gamma_v:	E
6assignvariableop_118_adam_batch_normalization_5_beta_v:	Q
5assignvariableop_119_adam_conv2d_transpose_1_kernel_v:B
3assignvariableop_120_adam_conv2d_transpose_1_bias_v:	F
7assignvariableop_121_adam_batch_normalization_6_gamma_v:	E
6assignvariableop_122_adam_batch_normalization_6_beta_v:	Q
5assignvariableop_123_adam_conv2d_transpose_2_kernel_v:B
3assignvariableop_124_adam_conv2d_transpose_2_bias_v:	F
7assignvariableop_125_adam_batch_normalization_7_gamma_v:	E
6assignvariableop_126_adam_batch_normalization_7_beta_v:	P
5assignvariableop_127_adam_conv2d_transpose_3_kernel_v:@ĄA
3assignvariableop_128_adam_conv2d_transpose_3_bias_v:@E
7assignvariableop_129_adam_batch_normalization_8_gamma_v:@D
6assignvariableop_130_adam_batch_normalization_8_beta_v:@O
5assignvariableop_131_adam_conv2d_transpose_4_kernel_v:DA
3assignvariableop_132_adam_conv2d_transpose_4_bias_v:
identity_134¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_100¢AssignVariableOp_101¢AssignVariableOp_102¢AssignVariableOp_103¢AssignVariableOp_104¢AssignVariableOp_105¢AssignVariableOp_106¢AssignVariableOp_107¢AssignVariableOp_108¢AssignVariableOp_109¢AssignVariableOp_11¢AssignVariableOp_110¢AssignVariableOp_111¢AssignVariableOp_112¢AssignVariableOp_113¢AssignVariableOp_114¢AssignVariableOp_115¢AssignVariableOp_116¢AssignVariableOp_117¢AssignVariableOp_118¢AssignVariableOp_119¢AssignVariableOp_12¢AssignVariableOp_120¢AssignVariableOp_121¢AssignVariableOp_122¢AssignVariableOp_123¢AssignVariableOp_124¢AssignVariableOp_125¢AssignVariableOp_126¢AssignVariableOp_127¢AssignVariableOp_128¢AssignVariableOp_129¢AssignVariableOp_13¢AssignVariableOp_130¢AssignVariableOp_131¢AssignVariableOp_132¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_80¢AssignVariableOp_81¢AssignVariableOp_82¢AssignVariableOp_83¢AssignVariableOp_84¢AssignVariableOp_85¢AssignVariableOp_86¢AssignVariableOp_87¢AssignVariableOp_88¢AssignVariableOp_89¢AssignVariableOp_9¢AssignVariableOp_90¢AssignVariableOp_91¢AssignVariableOp_92¢AssignVariableOp_93¢AssignVariableOp_94¢AssignVariableOp_95¢AssignVariableOp_96¢AssignVariableOp_97¢AssignVariableOp_98¢AssignVariableOp_99’J
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*¤J
valueJBJB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*¢
valueBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ć
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*®
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp*assignvariableop_batch_normalization_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp+assignvariableop_1_batch_normalization_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:”
AssignVariableOp_2AssignVariableOp2assignvariableop_2_batch_normalization_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:„
AssignVariableOp_3AssignVariableOp6assignvariableop_3_batch_normalization_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp assignvariableop_4_conv2d_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_conv2d_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp.assignvariableop_6_batch_normalization_1_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp-assignvariableop_7_batch_normalization_1_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_8AssignVariableOp4assignvariableop_8_batch_normalization_1_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_9AssignVariableOp8assignvariableop_9_batch_normalization_1_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv2d_1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv2d_1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_12AssignVariableOp/assignvariableop_12_batch_normalization_2_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp.assignvariableop_13_batch_normalization_2_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_14AssignVariableOp5assignvariableop_14_batch_normalization_2_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ŗ
AssignVariableOp_15AssignVariableOp9assignvariableop_15_batch_normalization_2_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp#assignvariableop_16_conv2d_2_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp!assignvariableop_17_conv2d_2_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_18AssignVariableOp/assignvariableop_18_batch_normalization_3_gammaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp.assignvariableop_19_batch_normalization_3_betaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_20AssignVariableOp5assignvariableop_20_batch_normalization_3_moving_meanIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Ŗ
AssignVariableOp_21AssignVariableOp9assignvariableop_21_batch_normalization_3_moving_varianceIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp#assignvariableop_22_conv2d_3_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp!assignvariableop_23_conv2d_3_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_24AssignVariableOp/assignvariableop_24_batch_normalization_4_gammaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp.assignvariableop_25_batch_normalization_4_betaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_26AssignVariableOp5assignvariableop_26_batch_normalization_4_moving_meanIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Ŗ
AssignVariableOp_27AssignVariableOp9assignvariableop_27_batch_normalization_4_moving_varianceIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp+assignvariableop_28_conv2d_transpose_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp)assignvariableop_29_conv2d_transpose_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_30AssignVariableOp/assignvariableop_30_batch_normalization_5_gammaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp.assignvariableop_31_batch_normalization_5_betaIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_32AssignVariableOp5assignvariableop_32_batch_normalization_5_moving_meanIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Ŗ
AssignVariableOp_33AssignVariableOp9assignvariableop_33_batch_normalization_5_moving_varianceIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp-assignvariableop_34_conv2d_transpose_1_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp+assignvariableop_35_conv2d_transpose_1_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_36AssignVariableOp/assignvariableop_36_batch_normalization_6_gammaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp.assignvariableop_37_batch_normalization_6_betaIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_38AssignVariableOp5assignvariableop_38_batch_normalization_6_moving_meanIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Ŗ
AssignVariableOp_39AssignVariableOp9assignvariableop_39_batch_normalization_6_moving_varianceIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp-assignvariableop_40_conv2d_transpose_2_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp+assignvariableop_41_conv2d_transpose_2_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_42AssignVariableOp/assignvariableop_42_batch_normalization_7_gammaIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp.assignvariableop_43_batch_normalization_7_betaIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_44AssignVariableOp5assignvariableop_44_batch_normalization_7_moving_meanIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:Ŗ
AssignVariableOp_45AssignVariableOp9assignvariableop_45_batch_normalization_7_moving_varianceIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp-assignvariableop_46_conv2d_transpose_3_kernelIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp+assignvariableop_47_conv2d_transpose_3_biasIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_48AssignVariableOp/assignvariableop_48_batch_normalization_8_gammaIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_49AssignVariableOp.assignvariableop_49_batch_normalization_8_betaIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_50AssignVariableOp5assignvariableop_50_batch_normalization_8_moving_meanIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:Ŗ
AssignVariableOp_51AssignVariableOp9assignvariableop_51_batch_normalization_8_moving_varianceIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOp-assignvariableop_52_conv2d_transpose_4_kernelIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_53AssignVariableOp+assignvariableop_53_conv2d_transpose_4_biasIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_54AssignVariableOpassignvariableop_54_adam_iterIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_55AssignVariableOpassignvariableop_55_adam_beta_1Identity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_56AssignVariableOpassignvariableop_56_adam_beta_2Identity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_57AssignVariableOpassignvariableop_57_adam_decayIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_58AssignVariableOp&assignvariableop_58_adam_learning_rateIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_59AssignVariableOpassignvariableop_59_totalIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_60AssignVariableOpassignvariableop_60_countIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:„
AssignVariableOp_61AssignVariableOp4assignvariableop_61_adam_batch_normalization_gamma_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_62AssignVariableOp3assignvariableop_62_adam_batch_normalization_beta_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_63AssignVariableOp(assignvariableop_63_adam_conv2d_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_64AssignVariableOp&assignvariableop_64_adam_conv2d_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_65AssignVariableOp6assignvariableop_65_adam_batch_normalization_1_gamma_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_66AssignVariableOp5assignvariableop_66_adam_batch_normalization_1_beta_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_67AssignVariableOp*assignvariableop_67_adam_conv2d_1_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_68AssignVariableOp(assignvariableop_68_adam_conv2d_1_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_69AssignVariableOp6assignvariableop_69_adam_batch_normalization_2_gamma_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_70AssignVariableOp5assignvariableop_70_adam_batch_normalization_2_beta_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_71AssignVariableOp*assignvariableop_71_adam_conv2d_2_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_72AssignVariableOp(assignvariableop_72_adam_conv2d_2_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_73AssignVariableOp6assignvariableop_73_adam_batch_normalization_3_gamma_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_74AssignVariableOp5assignvariableop_74_adam_batch_normalization_3_beta_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_75AssignVariableOp*assignvariableop_75_adam_conv2d_3_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_76AssignVariableOp(assignvariableop_76_adam_conv2d_3_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_77AssignVariableOp6assignvariableop_77_adam_batch_normalization_4_gamma_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_78AssignVariableOp5assignvariableop_78_adam_batch_normalization_4_beta_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_79AssignVariableOp2assignvariableop_79_adam_conv2d_transpose_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:”
AssignVariableOp_80AssignVariableOp0assignvariableop_80_adam_conv2d_transpose_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_81AssignVariableOp6assignvariableop_81_adam_batch_normalization_5_gamma_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_82AssignVariableOp5assignvariableop_82_adam_batch_normalization_5_beta_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:„
AssignVariableOp_83AssignVariableOp4assignvariableop_83_adam_conv2d_transpose_1_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_84AssignVariableOp2assignvariableop_84_adam_conv2d_transpose_1_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_85AssignVariableOp6assignvariableop_85_adam_batch_normalization_6_gamma_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_86AssignVariableOp5assignvariableop_86_adam_batch_normalization_6_beta_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:„
AssignVariableOp_87AssignVariableOp4assignvariableop_87_adam_conv2d_transpose_2_kernel_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_88AssignVariableOp2assignvariableop_88_adam_conv2d_transpose_2_bias_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_89AssignVariableOp6assignvariableop_89_adam_batch_normalization_7_gamma_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_90AssignVariableOp5assignvariableop_90_adam_batch_normalization_7_beta_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:„
AssignVariableOp_91AssignVariableOp4assignvariableop_91_adam_conv2d_transpose_3_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_92AssignVariableOp2assignvariableop_92_adam_conv2d_transpose_3_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_93AssignVariableOp6assignvariableop_93_adam_batch_normalization_8_gamma_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_94AssignVariableOp5assignvariableop_94_adam_batch_normalization_8_beta_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:„
AssignVariableOp_95AssignVariableOp4assignvariableop_95_adam_conv2d_transpose_4_kernel_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_96AssignVariableOp2assignvariableop_96_adam_conv2d_transpose_4_bias_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:„
AssignVariableOp_97AssignVariableOp4assignvariableop_97_adam_batch_normalization_gamma_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_98AssignVariableOp3assignvariableop_98_adam_batch_normalization_beta_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_99AssignVariableOp(assignvariableop_99_adam_conv2d_kernel_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_100AssignVariableOp'assignvariableop_100_adam_conv2d_bias_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:Ŗ
AssignVariableOp_101AssignVariableOp7assignvariableop_101_adam_batch_normalization_1_gamma_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_102AssignVariableOp6assignvariableop_102_adam_batch_normalization_1_beta_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_103AssignVariableOp+assignvariableop_103_adam_conv2d_1_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_104AssignVariableOp)assignvariableop_104_adam_conv2d_1_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:Ŗ
AssignVariableOp_105AssignVariableOp7assignvariableop_105_adam_batch_normalization_2_gamma_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_106AssignVariableOp6assignvariableop_106_adam_batch_normalization_2_beta_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_107AssignVariableOp+assignvariableop_107_adam_conv2d_2_kernel_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_108AssignVariableOp)assignvariableop_108_adam_conv2d_2_bias_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:Ŗ
AssignVariableOp_109AssignVariableOp7assignvariableop_109_adam_batch_normalization_3_gamma_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_110AssignVariableOp6assignvariableop_110_adam_batch_normalization_3_beta_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_111AssignVariableOp+assignvariableop_111_adam_conv2d_3_kernel_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_112AssignVariableOp)assignvariableop_112_adam_conv2d_3_bias_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:Ŗ
AssignVariableOp_113AssignVariableOp7assignvariableop_113_adam_batch_normalization_4_gamma_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_114AssignVariableOp6assignvariableop_114_adam_batch_normalization_4_beta_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_115AssignVariableOp3assignvariableop_115_adam_conv2d_transpose_kernel_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_116AssignVariableOp1assignvariableop_116_adam_conv2d_transpose_bias_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:Ŗ
AssignVariableOp_117AssignVariableOp7assignvariableop_117_adam_batch_normalization_5_gamma_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_118AssignVariableOp6assignvariableop_118_adam_batch_normalization_5_beta_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:Ø
AssignVariableOp_119AssignVariableOp5assignvariableop_119_adam_conv2d_transpose_1_kernel_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_120AssignVariableOp3assignvariableop_120_adam_conv2d_transpose_1_bias_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:Ŗ
AssignVariableOp_121AssignVariableOp7assignvariableop_121_adam_batch_normalization_6_gamma_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_122AssignVariableOp6assignvariableop_122_adam_batch_normalization_6_beta_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:Ø
AssignVariableOp_123AssignVariableOp5assignvariableop_123_adam_conv2d_transpose_2_kernel_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_124AssignVariableOp3assignvariableop_124_adam_conv2d_transpose_2_bias_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:Ŗ
AssignVariableOp_125AssignVariableOp7assignvariableop_125_adam_batch_normalization_7_gamma_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_126AssignVariableOp6assignvariableop_126_adam_batch_normalization_7_beta_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:Ø
AssignVariableOp_127AssignVariableOp5assignvariableop_127_adam_conv2d_transpose_3_kernel_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_128AssignVariableOp3assignvariableop_128_adam_conv2d_transpose_3_bias_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:Ŗ
AssignVariableOp_129AssignVariableOp7assignvariableop_129_adam_batch_normalization_8_gamma_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_130AssignVariableOp6assignvariableop_130_adam_batch_normalization_8_beta_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:Ø
AssignVariableOp_131AssignVariableOp5assignvariableop_131_adam_conv2d_transpose_4_kernel_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_132AssignVariableOp3assignvariableop_132_adam_conv2d_transpose_4_bias_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ß
Identity_133Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_134IdentityIdentity_133:output:0^NoOp_1*
T0*
_output_shapes
: Ė
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_134Identity_134:output:0*”
_input_shapes
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
É
c
G__inference_cropping2d_layer_call_and_return_conditional_losses_1285708

inputs
identityl
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"              n
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"    ų’’’ų’’’    n
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*

begin_mask	*
end_mask	
IdentityIdentitystrided_slice:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:r n
J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
ü
t
J__inference_concatenate_4_layer_call_and_return_conditional_losses_1286308

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*2
_output_shapes 
:’’’’’’’’’Ąb
IdentityIdentityconcat:output:0*
T0*2
_output_shapes 
:’’’’’’’’’Ą"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:’’’’’’’’’@:’’’’’’’’’:Y U
1
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs:ZV
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
ė
Å
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1289915

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ū
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs

m
Q__inference_periodic_padding2d_3_layer_call_and_return_conditional_losses_1289158

inputs
identityh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"        ų’’’j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         é
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskj
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ń
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :£
concatConcatV2strided_slice:output:0inputsstrided_slice_1:output:0concat/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’j
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ų’’’    l
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ś
strided_slice_2StridedSliceconcat:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskj
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ś
strided_slice_3StridedSliceconcat:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :²
concat_1ConcatV2strided_slice_2:output:0concat:output:0strided_slice_3:output:0concat_1/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’_
IdentityIdentityconcat_1:output:0*
T0*-
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:’’’’’’’’’:U Q
-
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¤

H__inference_concatenate_layer_call_and_return_conditional_losses_1289251
inputs_0
inputs_1
inputs_2
inputs_3
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1inputs_2inputs_3concat/axis:output:0*
N*
T0*1
_output_shapes
:’’’’’’’’’a
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesv
t:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:[ W
1
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/1:[W
1
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/2:[W
1
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/3
	
u
-__inference_concatenate_layer_call_fn_1289242
inputs_0
inputs_1
inputs_2
inputs_3
identityć
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_1285901j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesv
t:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:[ W
1
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/1:[W
1
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/2:[W
1
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/3
ó

(__inference_conv2d_layer_call_fn_1289384

inputs!
unknown:@
	unknown_0:@
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_1285941y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ė
Å
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1285185

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ū
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
«
Å
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1290385

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ė
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
Ż
”
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1290548

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ķ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs

v
J__inference_concatenate_4_layer_call_and_return_conditional_losses_1290615
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*2
_output_shapes 
:’’’’’’’’’Ąb
IdentityIdentityconcat:output:0*
T0*2
_output_shapes 
:’’’’’’’’’Ą"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:’’’’’’’’’@:’’’’’’’’’:[ W
1
_output_shapes
:’’’’’’’’’@
"
_user_specified_name
inputs/0:\X
2
_output_shapes 
:’’’’’’’’’
"
_user_specified_name
inputs/1
Ģ
©
4__inference_conv2d_transpose_4_layer_call_fn_1290841

inputs!
unknown:D
	unknown_0:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1285689
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+’’’’’’’’’’’’’’’’’’’’’’’’’’’D: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’D
 
_user_specified_nameinputs
Ż
”
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1289753

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ķ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ä
G
+__inference_reshape_3_layer_call_fn_1289220

inputs
identity¾
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_reshape_3_layer_call_and_return_conditional_losses_1285890j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:’’’’’’’’’:U Q
-
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
³
H
,__inference_cropping2d_layer_call_fn_1290915

inputs
identityŲ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_cropping2d_layer_call_and_return_conditional_losses_1285708
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:r n
J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ū
Ö
7__inference_batch_normalization_5_layer_call_fn_1290096

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1286758z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
ė
Å
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1285057

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ū
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ż
”
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1289609

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ķ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ė

M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1286138

inputsD
(conv2d_transpose_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskJ
stack/1Const*
_output_shapes
: *
dtype0*
value
B :J
stack/2Const*
_output_shapes
: *
dtype0*
value
B :J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype0Ī
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*2
_output_shapes 
:’’’’’’’’’*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’Y
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:’’’’’’’’’k
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
å

O__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1286397

inputsB
(conv2d_transpose_readvariableop_resource:D-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskJ
stack/1Const*
_output_shapes
: *
dtype0*
value
B :J
stack/2Const*
_output_shapes
: *
dtype0*
value
B :I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:D*
dtype0Ķ
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:’’’’’’’’’D: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:’’’’’’’’’D
 
_user_specified_nameinputs
Ą
E
)__inference_reshape_layer_call_fn_1289163

inputs
identity¼
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_1285842j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:’’’’’’’’’:U Q
-
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ß
b
F__inference_reshape_1_layer_call_and_return_conditional_losses_1289196

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :R
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :©
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’b
IdentityIdentityReshape:output:0*
T0*1
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:’’’’’’’’’:U Q
-
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Õ
Ņ
7__inference_batch_normalization_1_layer_call_fn_1289434

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1285964y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
«
Å
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1289663

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ė
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
ļµ
ł7
B__inference_model_layer_call_and_return_conditional_losses_1289042
inputs_0
inputs_1
inputs_2
inputs_39
+batch_normalization_readvariableop_resource:;
-batch_normalization_readvariableop_1_resource:J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:?
%conv2d_conv2d_readvariableop_resource:@4
&conv2d_biasadd_readvariableop_resource:@;
-batch_normalization_1_readvariableop_resource:@=
/batch_normalization_1_readvariableop_1_resource:@L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@B
'conv2d_1_conv2d_readvariableop_resource:@7
(conv2d_1_biasadd_readvariableop_resource:	<
-batch_normalization_2_readvariableop_resource:	>
/batch_normalization_2_readvariableop_1_resource:	M
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	C
'conv2d_2_conv2d_readvariableop_resource:7
(conv2d_2_biasadd_readvariableop_resource:	<
-batch_normalization_3_readvariableop_resource:	>
/batch_normalization_3_readvariableop_1_resource:	M
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	C
'conv2d_3_conv2d_readvariableop_resource:7
(conv2d_3_biasadd_readvariableop_resource:	<
-batch_normalization_4_readvariableop_resource:	>
/batch_normalization_4_readvariableop_1_resource:	M
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	U
9conv2d_transpose_conv2d_transpose_readvariableop_resource:?
0conv2d_transpose_biasadd_readvariableop_resource:	<
-batch_normalization_5_readvariableop_resource:	>
/batch_normalization_5_readvariableop_1_resource:	M
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:	W
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:A
2conv2d_transpose_1_biasadd_readvariableop_resource:	<
-batch_normalization_6_readvariableop_resource:	>
/batch_normalization_6_readvariableop_1_resource:	M
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:	W
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource:A
2conv2d_transpose_2_biasadd_readvariableop_resource:	<
-batch_normalization_7_readvariableop_resource:	>
/batch_normalization_7_readvariableop_1_resource:	M
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:	V
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource:@Ą@
2conv2d_transpose_3_biasadd_readvariableop_resource:@;
-batch_normalization_8_readvariableop_resource:@=
/batch_normalization_8_readvariableop_1_resource:@L
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:@U
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource:D@
2conv2d_transpose_4_biasadd_readvariableop_resource:
identity¢"batch_normalization/AssignNewValue¢$batch_normalization/AssignNewValue_1¢3batch_normalization/FusedBatchNormV3/ReadVariableOp¢5batch_normalization/FusedBatchNormV3/ReadVariableOp_1¢"batch_normalization/ReadVariableOp¢$batch_normalization/ReadVariableOp_1¢$batch_normalization_1/AssignNewValue¢&batch_normalization_1/AssignNewValue_1¢5batch_normalization_1/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_1/ReadVariableOp¢&batch_normalization_1/ReadVariableOp_1¢$batch_normalization_2/AssignNewValue¢&batch_normalization_2/AssignNewValue_1¢5batch_normalization_2/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_2/ReadVariableOp¢&batch_normalization_2/ReadVariableOp_1¢$batch_normalization_3/AssignNewValue¢&batch_normalization_3/AssignNewValue_1¢5batch_normalization_3/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_3/ReadVariableOp¢&batch_normalization_3/ReadVariableOp_1¢$batch_normalization_4/AssignNewValue¢&batch_normalization_4/AssignNewValue_1¢5batch_normalization_4/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_4/ReadVariableOp¢&batch_normalization_4/ReadVariableOp_1¢$batch_normalization_5/AssignNewValue¢&batch_normalization_5/AssignNewValue_1¢5batch_normalization_5/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_5/ReadVariableOp¢&batch_normalization_5/ReadVariableOp_1¢$batch_normalization_6/AssignNewValue¢&batch_normalization_6/AssignNewValue_1¢5batch_normalization_6/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_6/ReadVariableOp¢&batch_normalization_6/ReadVariableOp_1¢$batch_normalization_7/AssignNewValue¢&batch_normalization_7/AssignNewValue_1¢5batch_normalization_7/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_7/ReadVariableOp¢&batch_normalization_7/ReadVariableOp_1¢$batch_normalization_8/AssignNewValue¢&batch_normalization_8/AssignNewValue_1¢5batch_normalization_8/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_8/ReadVariableOp¢&batch_normalization_8/ReadVariableOp_1¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOp¢conv2d_2/BiasAdd/ReadVariableOp¢conv2d_2/Conv2D/ReadVariableOp¢conv2d_3/BiasAdd/ReadVariableOp¢conv2d_3/Conv2D/ReadVariableOp¢'conv2d_transpose/BiasAdd/ReadVariableOp¢0conv2d_transpose/conv2d_transpose/ReadVariableOp¢)conv2d_transpose_1/BiasAdd/ReadVariableOp¢2conv2d_transpose_1/conv2d_transpose/ReadVariableOp¢)conv2d_transpose_2/BiasAdd/ReadVariableOp¢2conv2d_transpose_2/conv2d_transpose/ReadVariableOp¢)conv2d_transpose_3/BiasAdd/ReadVariableOp¢2conv2d_transpose_3/conv2d_transpose/ReadVariableOp¢)conv2d_transpose_4/BiasAdd/ReadVariableOp¢2conv2d_transpose_4/conv2d_transpose/ReadVariableOp}
(periodic_padding2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"        ų’’’
*periodic_padding2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            
*periodic_padding2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         æ
"periodic_padding2d_3/strided_sliceStridedSliceinputs_31periodic_padding2d_3/strided_slice/stack:output:03periodic_padding2d_3/strided_slice/stack_1:output:03periodic_padding2d_3/strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_mask
*periodic_padding2d_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
,periodic_padding2d_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
,periodic_padding2d_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ē
$periodic_padding2d_3/strided_slice_1StridedSliceinputs_33periodic_padding2d_3/strided_slice_1/stack:output:05periodic_padding2d_3/strided_slice_1/stack_1:output:05periodic_padding2d_3/strided_slice_1/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskb
 periodic_padding2d_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ł
periodic_padding2d_3/concatConcatV2+periodic_padding2d_3/strided_slice:output:0inputs_3-periodic_padding2d_3/strided_slice_1:output:0)periodic_padding2d_3/concat/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’
*periodic_padding2d_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ų’’’    
,periodic_padding2d_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            
,periodic_padding2d_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ć
$periodic_padding2d_3/strided_slice_2StridedSlice$periodic_padding2d_3/concat:output:03periodic_padding2d_3/strided_slice_2/stack:output:05periodic_padding2d_3/strided_slice_2/stack_1:output:05periodic_padding2d_3/strided_slice_2/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_mask
*periodic_padding2d_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
,periodic_padding2d_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
,periodic_padding2d_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ć
$periodic_padding2d_3/strided_slice_3StridedSlice$periodic_padding2d_3/concat:output:03periodic_padding2d_3/strided_slice_3/stack:output:05periodic_padding2d_3/strided_slice_3/stack_1:output:05periodic_padding2d_3/strided_slice_3/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskd
"periodic_padding2d_3/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :
periodic_padding2d_3/concat_1ConcatV2-periodic_padding2d_3/strided_slice_2:output:0$periodic_padding2d_3/concat:output:0-periodic_padding2d_3/strided_slice_3:output:0+periodic_padding2d_3/concat_1/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’}
(periodic_padding2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"        ų’’’
*periodic_padding2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            
*periodic_padding2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         æ
"periodic_padding2d_2/strided_sliceStridedSliceinputs_21periodic_padding2d_2/strided_slice/stack:output:03periodic_padding2d_2/strided_slice/stack_1:output:03periodic_padding2d_2/strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_mask
*periodic_padding2d_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
,periodic_padding2d_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
,periodic_padding2d_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ē
$periodic_padding2d_2/strided_slice_1StridedSliceinputs_23periodic_padding2d_2/strided_slice_1/stack:output:05periodic_padding2d_2/strided_slice_1/stack_1:output:05periodic_padding2d_2/strided_slice_1/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskb
 periodic_padding2d_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ł
periodic_padding2d_2/concatConcatV2+periodic_padding2d_2/strided_slice:output:0inputs_2-periodic_padding2d_2/strided_slice_1:output:0)periodic_padding2d_2/concat/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’
*periodic_padding2d_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ų’’’    
,periodic_padding2d_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            
,periodic_padding2d_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ć
$periodic_padding2d_2/strided_slice_2StridedSlice$periodic_padding2d_2/concat:output:03periodic_padding2d_2/strided_slice_2/stack:output:05periodic_padding2d_2/strided_slice_2/stack_1:output:05periodic_padding2d_2/strided_slice_2/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_mask
*periodic_padding2d_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
,periodic_padding2d_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
,periodic_padding2d_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ć
$periodic_padding2d_2/strided_slice_3StridedSlice$periodic_padding2d_2/concat:output:03periodic_padding2d_2/strided_slice_3/stack:output:05periodic_padding2d_2/strided_slice_3/stack_1:output:05periodic_padding2d_2/strided_slice_3/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskd
"periodic_padding2d_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :
periodic_padding2d_2/concat_1ConcatV2-periodic_padding2d_2/strided_slice_2:output:0$periodic_padding2d_2/concat:output:0-periodic_padding2d_2/strided_slice_3:output:0+periodic_padding2d_2/concat_1/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’}
(periodic_padding2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"        ų’’’
*periodic_padding2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            
*periodic_padding2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         æ
"periodic_padding2d_1/strided_sliceStridedSliceinputs_11periodic_padding2d_1/strided_slice/stack:output:03periodic_padding2d_1/strided_slice/stack_1:output:03periodic_padding2d_1/strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_mask
*periodic_padding2d_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
,periodic_padding2d_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
,periodic_padding2d_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ē
$periodic_padding2d_1/strided_slice_1StridedSliceinputs_13periodic_padding2d_1/strided_slice_1/stack:output:05periodic_padding2d_1/strided_slice_1/stack_1:output:05periodic_padding2d_1/strided_slice_1/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskb
 periodic_padding2d_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ł
periodic_padding2d_1/concatConcatV2+periodic_padding2d_1/strided_slice:output:0inputs_1-periodic_padding2d_1/strided_slice_1:output:0)periodic_padding2d_1/concat/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’
*periodic_padding2d_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ų’’’    
,periodic_padding2d_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            
,periodic_padding2d_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ć
$periodic_padding2d_1/strided_slice_2StridedSlice$periodic_padding2d_1/concat:output:03periodic_padding2d_1/strided_slice_2/stack:output:05periodic_padding2d_1/strided_slice_2/stack_1:output:05periodic_padding2d_1/strided_slice_2/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_mask
*periodic_padding2d_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
,periodic_padding2d_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
,periodic_padding2d_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ć
$periodic_padding2d_1/strided_slice_3StridedSlice$periodic_padding2d_1/concat:output:03periodic_padding2d_1/strided_slice_3/stack:output:05periodic_padding2d_1/strided_slice_3/stack_1:output:05periodic_padding2d_1/strided_slice_3/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskd
"periodic_padding2d_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :
periodic_padding2d_1/concat_1ConcatV2-periodic_padding2d_1/strided_slice_2:output:0$periodic_padding2d_1/concat:output:0-periodic_padding2d_1/strided_slice_3:output:0+periodic_padding2d_1/concat_1/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’{
&periodic_padding2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"        ų’’’}
(periodic_padding2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            }
(periodic_padding2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ·
 periodic_padding2d/strided_sliceStridedSliceinputs_0/periodic_padding2d/strided_slice/stack:output:01periodic_padding2d/strided_slice/stack_1:output:01periodic_padding2d/strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_mask}
(periodic_padding2d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
*periodic_padding2d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
*periodic_padding2d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         æ
"periodic_padding2d/strided_slice_1StridedSliceinputs_01periodic_padding2d/strided_slice_1/stack:output:03periodic_padding2d/strided_slice_1/stack_1:output:03periodic_padding2d/strided_slice_1/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_mask`
periodic_padding2d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ń
periodic_padding2d/concatConcatV2)periodic_padding2d/strided_slice:output:0inputs_0+periodic_padding2d/strided_slice_1:output:0'periodic_padding2d/concat/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’}
(periodic_padding2d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ų’’’    
*periodic_padding2d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            
*periodic_padding2d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ł
"periodic_padding2d/strided_slice_2StridedSlice"periodic_padding2d/concat:output:01periodic_padding2d/strided_slice_2/stack:output:03periodic_padding2d/strided_slice_2/stack_1:output:03periodic_padding2d/strided_slice_2/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_mask}
(periodic_padding2d/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
*periodic_padding2d/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
*periodic_padding2d/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ł
"periodic_padding2d/strided_slice_3StridedSlice"periodic_padding2d/concat:output:01periodic_padding2d/strided_slice_3/stack:output:03periodic_padding2d/strided_slice_3/stack_1:output:03periodic_padding2d/strided_slice_3/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskb
 periodic_padding2d/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :
periodic_padding2d/concat_1ConcatV2+periodic_padding2d/strided_slice_2:output:0"periodic_padding2d/concat:output:0+periodic_padding2d/strided_slice_3:output:0)periodic_padding2d/concat_1/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’a
reshape/ShapeShape$periodic_padding2d/concat_1:output:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ł
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :Z
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :Y
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Ń
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
reshape/ReshapeReshape$periodic_padding2d/concat_1:output:0reshape/Reshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’e
reshape_1/ShapeShape&periodic_padding2d_1/concat_1:output:0*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :\
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :[
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Ū
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:¢
reshape_1/ReshapeReshape&periodic_padding2d_1/concat_1:output:0 reshape_1/Reshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’e
reshape_2/ShapeShape&periodic_padding2d_2/concat_1:output:0*
T0*
_output_shapes
:g
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :\
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :[
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Ū
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:¢
reshape_2/ReshapeReshape&periodic_padding2d_2/concat_1:output:0 reshape_2/Reshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’e
reshape_3/ShapeShape&periodic_padding2d_3/concat_1:output:0*
T0*
_output_shapes
:g
reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_3/strided_sliceStridedSlicereshape_3/Shape:output:0&reshape_3/strided_slice/stack:output:0(reshape_3/strided_slice/stack_1:output:0(reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :\
reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :[
reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Ū
reshape_3/Reshape/shapePack reshape_3/strided_slice:output:0"reshape_3/Reshape/shape/1:output:0"reshape_3/Reshape/shape/2:output:0"reshape_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:¢
reshape_3/ReshapeReshape&periodic_padding2d_3/concat_1:output:0 reshape_3/Reshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ó
concatenate/concatConcatV2reshape/Reshape:output:0reshape_1/Reshape:output:0reshape_2/Reshape:output:0reshape_3/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:’’’’’’’’’
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype0
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype0¬
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0°
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0æ
$batch_normalization/FusedBatchNormV3FusedBatchNormV3concatenate/concat:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ģ
conv2d/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’@*
paddingVALID*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’@f

conv2d/EluEluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:’’’’’’’’’@
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0°
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0“
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ę
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d/Elu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ó
conv2d_1/Conv2DConv2D*batch_normalization_1/FusedBatchNormV3:y:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’*
paddingVALID*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’k
conv2d_1/EluEluconv2d_1/BiasAdd:output:0*
T0*2
_output_shapes 
:’’’’’’’’’
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ķ
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_1/Elu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ó
conv2d_2/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’*
paddingVALID*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’k
conv2d_2/EluEluconv2d_2/BiasAdd:output:0*
T0*2
_output_shapes 
:’’’’’’’’’
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ķ
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_2/Elu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ó
conv2d_3/Conv2DConv2D*batch_normalization_3/FusedBatchNormV3:y:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’*
paddingVALID*
strides

conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’k
conv2d_3/EluEluconv2d_3/BiasAdd:output:0*
T0*2
_output_shapes 
:’’’’’’’’’
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ķ
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_3/Elu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ņ
concatenate_1/concatConcatV2conv2d_3/Elu:activations:0*batch_normalization_4/FusedBatchNormV3:y:0"concatenate_1/concat/axis:output:0*
N*
T0*2
_output_shapes 
:’’’’’’’’’c
conv2d_transpose/ShapeShapeconcatenate_1/concat:output:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¦
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value
B :[
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :[
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value
B :Ž
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:®
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask“
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype0
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0concatenate_1/concat:output:0*
T0*2
_output_shapes 
:’’’’’’’’’*
paddingVALID*
strides

'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0½
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’{
conv2d_transpose/EluElu!conv2d_transpose/BiasAdd:output:0*
T0*2
_output_shapes 
:’’’’’’’’’
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Õ
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3"conv2d_transpose/Elu:activations:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0[
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ņ
concatenate_2/concatConcatV2conv2d_2/Elu:activations:0*batch_normalization_5/FusedBatchNormV3:y:0"concatenate_2/concat/axis:output:0*
N*
T0*2
_output_shapes 
:’’’’’’’’’e
conv2d_transpose_1/ShapeShapeconcatenate_2/concat:output:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value
B :]
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :]
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :č
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ø
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskø
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype0
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0concatenate_2/concat:output:0*
T0*2
_output_shapes 
:’’’’’’’’’*
paddingVALID*
strides

)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ć
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’
conv2d_transpose_1/EluElu#conv2d_transpose_1/BiasAdd:output:0*
T0*2
_output_shapes 
:’’’’’’’’’
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0×
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_1/Elu:activations:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_6/AssignNewValueAssignVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
&batch_normalization_6/AssignNewValue_1AssignVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0[
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ņ
concatenate_3/concatConcatV2conv2d_1/Elu:activations:0*batch_normalization_6/FusedBatchNormV3:y:0"concatenate_3/concat/axis:output:0*
N*
T0*2
_output_shapes 
:’’’’’’’’’e
conv2d_transpose_2/ShapeShapeconcatenate_3/concat:output:0*
T0*
_output_shapes
:p
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value
B :]
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value
B :]
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value
B :č
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ø
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskø
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype0
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0concatenate_3/concat:output:0*
T0*2
_output_shapes 
:’’’’’’’’’*
paddingVALID*
strides

)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ć
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’
conv2d_transpose_2/EluElu#conv2d_transpose_2/BiasAdd:output:0*
T0*2
_output_shapes 
:’’’’’’’’’
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0×
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_2/Elu:activations:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_7/AssignNewValueAssignVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
&batch_normalization_7/AssignNewValue_1AssignVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0[
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Š
concatenate_4/concatConcatV2conv2d/Elu:activations:0*batch_normalization_7/FusedBatchNormV3:y:0"concatenate_4/concat/axis:output:0*
N*
T0*2
_output_shapes 
:’’’’’’’’’Ąe
conv2d_transpose_3/ShapeShapeconcatenate_4/concat:output:0*
T0*
_output_shapes
:p
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value
B :]
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value
B :\
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@č
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ø
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask·
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@Ą*
dtype0
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0concatenate_4/concat:output:0*
T0*1
_output_shapes
:’’’’’’’’’@*
paddingVALID*
strides

)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ā
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’@~
conv2d_transpose_3/EluElu#conv2d_transpose_3/BiasAdd:output:0*
T0*1
_output_shapes
:’’’’’’’’’@
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:@*
dtype0
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:@*
dtype0°
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0“
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ņ
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_3/Elu:activations:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_8/AssignNewValueAssignVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0
&batch_normalization_8/AssignNewValue_1AssignVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0[
concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ņ
concatenate_5/concatConcatV2concatenate/concat:output:0*batch_normalization_8/FusedBatchNormV3:y:0"concatenate_5/concat/axis:output:0*
N*
T0*1
_output_shapes
:’’’’’’’’’De
conv2d_transpose_4/ShapeShapeconcatenate_5/concat:output:0*
T0*
_output_shapes
:p
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value
B :]
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value
B :\
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :č
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0#conv2d_transpose_4/stack/1:output:0#conv2d_transpose_4/stack/2:output:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ø
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¶
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:D*
dtype0
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0concatenate_5/concat:output:0*
T0*1
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides

)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ā
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’w
cropping2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"              y
 cropping2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"    ų’’’ų’’’    y
 cropping2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ·
cropping2d/strided_sliceStridedSlice#conv2d_transpose_4/BiasAdd:output:0'cropping2d/strided_slice/stack:output:0)cropping2d/strided_slice/stack_1:output:0)cropping2d/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:’’’’’’’’’*

begin_mask	*
end_mask	z
IdentityIdentity!cropping2d/strided_slice:output:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’Å
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*å
_input_shapesÓ
Š:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12L
$batch_normalization_6/AssignNewValue$batch_normalization_6/AssignNewValue2P
&batch_normalization_6/AssignNewValue_1&batch_normalization_6/AssignNewValue_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12L
$batch_normalization_7/AssignNewValue$batch_normalization_7/AssignNewValue2P
&batch_normalization_7/AssignNewValue_1&batch_normalization_7/AssignNewValue_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12L
$batch_normalization_8/AssignNewValue$batch_normalization_8/AssignNewValue2P
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp:W S
-
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/0:WS
-
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/1:WS
-
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/2:WS
-
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/3
Ļ
Ŗ
4__inference_conv2d_transpose_3_layer_call_fn_1290624

inputs"
unknown:@Ą
	unknown_0:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1285577
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ą: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ą
 
_user_specified_nameinputs
ų#

M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1290020

inputsD
(conv2d_transpose_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B :F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype0Ž
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’i
EluEluBiasAdd:output:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’{
IdentityIdentityElu:activations:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
	
Ö
7__inference_batch_normalization_7_layer_call_fn_1290491

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1285493
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
ń#

O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1285577

inputsC
(conv2d_transpose_readvariableop_resource:@Ą-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B :F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@Ą*
dtype0Ż
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@h
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@z
IdentityIdentityElu:activations:0^NoOp*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ą: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ą
 
_user_specified_nameinputs
±£
7
"__inference__wrapped_model_1284876	
u_vel	
v_vel	
w_vel

pr0_71?
1model_batch_normalization_readvariableop_resource:A
3model_batch_normalization_readvariableop_1_resource:P
Bmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource:R
Dmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:E
+model_conv2d_conv2d_readvariableop_resource:@:
,model_conv2d_biasadd_readvariableop_resource:@A
3model_batch_normalization_1_readvariableop_resource:@C
5model_batch_normalization_1_readvariableop_1_resource:@R
Dmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@T
Fmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@H
-model_conv2d_1_conv2d_readvariableop_resource:@=
.model_conv2d_1_biasadd_readvariableop_resource:	B
3model_batch_normalization_2_readvariableop_resource:	D
5model_batch_normalization_2_readvariableop_1_resource:	S
Dmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	U
Fmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	I
-model_conv2d_2_conv2d_readvariableop_resource:=
.model_conv2d_2_biasadd_readvariableop_resource:	B
3model_batch_normalization_3_readvariableop_resource:	D
5model_batch_normalization_3_readvariableop_1_resource:	S
Dmodel_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	U
Fmodel_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	I
-model_conv2d_3_conv2d_readvariableop_resource:=
.model_conv2d_3_biasadd_readvariableop_resource:	B
3model_batch_normalization_4_readvariableop_resource:	D
5model_batch_normalization_4_readvariableop_1_resource:	S
Dmodel_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	U
Fmodel_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	[
?model_conv2d_transpose_conv2d_transpose_readvariableop_resource:E
6model_conv2d_transpose_biasadd_readvariableop_resource:	B
3model_batch_normalization_5_readvariableop_resource:	D
5model_batch_normalization_5_readvariableop_1_resource:	S
Dmodel_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:	U
Fmodel_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:	]
Amodel_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:G
8model_conv2d_transpose_1_biasadd_readvariableop_resource:	B
3model_batch_normalization_6_readvariableop_resource:	D
5model_batch_normalization_6_readvariableop_1_resource:	S
Dmodel_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:	U
Fmodel_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:	]
Amodel_conv2d_transpose_2_conv2d_transpose_readvariableop_resource:G
8model_conv2d_transpose_2_biasadd_readvariableop_resource:	B
3model_batch_normalization_7_readvariableop_resource:	D
5model_batch_normalization_7_readvariableop_1_resource:	S
Dmodel_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:	U
Fmodel_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:	\
Amodel_conv2d_transpose_3_conv2d_transpose_readvariableop_resource:@ĄF
8model_conv2d_transpose_3_biasadd_readvariableop_resource:@A
3model_batch_normalization_8_readvariableop_resource:@C
5model_batch_normalization_8_readvariableop_1_resource:@R
Dmodel_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:@T
Fmodel_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:@[
Amodel_conv2d_transpose_4_conv2d_transpose_readvariableop_resource:DF
8model_conv2d_transpose_4_biasadd_readvariableop_resource:
identity¢9model/batch_normalization/FusedBatchNormV3/ReadVariableOp¢;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1¢(model/batch_normalization/ReadVariableOp¢*model/batch_normalization/ReadVariableOp_1¢;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp¢=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1¢*model/batch_normalization_1/ReadVariableOp¢,model/batch_normalization_1/ReadVariableOp_1¢;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp¢=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1¢*model/batch_normalization_2/ReadVariableOp¢,model/batch_normalization_2/ReadVariableOp_1¢;model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp¢=model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1¢*model/batch_normalization_3/ReadVariableOp¢,model/batch_normalization_3/ReadVariableOp_1¢;model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp¢=model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1¢*model/batch_normalization_4/ReadVariableOp¢,model/batch_normalization_4/ReadVariableOp_1¢;model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp¢=model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1¢*model/batch_normalization_5/ReadVariableOp¢,model/batch_normalization_5/ReadVariableOp_1¢;model/batch_normalization_6/FusedBatchNormV3/ReadVariableOp¢=model/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1¢*model/batch_normalization_6/ReadVariableOp¢,model/batch_normalization_6/ReadVariableOp_1¢;model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp¢=model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1¢*model/batch_normalization_7/ReadVariableOp¢,model/batch_normalization_7/ReadVariableOp_1¢;model/batch_normalization_8/FusedBatchNormV3/ReadVariableOp¢=model/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1¢*model/batch_normalization_8/ReadVariableOp¢,model/batch_normalization_8/ReadVariableOp_1¢#model/conv2d/BiasAdd/ReadVariableOp¢"model/conv2d/Conv2D/ReadVariableOp¢%model/conv2d_1/BiasAdd/ReadVariableOp¢$model/conv2d_1/Conv2D/ReadVariableOp¢%model/conv2d_2/BiasAdd/ReadVariableOp¢$model/conv2d_2/Conv2D/ReadVariableOp¢%model/conv2d_3/BiasAdd/ReadVariableOp¢$model/conv2d_3/Conv2D/ReadVariableOp¢-model/conv2d_transpose/BiasAdd/ReadVariableOp¢6model/conv2d_transpose/conv2d_transpose/ReadVariableOp¢/model/conv2d_transpose_1/BiasAdd/ReadVariableOp¢8model/conv2d_transpose_1/conv2d_transpose/ReadVariableOp¢/model/conv2d_transpose_2/BiasAdd/ReadVariableOp¢8model/conv2d_transpose_2/conv2d_transpose/ReadVariableOp¢/model/conv2d_transpose_3/BiasAdd/ReadVariableOp¢8model/conv2d_transpose_3/conv2d_transpose/ReadVariableOp¢/model/conv2d_transpose_4/BiasAdd/ReadVariableOp¢8model/conv2d_transpose_4/conv2d_transpose/ReadVariableOp
.model/periodic_padding2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"        ų’’’
0model/periodic_padding2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            
0model/periodic_padding2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Õ
(model/periodic_padding2d_3/strided_sliceStridedSlicepr0_717model/periodic_padding2d_3/strided_slice/stack:output:09model/periodic_padding2d_3/strided_slice/stack_1:output:09model/periodic_padding2d_3/strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_mask
0model/periodic_padding2d_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
2model/periodic_padding2d_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
2model/periodic_padding2d_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ż
*model/periodic_padding2d_3/strided_slice_1StridedSlicepr0_719model/periodic_padding2d_3/strided_slice_1/stack:output:0;model/periodic_padding2d_3/strided_slice_1/stack_1:output:0;model/periodic_padding2d_3/strided_slice_1/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskh
&model/periodic_padding2d_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
!model/periodic_padding2d_3/concatConcatV21model/periodic_padding2d_3/strided_slice:output:0pr0_713model/periodic_padding2d_3/strided_slice_1:output:0/model/periodic_padding2d_3/concat/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’
0model/periodic_padding2d_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ų’’’    
2model/periodic_padding2d_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            
2model/periodic_padding2d_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
*model/periodic_padding2d_3/strided_slice_2StridedSlice*model/periodic_padding2d_3/concat:output:09model/periodic_padding2d_3/strided_slice_2/stack:output:0;model/periodic_padding2d_3/strided_slice_2/stack_1:output:0;model/periodic_padding2d_3/strided_slice_2/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_mask
0model/periodic_padding2d_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
2model/periodic_padding2d_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
2model/periodic_padding2d_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
*model/periodic_padding2d_3/strided_slice_3StridedSlice*model/periodic_padding2d_3/concat:output:09model/periodic_padding2d_3/strided_slice_3/stack:output:0;model/periodic_padding2d_3/strided_slice_3/stack_1:output:0;model/periodic_padding2d_3/strided_slice_3/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskj
(model/periodic_padding2d_3/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :¹
#model/periodic_padding2d_3/concat_1ConcatV23model/periodic_padding2d_3/strided_slice_2:output:0*model/periodic_padding2d_3/concat:output:03model/periodic_padding2d_3/strided_slice_3:output:01model/periodic_padding2d_3/concat_1/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’
.model/periodic_padding2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"        ų’’’
0model/periodic_padding2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            
0model/periodic_padding2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ō
(model/periodic_padding2d_2/strided_sliceStridedSlicew_vel7model/periodic_padding2d_2/strided_slice/stack:output:09model/periodic_padding2d_2/strided_slice/stack_1:output:09model/periodic_padding2d_2/strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_mask
0model/periodic_padding2d_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
2model/periodic_padding2d_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
2model/periodic_padding2d_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ü
*model/periodic_padding2d_2/strided_slice_1StridedSlicew_vel9model/periodic_padding2d_2/strided_slice_1/stack:output:0;model/periodic_padding2d_2/strided_slice_1/stack_1:output:0;model/periodic_padding2d_2/strided_slice_1/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskh
&model/periodic_padding2d_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
!model/periodic_padding2d_2/concatConcatV21model/periodic_padding2d_2/strided_slice:output:0w_vel3model/periodic_padding2d_2/strided_slice_1:output:0/model/periodic_padding2d_2/concat/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’
0model/periodic_padding2d_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ų’’’    
2model/periodic_padding2d_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            
2model/periodic_padding2d_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
*model/periodic_padding2d_2/strided_slice_2StridedSlice*model/periodic_padding2d_2/concat:output:09model/periodic_padding2d_2/strided_slice_2/stack:output:0;model/periodic_padding2d_2/strided_slice_2/stack_1:output:0;model/periodic_padding2d_2/strided_slice_2/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_mask
0model/periodic_padding2d_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
2model/periodic_padding2d_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
2model/periodic_padding2d_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
*model/periodic_padding2d_2/strided_slice_3StridedSlice*model/periodic_padding2d_2/concat:output:09model/periodic_padding2d_2/strided_slice_3/stack:output:0;model/periodic_padding2d_2/strided_slice_3/stack_1:output:0;model/periodic_padding2d_2/strided_slice_3/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskj
(model/periodic_padding2d_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :¹
#model/periodic_padding2d_2/concat_1ConcatV23model/periodic_padding2d_2/strided_slice_2:output:0*model/periodic_padding2d_2/concat:output:03model/periodic_padding2d_2/strided_slice_3:output:01model/periodic_padding2d_2/concat_1/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’
.model/periodic_padding2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"        ų’’’
0model/periodic_padding2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            
0model/periodic_padding2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ō
(model/periodic_padding2d_1/strided_sliceStridedSlicev_vel7model/periodic_padding2d_1/strided_slice/stack:output:09model/periodic_padding2d_1/strided_slice/stack_1:output:09model/periodic_padding2d_1/strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_mask
0model/periodic_padding2d_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
2model/periodic_padding2d_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
2model/periodic_padding2d_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ü
*model/periodic_padding2d_1/strided_slice_1StridedSlicev_vel9model/periodic_padding2d_1/strided_slice_1/stack:output:0;model/periodic_padding2d_1/strided_slice_1/stack_1:output:0;model/periodic_padding2d_1/strided_slice_1/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskh
&model/periodic_padding2d_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
!model/periodic_padding2d_1/concatConcatV21model/periodic_padding2d_1/strided_slice:output:0v_vel3model/periodic_padding2d_1/strided_slice_1:output:0/model/periodic_padding2d_1/concat/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’
0model/periodic_padding2d_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ų’’’    
2model/periodic_padding2d_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            
2model/periodic_padding2d_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
*model/periodic_padding2d_1/strided_slice_2StridedSlice*model/periodic_padding2d_1/concat:output:09model/periodic_padding2d_1/strided_slice_2/stack:output:0;model/periodic_padding2d_1/strided_slice_2/stack_1:output:0;model/periodic_padding2d_1/strided_slice_2/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_mask
0model/periodic_padding2d_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
2model/periodic_padding2d_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
2model/periodic_padding2d_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
*model/periodic_padding2d_1/strided_slice_3StridedSlice*model/periodic_padding2d_1/concat:output:09model/periodic_padding2d_1/strided_slice_3/stack:output:0;model/periodic_padding2d_1/strided_slice_3/stack_1:output:0;model/periodic_padding2d_1/strided_slice_3/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskj
(model/periodic_padding2d_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :¹
#model/periodic_padding2d_1/concat_1ConcatV23model/periodic_padding2d_1/strided_slice_2:output:0*model/periodic_padding2d_1/concat:output:03model/periodic_padding2d_1/strided_slice_3:output:01model/periodic_padding2d_1/concat_1/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’
,model/periodic_padding2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"        ų’’’
.model/periodic_padding2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            
.model/periodic_padding2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ģ
&model/periodic_padding2d/strided_sliceStridedSliceu_vel5model/periodic_padding2d/strided_slice/stack:output:07model/periodic_padding2d/strided_slice/stack_1:output:07model/periodic_padding2d/strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_mask
.model/periodic_padding2d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
0model/periodic_padding2d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
0model/periodic_padding2d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ō
(model/periodic_padding2d/strided_slice_1StridedSliceu_vel7model/periodic_padding2d/strided_slice_1/stack:output:09model/periodic_padding2d/strided_slice_1/stack_1:output:09model/periodic_padding2d/strided_slice_1/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskf
$model/periodic_padding2d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
model/periodic_padding2d/concatConcatV2/model/periodic_padding2d/strided_slice:output:0u_vel1model/periodic_padding2d/strided_slice_1:output:0-model/periodic_padding2d/concat/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’
.model/periodic_padding2d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ų’’’    
0model/periodic_padding2d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            
0model/periodic_padding2d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ÷
(model/periodic_padding2d/strided_slice_2StridedSlice(model/periodic_padding2d/concat:output:07model/periodic_padding2d/strided_slice_2/stack:output:09model/periodic_padding2d/strided_slice_2/stack_1:output:09model/periodic_padding2d/strided_slice_2/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_mask
.model/periodic_padding2d/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
0model/periodic_padding2d/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
0model/periodic_padding2d/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ÷
(model/periodic_padding2d/strided_slice_3StridedSlice(model/periodic_padding2d/concat:output:07model/periodic_padding2d/strided_slice_3/stack:output:09model/periodic_padding2d/strided_slice_3/stack_1:output:09model/periodic_padding2d/strided_slice_3/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskh
&model/periodic_padding2d/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :Æ
!model/periodic_padding2d/concat_1ConcatV21model/periodic_padding2d/strided_slice_2:output:0(model/periodic_padding2d/concat:output:01model/periodic_padding2d/strided_slice_3:output:0/model/periodic_padding2d/concat_1/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’m
model/reshape/ShapeShape*model/periodic_padding2d/concat_1:output:0*
T0*
_output_shapes
:k
!model/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#model/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#model/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
model/reshape/strided_sliceStridedSlicemodel/reshape/Shape:output:0*model/reshape/strided_slice/stack:output:0,model/reshape/strided_slice/stack_1:output:0,model/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
model/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :`
model/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :_
model/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ļ
model/reshape/Reshape/shapePack$model/reshape/strided_slice:output:0&model/reshape/Reshape/shape/1:output:0&model/reshape/Reshape/shape/2:output:0&model/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:®
model/reshape/ReshapeReshape*model/periodic_padding2d/concat_1:output:0$model/reshape/Reshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’q
model/reshape_1/ShapeShape,model/periodic_padding2d_1/concat_1:output:0*
T0*
_output_shapes
:m
#model/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%model/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%model/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:”
model/reshape_1/strided_sliceStridedSlicemodel/reshape_1/Shape:output:0,model/reshape_1/strided_slice/stack:output:0.model/reshape_1/strided_slice/stack_1:output:0.model/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
model/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :b
model/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :a
model/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ł
model/reshape_1/Reshape/shapePack&model/reshape_1/strided_slice:output:0(model/reshape_1/Reshape/shape/1:output:0(model/reshape_1/Reshape/shape/2:output:0(model/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:“
model/reshape_1/ReshapeReshape,model/periodic_padding2d_1/concat_1:output:0&model/reshape_1/Reshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’q
model/reshape_2/ShapeShape,model/periodic_padding2d_2/concat_1:output:0*
T0*
_output_shapes
:m
#model/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%model/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%model/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:”
model/reshape_2/strided_sliceStridedSlicemodel/reshape_2/Shape:output:0,model/reshape_2/strided_slice/stack:output:0.model/reshape_2/strided_slice/stack_1:output:0.model/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
model/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :b
model/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :a
model/reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ł
model/reshape_2/Reshape/shapePack&model/reshape_2/strided_slice:output:0(model/reshape_2/Reshape/shape/1:output:0(model/reshape_2/Reshape/shape/2:output:0(model/reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:“
model/reshape_2/ReshapeReshape,model/periodic_padding2d_2/concat_1:output:0&model/reshape_2/Reshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’q
model/reshape_3/ShapeShape,model/periodic_padding2d_3/concat_1:output:0*
T0*
_output_shapes
:m
#model/reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%model/reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%model/reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:”
model/reshape_3/strided_sliceStridedSlicemodel/reshape_3/Shape:output:0,model/reshape_3/strided_slice/stack:output:0.model/reshape_3/strided_slice/stack_1:output:0.model/reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
model/reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :b
model/reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :a
model/reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ł
model/reshape_3/Reshape/shapePack&model/reshape_3/strided_slice:output:0(model/reshape_3/Reshape/shape/1:output:0(model/reshape_3/Reshape/shape/2:output:0(model/reshape_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:“
model/reshape_3/ReshapeReshape,model/periodic_padding2d_3/concat_1:output:0&model/reshape_3/Reshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’_
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
model/concatenate/concatConcatV2model/reshape/Reshape:output:0 model/reshape_1/Reshape:output:0 model/reshape_2/Reshape:output:0 model/reshape_3/Reshape:output:0&model/concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:’’’’’’’’’
(model/batch_normalization/ReadVariableOpReadVariableOp1model_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype0
*model/batch_normalization/ReadVariableOp_1ReadVariableOp3model_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype0ø
9model/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpBmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0¼
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Õ
*model/batch_normalization/FusedBatchNormV3FusedBatchNormV3!model/concatenate/concat:output:00model/batch_normalization/ReadVariableOp:value:02model/batch_normalization/ReadVariableOp_1:value:0Amodel/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Cmodel/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:’’’’’’’’’:::::*
epsilon%o:*
is_training( 
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ž
model/conv2d/Conv2DConv2D.model/batch_normalization/FusedBatchNormV3:y:0*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’@*
paddingVALID*
strides

#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¦
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’@r
model/conv2d/EluElumodel/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:’’’’’’’’’@
*model/batch_normalization_1/ReadVariableOpReadVariableOp3model_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0
,model/batch_normalization_1/ReadVariableOp_1ReadVariableOp5model_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0¼
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ą
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ü
,model/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3model/conv2d/Elu:activations:02model/batch_normalization_1/ReadVariableOp:value:04model/batch_normalization_1/ReadVariableOp_1:value:0Cmodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
is_training( 
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0å
model/conv2d_1/Conv2DConv2D0model/batch_normalization_1/FusedBatchNormV3:y:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’*
paddingVALID*
strides

%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’w
model/conv2d_1/EluElumodel/conv2d_1/BiasAdd:output:0*
T0*2
_output_shapes 
:’’’’’’’’’
*model/batch_normalization_2/ReadVariableOpReadVariableOp3model_batch_normalization_2_readvariableop_resource*
_output_shapes	
:*
dtype0
,model/batch_normalization_2/ReadVariableOp_1ReadVariableOp5model_batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:*
dtype0½
;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Į
=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0ć
,model/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3 model/conv2d_1/Elu:activations:02model/batch_normalization_2/ReadVariableOp:value:04model/batch_normalization_2/ReadVariableOp_1:value:0Cmodel/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
is_training( 
$model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0å
model/conv2d_2/Conv2DConv2D0model/batch_normalization_2/FusedBatchNormV3:y:0,model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’*
paddingVALID*
strides

%model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
model/conv2d_2/BiasAddBiasAddmodel/conv2d_2/Conv2D:output:0-model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’w
model/conv2d_2/EluElumodel/conv2d_2/BiasAdd:output:0*
T0*2
_output_shapes 
:’’’’’’’’’
*model/batch_normalization_3/ReadVariableOpReadVariableOp3model_batch_normalization_3_readvariableop_resource*
_output_shapes	
:*
dtype0
,model/batch_normalization_3/ReadVariableOp_1ReadVariableOp5model_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:*
dtype0½
;model/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Į
=model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0ć
,model/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3 model/conv2d_2/Elu:activations:02model/batch_normalization_3/ReadVariableOp:value:04model/batch_normalization_3/ReadVariableOp_1:value:0Cmodel/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
is_training( 
$model/conv2d_3/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0å
model/conv2d_3/Conv2DConv2D0model/batch_normalization_3/FusedBatchNormV3:y:0,model/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’*
paddingVALID*
strides

%model/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
model/conv2d_3/BiasAddBiasAddmodel/conv2d_3/Conv2D:output:0-model/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’w
model/conv2d_3/EluElumodel/conv2d_3/BiasAdd:output:0*
T0*2
_output_shapes 
:’’’’’’’’’
*model/batch_normalization_4/ReadVariableOpReadVariableOp3model_batch_normalization_4_readvariableop_resource*
_output_shapes	
:*
dtype0
,model/batch_normalization_4/ReadVariableOp_1ReadVariableOp5model_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:*
dtype0½
;model/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Į
=model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0ć
,model/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3 model/conv2d_3/Elu:activations:02model/batch_normalization_4/ReadVariableOp:value:04model/batch_normalization_4/ReadVariableOp_1:value:0Cmodel/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
is_training( a
model/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ź
model/concatenate_1/concatConcatV2 model/conv2d_3/Elu:activations:00model/batch_normalization_4/FusedBatchNormV3:y:0(model/concatenate_1/concat/axis:output:0*
N*
T0*2
_output_shapes 
:’’’’’’’’’o
model/conv2d_transpose/ShapeShape#model/concatenate_1/concat:output:0*
T0*
_output_shapes
:t
*model/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,model/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,model/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$model/conv2d_transpose/strided_sliceStridedSlice%model/conv2d_transpose/Shape:output:03model/conv2d_transpose/strided_slice/stack:output:05model/conv2d_transpose/strided_slice/stack_1:output:05model/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
model/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value
B :a
model/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :a
model/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value
B :ü
model/conv2d_transpose/stackPack-model/conv2d_transpose/strided_slice:output:0'model/conv2d_transpose/stack/1:output:0'model/conv2d_transpose/stack/2:output:0'model/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:v
,model/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ģ
&model/conv2d_transpose/strided_slice_1StridedSlice%model/conv2d_transpose/stack:output:05model/conv2d_transpose/strided_slice_1/stack:output:07model/conv2d_transpose/strided_slice_1/stack_1:output:07model/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskĄ
6model/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp?model_conv2d_transpose_conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype0°
'model/conv2d_transpose/conv2d_transposeConv2DBackpropInput%model/conv2d_transpose/stack:output:0>model/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0#model/concatenate_1/concat:output:0*
T0*2
_output_shapes 
:’’’’’’’’’*
paddingVALID*
strides
”
-model/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp6model_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ļ
model/conv2d_transpose/BiasAddBiasAdd0model/conv2d_transpose/conv2d_transpose:output:05model/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’
model/conv2d_transpose/EluElu'model/conv2d_transpose/BiasAdd:output:0*
T0*2
_output_shapes 
:’’’’’’’’’
*model/batch_normalization_5/ReadVariableOpReadVariableOp3model_batch_normalization_5_readvariableop_resource*
_output_shapes	
:*
dtype0
,model/batch_normalization_5/ReadVariableOp_1ReadVariableOp5model_batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:*
dtype0½
;model/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Į
=model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0ė
,model/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3(model/conv2d_transpose/Elu:activations:02model/batch_normalization_5/ReadVariableOp:value:04model/batch_normalization_5/ReadVariableOp_1:value:0Cmodel/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
is_training( a
model/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ź
model/concatenate_2/concatConcatV2 model/conv2d_2/Elu:activations:00model/batch_normalization_5/FusedBatchNormV3:y:0(model/concatenate_2/concat/axis:output:0*
N*
T0*2
_output_shapes 
:’’’’’’’’’q
model/conv2d_transpose_1/ShapeShape#model/concatenate_2/concat:output:0*
T0*
_output_shapes
:v
,model/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ī
&model/conv2d_transpose_1/strided_sliceStridedSlice'model/conv2d_transpose_1/Shape:output:05model/conv2d_transpose_1/strided_slice/stack:output:07model/conv2d_transpose_1/strided_slice/stack_1:output:07model/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
 model/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value
B :c
 model/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :c
 model/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :
model/conv2d_transpose_1/stackPack/model/conv2d_transpose_1/strided_slice:output:0)model/conv2d_transpose_1/stack/1:output:0)model/conv2d_transpose_1/stack/2:output:0)model/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:x
.model/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0model/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ö
(model/conv2d_transpose_1/strided_slice_1StridedSlice'model/conv2d_transpose_1/stack:output:07model/conv2d_transpose_1/strided_slice_1/stack:output:09model/conv2d_transpose_1/strided_slice_1/stack_1:output:09model/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÄ
8model/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpAmodel_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype0¶
)model/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput'model/conv2d_transpose_1/stack:output:0@model/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#model/concatenate_2/concat:output:0*
T0*2
_output_shapes 
:’’’’’’’’’*
paddingVALID*
strides
„
/model/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp8model_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
 model/conv2d_transpose_1/BiasAddBiasAdd2model/conv2d_transpose_1/conv2d_transpose:output:07model/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’
model/conv2d_transpose_1/EluElu)model/conv2d_transpose_1/BiasAdd:output:0*
T0*2
_output_shapes 
:’’’’’’’’’
*model/batch_normalization_6/ReadVariableOpReadVariableOp3model_batch_normalization_6_readvariableop_resource*
_output_shapes	
:*
dtype0
,model/batch_normalization_6/ReadVariableOp_1ReadVariableOp5model_batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:*
dtype0½
;model/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Į
=model/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0ķ
,model/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3*model/conv2d_transpose_1/Elu:activations:02model/batch_normalization_6/ReadVariableOp:value:04model/batch_normalization_6/ReadVariableOp_1:value:0Cmodel/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
is_training( a
model/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ź
model/concatenate_3/concatConcatV2 model/conv2d_1/Elu:activations:00model/batch_normalization_6/FusedBatchNormV3:y:0(model/concatenate_3/concat/axis:output:0*
N*
T0*2
_output_shapes 
:’’’’’’’’’q
model/conv2d_transpose_2/ShapeShape#model/concatenate_3/concat:output:0*
T0*
_output_shapes
:v
,model/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ī
&model/conv2d_transpose_2/strided_sliceStridedSlice'model/conv2d_transpose_2/Shape:output:05model/conv2d_transpose_2/strided_slice/stack:output:07model/conv2d_transpose_2/strided_slice/stack_1:output:07model/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
 model/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value
B :c
 model/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value
B :c
 model/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value
B :
model/conv2d_transpose_2/stackPack/model/conv2d_transpose_2/strided_slice:output:0)model/conv2d_transpose_2/stack/1:output:0)model/conv2d_transpose_2/stack/2:output:0)model/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:x
.model/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0model/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ö
(model/conv2d_transpose_2/strided_slice_1StridedSlice'model/conv2d_transpose_2/stack:output:07model/conv2d_transpose_2/strided_slice_1/stack:output:09model/conv2d_transpose_2/strided_slice_1/stack_1:output:09model/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÄ
8model/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpAmodel_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype0¶
)model/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput'model/conv2d_transpose_2/stack:output:0@model/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0#model/concatenate_3/concat:output:0*
T0*2
_output_shapes 
:’’’’’’’’’*
paddingVALID*
strides
„
/model/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp8model_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
 model/conv2d_transpose_2/BiasAddBiasAdd2model/conv2d_transpose_2/conv2d_transpose:output:07model/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’
model/conv2d_transpose_2/EluElu)model/conv2d_transpose_2/BiasAdd:output:0*
T0*2
_output_shapes 
:’’’’’’’’’
*model/batch_normalization_7/ReadVariableOpReadVariableOp3model_batch_normalization_7_readvariableop_resource*
_output_shapes	
:*
dtype0
,model/batch_normalization_7/ReadVariableOp_1ReadVariableOp5model_batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:*
dtype0½
;model/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Į
=model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0ķ
,model/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3*model/conv2d_transpose_2/Elu:activations:02model/batch_normalization_7/ReadVariableOp:value:04model/batch_normalization_7/ReadVariableOp_1:value:0Cmodel/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
is_training( a
model/concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :č
model/concatenate_4/concatConcatV2model/conv2d/Elu:activations:00model/batch_normalization_7/FusedBatchNormV3:y:0(model/concatenate_4/concat/axis:output:0*
N*
T0*2
_output_shapes 
:’’’’’’’’’Ąq
model/conv2d_transpose_3/ShapeShape#model/concatenate_4/concat:output:0*
T0*
_output_shapes
:v
,model/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ī
&model/conv2d_transpose_3/strided_sliceStridedSlice'model/conv2d_transpose_3/Shape:output:05model/conv2d_transpose_3/strided_slice/stack:output:07model/conv2d_transpose_3/strided_slice/stack_1:output:07model/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
 model/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value
B :c
 model/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value
B :b
 model/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@
model/conv2d_transpose_3/stackPack/model/conv2d_transpose_3/strided_slice:output:0)model/conv2d_transpose_3/stack/1:output:0)model/conv2d_transpose_3/stack/2:output:0)model/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:x
.model/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0model/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ö
(model/conv2d_transpose_3/strided_slice_1StridedSlice'model/conv2d_transpose_3/stack:output:07model/conv2d_transpose_3/strided_slice_1/stack:output:09model/conv2d_transpose_3/strided_slice_1/stack_1:output:09model/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskĆ
8model/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpAmodel_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@Ą*
dtype0µ
)model/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput'model/conv2d_transpose_3/stack:output:0@model/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0#model/concatenate_4/concat:output:0*
T0*1
_output_shapes
:’’’’’’’’’@*
paddingVALID*
strides
¤
/model/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp8model_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ō
 model/conv2d_transpose_3/BiasAddBiasAdd2model/conv2d_transpose_3/conv2d_transpose:output:07model/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’@
model/conv2d_transpose_3/EluElu)model/conv2d_transpose_3/BiasAdd:output:0*
T0*1
_output_shapes
:’’’’’’’’’@
*model/batch_normalization_8/ReadVariableOpReadVariableOp3model_batch_normalization_8_readvariableop_resource*
_output_shapes
:@*
dtype0
,model/batch_normalization_8/ReadVariableOp_1ReadVariableOp5model_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:@*
dtype0¼
;model/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ą
=model/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0č
,model/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3*model/conv2d_transpose_3/Elu:activations:02model/batch_normalization_8/ReadVariableOp:value:04model/batch_normalization_8/ReadVariableOp_1:value:0Cmodel/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
is_training( a
model/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ź
model/concatenate_5/concatConcatV2!model/concatenate/concat:output:00model/batch_normalization_8/FusedBatchNormV3:y:0(model/concatenate_5/concat/axis:output:0*
N*
T0*1
_output_shapes
:’’’’’’’’’Dq
model/conv2d_transpose_4/ShapeShape#model/concatenate_5/concat:output:0*
T0*
_output_shapes
:v
,model/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ī
&model/conv2d_transpose_4/strided_sliceStridedSlice'model/conv2d_transpose_4/Shape:output:05model/conv2d_transpose_4/strided_slice/stack:output:07model/conv2d_transpose_4/strided_slice/stack_1:output:07model/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
 model/conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value
B :c
 model/conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value
B :b
 model/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :
model/conv2d_transpose_4/stackPack/model/conv2d_transpose_4/strided_slice:output:0)model/conv2d_transpose_4/stack/1:output:0)model/conv2d_transpose_4/stack/2:output:0)model/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:x
.model/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0model/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ö
(model/conv2d_transpose_4/strided_slice_1StridedSlice'model/conv2d_transpose_4/stack:output:07model/conv2d_transpose_4/strided_slice_1/stack:output:09model/conv2d_transpose_4/strided_slice_1/stack_1:output:09model/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskĀ
8model/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpAmodel_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:D*
dtype0µ
)model/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput'model/conv2d_transpose_4/stack:output:0@model/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0#model/concatenate_5/concat:output:0*
T0*1
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides
¤
/model/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp8model_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ō
 model/conv2d_transpose_4/BiasAddBiasAdd2model/conv2d_transpose_4/conv2d_transpose:output:07model/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’}
$model/cropping2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"              
&model/cropping2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"    ų’’’ų’’’    
&model/cropping2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Õ
model/cropping2d/strided_sliceStridedSlice)model/conv2d_transpose_4/BiasAdd:output:0-model/cropping2d/strided_slice/stack:output:0/model/cropping2d/strided_slice/stack_1:output:0/model/cropping2d/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:’’’’’’’’’*

begin_mask	*
end_mask	
IdentityIdentity'model/cropping2d/strided_slice:output:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’½
NoOpNoOp:^model/batch_normalization/FusedBatchNormV3/ReadVariableOp<^model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1)^model/batch_normalization/ReadVariableOp+^model/batch_normalization/ReadVariableOp_1<^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_1/ReadVariableOp-^model/batch_normalization_1/ReadVariableOp_1<^model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_2/ReadVariableOp-^model/batch_normalization_2/ReadVariableOp_1<^model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_3/ReadVariableOp-^model/batch_normalization_3/ReadVariableOp_1<^model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_4/ReadVariableOp-^model/batch_normalization_4/ReadVariableOp_1<^model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_5/ReadVariableOp-^model/batch_normalization_5/ReadVariableOp_1<^model/batch_normalization_6/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_6/ReadVariableOp-^model/batch_normalization_6/ReadVariableOp_1<^model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_7/ReadVariableOp-^model/batch_normalization_7/ReadVariableOp_1<^model/batch_normalization_8/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_8/ReadVariableOp-^model/batch_normalization_8/ReadVariableOp_1$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp&^model/conv2d_3/BiasAdd/ReadVariableOp%^model/conv2d_3/Conv2D/ReadVariableOp.^model/conv2d_transpose/BiasAdd/ReadVariableOp7^model/conv2d_transpose/conv2d_transpose/ReadVariableOp0^model/conv2d_transpose_1/BiasAdd/ReadVariableOp9^model/conv2d_transpose_1/conv2d_transpose/ReadVariableOp0^model/conv2d_transpose_2/BiasAdd/ReadVariableOp9^model/conv2d_transpose_2/conv2d_transpose/ReadVariableOp0^model/conv2d_transpose_3/BiasAdd/ReadVariableOp9^model/conv2d_transpose_3/conv2d_transpose/ReadVariableOp0^model/conv2d_transpose_4/BiasAdd/ReadVariableOp9^model/conv2d_transpose_4/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*å
_input_shapesÓ
Š:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2v
9model/batch_normalization/FusedBatchNormV3/ReadVariableOp9model/batch_normalization/FusedBatchNormV3/ReadVariableOp2z
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_12T
(model/batch_normalization/ReadVariableOp(model/batch_normalization/ReadVariableOp2X
*model/batch_normalization/ReadVariableOp_1*model/batch_normalization/ReadVariableOp_12z
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_1/ReadVariableOp*model/batch_normalization_1/ReadVariableOp2\
,model/batch_normalization_1/ReadVariableOp_1,model/batch_normalization_1/ReadVariableOp_12z
;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_2/ReadVariableOp*model/batch_normalization_2/ReadVariableOp2\
,model/batch_normalization_2/ReadVariableOp_1,model/batch_normalization_2/ReadVariableOp_12z
;model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_3/ReadVariableOp*model/batch_normalization_3/ReadVariableOp2\
,model/batch_normalization_3/ReadVariableOp_1,model/batch_normalization_3/ReadVariableOp_12z
;model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_4/ReadVariableOp*model/batch_normalization_4/ReadVariableOp2\
,model/batch_normalization_4/ReadVariableOp_1,model/batch_normalization_4/ReadVariableOp_12z
;model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_5/ReadVariableOp*model/batch_normalization_5/ReadVariableOp2\
,model/batch_normalization_5/ReadVariableOp_1,model/batch_normalization_5/ReadVariableOp_12z
;model/batch_normalization_6/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_6/ReadVariableOp*model/batch_normalization_6/ReadVariableOp2\
,model/batch_normalization_6/ReadVariableOp_1,model/batch_normalization_6/ReadVariableOp_12z
;model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_7/ReadVariableOp*model/batch_normalization_7/ReadVariableOp2\
,model/batch_normalization_7/ReadVariableOp_1,model/batch_normalization_7/ReadVariableOp_12z
;model/batch_normalization_8/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_8/ReadVariableOp*model/batch_normalization_8/ReadVariableOp2\
,model/batch_normalization_8/ReadVariableOp_1,model/batch_normalization_8/ReadVariableOp_12J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2N
%model/conv2d_1/BiasAdd/ReadVariableOp%model/conv2d_1/BiasAdd/ReadVariableOp2L
$model/conv2d_1/Conv2D/ReadVariableOp$model/conv2d_1/Conv2D/ReadVariableOp2N
%model/conv2d_2/BiasAdd/ReadVariableOp%model/conv2d_2/BiasAdd/ReadVariableOp2L
$model/conv2d_2/Conv2D/ReadVariableOp$model/conv2d_2/Conv2D/ReadVariableOp2N
%model/conv2d_3/BiasAdd/ReadVariableOp%model/conv2d_3/BiasAdd/ReadVariableOp2L
$model/conv2d_3/Conv2D/ReadVariableOp$model/conv2d_3/Conv2D/ReadVariableOp2^
-model/conv2d_transpose/BiasAdd/ReadVariableOp-model/conv2d_transpose/BiasAdd/ReadVariableOp2p
6model/conv2d_transpose/conv2d_transpose/ReadVariableOp6model/conv2d_transpose/conv2d_transpose/ReadVariableOp2b
/model/conv2d_transpose_1/BiasAdd/ReadVariableOp/model/conv2d_transpose_1/BiasAdd/ReadVariableOp2t
8model/conv2d_transpose_1/conv2d_transpose/ReadVariableOp8model/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2b
/model/conv2d_transpose_2/BiasAdd/ReadVariableOp/model/conv2d_transpose_2/BiasAdd/ReadVariableOp2t
8model/conv2d_transpose_2/conv2d_transpose/ReadVariableOp8model/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2b
/model/conv2d_transpose_3/BiasAdd/ReadVariableOp/model/conv2d_transpose_3/BiasAdd/ReadVariableOp2t
8model/conv2d_transpose_3/conv2d_transpose/ReadVariableOp8model/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2b
/model/conv2d_transpose_4/BiasAdd/ReadVariableOp/model/conv2d_transpose_4/BiasAdd/ReadVariableOp2t
8model/conv2d_transpose_4/conv2d_transpose/ReadVariableOp8model/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:T P
-
_output_shapes
:’’’’’’’’’

_user_specified_nameu_vel:TP
-
_output_shapes
:’’’’’’’’’

_user_specified_namev_vel:TP
-
_output_shapes
:’’’’’’’’’

_user_specified_namew_vel:UQ
-
_output_shapes
:’’’’’’’’’
 
_user_specified_namepr0.71
Ó¦

B__inference_model_layer_call_and_return_conditional_losses_1287894	
u_vel	
v_vel	
w_vel

pr0_71)
batch_normalization_1287761:)
batch_normalization_1287763:)
batch_normalization_1287765:)
batch_normalization_1287767:(
conv2d_1287770:@
conv2d_1287772:@+
batch_normalization_1_1287775:@+
batch_normalization_1_1287777:@+
batch_normalization_1_1287779:@+
batch_normalization_1_1287781:@+
conv2d_1_1287784:@
conv2d_1_1287786:	,
batch_normalization_2_1287789:	,
batch_normalization_2_1287791:	,
batch_normalization_2_1287793:	,
batch_normalization_2_1287795:	,
conv2d_2_1287798:
conv2d_2_1287800:	,
batch_normalization_3_1287803:	,
batch_normalization_3_1287805:	,
batch_normalization_3_1287807:	,
batch_normalization_3_1287809:	,
conv2d_3_1287812:
conv2d_3_1287814:	,
batch_normalization_4_1287817:	,
batch_normalization_4_1287819:	,
batch_normalization_4_1287821:	,
batch_normalization_4_1287823:	4
conv2d_transpose_1287827:'
conv2d_transpose_1287829:	,
batch_normalization_5_1287832:	,
batch_normalization_5_1287834:	,
batch_normalization_5_1287836:	,
batch_normalization_5_1287838:	6
conv2d_transpose_1_1287842:)
conv2d_transpose_1_1287844:	,
batch_normalization_6_1287847:	,
batch_normalization_6_1287849:	,
batch_normalization_6_1287851:	,
batch_normalization_6_1287853:	6
conv2d_transpose_2_1287857:)
conv2d_transpose_2_1287859:	,
batch_normalization_7_1287862:	,
batch_normalization_7_1287864:	,
batch_normalization_7_1287866:	,
batch_normalization_7_1287868:	5
conv2d_transpose_3_1287872:@Ą(
conv2d_transpose_3_1287874:@+
batch_normalization_8_1287877:@+
batch_normalization_8_1287879:@+
batch_normalization_8_1287881:@+
batch_normalization_8_1287883:@4
conv2d_transpose_4_1287887:D(
conv2d_transpose_4_1287889:
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢-batch_normalization_6/StatefulPartitionedCall¢-batch_normalization_7/StatefulPartitionedCall¢-batch_normalization_8/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¢(conv2d_transpose/StatefulPartitionedCall¢*conv2d_transpose_1/StatefulPartitionedCall¢*conv2d_transpose_2/StatefulPartitionedCall¢*conv2d_transpose_3/StatefulPartitionedCall¢*conv2d_transpose_4/StatefulPartitionedCallŚ
$periodic_padding2d_3/PartitionedCallPartitionedCallpr0_71*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_periodic_padding2d_3_layer_call_and_return_conditional_losses_1285748Ł
$periodic_padding2d_2/PartitionedCallPartitionedCallw_vel*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_periodic_padding2d_2_layer_call_and_return_conditional_losses_1285774Ł
$periodic_padding2d_1/PartitionedCallPartitionedCallv_vel*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_periodic_padding2d_1_layer_call_and_return_conditional_losses_1285800Õ
"periodic_padding2d/PartitionedCallPartitionedCallu_vel*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_periodic_padding2d_layer_call_and_return_conditional_losses_1285826é
reshape/PartitionedCallPartitionedCall+periodic_padding2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_1285842ļ
reshape_1/PartitionedCallPartitionedCall-periodic_padding2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_reshape_1_layer_call_and_return_conditional_losses_1285858ļ
reshape_2/PartitionedCallPartitionedCall-periodic_padding2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_1285874ļ
reshape_3/PartitionedCallPartitionedCall-periodic_padding2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_reshape_3_layer_call_and_return_conditional_losses_1285890Õ
concatenate/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0"reshape_1/PartitionedCall:output:0"reshape_2/PartitionedCall:output:0"reshape_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_1285901
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0batch_normalization_1287761batch_normalization_1287763batch_normalization_1287765batch_normalization_1287767*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1287034¦
conv2d/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_1287770conv2d_1287772*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_1285941
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_1_1287775batch_normalization_1_1287777batch_normalization_1_1287779batch_normalization_1_1287781*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1286980±
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_1_1287784conv2d_1_1287786*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1285985
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_2_1287789batch_normalization_2_1287791batch_normalization_2_1287793batch_normalization_2_1287795*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1286926±
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_2_1287798conv2d_2_1287800*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1286029
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_3_1287803batch_normalization_3_1287805batch_normalization_3_1287807batch_normalization_3_1287809*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1286872±
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_3_1287812conv2d_3_1287814*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1286073
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_4_1287817batch_normalization_4_1287819batch_normalization_4_1287821batch_normalization_4_1287823*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1286818­
concatenate_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:06batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1286113Į
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0conv2d_transpose_1287827conv2d_transpose_1287829*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1286138 
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0batch_normalization_5_1287832batch_normalization_5_1287834batch_normalization_5_1287836batch_normalization_5_1287838*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1286758­
concatenate_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:06batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_1286178É
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0conv2d_transpose_1_1287842conv2d_transpose_1_1287844*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1286203¢
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0batch_normalization_6_1287847batch_normalization_6_1287849batch_normalization_6_1287851batch_normalization_6_1287853*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1286698­
concatenate_3/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:06batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_3_layer_call_and_return_conditional_losses_1286243É
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0conv2d_transpose_2_1287857conv2d_transpose_2_1287859*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1286268¢
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0batch_normalization_7_1287862batch_normalization_7_1287864batch_normalization_7_1287866batch_normalization_7_1287868*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1286638«
concatenate_4/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:06batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’Ą* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_4_layer_call_and_return_conditional_losses_1286308Č
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv2d_transpose_3_1287872conv2d_transpose_3_1287874*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1286333”
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0batch_normalization_8_1287877batch_normalization_8_1287879batch_normalization_8_1287881batch_normalization_8_1287883*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1286578§
concatenate_5/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:06batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’D* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_5_layer_call_and_return_conditional_losses_1286373Č
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0conv2d_transpose_4_1287887conv2d_transpose_4_1287889*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1286397÷
cropping2d/PartitionedCallPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_cropping2d_layer_call_and_return_conditional_losses_1286410|
IdentityIdentity#cropping2d/PartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’Ż
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*å
_input_shapesÓ
Š:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall:T P
-
_output_shapes
:’’’’’’’’’

_user_specified_nameu_vel:TP
-
_output_shapes
:’’’’’’’’’

_user_specified_namev_vel:TP
-
_output_shapes
:’’’’’’’’’

_user_specified_namew_vel:UQ
-
_output_shapes
:’’’’’’’’’
 
_user_specified_namepr0.71

k
O__inference_periodic_padding2d_layer_call_and_return_conditional_losses_1289071

inputs
identityh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"        ų’’’j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         é
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskj
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ń
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :£
concatConcatV2strided_slice:output:0inputsstrided_slice_1:output:0concat/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’j
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ų’’’    l
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ś
strided_slice_2StridedSliceconcat:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskj
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ś
strided_slice_3StridedSliceconcat:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :²
concat_1ConcatV2strided_slice_2:output:0concat:output:0strided_slice_3:output:0concat_1/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’_
IdentityIdentityconcat_1:output:0*
T0*-
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:’’’’’’’’’:U Q
-
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ż
Ö
7__inference_batch_normalization_2_layer_call_fn_1289578

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1286008z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
«
Å
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1286872

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ė
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
č¦
 
B__inference_model_layer_call_and_return_conditional_losses_1287371

inputs
inputs_1
inputs_2
inputs_3)
batch_normalization_1287238:)
batch_normalization_1287240:)
batch_normalization_1287242:)
batch_normalization_1287244:(
conv2d_1287247:@
conv2d_1287249:@+
batch_normalization_1_1287252:@+
batch_normalization_1_1287254:@+
batch_normalization_1_1287256:@+
batch_normalization_1_1287258:@+
conv2d_1_1287261:@
conv2d_1_1287263:	,
batch_normalization_2_1287266:	,
batch_normalization_2_1287268:	,
batch_normalization_2_1287270:	,
batch_normalization_2_1287272:	,
conv2d_2_1287275:
conv2d_2_1287277:	,
batch_normalization_3_1287280:	,
batch_normalization_3_1287282:	,
batch_normalization_3_1287284:	,
batch_normalization_3_1287286:	,
conv2d_3_1287289:
conv2d_3_1287291:	,
batch_normalization_4_1287294:	,
batch_normalization_4_1287296:	,
batch_normalization_4_1287298:	,
batch_normalization_4_1287300:	4
conv2d_transpose_1287304:'
conv2d_transpose_1287306:	,
batch_normalization_5_1287309:	,
batch_normalization_5_1287311:	,
batch_normalization_5_1287313:	,
batch_normalization_5_1287315:	6
conv2d_transpose_1_1287319:)
conv2d_transpose_1_1287321:	,
batch_normalization_6_1287324:	,
batch_normalization_6_1287326:	,
batch_normalization_6_1287328:	,
batch_normalization_6_1287330:	6
conv2d_transpose_2_1287334:)
conv2d_transpose_2_1287336:	,
batch_normalization_7_1287339:	,
batch_normalization_7_1287341:	,
batch_normalization_7_1287343:	,
batch_normalization_7_1287345:	5
conv2d_transpose_3_1287349:@Ą(
conv2d_transpose_3_1287351:@+
batch_normalization_8_1287354:@+
batch_normalization_8_1287356:@+
batch_normalization_8_1287358:@+
batch_normalization_8_1287360:@4
conv2d_transpose_4_1287364:D(
conv2d_transpose_4_1287366:
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢-batch_normalization_6/StatefulPartitionedCall¢-batch_normalization_7/StatefulPartitionedCall¢-batch_normalization_8/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¢(conv2d_transpose/StatefulPartitionedCall¢*conv2d_transpose_1/StatefulPartitionedCall¢*conv2d_transpose_2/StatefulPartitionedCall¢*conv2d_transpose_3/StatefulPartitionedCall¢*conv2d_transpose_4/StatefulPartitionedCallÜ
$periodic_padding2d_3/PartitionedCallPartitionedCallinputs_3*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_periodic_padding2d_3_layer_call_and_return_conditional_losses_1285748Ü
$periodic_padding2d_2/PartitionedCallPartitionedCallinputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_periodic_padding2d_2_layer_call_and_return_conditional_losses_1285774Ü
$periodic_padding2d_1/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_periodic_padding2d_1_layer_call_and_return_conditional_losses_1285800Ö
"periodic_padding2d/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_periodic_padding2d_layer_call_and_return_conditional_losses_1285826é
reshape/PartitionedCallPartitionedCall+periodic_padding2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_1285842ļ
reshape_1/PartitionedCallPartitionedCall-periodic_padding2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_reshape_1_layer_call_and_return_conditional_losses_1285858ļ
reshape_2/PartitionedCallPartitionedCall-periodic_padding2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_1285874ļ
reshape_3/PartitionedCallPartitionedCall-periodic_padding2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_reshape_3_layer_call_and_return_conditional_losses_1285890Õ
concatenate/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0"reshape_1/PartitionedCall:output:0"reshape_2/PartitionedCall:output:0"reshape_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_1285901
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0batch_normalization_1287238batch_normalization_1287240batch_normalization_1287242batch_normalization_1287244*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1287034¦
conv2d/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_1287247conv2d_1287249*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_1285941
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_1_1287252batch_normalization_1_1287254batch_normalization_1_1287256batch_normalization_1_1287258*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1286980±
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_1_1287261conv2d_1_1287263*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1285985
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_2_1287266batch_normalization_2_1287268batch_normalization_2_1287270batch_normalization_2_1287272*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1286926±
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_2_1287275conv2d_2_1287277*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1286029
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_3_1287280batch_normalization_3_1287282batch_normalization_3_1287284batch_normalization_3_1287286*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1286872±
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_3_1287289conv2d_3_1287291*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1286073
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_4_1287294batch_normalization_4_1287296batch_normalization_4_1287298batch_normalization_4_1287300*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1286818­
concatenate_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:06batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1286113Į
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0conv2d_transpose_1287304conv2d_transpose_1287306*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1286138 
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0batch_normalization_5_1287309batch_normalization_5_1287311batch_normalization_5_1287313batch_normalization_5_1287315*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1286758­
concatenate_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:06batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_1286178É
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0conv2d_transpose_1_1287319conv2d_transpose_1_1287321*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1286203¢
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0batch_normalization_6_1287324batch_normalization_6_1287326batch_normalization_6_1287328batch_normalization_6_1287330*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1286698­
concatenate_3/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:06batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_3_layer_call_and_return_conditional_losses_1286243É
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0conv2d_transpose_2_1287334conv2d_transpose_2_1287336*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1286268¢
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0batch_normalization_7_1287339batch_normalization_7_1287341batch_normalization_7_1287343batch_normalization_7_1287345*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1286638«
concatenate_4/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:06batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’Ą* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_4_layer_call_and_return_conditional_losses_1286308Č
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv2d_transpose_3_1287349conv2d_transpose_3_1287351*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1286333”
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0batch_normalization_8_1287354batch_normalization_8_1287356batch_normalization_8_1287358batch_normalization_8_1287360*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1286578§
concatenate_5/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:06batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’D* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_5_layer_call_and_return_conditional_losses_1286373Č
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0conv2d_transpose_4_1287364conv2d_transpose_4_1287366*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1286397÷
cropping2d/PartitionedCallPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_cropping2d_layer_call_and_return_conditional_losses_1286410|
IdentityIdentity#cropping2d/PartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’Ż
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*å
_input_shapesÓ
Š:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall:U Q
-
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:UQ
-
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:UQ
-
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:UQ
-
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ž
t
J__inference_concatenate_2_layer_call_and_return_conditional_losses_1286178

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*2
_output_shapes 
:’’’’’’’’’b
IdentityIdentityconcat:output:0*
T0*2
_output_shapes 
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:’’’’’’’’’:’’’’’’’’’:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs:ZV
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
Ł
æ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1284929

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ķ

R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1285606

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Č
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 
_user_specified_nameinputs
«
Å
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1286926

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ė
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
Ņ
R
6__inference_periodic_padding2d_3_layer_call_fn_1289134

inputs
identityÅ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_periodic_padding2d_3_layer_call_and_return_conditional_losses_1285748f
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:’’’’’’’’’:U Q
-
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

m
Q__inference_periodic_padding2d_2_layer_call_and_return_conditional_losses_1285774

inputs
identityh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"        ų’’’j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         é
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskj
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ń
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :£
concatConcatV2strided_slice:output:0inputsstrided_slice_1:output:0concat/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’j
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ų’’’    l
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ś
strided_slice_2StridedSliceconcat:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskj
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ś
strided_slice_3StridedSliceconcat:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*

begin_mask*
end_maskO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :²
concat_1ConcatV2strided_slice_2:output:0concat:output:0strided_slice_3:output:0concat_1/axis:output:0*
N*
T0*-
_output_shapes
:’’’’’’’’’_
IdentityIdentityconcat_1:output:0*
T0*-
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:’’’’’’’’’:U Q
-
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ū
Ö
7__inference_batch_normalization_6_layer_call_fn_1290313

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1286698z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs
Ä

O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1286333

inputsC
(conv2d_transpose_readvariableop_resource:@Ą-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskJ
stack/1Const*
_output_shapes
: *
dtype0*
value
B :J
stack/2Const*
_output_shapes
: *
dtype0*
value
B :I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@Ą*
dtype0Ķ
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:’’’’’’’’’@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’@X
EluEluBiasAdd:output:0*
T0*1
_output_shapes
:’’’’’’’’’@j
IdentityIdentityElu:activations:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’@
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":’’’’’’’’’Ą: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Z V
2
_output_shapes 
:’’’’’’’’’Ą
 
_user_specified_nameinputs
	
Š
5__inference_batch_normalization_layer_call_fn_1289264

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1284898
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
ė
Å
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1289771

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ū
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,’’’’’’’’’’’’’’’’’’’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs

æ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1287034

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ę
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:’’’’’’’’’:::::*
epsilon%o:*
exponential_avg_factor%
×#<°
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0ŗ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0m
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’Ō
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ī
H
,__inference_cropping2d_layer_call_fn_1290920

inputs
identityæ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_cropping2d_layer_call_and_return_conditional_losses_1286410j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:’’’’’’’’’:Y U
1
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ū
Ö
7__inference_batch_normalization_7_layer_call_fn_1290530

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1286638z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:’’’’’’’’’
 
_user_specified_nameinputs


'__inference_model_layer_call_fn_1288250
inputs_0
inputs_1
inputs_2
inputs_3
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@$
	unknown_9:@

unknown_10:	

unknown_11:	

unknown_12:	

unknown_13:	

unknown_14:	&

unknown_15:

unknown_16:	

unknown_17:	

unknown_18:	

unknown_19:	

unknown_20:	&

unknown_21:

unknown_22:	

unknown_23:	

unknown_24:	

unknown_25:	

unknown_26:	&

unknown_27:

unknown_28:	

unknown_29:	

unknown_30:	

unknown_31:	

unknown_32:	&

unknown_33:

unknown_34:	

unknown_35:	

unknown_36:	

unknown_37:	

unknown_38:	&

unknown_39:

unknown_40:	

unknown_41:	

unknown_42:	

unknown_43:	

unknown_44:	%

unknown_45:@Ą

unknown_46:@

unknown_47:@

unknown_48:@

unknown_49:@

unknown_50:@$

unknown_51:D

unknown_52:
identity¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*E
Tin>
<2:*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’*F
_read_only_resource_inputs(
&$	
 !"#&'(),-./234589*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1287371y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*å
_input_shapesÓ
Š:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
-
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/0:WS
-
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/1:WS
-
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/2:WS
-
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/3


E__inference_conv2d_1_layer_call_and_return_conditional_losses_1289539

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:’’’’’’’’’Y
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:’’’’’’’’’k
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:’’’’’’’’’w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:’’’’’’’’’@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs
É
c
G__inference_cropping2d_layer_call_and_return_conditional_losses_1290928

inputs
identityl
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"              n
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"    ų’’’ų’’’    n
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*

begin_mask	*
end_mask	
IdentityIdentitystrided_slice:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:r n
J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
ś#

O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1285351

inputsD
(conv2d_transpose_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B :F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype0Ž
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’i
EluEluBiasAdd:output:0*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’{
IdentityIdentityElu:activations:0^NoOp*
T0*B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,’’’’’’’’’’’’’’’’’’’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
ą

'__inference_model_layer_call_fn_1287598	
u_vel	
v_vel	
w_vel

pr0_71
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@$
	unknown_9:@

unknown_10:	

unknown_11:	

unknown_12:	

unknown_13:	

unknown_14:	&

unknown_15:

unknown_16:	

unknown_17:	

unknown_18:	

unknown_19:	

unknown_20:	&

unknown_21:

unknown_22:	

unknown_23:	

unknown_24:	

unknown_25:	

unknown_26:	&

unknown_27:

unknown_28:	

unknown_29:	

unknown_30:	

unknown_31:	

unknown_32:	&

unknown_33:

unknown_34:	

unknown_35:	

unknown_36:	

unknown_37:	

unknown_38:	&

unknown_39:

unknown_40:	

unknown_41:	

unknown_42:	

unknown_43:	

unknown_44:	%

unknown_45:@Ą

unknown_46:@

unknown_47:@

unknown_48:@

unknown_49:@

unknown_50:@$

unknown_51:D

unknown_52:
identity¢StatefulPartitionedCall¹
StatefulPartitionedCallStatefulPartitionedCallu_velv_velw_velpr0_71unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*E
Tin>
<2:*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’*F
_read_only_resource_inputs(
&$	
 !"#&'(),-./234589*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1287371y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*å
_input_shapesÓ
Š:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
-
_output_shapes
:’’’’’’’’’

_user_specified_nameu_vel:TP
-
_output_shapes
:’’’’’’’’’

_user_specified_namev_vel:TP
-
_output_shapes
:’’’’’’’’’

_user_specified_namew_vel:UQ
-
_output_shapes
:’’’’’’’’’
 
_user_specified_namepr0.71


R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1289501

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:’’’’’’’’’@:@:@:@:@:*
epsilon%o:*
is_training( m
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:’’’’’’’’’@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:’’’’’’’’’@
 
_user_specified_nameinputs

©
4__inference_conv2d_transpose_4_layer_call_fn_1290850

inputs!
unknown:D
	unknown_0:
identity¢StatefulPartitionedCallń
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1286397y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:’’’’’’’’’D: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:’’’’’’’’’D
 
_user_specified_nameinputs"L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ų
serving_defaultä
?
pr0.715
serving_default_pr0.71:0’’’’’’’’’
=
u_vel4
serving_default_u_vel:0’’’’’’’’’
=
v_vel4
serving_default_v_vel:0’’’’’’’’’
=
w_vel4
serving_default_w_vel:0’’’’’’’’’H

cropping2d:
StatefulPartitionedCall:0’’’’’’’’’tensorflow/serving/predict:»
¾	
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer_with_weights-0
layer-13
layer_with_weights-1
layer-14
layer_with_weights-2
layer-15
layer_with_weights-3
layer-16
layer_with_weights-4
layer-17
layer_with_weights-5
layer-18
layer_with_weights-6
layer-19
layer_with_weights-7
layer-20
layer_with_weights-8
layer-21
layer-22
layer_with_weights-9
layer-23
layer_with_weights-10
layer-24
layer-25
layer_with_weights-11
layer-26
layer_with_weights-12
layer-27
layer-28
layer_with_weights-13
layer-29
layer_with_weights-14
layer-30
 layer-31
!layer_with_weights-15
!layer-32
"layer_with_weights-16
"layer-33
#layer-34
$layer_with_weights-17
$layer-35
%layer-36
&	optimizer
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+
signatures
ė__call__
+ģ&call_and_return_all_conditional_losses
ķ_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
§
,	variables
-trainable_variables
.regularization_losses
/	keras_api
ī__call__
+ļ&call_and_return_all_conditional_losses"
_tf_keras_layer
§
0	variables
1trainable_variables
2regularization_losses
3	keras_api
š__call__
+ń&call_and_return_all_conditional_losses"
_tf_keras_layer
§
4	variables
5trainable_variables
6regularization_losses
7	keras_api
ņ__call__
+ó&call_and_return_all_conditional_losses"
_tf_keras_layer
§
8	variables
9trainable_variables
:regularization_losses
;	keras_api
ō__call__
+õ&call_and_return_all_conditional_losses"
_tf_keras_layer
§
<	variables
=trainable_variables
>regularization_losses
?	keras_api
ö__call__
+÷&call_and_return_all_conditional_losses"
_tf_keras_layer
§
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
ų__call__
+ł&call_and_return_all_conditional_losses"
_tf_keras_layer
§
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
ś__call__
+ū&call_and_return_all_conditional_losses"
_tf_keras_layer
§
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
ü__call__
+ż&call_and_return_all_conditional_losses"
_tf_keras_layer
§
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
ž__call__
+’&call_and_return_all_conditional_losses"
_tf_keras_layer
ģ
Paxis
	Qgamma
Rbeta
Smoving_mean
Tmoving_variance
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
½

Ykernel
Zbias
[	variables
\trainable_variables
]regularization_losses
^	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ģ
_axis
	`gamma
abeta
bmoving_mean
cmoving_variance
d	variables
etrainable_variables
fregularization_losses
g	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
½

hkernel
ibias
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ģ
naxis
	ogamma
pbeta
qmoving_mean
rmoving_variance
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
½

wkernel
xbias
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ņ
}axis
	~gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ć
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ć
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	axis

 gamma
	”beta
¢moving_mean
£moving_variance
¤	variables
„trainable_variables
¦regularization_losses
§	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
Ø	variables
©trainable_variables
Ŗregularization_losses
«	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ć
¬kernel
	­bias
®	variables
Ætrainable_variables
°regularization_losses
±	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	²axis

³gamma
	“beta
µmoving_mean
¶moving_variance
·	variables
øtrainable_variables
¹regularization_losses
ŗ	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
»	variables
¼trainable_variables
½regularization_losses
¾	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ć
ækernel
	Ąbias
Į	variables
Ātrainable_variables
Ćregularization_losses
Ä	keras_api
 __call__
+”&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	Åaxis

Ęgamma
	Ēbeta
Čmoving_mean
Émoving_variance
Ź	variables
Ėtrainable_variables
Ģregularization_losses
Ķ	keras_api
¢__call__
+£&call_and_return_all_conditional_losses"
_tf_keras_layer
«
Ī	variables
Ļtrainable_variables
Šregularization_losses
Ń	keras_api
¤__call__
+„&call_and_return_all_conditional_losses"
_tf_keras_layer
Ć
Ņkernel
	Óbias
Ō	variables
Õtrainable_variables
Öregularization_losses
×	keras_api
¦__call__
+§&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	Ųaxis

Łgamma
	Śbeta
Ūmoving_mean
Ümoving_variance
Ż	variables
Žtrainable_variables
ßregularization_losses
ą	keras_api
Ø__call__
+©&call_and_return_all_conditional_losses"
_tf_keras_layer
«
į	variables
ātrainable_variables
ćregularization_losses
ä	keras_api
Ŗ__call__
+«&call_and_return_all_conditional_losses"
_tf_keras_layer
Ć
åkernel
	ębias
ē	variables
čtrainable_variables
éregularization_losses
ź	keras_api
¬__call__
+­&call_and_return_all_conditional_losses"
_tf_keras_layer
«
ė	variables
ģtrainable_variables
ķregularization_losses
ī	keras_api
®__call__
+Æ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ō
	ļiter
šbeta_1
ńbeta_2

ņdecay
ólearning_rateQm£Rm¤Ym„Zm¦`m§amØhm©imŖom«pm¬wm­xm®~mÆm°	m±	m²	m³	m“	mµ	m¶	 m·	”mø	¬m¹	­mŗ	³m»	“m¼	æm½	Ąm¾	Ęmæ	ĒmĄ	ŅmĮ	ÓmĀ	ŁmĆ	ŚmÄ	åmÅ	ęmĘQvĒRvČYvÉZvŹ`vĖavĢhvĶivĪovĻpvŠwvŃxvŅ~vÓvŌ	vÕ	vÖ	v×	vŲ	vŁ	vŚ	 vŪ	”vÜ	¬vŻ	­vŽ	³vß	“vą	ævį	Ąvā	Ęvć	Ēvä	Ņvå	Óvę	Łvē	Śvč	åvé	ęvź"
	optimizer
č
Q0
R1
S2
T3
Y4
Z5
`6
a7
b8
c9
h10
i11
o12
p13
q14
r15
w16
x17
~18
19
20
21
22
23
24
25
26
27
28
29
 30
”31
¢32
£33
¬34
­35
³36
“37
µ38
¶39
æ40
Ą41
Ę42
Ē43
Č44
É45
Ņ46
Ó47
Ł48
Ś49
Ū50
Ü51
å52
ę53"
trackable_list_wrapper
Ģ
Q0
R1
Y2
Z3
`4
a5
h6
i7
o8
p9
w10
x11
~12
13
14
15
16
17
18
19
 20
”21
¬22
­23
³24
“25
æ26
Ą27
Ę28
Ē29
Ņ30
Ó31
Ł32
Ś33
å34
ę35"
trackable_list_wrapper
 "
trackable_list_wrapper
Ó
ōnon_trainable_variables
õlayers
ömetrics
 ÷layer_regularization_losses
ųlayer_metrics
'	variables
(trainable_variables
)regularization_losses
ė__call__
ķ_default_save_signature
+ģ&call_and_return_all_conditional_losses
'ģ"call_and_return_conditional_losses"
_generic_user_object
-
°serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
łnon_trainable_variables
ślayers
ūmetrics
 ülayer_regularization_losses
żlayer_metrics
,	variables
-trainable_variables
.regularization_losses
ī__call__
+ļ&call_and_return_all_conditional_losses
'ļ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
žnon_trainable_variables
’layers
metrics
 layer_regularization_losses
layer_metrics
0	variables
1trainable_variables
2regularization_losses
š__call__
+ń&call_and_return_all_conditional_losses
'ń"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
4	variables
5trainable_variables
6regularization_losses
ņ__call__
+ó&call_and_return_all_conditional_losses
'ó"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
8	variables
9trainable_variables
:regularization_losses
ō__call__
+õ&call_and_return_all_conditional_losses
'õ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
<	variables
=trainable_variables
>regularization_losses
ö__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
ų__call__
+ł&call_and_return_all_conditional_losses
'ł"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
ś__call__
+ū&call_and_return_all_conditional_losses
'ū"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
ü__call__
+ż&call_and_return_all_conditional_losses
'ż"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
”non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
„layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
ž__call__
+’&call_and_return_all_conditional_losses
'’"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%2batch_normalization/gamma
&:$2batch_normalization/beta
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
<
Q0
R1
S2
T3"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¦non_trainable_variables
§layers
Ømetrics
 ©layer_regularization_losses
Ŗlayer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
':%@2conv2d/kernel
:@2conv2d/bias
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
«non_trainable_variables
¬layers
­metrics
 ®layer_regularization_losses
Ælayer_metrics
[	variables
\trainable_variables
]regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_1/gamma
(:&@2batch_normalization_1/beta
1:/@ (2!batch_normalization_1/moving_mean
5:3@ (2%batch_normalization_1/moving_variance
<
`0
a1
b2
c3"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
°non_trainable_variables
±layers
²metrics
 ³layer_regularization_losses
“layer_metrics
d	variables
etrainable_variables
fregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
*:(@2conv2d_1/kernel
:2conv2d_1/bias
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
µnon_trainable_variables
¶layers
·metrics
 ølayer_regularization_losses
¹layer_metrics
j	variables
ktrainable_variables
lregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_2/gamma
):'2batch_normalization_2/beta
2:0 (2!batch_normalization_2/moving_mean
6:4 (2%batch_normalization_2/moving_variance
<
o0
p1
q2
r3"
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ŗnon_trainable_variables
»layers
¼metrics
 ½layer_regularization_losses
¾layer_metrics
s	variables
ttrainable_variables
uregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_2/kernel
:2conv2d_2/bias
.
w0
x1"
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ænon_trainable_variables
Ąlayers
Įmetrics
 Ālayer_regularization_losses
Ćlayer_metrics
y	variables
ztrainable_variables
{regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_3/gamma
):'2batch_normalization_3/beta
2:0 (2!batch_normalization_3/moving_mean
6:4 (2%batch_normalization_3/moving_variance
>
~0
1
2
3"
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
Änon_trainable_variables
Ålayers
Ęmetrics
 Ēlayer_regularization_losses
Člayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_3/kernel
:2conv2d_3/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
Énon_trainable_variables
Źlayers
Ėmetrics
 Ģlayer_regularization_losses
Ķlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_4/gamma
):'2batch_normalization_4/beta
2:0 (2!batch_normalization_4/moving_mean
6:4 (2%batch_normalization_4/moving_variance
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
Īnon_trainable_variables
Ļlayers
Šmetrics
 Ńlayer_regularization_losses
Ņlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
Ónon_trainable_variables
Ōlayers
Õmetrics
 Ölayer_regularization_losses
×layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
3:12conv2d_transpose/kernel
$:"2conv2d_transpose/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
Ųnon_trainable_variables
Łlayers
Śmetrics
 Ūlayer_regularization_losses
Ülayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_5/gamma
):'2batch_normalization_5/beta
2:0 (2!batch_normalization_5/moving_mean
6:4 (2%batch_normalization_5/moving_variance
@
 0
”1
¢2
£3"
trackable_list_wrapper
0
 0
”1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
Żnon_trainable_variables
Žlayers
ßmetrics
 ąlayer_regularization_losses
įlayer_metrics
¤	variables
„trainable_variables
¦regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
ānon_trainable_variables
ćlayers
ämetrics
 ålayer_regularization_losses
ęlayer_metrics
Ø	variables
©trainable_variables
Ŗregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
5:32conv2d_transpose_1/kernel
&:$2conv2d_transpose_1/bias
0
¬0
­1"
trackable_list_wrapper
0
¬0
­1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
ēnon_trainable_variables
člayers
émetrics
 źlayer_regularization_losses
ėlayer_metrics
®	variables
Ætrainable_variables
°regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_6/gamma
):'2batch_normalization_6/beta
2:0 (2!batch_normalization_6/moving_mean
6:4 (2%batch_normalization_6/moving_variance
@
³0
“1
µ2
¶3"
trackable_list_wrapper
0
³0
“1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
ģnon_trainable_variables
ķlayers
īmetrics
 ļlayer_regularization_losses
šlayer_metrics
·	variables
øtrainable_variables
¹regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
ńnon_trainable_variables
ņlayers
ómetrics
 ōlayer_regularization_losses
õlayer_metrics
»	variables
¼trainable_variables
½regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
5:32conv2d_transpose_2/kernel
&:$2conv2d_transpose_2/bias
0
æ0
Ą1"
trackable_list_wrapper
0
æ0
Ą1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
önon_trainable_variables
÷layers
ųmetrics
 łlayer_regularization_losses
ślayer_metrics
Į	variables
Ātrainable_variables
Ćregularization_losses
 __call__
+”&call_and_return_all_conditional_losses
'”"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_7/gamma
):'2batch_normalization_7/beta
2:0 (2!batch_normalization_7/moving_mean
6:4 (2%batch_normalization_7/moving_variance
@
Ę0
Ē1
Č2
É3"
trackable_list_wrapper
0
Ę0
Ē1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
ūnon_trainable_variables
ülayers
żmetrics
 žlayer_regularization_losses
’layer_metrics
Ź	variables
Ėtrainable_variables
Ģregularization_losses
¢__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ī	variables
Ļtrainable_variables
Šregularization_losses
¤__call__
+„&call_and_return_all_conditional_losses
'„"call_and_return_conditional_losses"
_generic_user_object
4:2@Ą2conv2d_transpose_3/kernel
%:#@2conv2d_transpose_3/bias
0
Ņ0
Ó1"
trackable_list_wrapper
0
Ņ0
Ó1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ō	variables
Õtrainable_variables
Öregularization_losses
¦__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_8/gamma
(:&@2batch_normalization_8/beta
1:/@ (2!batch_normalization_8/moving_mean
5:3@ (2%batch_normalization_8/moving_variance
@
Ł0
Ś1
Ū2
Ü3"
trackable_list_wrapper
0
Ł0
Ś1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ż	variables
Žtrainable_variables
ßregularization_losses
Ø__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
į	variables
ātrainable_variables
ćregularization_losses
Ŗ__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
3:1D2conv2d_transpose_4/kernel
%:#2conv2d_transpose_4/bias
0
å0
ę1"
trackable_list_wrapper
0
å0
ę1"
trackable_list_wrapper
 "
trackable_list_wrapper
ø
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ē	variables
čtrainable_variables
éregularization_losses
¬__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ø
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ė	variables
ģtrainable_variables
ķregularization_losses
®__call__
+Æ&call_and_return_all_conditional_losses
'Æ"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
²
S0
T1
b2
c3
q4
r5
6
7
8
9
¢10
£11
µ12
¶13
Č14
É15
Ū16
Ü17"
trackable_list_wrapper
¾
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
¢0
£1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
µ0
¶1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
Č0
É1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
Ū0
Ü1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

total

 count
”	variables
¢	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
0
 1"
trackable_list_wrapper
.
”	variables"
_generic_user_object
,:*2 Adam/batch_normalization/gamma/m
+:)2Adam/batch_normalization/beta/m
,:*@2Adam/conv2d/kernel/m
:@2Adam/conv2d/bias/m
.:,@2"Adam/batch_normalization_1/gamma/m
-:+@2!Adam/batch_normalization_1/beta/m
/:-@2Adam/conv2d_1/kernel/m
!:2Adam/conv2d_1/bias/m
/:-2"Adam/batch_normalization_2/gamma/m
.:,2!Adam/batch_normalization_2/beta/m
0:.2Adam/conv2d_2/kernel/m
!:2Adam/conv2d_2/bias/m
/:-2"Adam/batch_normalization_3/gamma/m
.:,2!Adam/batch_normalization_3/beta/m
0:.2Adam/conv2d_3/kernel/m
!:2Adam/conv2d_3/bias/m
/:-2"Adam/batch_normalization_4/gamma/m
.:,2!Adam/batch_normalization_4/beta/m
8:62Adam/conv2d_transpose/kernel/m
):'2Adam/conv2d_transpose/bias/m
/:-2"Adam/batch_normalization_5/gamma/m
.:,2!Adam/batch_normalization_5/beta/m
::82 Adam/conv2d_transpose_1/kernel/m
+:)2Adam/conv2d_transpose_1/bias/m
/:-2"Adam/batch_normalization_6/gamma/m
.:,2!Adam/batch_normalization_6/beta/m
::82 Adam/conv2d_transpose_2/kernel/m
+:)2Adam/conv2d_transpose_2/bias/m
/:-2"Adam/batch_normalization_7/gamma/m
.:,2!Adam/batch_normalization_7/beta/m
9:7@Ą2 Adam/conv2d_transpose_3/kernel/m
*:(@2Adam/conv2d_transpose_3/bias/m
.:,@2"Adam/batch_normalization_8/gamma/m
-:+@2!Adam/batch_normalization_8/beta/m
8:6D2 Adam/conv2d_transpose_4/kernel/m
*:(2Adam/conv2d_transpose_4/bias/m
,:*2 Adam/batch_normalization/gamma/v
+:)2Adam/batch_normalization/beta/v
,:*@2Adam/conv2d/kernel/v
:@2Adam/conv2d/bias/v
.:,@2"Adam/batch_normalization_1/gamma/v
-:+@2!Adam/batch_normalization_1/beta/v
/:-@2Adam/conv2d_1/kernel/v
!:2Adam/conv2d_1/bias/v
/:-2"Adam/batch_normalization_2/gamma/v
.:,2!Adam/batch_normalization_2/beta/v
0:.2Adam/conv2d_2/kernel/v
!:2Adam/conv2d_2/bias/v
/:-2"Adam/batch_normalization_3/gamma/v
.:,2!Adam/batch_normalization_3/beta/v
0:.2Adam/conv2d_3/kernel/v
!:2Adam/conv2d_3/bias/v
/:-2"Adam/batch_normalization_4/gamma/v
.:,2!Adam/batch_normalization_4/beta/v
8:62Adam/conv2d_transpose/kernel/v
):'2Adam/conv2d_transpose/bias/v
/:-2"Adam/batch_normalization_5/gamma/v
.:,2!Adam/batch_normalization_5/beta/v
::82 Adam/conv2d_transpose_1/kernel/v
+:)2Adam/conv2d_transpose_1/bias/v
/:-2"Adam/batch_normalization_6/gamma/v
.:,2!Adam/batch_normalization_6/beta/v
::82 Adam/conv2d_transpose_2/kernel/v
+:)2Adam/conv2d_transpose_2/bias/v
/:-2"Adam/batch_normalization_7/gamma/v
.:,2!Adam/batch_normalization_7/beta/v
9:7@Ą2 Adam/conv2d_transpose_3/kernel/v
*:(@2Adam/conv2d_transpose_3/bias/v
.:,@2"Adam/batch_normalization_8/gamma/v
-:+@2!Adam/batch_normalization_8/beta/v
8:6D2 Adam/conv2d_transpose_4/kernel/v
*:(2Adam/conv2d_transpose_4/bias/v
ź2ē
'__inference_model_layer_call_fn_1286524
'__inference_model_layer_call_fn_1288134
'__inference_model_layer_call_fn_1288250
'__inference_model_layer_call_fn_1287598Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Ö2Ó
B__inference_model_layer_call_and_return_conditional_losses_1288646
B__inference_model_layer_call_and_return_conditional_losses_1289042
B__inference_model_layer_call_and_return_conditional_losses_1287746
B__inference_model_layer_call_and_return_conditional_losses_1287894Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
įBŽ
"__inference__wrapped_model_1284876u_velv_velw_velpr0.71"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ž2Ū
4__inference_periodic_padding2d_layer_call_fn_1289047¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ł2ö
O__inference_periodic_padding2d_layer_call_and_return_conditional_losses_1289071¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ą2Ż
6__inference_periodic_padding2d_1_layer_call_fn_1289076¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ū2ų
Q__inference_periodic_padding2d_1_layer_call_and_return_conditional_losses_1289100¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ą2Ż
6__inference_periodic_padding2d_2_layer_call_fn_1289105¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ū2ų
Q__inference_periodic_padding2d_2_layer_call_and_return_conditional_losses_1289129¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ą2Ż
6__inference_periodic_padding2d_3_layer_call_fn_1289134¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ū2ų
Q__inference_periodic_padding2d_3_layer_call_and_return_conditional_losses_1289158¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ó2Š
)__inference_reshape_layer_call_fn_1289163¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ī2ė
D__inference_reshape_layer_call_and_return_conditional_losses_1289177¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Õ2Ņ
+__inference_reshape_1_layer_call_fn_1289182¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
š2ķ
F__inference_reshape_1_layer_call_and_return_conditional_losses_1289196¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Õ2Ņ
+__inference_reshape_2_layer_call_fn_1289201¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
š2ķ
F__inference_reshape_2_layer_call_and_return_conditional_losses_1289215¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Õ2Ņ
+__inference_reshape_3_layer_call_fn_1289220¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
š2ķ
F__inference_reshape_3_layer_call_and_return_conditional_losses_1289234¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
×2Ō
-__inference_concatenate_layer_call_fn_1289242¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ņ2ļ
H__inference_concatenate_layer_call_and_return_conditional_losses_1289251¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
2
5__inference_batch_normalization_layer_call_fn_1289264
5__inference_batch_normalization_layer_call_fn_1289277
5__inference_batch_normalization_layer_call_fn_1289290
5__inference_batch_normalization_layer_call_fn_1289303“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2’
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1289321
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1289339
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1289357
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1289375“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Ņ2Ļ
(__inference_conv2d_layer_call_fn_1289384¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ķ2ź
C__inference_conv2d_layer_call_and_return_conditional_losses_1289395¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
2
7__inference_batch_normalization_1_layer_call_fn_1289408
7__inference_batch_normalization_1_layer_call_fn_1289421
7__inference_batch_normalization_1_layer_call_fn_1289434
7__inference_batch_normalization_1_layer_call_fn_1289447“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1289465
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1289483
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1289501
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1289519“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Ō2Ń
*__inference_conv2d_1_layer_call_fn_1289528¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ļ2ģ
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1289539¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
2
7__inference_batch_normalization_2_layer_call_fn_1289552
7__inference_batch_normalization_2_layer_call_fn_1289565
7__inference_batch_normalization_2_layer_call_fn_1289578
7__inference_batch_normalization_2_layer_call_fn_1289591“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1289609
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1289627
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1289645
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1289663“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Ō2Ń
*__inference_conv2d_2_layer_call_fn_1289672¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ļ2ģ
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1289683¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
2
7__inference_batch_normalization_3_layer_call_fn_1289696
7__inference_batch_normalization_3_layer_call_fn_1289709
7__inference_batch_normalization_3_layer_call_fn_1289722
7__inference_batch_normalization_3_layer_call_fn_1289735“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1289753
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1289771
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1289789
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1289807“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Ō2Ń
*__inference_conv2d_3_layer_call_fn_1289816¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ļ2ģ
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1289827¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
2
7__inference_batch_normalization_4_layer_call_fn_1289840
7__inference_batch_normalization_4_layer_call_fn_1289853
7__inference_batch_normalization_4_layer_call_fn_1289866
7__inference_batch_normalization_4_layer_call_fn_1289879“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1289897
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1289915
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1289933
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1289951“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Ł2Ö
/__inference_concatenate_1_layer_call_fn_1289957¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ō2ń
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1289964¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
2
2__inference_conv2d_transpose_layer_call_fn_1289973
2__inference_conv2d_transpose_layer_call_fn_1289982¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ę2Ć
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1290020
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1290044¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
2
7__inference_batch_normalization_5_layer_call_fn_1290057
7__inference_batch_normalization_5_layer_call_fn_1290070
7__inference_batch_normalization_5_layer_call_fn_1290083
7__inference_batch_normalization_5_layer_call_fn_1290096“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1290114
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1290132
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1290150
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1290168“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Ł2Ö
/__inference_concatenate_2_layer_call_fn_1290174¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ō2ń
J__inference_concatenate_2_layer_call_and_return_conditional_losses_1290181¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
2
4__inference_conv2d_transpose_1_layer_call_fn_1290190
4__inference_conv2d_transpose_1_layer_call_fn_1290199¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ź2Ē
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1290237
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1290261¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
2
7__inference_batch_normalization_6_layer_call_fn_1290274
7__inference_batch_normalization_6_layer_call_fn_1290287
7__inference_batch_normalization_6_layer_call_fn_1290300
7__inference_batch_normalization_6_layer_call_fn_1290313“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1290331
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1290349
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1290367
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1290385“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Ł2Ö
/__inference_concatenate_3_layer_call_fn_1290391¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ō2ń
J__inference_concatenate_3_layer_call_and_return_conditional_losses_1290398¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
2
4__inference_conv2d_transpose_2_layer_call_fn_1290407
4__inference_conv2d_transpose_2_layer_call_fn_1290416¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ź2Ē
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1290454
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1290478¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
2
7__inference_batch_normalization_7_layer_call_fn_1290491
7__inference_batch_normalization_7_layer_call_fn_1290504
7__inference_batch_normalization_7_layer_call_fn_1290517
7__inference_batch_normalization_7_layer_call_fn_1290530“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1290548
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1290566
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1290584
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1290602“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Ł2Ö
/__inference_concatenate_4_layer_call_fn_1290608¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ō2ń
J__inference_concatenate_4_layer_call_and_return_conditional_losses_1290615¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
2
4__inference_conv2d_transpose_3_layer_call_fn_1290624
4__inference_conv2d_transpose_3_layer_call_fn_1290633¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ź2Ē
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1290671
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1290695¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
2
7__inference_batch_normalization_8_layer_call_fn_1290708
7__inference_batch_normalization_8_layer_call_fn_1290721
7__inference_batch_normalization_8_layer_call_fn_1290734
7__inference_batch_normalization_8_layer_call_fn_1290747“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1290765
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1290783
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1290801
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1290819“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Ł2Ö
/__inference_concatenate_5_layer_call_fn_1290825¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ō2ń
J__inference_concatenate_5_layer_call_and_return_conditional_losses_1290832¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
2
4__inference_conv2d_transpose_4_layer_call_fn_1290841
4__inference_conv2d_transpose_4_layer_call_fn_1290850¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ź2Ē
O__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1290887
O__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1290910¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
2
,__inference_cropping2d_layer_call_fn_1290915
,__inference_cropping2d_layer_call_fn_1290920¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ŗ2·
G__inference_cropping2d_layer_call_and_return_conditional_losses_1290928
G__inference_cropping2d_layer_call_and_return_conditional_losses_1290936¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ŽBŪ
%__inference_signature_wrapper_1288018pr0.71u_velv_velw_vel"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ū
"__inference__wrapped_model_1284876ŌXQRSTYZ`abchiopqrwx~ ”¢£¬­³“µ¶æĄĘĒČÉŅÓŁŚŪÜåę“¢°
Ø¢¤
”
%"
u_vel’’’’’’’’’
%"
v_vel’’’’’’’’’
%"
w_vel’’’’’’’’’
&#
pr0.71’’’’’’’’’
Ŗ "AŖ>
<

cropping2d.+

cropping2d’’’’’’’’’ķ
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1289465`abcM¢J
C¢@
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
p 
Ŗ "?¢<
52
0+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 ķ
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1289483`abcM¢J
C¢@
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
p
Ŗ "?¢<
52
0+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 Ģ
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1289501v`abc=¢:
3¢0
*'
inputs’’’’’’’’’@
p 
Ŗ "/¢,
%"
0’’’’’’’’’@
 Ģ
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1289519v`abc=¢:
3¢0
*'
inputs’’’’’’’’’@
p
Ŗ "/¢,
%"
0’’’’’’’’’@
 Å
7__inference_batch_normalization_1_layer_call_fn_1289408`abcM¢J
C¢@
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
p 
Ŗ "2/+’’’’’’’’’’’’’’’’’’’’’’’’’’’@Å
7__inference_batch_normalization_1_layer_call_fn_1289421`abcM¢J
C¢@
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
p
Ŗ "2/+’’’’’’’’’’’’’’’’’’’’’’’’’’’@¤
7__inference_batch_normalization_1_layer_call_fn_1289434i`abc=¢:
3¢0
*'
inputs’’’’’’’’’@
p 
Ŗ ""’’’’’’’’’@¤
7__inference_batch_normalization_1_layer_call_fn_1289447i`abc=¢:
3¢0
*'
inputs’’’’’’’’’@
p
Ŗ ""’’’’’’’’’@ļ
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1289609opqrN¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p 
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 ļ
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1289627opqrN¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Ī
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1289645xopqr>¢;
4¢1
+(
inputs’’’’’’’’’
p 
Ŗ "0¢-
&#
0’’’’’’’’’
 Ī
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1289663xopqr>¢;
4¢1
+(
inputs’’’’’’’’’
p
Ŗ "0¢-
&#
0’’’’’’’’’
 Ē
7__inference_batch_normalization_2_layer_call_fn_1289552opqrN¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p 
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ē
7__inference_batch_normalization_2_layer_call_fn_1289565opqrN¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’¦
7__inference_batch_normalization_2_layer_call_fn_1289578kopqr>¢;
4¢1
+(
inputs’’’’’’’’’
p 
Ŗ "# ’’’’’’’’’¦
7__inference_batch_normalization_2_layer_call_fn_1289591kopqr>¢;
4¢1
+(
inputs’’’’’’’’’
p
Ŗ "# ’’’’’’’’’ń
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1289753~N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p 
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 ń
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1289771~N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Š
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1289789z~>¢;
4¢1
+(
inputs’’’’’’’’’
p 
Ŗ "0¢-
&#
0’’’’’’’’’
 Š
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1289807z~>¢;
4¢1
+(
inputs’’’’’’’’’
p
Ŗ "0¢-
&#
0’’’’’’’’’
 É
7__inference_batch_normalization_3_layer_call_fn_1289696~N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p 
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’É
7__inference_batch_normalization_3_layer_call_fn_1289709~N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ø
7__inference_batch_normalization_3_layer_call_fn_1289722m~>¢;
4¢1
+(
inputs’’’’’’’’’
p 
Ŗ "# ’’’’’’’’’Ø
7__inference_batch_normalization_3_layer_call_fn_1289735m~>¢;
4¢1
+(
inputs’’’’’’’’’
p
Ŗ "# ’’’’’’’’’ó
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1289897N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p 
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 ó
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1289915N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Ņ
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1289933|>¢;
4¢1
+(
inputs’’’’’’’’’
p 
Ŗ "0¢-
&#
0’’’’’’’’’
 Ņ
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1289951|>¢;
4¢1
+(
inputs’’’’’’’’’
p
Ŗ "0¢-
&#
0’’’’’’’’’
 Ė
7__inference_batch_normalization_4_layer_call_fn_1289840N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p 
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ė
7__inference_batch_normalization_4_layer_call_fn_1289853N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ŗ
7__inference_batch_normalization_4_layer_call_fn_1289866o>¢;
4¢1
+(
inputs’’’’’’’’’
p 
Ŗ "# ’’’’’’’’’Ŗ
7__inference_batch_normalization_4_layer_call_fn_1289879o>¢;
4¢1
+(
inputs’’’’’’’’’
p
Ŗ "# ’’’’’’’’’ó
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1290114 ”¢£N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p 
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 ó
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1290132 ”¢£N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Ņ
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1290150| ”¢£>¢;
4¢1
+(
inputs’’’’’’’’’
p 
Ŗ "0¢-
&#
0’’’’’’’’’
 Ņ
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1290168| ”¢£>¢;
4¢1
+(
inputs’’’’’’’’’
p
Ŗ "0¢-
&#
0’’’’’’’’’
 Ė
7__inference_batch_normalization_5_layer_call_fn_1290057 ”¢£N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p 
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ė
7__inference_batch_normalization_5_layer_call_fn_1290070 ”¢£N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ŗ
7__inference_batch_normalization_5_layer_call_fn_1290083o ”¢£>¢;
4¢1
+(
inputs’’’’’’’’’
p 
Ŗ "# ’’’’’’’’’Ŗ
7__inference_batch_normalization_5_layer_call_fn_1290096o ”¢£>¢;
4¢1
+(
inputs’’’’’’’’’
p
Ŗ "# ’’’’’’’’’ó
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1290331³“µ¶N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p 
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 ó
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1290349³“µ¶N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Ņ
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1290367|³“µ¶>¢;
4¢1
+(
inputs’’’’’’’’’
p 
Ŗ "0¢-
&#
0’’’’’’’’’
 Ņ
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1290385|³“µ¶>¢;
4¢1
+(
inputs’’’’’’’’’
p
Ŗ "0¢-
&#
0’’’’’’’’’
 Ė
7__inference_batch_normalization_6_layer_call_fn_1290274³“µ¶N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p 
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ė
7__inference_batch_normalization_6_layer_call_fn_1290287³“µ¶N¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ŗ
7__inference_batch_normalization_6_layer_call_fn_1290300o³“µ¶>¢;
4¢1
+(
inputs’’’’’’’’’
p 
Ŗ "# ’’’’’’’’’Ŗ
7__inference_batch_normalization_6_layer_call_fn_1290313o³“µ¶>¢;
4¢1
+(
inputs’’’’’’’’’
p
Ŗ "# ’’’’’’’’’ó
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1290548ĘĒČÉN¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p 
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 ó
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1290566ĘĒČÉN¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Ņ
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1290584|ĘĒČÉ>¢;
4¢1
+(
inputs’’’’’’’’’
p 
Ŗ "0¢-
&#
0’’’’’’’’’
 Ņ
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1290602|ĘĒČÉ>¢;
4¢1
+(
inputs’’’’’’’’’
p
Ŗ "0¢-
&#
0’’’’’’’’’
 Ė
7__inference_batch_normalization_7_layer_call_fn_1290491ĘĒČÉN¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p 
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ė
7__inference_batch_normalization_7_layer_call_fn_1290504ĘĒČÉN¢K
D¢A
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ŗ
7__inference_batch_normalization_7_layer_call_fn_1290517oĘĒČÉ>¢;
4¢1
+(
inputs’’’’’’’’’
p 
Ŗ "# ’’’’’’’’’Ŗ
7__inference_batch_normalization_7_layer_call_fn_1290530oĘĒČÉ>¢;
4¢1
+(
inputs’’’’’’’’’
p
Ŗ "# ’’’’’’’’’ń
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1290765ŁŚŪÜM¢J
C¢@
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
p 
Ŗ "?¢<
52
0+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 ń
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1290783ŁŚŪÜM¢J
C¢@
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
p
Ŗ "?¢<
52
0+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 Š
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1290801zŁŚŪÜ=¢:
3¢0
*'
inputs’’’’’’’’’@
p 
Ŗ "/¢,
%"
0’’’’’’’’’@
 Š
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1290819zŁŚŪÜ=¢:
3¢0
*'
inputs’’’’’’’’’@
p
Ŗ "/¢,
%"
0’’’’’’’’’@
 É
7__inference_batch_normalization_8_layer_call_fn_1290708ŁŚŪÜM¢J
C¢@
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
p 
Ŗ "2/+’’’’’’’’’’’’’’’’’’’’’’’’’’’@É
7__inference_batch_normalization_8_layer_call_fn_1290721ŁŚŪÜM¢J
C¢@
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
p
Ŗ "2/+’’’’’’’’’’’’’’’’’’’’’’’’’’’@Ø
7__inference_batch_normalization_8_layer_call_fn_1290734mŁŚŪÜ=¢:
3¢0
*'
inputs’’’’’’’’’@
p 
Ŗ ""’’’’’’’’’@Ø
7__inference_batch_normalization_8_layer_call_fn_1290747mŁŚŪÜ=¢:
3¢0
*'
inputs’’’’’’’’’@
p
Ŗ ""’’’’’’’’’@ė
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1289321QRSTM¢J
C¢@
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’
p 
Ŗ "?¢<
52
0+’’’’’’’’’’’’’’’’’’’’’’’’’’’
 ė
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1289339QRSTM¢J
C¢@
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
Ŗ "?¢<
52
0+’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Ź
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1289357vQRST=¢:
3¢0
*'
inputs’’’’’’’’’
p 
Ŗ "/¢,
%"
0’’’’’’’’’
 Ź
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1289375vQRST=¢:
3¢0
*'
inputs’’’’’’’’’
p
Ŗ "/¢,
%"
0’’’’’’’’’
 Ć
5__inference_batch_normalization_layer_call_fn_1289264QRSTM¢J
C¢@
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’
p 
Ŗ "2/+’’’’’’’’’’’’’’’’’’’’’’’’’’’Ć
5__inference_batch_normalization_layer_call_fn_1289277QRSTM¢J
C¢@
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
Ŗ "2/+’’’’’’’’’’’’’’’’’’’’’’’’’’’¢
5__inference_batch_normalization_layer_call_fn_1289290iQRST=¢:
3¢0
*'
inputs’’’’’’’’’
p 
Ŗ ""’’’’’’’’’¢
5__inference_batch_normalization_layer_call_fn_1289303iQRST=¢:
3¢0
*'
inputs’’’’’’’’’
p
Ŗ ""’’’’’’’’’ó
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1289964¤p¢m
f¢c
a^
-*
inputs/0’’’’’’’’’
-*
inputs/1’’’’’’’’’
Ŗ "0¢-
&#
0’’’’’’’’’
 Ė
/__inference_concatenate_1_layer_call_fn_1289957p¢m
f¢c
a^
-*
inputs/0’’’’’’’’’
-*
inputs/1’’’’’’’’’
Ŗ "# ’’’’’’’’’ó
J__inference_concatenate_2_layer_call_and_return_conditional_losses_1290181¤p¢m
f¢c
a^
-*
inputs/0’’’’’’’’’
-*
inputs/1’’’’’’’’’
Ŗ "0¢-
&#
0’’’’’’’’’
 Ė
/__inference_concatenate_2_layer_call_fn_1290174p¢m
f¢c
a^
-*
inputs/0’’’’’’’’’
-*
inputs/1’’’’’’’’’
Ŗ "# ’’’’’’’’’ó
J__inference_concatenate_3_layer_call_and_return_conditional_losses_1290398¤p¢m
f¢c
a^
-*
inputs/0’’’’’’’’’
-*
inputs/1’’’’’’’’’
Ŗ "0¢-
&#
0’’’’’’’’’
 Ė
/__inference_concatenate_3_layer_call_fn_1290391p¢m
f¢c
a^
-*
inputs/0’’’’’’’’’
-*
inputs/1’’’’’’’’’
Ŗ "# ’’’’’’’’’ņ
J__inference_concatenate_4_layer_call_and_return_conditional_losses_1290615£o¢l
e¢b
`]
,)
inputs/0’’’’’’’’’@
-*
inputs/1’’’’’’’’’
Ŗ "0¢-
&#
0’’’’’’’’’Ą
 Ź
/__inference_concatenate_4_layer_call_fn_1290608o¢l
e¢b
`]
,)
inputs/0’’’’’’’’’@
-*
inputs/1’’’’’’’’’
Ŗ "# ’’’’’’’’’Ąš
J__inference_concatenate_5_layer_call_and_return_conditional_losses_1290832”n¢k
d¢a
_\
,)
inputs/0’’’’’’’’’
,)
inputs/1’’’’’’’’’@
Ŗ "/¢,
%"
0’’’’’’’’’D
 Č
/__inference_concatenate_5_layer_call_fn_1290825n¢k
d¢a
_\
,)
inputs/0’’’’’’’’’
,)
inputs/1’’’’’’’’’@
Ŗ ""’’’’’’’’’DŠ
H__inference_concatenate_layer_call_and_return_conditional_losses_1289251Ļ¢Ė
Ć¢æ
¼ø
,)
inputs/0’’’’’’’’’
,)
inputs/1’’’’’’’’’
,)
inputs/2’’’’’’’’’
,)
inputs/3’’’’’’’’’
Ŗ "/¢,
%"
0’’’’’’’’’
 Ø
-__inference_concatenate_layer_call_fn_1289242öĻ¢Ė
Ć¢æ
¼ø
,)
inputs/0’’’’’’’’’
,)
inputs/1’’’’’’’’’
,)
inputs/2’’’’’’’’’
,)
inputs/3’’’’’’’’’
Ŗ ""’’’’’’’’’ŗ
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1289539qhi9¢6
/¢,
*'
inputs’’’’’’’’’@
Ŗ "0¢-
&#
0’’’’’’’’’
 
*__inference_conv2d_1_layer_call_fn_1289528dhi9¢6
/¢,
*'
inputs’’’’’’’’’@
Ŗ "# ’’’’’’’’’»
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1289683rwx:¢7
0¢-
+(
inputs’’’’’’’’’
Ŗ "0¢-
&#
0’’’’’’’’’
 
*__inference_conv2d_2_layer_call_fn_1289672ewx:¢7
0¢-
+(
inputs’’’’’’’’’
Ŗ "# ’’’’’’’’’½
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1289827t:¢7
0¢-
+(
inputs’’’’’’’’’
Ŗ "0¢-
&#
0’’’’’’’’’
 
*__inference_conv2d_3_layer_call_fn_1289816g:¢7
0¢-
+(
inputs’’’’’’’’’
Ŗ "# ’’’’’’’’’·
C__inference_conv2d_layer_call_and_return_conditional_losses_1289395pYZ9¢6
/¢,
*'
inputs’’’’’’’’’
Ŗ "/¢,
%"
0’’’’’’’’’@
 
(__inference_conv2d_layer_call_fn_1289384cYZ9¢6
/¢,
*'
inputs’’’’’’’’’
Ŗ ""’’’’’’’’’@č
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1290237¬­J¢G
@¢=
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Ē
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1290261t¬­:¢7
0¢-
+(
inputs’’’’’’’’’
Ŗ "0¢-
&#
0’’’’’’’’’
 Ą
4__inference_conv2d_transpose_1_layer_call_fn_1290190¬­J¢G
@¢=
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’
4__inference_conv2d_transpose_1_layer_call_fn_1290199g¬­:¢7
0¢-
+(
inputs’’’’’’’’’
Ŗ "# ’’’’’’’’’č
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1290454æĄJ¢G
@¢=
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Ē
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1290478tæĄ:¢7
0¢-
+(
inputs’’’’’’’’’
Ŗ "0¢-
&#
0’’’’’’’’’
 Ą
4__inference_conv2d_transpose_2_layer_call_fn_1290407æĄJ¢G
@¢=
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’
4__inference_conv2d_transpose_2_layer_call_fn_1290416gæĄ:¢7
0¢-
+(
inputs’’’’’’’’’
Ŗ "# ’’’’’’’’’ē
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1290671ŅÓJ¢G
@¢=
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ą
Ŗ "?¢<
52
0+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
 Ę
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1290695sŅÓ:¢7
0¢-
+(
inputs’’’’’’’’’Ą
Ŗ "/¢,
%"
0’’’’’’’’’@
 æ
4__inference_conv2d_transpose_3_layer_call_fn_1290624ŅÓJ¢G
@¢=
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’Ą
Ŗ "2/+’’’’’’’’’’’’’’’’’’’’’’’’’’’@
4__inference_conv2d_transpose_3_layer_call_fn_1290633fŅÓ:¢7
0¢-
+(
inputs’’’’’’’’’Ą
Ŗ ""’’’’’’’’’@ę
O__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1290887åęI¢F
?¢<
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’D
Ŗ "?¢<
52
0+’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Å
O__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1290910råę9¢6
/¢,
*'
inputs’’’’’’’’’D
Ŗ "/¢,
%"
0’’’’’’’’’
 ¾
4__inference_conv2d_transpose_4_layer_call_fn_1290841åęI¢F
?¢<
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’D
Ŗ "2/+’’’’’’’’’’’’’’’’’’’’’’’’’’’
4__inference_conv2d_transpose_4_layer_call_fn_1290850eåę9¢6
/¢,
*'
inputs’’’’’’’’’D
Ŗ ""’’’’’’’’’ę
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1290020J¢G
@¢=
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "@¢=
63
0,’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Å
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1290044t:¢7
0¢-
+(
inputs’’’’’’’’’
Ŗ "0¢-
&#
0’’’’’’’’’
 ¾
2__inference_conv2d_transpose_layer_call_fn_1289973J¢G
@¢=
;8
inputs,’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "30,’’’’’’’’’’’’’’’’’’’’’’’’’’’
2__inference_conv2d_transpose_layer_call_fn_1289982g:¢7
0¢-
+(
inputs’’’’’’’’’
Ŗ "# ’’’’’’’’’ź
G__inference_cropping2d_layer_call_and_return_conditional_losses_1290928R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "H¢E
>;
04’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 ·
G__inference_cropping2d_layer_call_and_return_conditional_losses_1290936l9¢6
/¢,
*'
inputs’’’’’’’’’
Ŗ "/¢,
%"
0’’’’’’’’’
 Ā
,__inference_cropping2d_layer_call_fn_1290915R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ ";84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
,__inference_cropping2d_layer_call_fn_1290920_9¢6
/¢,
*'
inputs’’’’’’’’’
Ŗ ""’’’’’’’’’
B__inference_model_layer_call_and_return_conditional_losses_1287746ŹXQRSTYZ`abchiopqrwx~ ”¢£¬­³“µ¶æĄĘĒČÉŅÓŁŚŪÜåę¼¢ø
°¢¬
”
%"
u_vel’’’’’’’’’
%"
v_vel’’’’’’’’’
%"
w_vel’’’’’’’’’
&#
pr0.71’’’’’’’’’
p 

 
Ŗ "/¢,
%"
0’’’’’’’’’
 
B__inference_model_layer_call_and_return_conditional_losses_1287894ŹXQRSTYZ`abchiopqrwx~ ”¢£¬­³“µ¶æĄĘĒČÉŅÓŁŚŪÜåę¼¢ø
°¢¬
”
%"
u_vel’’’’’’’’’
%"
v_vel’’’’’’’’’
%"
w_vel’’’’’’’’’
&#
pr0.71’’’’’’’’’
p

 
Ŗ "/¢,
%"
0’’’’’’’’’
 
B__inference_model_layer_call_and_return_conditional_losses_1288646ÕXQRSTYZ`abchiopqrwx~ ”¢£¬­³“µ¶æĄĘĒČÉŅÓŁŚŪÜåęĒ¢Ć
»¢·
¬Ø
(%
inputs/0’’’’’’’’’
(%
inputs/1’’’’’’’’’
(%
inputs/2’’’’’’’’’
(%
inputs/3’’’’’’’’’
p 

 
Ŗ "/¢,
%"
0’’’’’’’’’
 
B__inference_model_layer_call_and_return_conditional_losses_1289042ÕXQRSTYZ`abchiopqrwx~ ”¢£¬­³“µ¶æĄĘĒČÉŅÓŁŚŪÜåęĒ¢Ć
»¢·
¬Ø
(%
inputs/0’’’’’’’’’
(%
inputs/1’’’’’’’’’
(%
inputs/2’’’’’’’’’
(%
inputs/3’’’’’’’’’
p

 
Ŗ "/¢,
%"
0’’’’’’’’’
 é
'__inference_model_layer_call_fn_1286524½XQRSTYZ`abchiopqrwx~ ”¢£¬­³“µ¶æĄĘĒČÉŅÓŁŚŪÜåę¼¢ø
°¢¬
”
%"
u_vel’’’’’’’’’
%"
v_vel’’’’’’’’’
%"
w_vel’’’’’’’’’
&#
pr0.71’’’’’’’’’
p 

 
Ŗ ""’’’’’’’’’é
'__inference_model_layer_call_fn_1287598½XQRSTYZ`abchiopqrwx~ ”¢£¬­³“µ¶æĄĘĒČÉŅÓŁŚŪÜåę¼¢ø
°¢¬
”
%"
u_vel’’’’’’’’’
%"
v_vel’’’’’’’’’
%"
w_vel’’’’’’’’’
&#
pr0.71’’’’’’’’’
p

 
Ŗ ""’’’’’’’’’ō
'__inference_model_layer_call_fn_1288134ČXQRSTYZ`abchiopqrwx~ ”¢£¬­³“µ¶æĄĘĒČÉŅÓŁŚŪÜåęĒ¢Ć
»¢·
¬Ø
(%
inputs/0’’’’’’’’’
(%
inputs/1’’’’’’’’’
(%
inputs/2’’’’’’’’’
(%
inputs/3’’’’’’’’’
p 

 
Ŗ ""’’’’’’’’’ō
'__inference_model_layer_call_fn_1288250ČXQRSTYZ`abchiopqrwx~ ”¢£¬­³“µ¶æĄĘĒČÉŅÓŁŚŪÜåęĒ¢Ć
»¢·
¬Ø
(%
inputs/0’’’’’’’’’
(%
inputs/1’’’’’’’’’
(%
inputs/2’’’’’’’’’
(%
inputs/3’’’’’’’’’
p

 
Ŗ ""’’’’’’’’’¹
Q__inference_periodic_padding2d_1_layer_call_and_return_conditional_losses_1289100d5¢2
+¢(
&#
inputs’’’’’’’’’
Ŗ "+¢(
!
0’’’’’’’’’
 
6__inference_periodic_padding2d_1_layer_call_fn_1289076W5¢2
+¢(
&#
inputs’’’’’’’’’
Ŗ "’’’’’’’’’¹
Q__inference_periodic_padding2d_2_layer_call_and_return_conditional_losses_1289129d5¢2
+¢(
&#
inputs’’’’’’’’’
Ŗ "+¢(
!
0’’’’’’’’’
 
6__inference_periodic_padding2d_2_layer_call_fn_1289105W5¢2
+¢(
&#
inputs’’’’’’’’’
Ŗ "’’’’’’’’’¹
Q__inference_periodic_padding2d_3_layer_call_and_return_conditional_losses_1289158d5¢2
+¢(
&#
inputs’’’’’’’’’
Ŗ "+¢(
!
0’’’’’’’’’
 
6__inference_periodic_padding2d_3_layer_call_fn_1289134W5¢2
+¢(
&#
inputs’’’’’’’’’
Ŗ "’’’’’’’’’·
O__inference_periodic_padding2d_layer_call_and_return_conditional_losses_1289071d5¢2
+¢(
&#
inputs’’’’’’’’’
Ŗ "+¢(
!
0’’’’’’’’’
 
4__inference_periodic_padding2d_layer_call_fn_1289047W5¢2
+¢(
&#
inputs’’’’’’’’’
Ŗ "’’’’’’’’’²
F__inference_reshape_1_layer_call_and_return_conditional_losses_1289196h5¢2
+¢(
&#
inputs’’’’’’’’’
Ŗ "/¢,
%"
0’’’’’’’’’
 
+__inference_reshape_1_layer_call_fn_1289182[5¢2
+¢(
&#
inputs’’’’’’’’’
Ŗ ""’’’’’’’’’²
F__inference_reshape_2_layer_call_and_return_conditional_losses_1289215h5¢2
+¢(
&#
inputs’’’’’’’’’
Ŗ "/¢,
%"
0’’’’’’’’’
 
+__inference_reshape_2_layer_call_fn_1289201[5¢2
+¢(
&#
inputs’’’’’’’’’
Ŗ ""’’’’’’’’’²
F__inference_reshape_3_layer_call_and_return_conditional_losses_1289234h5¢2
+¢(
&#
inputs’’’’’’’’’
Ŗ "/¢,
%"
0’’’’’’’’’
 
+__inference_reshape_3_layer_call_fn_1289220[5¢2
+¢(
&#
inputs’’’’’’’’’
Ŗ ""’’’’’’’’’°
D__inference_reshape_layer_call_and_return_conditional_losses_1289177h5¢2
+¢(
&#
inputs’’’’’’’’’
Ŗ "/¢,
%"
0’’’’’’’’’
 
)__inference_reshape_layer_call_fn_1289163[5¢2
+¢(
&#
inputs’’’’’’’’’
Ŗ ""’’’’’’’’’
%__inference_signature_wrapper_1288018ņXQRSTYZ`abchiopqrwx~ ”¢£¬­³“µ¶æĄĘĒČÉŅÓŁŚŪÜåęŅ¢Ī
¢ 
ĘŖĀ
0
pr0.71&#
pr0.71’’’’’’’’’
.
u_vel%"
u_vel’’’’’’’’’
.
v_vel%"
v_vel’’’’’’’’’
.
w_vel%"
w_vel’’’’’’’’’"AŖ>
<

cropping2d.+

cropping2d’’’’’’’’’