Щљ3
╗ї
B
AssignVariableOp
resource
value"dtype"
dtypetypeѕ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Џ
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
└
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
Щ
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
epsilonfloat%иЛ8"&
exponential_avg_factorfloat%  ђ?";
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
delete_old_dirsbool(ѕ
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
dtypetypeѕ
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
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
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
Й
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
executor_typestring ѕ
@
StaticRegexFullMatch	
input

output
"
patternstring
Ш
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
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.6.02v2.6.0-rc2-32-g919f693420e8љё/
і
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma
Ѓ
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0
ѕ
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta
Ђ
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0
ќ
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean
Ј
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0
ъ
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance
Ќ
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:@*
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
ј
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_1/gamma
Є
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:@*
dtype0
ї
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_1/beta
Ё
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:@*
dtype0
џ
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_1/moving_mean
Њ
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:@*
dtype0
б
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_1/moving_variance
Џ
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:@*
dtype0
Ѓ
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ђ* 
shared_nameconv2d_1/kernel
|
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*'
_output_shapes
:@ђ*
dtype0
s
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_nameconv2d_1/bias
l
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes	
:ђ*
dtype0
Ј
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*,
shared_namebatch_normalization_2/gamma
ѕ
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes	
:ђ*
dtype0
Ї
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*+
shared_namebatch_normalization_2/beta
є
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes	
:ђ*
dtype0
Џ
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!batch_normalization_2/moving_mean
ћ
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes	
:ђ*
dtype0
Б
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*6
shared_name'%batch_normalization_2/moving_variance
ю
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes	
:ђ*
dtype0
ё
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ* 
shared_nameconv2d_2/kernel
}
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*(
_output_shapes
:ђђ*
dtype0
s
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_nameconv2d_2/bias
l
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes	
:ђ*
dtype0
Ј
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*,
shared_namebatch_normalization_3/gamma
ѕ
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes	
:ђ*
dtype0
Ї
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*+
shared_namebatch_normalization_3/beta
є
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes	
:ђ*
dtype0
Џ
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!batch_normalization_3/moving_mean
ћ
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes	
:ђ*
dtype0
Б
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*6
shared_name'%batch_normalization_3/moving_variance
ю
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes	
:ђ*
dtype0
ё
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ* 
shared_nameconv2d_3/kernel
}
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:ђђ*
dtype0
s
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_nameconv2d_3/bias
l
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes	
:ђ*
dtype0
Ј
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*,
shared_namebatch_normalization_4/gamma
ѕ
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes	
:ђ*
dtype0
Ї
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*+
shared_namebatch_normalization_4/beta
є
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes	
:ђ*
dtype0
Џ
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!batch_normalization_4/moving_mean
ћ
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes	
:ђ*
dtype0
Б
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*6
shared_name'%batch_normalization_4/moving_variance
ю
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes	
:ђ*
dtype0
ћ
conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*(
shared_nameconv2d_transpose/kernel
Ї
+conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose/kernel*(
_output_shapes
:ђђ*
dtype0
Ѓ
conv2d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameconv2d_transpose/bias
|
)conv2d_transpose/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose/bias*
_output_shapes	
:ђ*
dtype0
Ј
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*,
shared_namebatch_normalization_5/gamma
ѕ
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes	
:ђ*
dtype0
Ї
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*+
shared_namebatch_normalization_5/beta
є
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes	
:ђ*
dtype0
Џ
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!batch_normalization_5/moving_mean
ћ
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes	
:ђ*
dtype0
Б
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*6
shared_name'%batch_normalization_5/moving_variance
ю
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes	
:ђ*
dtype0
ў
conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ**
shared_nameconv2d_transpose_1/kernel
Љ
-conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/kernel*(
_output_shapes
:ђђ*
dtype0
Є
conv2d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*(
shared_nameconv2d_transpose_1/bias
ђ
+conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/bias*
_output_shapes	
:ђ*
dtype0
Ј
batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*,
shared_namebatch_normalization_6/gamma
ѕ
/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes	
:ђ*
dtype0
Ї
batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*+
shared_namebatch_normalization_6/beta
є
.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes	
:ђ*
dtype0
Џ
!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!batch_normalization_6/moving_mean
ћ
5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes	
:ђ*
dtype0
Б
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*6
shared_name'%batch_normalization_6/moving_variance
ю
9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes	
:ђ*
dtype0
ў
conv2d_transpose_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ**
shared_nameconv2d_transpose_2/kernel
Љ
-conv2d_transpose_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/kernel*(
_output_shapes
:ђђ*
dtype0
Є
conv2d_transpose_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*(
shared_nameconv2d_transpose_2/bias
ђ
+conv2d_transpose_2/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/bias*
_output_shapes	
:ђ*
dtype0
Ј
batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*,
shared_namebatch_normalization_7/gamma
ѕ
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes	
:ђ*
dtype0
Ї
batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*+
shared_namebatch_normalization_7/beta
є
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes	
:ђ*
dtype0
Џ
!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!batch_normalization_7/moving_mean
ћ
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes	
:ђ*
dtype0
Б
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*6
shared_name'%batch_normalization_7/moving_variance
ю
9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes	
:ђ*
dtype0
Ќ
conv2d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ђ**
shared_nameconv2d_transpose_3/kernel
љ
-conv2d_transpose_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/kernel*'
_output_shapes
:@ђ*
dtype0
є
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
ј
batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_8/gamma
Є
/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes
:@*
dtype0
ї
batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_8/beta
Ё
.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes
:@*
dtype0
џ
!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_8/moving_mean
Њ
5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes
:@*
dtype0
б
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_8/moving_variance
Џ
9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes
:@*
dtype0
ќ
conv2d_transpose_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameconv2d_transpose_4/kernel
Ј
-conv2d_transpose_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_4/kernel*&
_output_shapes
:@*
dtype0
є
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
ў
 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/batch_normalization/gamma/m
Љ
4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes
:*
dtype0
ќ
Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/batch_normalization/beta/m
Ј
3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes
:*
dtype0
ї
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d/kernel/m
Ё
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:@*
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
ю
"Adam/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_1/gamma/m
Ћ
6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/m*
_output_shapes
:@*
dtype0
џ
!Adam/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_1/beta/m
Њ
5Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/m*
_output_shapes
:@*
dtype0
Љ
Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ђ*'
shared_nameAdam/conv2d_1/kernel/m
і
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*'
_output_shapes
:@ђ*
dtype0
Ђ
Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*%
shared_nameAdam/conv2d_1/bias/m
z
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes	
:ђ*
dtype0
Ю
"Adam/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*3
shared_name$"Adam/batch_normalization_2/gamma/m
ќ
6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/m*
_output_shapes	
:ђ*
dtype0
Џ
!Adam/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!Adam/batch_normalization_2/beta/m
ћ
5Adam/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/m*
_output_shapes	
:ђ*
dtype0
њ
Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*'
shared_nameAdam/conv2d_2/kernel/m
І
*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*(
_output_shapes
:ђђ*
dtype0
Ђ
Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*%
shared_nameAdam/conv2d_2/bias/m
z
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes	
:ђ*
dtype0
Ю
"Adam/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*3
shared_name$"Adam/batch_normalization_3/gamma/m
ќ
6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/m*
_output_shapes	
:ђ*
dtype0
Џ
!Adam/batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!Adam/batch_normalization_3/beta/m
ћ
5Adam/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/m*
_output_shapes	
:ђ*
dtype0
њ
Adam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*'
shared_nameAdam/conv2d_3/kernel/m
І
*Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/m*(
_output_shapes
:ђђ*
dtype0
Ђ
Adam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*%
shared_nameAdam/conv2d_3/bias/m
z
(Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/m*
_output_shapes	
:ђ*
dtype0
Ю
"Adam/batch_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*3
shared_name$"Adam/batch_normalization_4/gamma/m
ќ
6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/m*
_output_shapes	
:ђ*
dtype0
Џ
!Adam/batch_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!Adam/batch_normalization_4/beta/m
ћ
5Adam/batch_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/m*
_output_shapes	
:ђ*
dtype0
б
Adam/conv2d_transpose/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*/
shared_name Adam/conv2d_transpose/kernel/m
Џ
2Adam/conv2d_transpose/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/kernel/m*(
_output_shapes
:ђђ*
dtype0
Љ
Adam/conv2d_transpose/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*-
shared_nameAdam/conv2d_transpose/bias/m
і
0Adam/conv2d_transpose/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/bias/m*
_output_shapes	
:ђ*
dtype0
Ю
"Adam/batch_normalization_5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*3
shared_name$"Adam/batch_normalization_5/gamma/m
ќ
6Adam/batch_normalization_5/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_5/gamma/m*
_output_shapes	
:ђ*
dtype0
Џ
!Adam/batch_normalization_5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!Adam/batch_normalization_5/beta/m
ћ
5Adam/batch_normalization_5/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_5/beta/m*
_output_shapes	
:ђ*
dtype0
д
 Adam/conv2d_transpose_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*1
shared_name" Adam/conv2d_transpose_1/kernel/m
Ъ
4Adam/conv2d_transpose_1/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_1/kernel/m*(
_output_shapes
:ђђ*
dtype0
Ћ
Adam/conv2d_transpose_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*/
shared_name Adam/conv2d_transpose_1/bias/m
ј
2Adam/conv2d_transpose_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_1/bias/m*
_output_shapes	
:ђ*
dtype0
Ю
"Adam/batch_normalization_6/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*3
shared_name$"Adam/batch_normalization_6/gamma/m
ќ
6Adam/batch_normalization_6/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_6/gamma/m*
_output_shapes	
:ђ*
dtype0
Џ
!Adam/batch_normalization_6/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!Adam/batch_normalization_6/beta/m
ћ
5Adam/batch_normalization_6/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_6/beta/m*
_output_shapes	
:ђ*
dtype0
д
 Adam/conv2d_transpose_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*1
shared_name" Adam/conv2d_transpose_2/kernel/m
Ъ
4Adam/conv2d_transpose_2/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_2/kernel/m*(
_output_shapes
:ђђ*
dtype0
Ћ
Adam/conv2d_transpose_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*/
shared_name Adam/conv2d_transpose_2/bias/m
ј
2Adam/conv2d_transpose_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_2/bias/m*
_output_shapes	
:ђ*
dtype0
Ю
"Adam/batch_normalization_7/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*3
shared_name$"Adam/batch_normalization_7/gamma/m
ќ
6Adam/batch_normalization_7/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/m*
_output_shapes	
:ђ*
dtype0
Џ
!Adam/batch_normalization_7/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!Adam/batch_normalization_7/beta/m
ћ
5Adam/batch_normalization_7/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/m*
_output_shapes	
:ђ*
dtype0
Ц
 Adam/conv2d_transpose_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ђ*1
shared_name" Adam/conv2d_transpose_3/kernel/m
ъ
4Adam/conv2d_transpose_3/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_3/kernel/m*'
_output_shapes
:@ђ*
dtype0
ћ
Adam/conv2d_transpose_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/conv2d_transpose_3/bias/m
Ї
2Adam/conv2d_transpose_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_3/bias/m*
_output_shapes
:@*
dtype0
ю
"Adam/batch_normalization_8/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_8/gamma/m
Ћ
6Adam/batch_normalization_8/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_8/gamma/m*
_output_shapes
:@*
dtype0
џ
!Adam/batch_normalization_8/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_8/beta/m
Њ
5Adam/batch_normalization_8/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_8/beta/m*
_output_shapes
:@*
dtype0
ц
 Adam/conv2d_transpose_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/conv2d_transpose_4/kernel/m
Ю
4Adam/conv2d_transpose_4/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_4/kernel/m*&
_output_shapes
:@*
dtype0
ћ
Adam/conv2d_transpose_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_4/bias/m
Ї
2Adam/conv2d_transpose_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_4/bias/m*
_output_shapes
:*
dtype0
ў
 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/batch_normalization/gamma/v
Љ
4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes
:*
dtype0
ќ
Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/batch_normalization/beta/v
Ј
3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes
:*
dtype0
ї
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d/kernel/v
Ё
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:@*
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
ю
"Adam/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_1/gamma/v
Ћ
6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/v*
_output_shapes
:@*
dtype0
џ
!Adam/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_1/beta/v
Њ
5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/v*
_output_shapes
:@*
dtype0
Љ
Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ђ*'
shared_nameAdam/conv2d_1/kernel/v
і
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*'
_output_shapes
:@ђ*
dtype0
Ђ
Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*%
shared_nameAdam/conv2d_1/bias/v
z
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes	
:ђ*
dtype0
Ю
"Adam/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*3
shared_name$"Adam/batch_normalization_2/gamma/v
ќ
6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/v*
_output_shapes	
:ђ*
dtype0
Џ
!Adam/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!Adam/batch_normalization_2/beta/v
ћ
5Adam/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/v*
_output_shapes	
:ђ*
dtype0
њ
Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*'
shared_nameAdam/conv2d_2/kernel/v
І
*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*(
_output_shapes
:ђђ*
dtype0
Ђ
Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*%
shared_nameAdam/conv2d_2/bias/v
z
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes	
:ђ*
dtype0
Ю
"Adam/batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*3
shared_name$"Adam/batch_normalization_3/gamma/v
ќ
6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/v*
_output_shapes	
:ђ*
dtype0
Џ
!Adam/batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!Adam/batch_normalization_3/beta/v
ћ
5Adam/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/v*
_output_shapes	
:ђ*
dtype0
њ
Adam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*'
shared_nameAdam/conv2d_3/kernel/v
І
*Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/v*(
_output_shapes
:ђђ*
dtype0
Ђ
Adam/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*%
shared_nameAdam/conv2d_3/bias/v
z
(Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/v*
_output_shapes	
:ђ*
dtype0
Ю
"Adam/batch_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*3
shared_name$"Adam/batch_normalization_4/gamma/v
ќ
6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/v*
_output_shapes	
:ђ*
dtype0
Џ
!Adam/batch_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!Adam/batch_normalization_4/beta/v
ћ
5Adam/batch_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/v*
_output_shapes	
:ђ*
dtype0
б
Adam/conv2d_transpose/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*/
shared_name Adam/conv2d_transpose/kernel/v
Џ
2Adam/conv2d_transpose/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/kernel/v*(
_output_shapes
:ђђ*
dtype0
Љ
Adam/conv2d_transpose/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*-
shared_nameAdam/conv2d_transpose/bias/v
і
0Adam/conv2d_transpose/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/bias/v*
_output_shapes	
:ђ*
dtype0
Ю
"Adam/batch_normalization_5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*3
shared_name$"Adam/batch_normalization_5/gamma/v
ќ
6Adam/batch_normalization_5/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_5/gamma/v*
_output_shapes	
:ђ*
dtype0
Џ
!Adam/batch_normalization_5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!Adam/batch_normalization_5/beta/v
ћ
5Adam/batch_normalization_5/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_5/beta/v*
_output_shapes	
:ђ*
dtype0
д
 Adam/conv2d_transpose_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*1
shared_name" Adam/conv2d_transpose_1/kernel/v
Ъ
4Adam/conv2d_transpose_1/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_1/kernel/v*(
_output_shapes
:ђђ*
dtype0
Ћ
Adam/conv2d_transpose_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*/
shared_name Adam/conv2d_transpose_1/bias/v
ј
2Adam/conv2d_transpose_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_1/bias/v*
_output_shapes	
:ђ*
dtype0
Ю
"Adam/batch_normalization_6/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*3
shared_name$"Adam/batch_normalization_6/gamma/v
ќ
6Adam/batch_normalization_6/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_6/gamma/v*
_output_shapes	
:ђ*
dtype0
Џ
!Adam/batch_normalization_6/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!Adam/batch_normalization_6/beta/v
ћ
5Adam/batch_normalization_6/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_6/beta/v*
_output_shapes	
:ђ*
dtype0
д
 Adam/conv2d_transpose_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*1
shared_name" Adam/conv2d_transpose_2/kernel/v
Ъ
4Adam/conv2d_transpose_2/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_2/kernel/v*(
_output_shapes
:ђђ*
dtype0
Ћ
Adam/conv2d_transpose_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*/
shared_name Adam/conv2d_transpose_2/bias/v
ј
2Adam/conv2d_transpose_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_2/bias/v*
_output_shapes	
:ђ*
dtype0
Ю
"Adam/batch_normalization_7/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*3
shared_name$"Adam/batch_normalization_7/gamma/v
ќ
6Adam/batch_normalization_7/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/v*
_output_shapes	
:ђ*
dtype0
Џ
!Adam/batch_normalization_7/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!Adam/batch_normalization_7/beta/v
ћ
5Adam/batch_normalization_7/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/v*
_output_shapes	
:ђ*
dtype0
Ц
 Adam/conv2d_transpose_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ђ*1
shared_name" Adam/conv2d_transpose_3/kernel/v
ъ
4Adam/conv2d_transpose_3/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_3/kernel/v*'
_output_shapes
:@ђ*
dtype0
ћ
Adam/conv2d_transpose_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/conv2d_transpose_3/bias/v
Ї
2Adam/conv2d_transpose_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_3/bias/v*
_output_shapes
:@*
dtype0
ю
"Adam/batch_normalization_8/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_8/gamma/v
Ћ
6Adam/batch_normalization_8/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_8/gamma/v*
_output_shapes
:@*
dtype0
џ
!Adam/batch_normalization_8/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_8/beta/v
Њ
5Adam/batch_normalization_8/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_8/beta/v*
_output_shapes
:@*
dtype0
ц
 Adam/conv2d_transpose_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/conv2d_transpose_4/kernel/v
Ю
4Adam/conv2d_transpose_4/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_4/kernel/v*&
_output_shapes
:@*
dtype0
ћ
Adam/conv2d_transpose_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_4/bias/v
Ї
2Adam/conv2d_transpose_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_4/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
Ї═
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*К╠
value╝╠BИ╠ B░╠
█
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
layer_with_weights-8
layer-10
layer_with_weights-9
layer-11
layer_with_weights-10
layer-12
layer_with_weights-11
layer-13
layer_with_weights-12
layer-14
layer_with_weights-13
layer-15
layer_with_weights-14
layer-16
layer_with_weights-15
layer-17
layer_with_weights-16
layer-18
layer_with_weights-17
layer-19
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
R
regularization_losses
	variables
trainable_variables
	keras_api
Ќ
axis
	 gamma
!beta
"moving_mean
#moving_variance
$regularization_losses
%	variables
&trainable_variables
'	keras_api
h

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
Ќ
.axis
	/gamma
0beta
1moving_mean
2moving_variance
3regularization_losses
4	variables
5trainable_variables
6	keras_api
h

7kernel
8bias
9regularization_losses
:	variables
;trainable_variables
<	keras_api
Ќ
=axis
	>gamma
?beta
@moving_mean
Amoving_variance
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
h

Fkernel
Gbias
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
Ќ
Laxis
	Mgamma
Nbeta
Omoving_mean
Pmoving_variance
Qregularization_losses
R	variables
Strainable_variables
T	keras_api
h

Ukernel
Vbias
Wregularization_losses
X	variables
Ytrainable_variables
Z	keras_api
Ќ
[axis
	\gamma
]beta
^moving_mean
_moving_variance
`regularization_losses
a	variables
btrainable_variables
c	keras_api
h

dkernel
ebias
fregularization_losses
g	variables
htrainable_variables
i	keras_api
Ќ
jaxis
	kgamma
lbeta
mmoving_mean
nmoving_variance
oregularization_losses
p	variables
qtrainable_variables
r	keras_api
h

skernel
tbias
uregularization_losses
v	variables
wtrainable_variables
x	keras_api
Ў
yaxis
	zgamma
{beta
|moving_mean
}moving_variance
~regularization_losses
	variables
ђtrainable_variables
Ђ	keras_api
n
ѓkernel
	Ѓbias
ёregularization_losses
Ё	variables
єtrainable_variables
Є	keras_api
а
	ѕaxis

Ѕgamma
	іbeta
Іmoving_mean
їmoving_variance
Їregularization_losses
ј	variables
Јtrainable_variables
љ	keras_api
n
Љkernel
	њbias
Њregularization_losses
ћ	variables
Ћtrainable_variables
ќ	keras_api
а
	Ќaxis

ўgamma
	Ўbeta
џmoving_mean
Џmoving_variance
юregularization_losses
Ю	variables
ъtrainable_variables
Ъ	keras_api
n
аkernel
	Аbias
бregularization_losses
Б	variables
цtrainable_variables
Ц	keras_api
Е
	дiter
Дbeta_1
еbeta_2

Еdecay
фlearning_rate mћ!mЋ(mќ)mЌ/mў0mЎ7mџ8mЏ>mю?mЮFmъGmЪMmаNmАUmбVmБ\mц]mЦdmдemДkmеlmЕsmфtmФzmг{mГ	ѓm«	Ѓm»	Ѕm░	іm▒	Љm▓	њm│	ўm┤	Ўmх	аmХ	Аmи vИ!v╣(v║)v╗/v╝0vй7vЙ8v┐>v└?v┴Fv┬Gv├Mv─Nv┼UvкVvК\v╚]v╔dv╩ev╦kv╠lv═sv╬tv¤zvл{vЛ	ѓvм	ЃvМ	Ѕvн	іvН	Љvо	њvО	ўvп	Ўv┘	аv┌	Аv█
 
┤
 0
!1
"2
#3
(4
)5
/6
07
18
29
710
811
>12
?13
@14
A15
F16
G17
M18
N19
O20
P21
U22
V23
\24
]25
^26
_27
d28
e29
k30
l31
m32
n33
s34
t35
z36
{37
|38
}39
ѓ40
Ѓ41
Ѕ42
і43
І44
ї45
Љ46
њ47
ў48
Ў49
џ50
Џ51
а52
А53
а
 0
!1
(2
)3
/4
05
76
87
>8
?9
F10
G11
M12
N13
U14
V15
\16
]17
d18
e19
k20
l21
s22
t23
z24
{25
ѓ26
Ѓ27
Ѕ28
і29
Љ30
њ31
ў32
Ў33
а34
А35
▓
regularization_losses
	variables
trainable_variables
Фnon_trainable_variables
гlayers
Гlayer_metrics
«metrics
 »layer_regularization_losses
 
 
 
 
▓
regularization_losses
	variables
trainable_variables
░non_trainable_variables
▒layers
▓layer_metrics
│metrics
 ┤layer_regularization_losses
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

 0
!1
"2
#3

 0
!1
▓
$regularization_losses
%	variables
&trainable_variables
хnon_trainable_variables
Хlayers
иlayer_metrics
Иmetrics
 ╣layer_regularization_losses
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

(0
)1

(0
)1
▓
*regularization_losses
+	variables
,trainable_variables
║non_trainable_variables
╗layers
╝layer_metrics
йmetrics
 Йlayer_regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

/0
01
12
23

/0
01
▓
3regularization_losses
4	variables
5trainable_variables
┐non_trainable_variables
└layers
┴layer_metrics
┬metrics
 ├layer_regularization_losses
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

70
81

70
81
▓
9regularization_losses
:	variables
;trainable_variables
─non_trainable_variables
┼layers
кlayer_metrics
Кmetrics
 ╚layer_regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

>0
?1
@2
A3

>0
?1
▓
Bregularization_losses
C	variables
Dtrainable_variables
╔non_trainable_variables
╩layers
╦layer_metrics
╠metrics
 ═layer_regularization_losses
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

F0
G1

F0
G1
▓
Hregularization_losses
I	variables
Jtrainable_variables
╬non_trainable_variables
¤layers
лlayer_metrics
Лmetrics
 мlayer_regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

M0
N1
O2
P3

M0
N1
▓
Qregularization_losses
R	variables
Strainable_variables
Мnon_trainable_variables
нlayers
Нlayer_metrics
оmetrics
 Оlayer_regularization_losses
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

U0
V1

U0
V1
▓
Wregularization_losses
X	variables
Ytrainable_variables
пnon_trainable_variables
┘layers
┌layer_metrics
█metrics
 ▄layer_regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

\0
]1
^2
_3

\0
]1
▓
`regularization_losses
a	variables
btrainable_variables
Пnon_trainable_variables
яlayers
▀layer_metrics
Яmetrics
 рlayer_regularization_losses
ca
VARIABLE_VALUEconv2d_transpose/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_transpose/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

d0
e1

d0
e1
▓
fregularization_losses
g	variables
htrainable_variables
Рnon_trainable_variables
сlayers
Сlayer_metrics
тmetrics
 Тlayer_regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_5/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_5/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_5/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_5/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

k0
l1
m2
n3

k0
l1
▓
oregularization_losses
p	variables
qtrainable_variables
уnon_trainable_variables
Уlayers
жlayer_metrics
Жmetrics
 вlayer_regularization_losses
fd
VARIABLE_VALUEconv2d_transpose_1/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_1/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE
 

s0
t1

s0
t1
▓
uregularization_losses
v	variables
wtrainable_variables
Вnon_trainable_variables
ьlayers
Ьlayer_metrics
№metrics
 ­layer_regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_6/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_6/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_6/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_6/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

z0
{1
|2
}3

z0
{1
│
~regularization_losses
	variables
ђtrainable_variables
ыnon_trainable_variables
Ыlayers
зlayer_metrics
Зmetrics
 шlayer_regularization_losses
fd
VARIABLE_VALUEconv2d_transpose_2/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_2/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE
 

ѓ0
Ѓ1

ѓ0
Ѓ1
х
ёregularization_losses
Ё	variables
єtrainable_variables
Шnon_trainable_variables
эlayers
Эlayer_metrics
щmetrics
 Щlayer_regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_7/gamma6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_7/beta5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_7/moving_mean<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_7/moving_variance@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 
Ѕ0
і1
І2
ї3

Ѕ0
і1
х
Їregularization_losses
ј	variables
Јtrainable_variables
чnon_trainable_variables
Чlayers
§layer_metrics
■metrics
  layer_regularization_losses
fd
VARIABLE_VALUEconv2d_transpose_3/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_3/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Љ0
њ1

Љ0
њ1
х
Њregularization_losses
ћ	variables
Ћtrainable_variables
ђnon_trainable_variables
Ђlayers
ѓlayer_metrics
Ѓmetrics
 ёlayer_regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_8/gamma6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_8/beta5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_8/moving_mean<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_8/moving_variance@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 
ў0
Ў1
џ2
Џ3

ў0
Ў1
х
юregularization_losses
Ю	variables
ъtrainable_variables
Ёnon_trainable_variables
єlayers
Єlayer_metrics
ѕmetrics
 Ѕlayer_regularization_losses
fd
VARIABLE_VALUEconv2d_transpose_4/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_4/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE
 

а0
А1

а0
А1
х
бregularization_losses
Б	variables
цtrainable_variables
іnon_trainable_variables
Іlayers
їlayer_metrics
Їmetrics
 јlayer_regularization_losses
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
і
"0
#1
12
23
@4
A5
O6
P7
^8
_9
m10
n11
|12
}13
І14
ї15
џ16
Џ17
ќ
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
 

Ј0
 
 
 
 
 
 

"0
#1
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
10
21
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
@0
A1
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
O0
P1
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
^0
_1
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
m0
n1
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
|0
}1
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
І0
ї1
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
џ0
Џ1
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

љtotal

Љcount
њ	variables
Њ	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

љ0
Љ1

њ	variables
ѕЁ
VARIABLE_VALUE Adam/batch_normalization/gamma/mQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
єЃ
VARIABLE_VALUEAdam/batch_normalization/beta/mPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ѕЁ
VARIABLE_VALUE!Adam/batch_normalization_1/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ѕЁ
VARIABLE_VALUE!Adam/batch_normalization_2/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ѕЁ
VARIABLE_VALUE!Adam/batch_normalization_3/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUE"Adam/batch_normalization_4/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ѕЁ
VARIABLE_VALUE!Adam/batch_normalization_4/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Єё
VARIABLE_VALUEAdam/conv2d_transpose/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ѓђ
VARIABLE_VALUEAdam/conv2d_transpose/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Іѕ
VARIABLE_VALUE"Adam/batch_normalization_5/gamma/mRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ѕє
VARIABLE_VALUE!Adam/batch_normalization_5/beta/mQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUE Adam/conv2d_transpose_1/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
єЃ
VARIABLE_VALUEAdam/conv2d_transpose_1/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Іѕ
VARIABLE_VALUE"Adam/batch_normalization_6/gamma/mRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ѕє
VARIABLE_VALUE!Adam/batch_normalization_6/beta/mQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUE Adam/conv2d_transpose_2/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
єЃ
VARIABLE_VALUEAdam/conv2d_transpose_2/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Іѕ
VARIABLE_VALUE"Adam/batch_normalization_7/gamma/mRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ѕє
VARIABLE_VALUE!Adam/batch_normalization_7/beta/mQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUE Adam/conv2d_transpose_3/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
єЃ
VARIABLE_VALUEAdam/conv2d_transpose_3/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Іѕ
VARIABLE_VALUE"Adam/batch_normalization_8/gamma/mRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ѕє
VARIABLE_VALUE!Adam/batch_normalization_8/beta/mQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUE Adam/conv2d_transpose_4/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
єЃ
VARIABLE_VALUEAdam/conv2d_transpose_4/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ѕЁ
VARIABLE_VALUE Adam/batch_normalization/gamma/vQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
єЃ
VARIABLE_VALUEAdam/batch_normalization/beta/vPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ѕЁ
VARIABLE_VALUE!Adam/batch_normalization_1/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ѕЁ
VARIABLE_VALUE!Adam/batch_normalization_2/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ѕЁ
VARIABLE_VALUE!Adam/batch_normalization_3/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUE"Adam/batch_normalization_4/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ѕЁ
VARIABLE_VALUE!Adam/batch_normalization_4/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Єё
VARIABLE_VALUEAdam/conv2d_transpose/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ѓђ
VARIABLE_VALUEAdam/conv2d_transpose/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Іѕ
VARIABLE_VALUE"Adam/batch_normalization_5/gamma/vRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ѕє
VARIABLE_VALUE!Adam/batch_normalization_5/beta/vQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUE Adam/conv2d_transpose_1/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
єЃ
VARIABLE_VALUEAdam/conv2d_transpose_1/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Іѕ
VARIABLE_VALUE"Adam/batch_normalization_6/gamma/vRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ѕє
VARIABLE_VALUE!Adam/batch_normalization_6/beta/vQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUE Adam/conv2d_transpose_2/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
єЃ
VARIABLE_VALUEAdam/conv2d_transpose_2/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Іѕ
VARIABLE_VALUE"Adam/batch_normalization_7/gamma/vRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ѕє
VARIABLE_VALUE!Adam/batch_normalization_7/beta/vQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUE Adam/conv2d_transpose_3/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
єЃ
VARIABLE_VALUEAdam/conv2d_transpose_3/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Іѕ
VARIABLE_VALUE"Adam/batch_normalization_8/gamma/vRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ѕє
VARIABLE_VALUE!Adam/batch_normalization_8/beta/vQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUE Adam/conv2d_transpose_4/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
єЃ
VARIABLE_VALUEAdam/conv2d_transpose_4/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ё
serving_default_u_velPlaceholder*-
_output_shapes
:         ђђ*
dtype0*"
shape:         ђђ
И
StatefulPartitionedCallStatefulPartitionedCallserving_default_u_velbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d/kernelconv2d/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_3/kernelconv2d_3/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceconv2d_transpose/kernelconv2d_transpose/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_transpose_1/kernelconv2d_transpose_1/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv2d_transpose_2/kernelconv2d_transpose_2/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_transpose_3/kernelconv2d_transpose_3/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv2d_transpose_4/kernelconv2d_transpose_4/bias*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456*0
config_proto 

CPU

GPU2*0J 8ѓ *-
f(R&
$__inference_signature_wrapper_163776
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
О6
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp+conv2d_transpose/kernel/Read/ReadVariableOp)conv2d_transpose/bias/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp-conv2d_transpose_1/kernel/Read/ReadVariableOp+conv2d_transpose_1/bias/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp-conv2d_transpose_2/kernel/Read/ReadVariableOp+conv2d_transpose_2/bias/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp-conv2d_transpose_3/kernel/Read/ReadVariableOp+conv2d_transpose_3/bias/Read/ReadVariableOp/batch_normalization_8/gamma/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOp-conv2d_transpose_4/kernel/Read/ReadVariableOp+conv2d_transpose_4/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_1/beta/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_2/beta/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_3/beta/m/Read/ReadVariableOp*Adam/conv2d_3/kernel/m/Read/ReadVariableOp(Adam/conv2d_3/bias/m/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_4/beta/m/Read/ReadVariableOp2Adam/conv2d_transpose/kernel/m/Read/ReadVariableOp0Adam/conv2d_transpose/bias/m/Read/ReadVariableOp6Adam/batch_normalization_5/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_5/beta/m/Read/ReadVariableOp4Adam/conv2d_transpose_1/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose_1/bias/m/Read/ReadVariableOp6Adam/batch_normalization_6/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_6/beta/m/Read/ReadVariableOp4Adam/conv2d_transpose_2/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose_2/bias/m/Read/ReadVariableOp6Adam/batch_normalization_7/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_7/beta/m/Read/ReadVariableOp4Adam/conv2d_transpose_3/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose_3/bias/m/Read/ReadVariableOp6Adam/batch_normalization_8/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_8/beta/m/Read/ReadVariableOp4Adam/conv2d_transpose_4/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose_4/bias/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_1/beta/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_2/beta/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_3/beta/v/Read/ReadVariableOp*Adam/conv2d_3/kernel/v/Read/ReadVariableOp(Adam/conv2d_3/bias/v/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_4/beta/v/Read/ReadVariableOp2Adam/conv2d_transpose/kernel/v/Read/ReadVariableOp0Adam/conv2d_transpose/bias/v/Read/ReadVariableOp6Adam/batch_normalization_5/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_5/beta/v/Read/ReadVariableOp4Adam/conv2d_transpose_1/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose_1/bias/v/Read/ReadVariableOp6Adam/batch_normalization_6/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_6/beta/v/Read/ReadVariableOp4Adam/conv2d_transpose_2/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose_2/bias/v/Read/ReadVariableOp6Adam/batch_normalization_7/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_7/beta/v/Read/ReadVariableOp4Adam/conv2d_transpose_3/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose_3/bias/v/Read/ReadVariableOp6Adam/batch_normalization_8/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_8/beta/v/Read/ReadVariableOp4Adam/conv2d_transpose_4/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose_4/bias/v/Read/ReadVariableOpConst*Ћ
TinЇ
і2Є	*
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
GPU2*0J 8ѓ *(
f#R!
__inference__traced_save_166571
Ь!
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d/kernelconv2d/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_3/kernelconv2d_3/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceconv2d_transpose/kernelconv2d_transpose/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_transpose_1/kernelconv2d_transpose_1/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv2d_transpose_2/kernelconv2d_transpose_2/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_transpose_3/kernelconv2d_transpose_3/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv2d_transpose_4/kernelconv2d_transpose_4/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/mAdam/conv2d/kernel/mAdam/conv2d/bias/m"Adam/batch_normalization_1/gamma/m!Adam/batch_normalization_1/beta/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/m"Adam/batch_normalization_2/gamma/m!Adam/batch_normalization_2/beta/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/m"Adam/batch_normalization_3/gamma/m!Adam/batch_normalization_3/beta/mAdam/conv2d_3/kernel/mAdam/conv2d_3/bias/m"Adam/batch_normalization_4/gamma/m!Adam/batch_normalization_4/beta/mAdam/conv2d_transpose/kernel/mAdam/conv2d_transpose/bias/m"Adam/batch_normalization_5/gamma/m!Adam/batch_normalization_5/beta/m Adam/conv2d_transpose_1/kernel/mAdam/conv2d_transpose_1/bias/m"Adam/batch_normalization_6/gamma/m!Adam/batch_normalization_6/beta/m Adam/conv2d_transpose_2/kernel/mAdam/conv2d_transpose_2/bias/m"Adam/batch_normalization_7/gamma/m!Adam/batch_normalization_7/beta/m Adam/conv2d_transpose_3/kernel/mAdam/conv2d_transpose_3/bias/m"Adam/batch_normalization_8/gamma/m!Adam/batch_normalization_8/beta/m Adam/conv2d_transpose_4/kernel/mAdam/conv2d_transpose_4/bias/m Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/vAdam/conv2d/kernel/vAdam/conv2d/bias/v"Adam/batch_normalization_1/gamma/v!Adam/batch_normalization_1/beta/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/v"Adam/batch_normalization_2/gamma/v!Adam/batch_normalization_2/beta/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/v"Adam/batch_normalization_3/gamma/v!Adam/batch_normalization_3/beta/vAdam/conv2d_3/kernel/vAdam/conv2d_3/bias/v"Adam/batch_normalization_4/gamma/v!Adam/batch_normalization_4/beta/vAdam/conv2d_transpose/kernel/vAdam/conv2d_transpose/bias/v"Adam/batch_normalization_5/gamma/v!Adam/batch_normalization_5/beta/v Adam/conv2d_transpose_1/kernel/vAdam/conv2d_transpose_1/bias/v"Adam/batch_normalization_6/gamma/v!Adam/batch_normalization_6/beta/v Adam/conv2d_transpose_2/kernel/vAdam/conv2d_transpose_2/bias/v"Adam/batch_normalization_7/gamma/v!Adam/batch_normalization_7/beta/v Adam/conv2d_transpose_3/kernel/vAdam/conv2d_transpose_3/bias/v"Adam/batch_normalization_8/gamma/v!Adam/batch_normalization_8/beta/v Adam/conv2d_transpose_4/kernel/vAdam/conv2d_transpose_4/bias/v*ћ
Tinї
Ѕ2є*
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
GPU2*0J 8ѓ *+
f&R$
"__inference__traced_restore_166980АЪ*
├	
Л
6__inference_batch_normalization_1_layer_call_fn_164712

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1603822
StatefulPartitionedCallЋ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
шн
█<
!__inference__wrapped_model_160234	
u_velF
8cnn_baseline_batch_normalization_readvariableop_resource:H
:cnn_baseline_batch_normalization_readvariableop_1_resource:W
Icnn_baseline_batch_normalization_fusedbatchnormv3_readvariableop_resource:Y
Kcnn_baseline_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:L
2cnn_baseline_conv2d_conv2d_readvariableop_resource:@A
3cnn_baseline_conv2d_biasadd_readvariableop_resource:@H
:cnn_baseline_batch_normalization_1_readvariableop_resource:@J
<cnn_baseline_batch_normalization_1_readvariableop_1_resource:@Y
Kcnn_baseline_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@[
Mcnn_baseline_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@O
4cnn_baseline_conv2d_1_conv2d_readvariableop_resource:@ђD
5cnn_baseline_conv2d_1_biasadd_readvariableop_resource:	ђI
:cnn_baseline_batch_normalization_2_readvariableop_resource:	ђK
<cnn_baseline_batch_normalization_2_readvariableop_1_resource:	ђZ
Kcnn_baseline_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	ђ\
Mcnn_baseline_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	ђP
4cnn_baseline_conv2d_2_conv2d_readvariableop_resource:ђђD
5cnn_baseline_conv2d_2_biasadd_readvariableop_resource:	ђI
:cnn_baseline_batch_normalization_3_readvariableop_resource:	ђK
<cnn_baseline_batch_normalization_3_readvariableop_1_resource:	ђZ
Kcnn_baseline_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	ђ\
Mcnn_baseline_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	ђP
4cnn_baseline_conv2d_3_conv2d_readvariableop_resource:ђђD
5cnn_baseline_conv2d_3_biasadd_readvariableop_resource:	ђI
:cnn_baseline_batch_normalization_4_readvariableop_resource:	ђK
<cnn_baseline_batch_normalization_4_readvariableop_1_resource:	ђZ
Kcnn_baseline_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	ђ\
Mcnn_baseline_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	ђb
Fcnn_baseline_conv2d_transpose_conv2d_transpose_readvariableop_resource:ђђL
=cnn_baseline_conv2d_transpose_biasadd_readvariableop_resource:	ђI
:cnn_baseline_batch_normalization_5_readvariableop_resource:	ђK
<cnn_baseline_batch_normalization_5_readvariableop_1_resource:	ђZ
Kcnn_baseline_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:	ђ\
Mcnn_baseline_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:	ђd
Hcnn_baseline_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:ђђN
?cnn_baseline_conv2d_transpose_1_biasadd_readvariableop_resource:	ђI
:cnn_baseline_batch_normalization_6_readvariableop_resource:	ђK
<cnn_baseline_batch_normalization_6_readvariableop_1_resource:	ђZ
Kcnn_baseline_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:	ђ\
Mcnn_baseline_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:	ђd
Hcnn_baseline_conv2d_transpose_2_conv2d_transpose_readvariableop_resource:ђђN
?cnn_baseline_conv2d_transpose_2_biasadd_readvariableop_resource:	ђI
:cnn_baseline_batch_normalization_7_readvariableop_resource:	ђK
<cnn_baseline_batch_normalization_7_readvariableop_1_resource:	ђZ
Kcnn_baseline_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:	ђ\
Mcnn_baseline_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:	ђc
Hcnn_baseline_conv2d_transpose_3_conv2d_transpose_readvariableop_resource:@ђM
?cnn_baseline_conv2d_transpose_3_biasadd_readvariableop_resource:@H
:cnn_baseline_batch_normalization_8_readvariableop_resource:@J
<cnn_baseline_batch_normalization_8_readvariableop_1_resource:@Y
Kcnn_baseline_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:@[
Mcnn_baseline_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:@b
Hcnn_baseline_conv2d_transpose_4_conv2d_transpose_readvariableop_resource:@M
?cnn_baseline_conv2d_transpose_4_biasadd_readvariableop_resource:
identityѕб@CNN_baseline/batch_normalization/FusedBatchNormV3/ReadVariableOpбBCNN_baseline/batch_normalization/FusedBatchNormV3/ReadVariableOp_1б/CNN_baseline/batch_normalization/ReadVariableOpб1CNN_baseline/batch_normalization/ReadVariableOp_1бBCNN_baseline/batch_normalization_1/FusedBatchNormV3/ReadVariableOpбDCNN_baseline/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1б1CNN_baseline/batch_normalization_1/ReadVariableOpб3CNN_baseline/batch_normalization_1/ReadVariableOp_1бBCNN_baseline/batch_normalization_2/FusedBatchNormV3/ReadVariableOpбDCNN_baseline/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1б1CNN_baseline/batch_normalization_2/ReadVariableOpб3CNN_baseline/batch_normalization_2/ReadVariableOp_1бBCNN_baseline/batch_normalization_3/FusedBatchNormV3/ReadVariableOpбDCNN_baseline/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1б1CNN_baseline/batch_normalization_3/ReadVariableOpб3CNN_baseline/batch_normalization_3/ReadVariableOp_1бBCNN_baseline/batch_normalization_4/FusedBatchNormV3/ReadVariableOpбDCNN_baseline/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б1CNN_baseline/batch_normalization_4/ReadVariableOpб3CNN_baseline/batch_normalization_4/ReadVariableOp_1бBCNN_baseline/batch_normalization_5/FusedBatchNormV3/ReadVariableOpбDCNN_baseline/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б1CNN_baseline/batch_normalization_5/ReadVariableOpб3CNN_baseline/batch_normalization_5/ReadVariableOp_1бBCNN_baseline/batch_normalization_6/FusedBatchNormV3/ReadVariableOpбDCNN_baseline/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1б1CNN_baseline/batch_normalization_6/ReadVariableOpб3CNN_baseline/batch_normalization_6/ReadVariableOp_1бBCNN_baseline/batch_normalization_7/FusedBatchNormV3/ReadVariableOpбDCNN_baseline/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1б1CNN_baseline/batch_normalization_7/ReadVariableOpб3CNN_baseline/batch_normalization_7/ReadVariableOp_1бBCNN_baseline/batch_normalization_8/FusedBatchNormV3/ReadVariableOpбDCNN_baseline/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1б1CNN_baseline/batch_normalization_8/ReadVariableOpб3CNN_baseline/batch_normalization_8/ReadVariableOp_1б*CNN_baseline/conv2d/BiasAdd/ReadVariableOpб)CNN_baseline/conv2d/Conv2D/ReadVariableOpб,CNN_baseline/conv2d_1/BiasAdd/ReadVariableOpб+CNN_baseline/conv2d_1/Conv2D/ReadVariableOpб,CNN_baseline/conv2d_2/BiasAdd/ReadVariableOpб+CNN_baseline/conv2d_2/Conv2D/ReadVariableOpб,CNN_baseline/conv2d_3/BiasAdd/ReadVariableOpб+CNN_baseline/conv2d_3/Conv2D/ReadVariableOpб4CNN_baseline/conv2d_transpose/BiasAdd/ReadVariableOpб=CNN_baseline/conv2d_transpose/conv2d_transpose/ReadVariableOpб6CNN_baseline/conv2d_transpose_1/BiasAdd/ReadVariableOpб?CNN_baseline/conv2d_transpose_1/conv2d_transpose/ReadVariableOpб6CNN_baseline/conv2d_transpose_2/BiasAdd/ReadVariableOpб?CNN_baseline/conv2d_transpose_2/conv2d_transpose/ReadVariableOpб6CNN_baseline/conv2d_transpose_3/BiasAdd/ReadVariableOpб?CNN_baseline/conv2d_transpose_3/conv2d_transpose/ReadVariableOpб6CNN_baseline/conv2d_transpose_4/BiasAdd/ReadVariableOpб?CNN_baseline/conv2d_transpose_4/conv2d_transpose/ReadVariableOpm
CNN_baseline/reshape/ShapeShapeu_vel*
T0*
_output_shapes
:2
CNN_baseline/reshape/Shapeъ
(CNN_baseline/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(CNN_baseline/reshape/strided_slice/stackб
*CNN_baseline/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*CNN_baseline/reshape/strided_slice/stack_1б
*CNN_baseline/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*CNN_baseline/reshape/strided_slice/stack_2Я
"CNN_baseline/reshape/strided_sliceStridedSlice#CNN_baseline/reshape/Shape:output:01CNN_baseline/reshape/strided_slice/stack:output:03CNN_baseline/reshape/strided_slice/stack_1:output:03CNN_baseline/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"CNN_baseline/reshape/strided_sliceЈ
$CNN_baseline/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :ђ2&
$CNN_baseline/reshape/Reshape/shape/1Ј
$CNN_baseline/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ђ2&
$CNN_baseline/reshape/Reshape/shape/2ј
$CNN_baseline/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2&
$CNN_baseline/reshape/Reshape/shape/3И
"CNN_baseline/reshape/Reshape/shapePack+CNN_baseline/reshape/strided_slice:output:0-CNN_baseline/reshape/Reshape/shape/1:output:0-CNN_baseline/reshape/Reshape/shape/2:output:0-CNN_baseline/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2$
"CNN_baseline/reshape/Reshape/shapeи
CNN_baseline/reshape/ReshapeReshapeu_vel+CNN_baseline/reshape/Reshape/shape:output:0*
T0*1
_output_shapes
:         ђђ2
CNN_baseline/reshape/ReshapeО
/CNN_baseline/batch_normalization/ReadVariableOpReadVariableOp8cnn_baseline_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype021
/CNN_baseline/batch_normalization/ReadVariableOpП
1CNN_baseline/batch_normalization/ReadVariableOp_1ReadVariableOp:cnn_baseline_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype023
1CNN_baseline/batch_normalization/ReadVariableOp_1і
@CNN_baseline/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpIcnn_baseline_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02B
@CNN_baseline/batch_normalization/FusedBatchNormV3/ReadVariableOpљ
BCNN_baseline/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKcnn_baseline_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02D
BCNN_baseline/batch_normalization/FusedBatchNormV3/ReadVariableOp_1▒
1CNN_baseline/batch_normalization/FusedBatchNormV3FusedBatchNormV3%CNN_baseline/reshape/Reshape:output:07CNN_baseline/batch_normalization/ReadVariableOp:value:09CNN_baseline/batch_normalization/ReadVariableOp_1:value:0HCNN_baseline/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0JCNN_baseline/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ:::::*
epsilon%oЃ:*
is_training( 23
1CNN_baseline/batch_normalization/FusedBatchNormV3Л
)CNN_baseline/conv2d/Conv2D/ReadVariableOpReadVariableOp2cnn_baseline_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02+
)CNN_baseline/conv2d/Conv2D/ReadVariableOpЉ
CNN_baseline/conv2d/Conv2DConv2D5CNN_baseline/batch_normalization/FusedBatchNormV3:y:01CNN_baseline/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЧЧ@*
paddingVALID*
strides
2
CNN_baseline/conv2d/Conv2D╚
*CNN_baseline/conv2d/BiasAdd/ReadVariableOpReadVariableOp3cnn_baseline_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*CNN_baseline/conv2d/BiasAdd/ReadVariableOp┌
CNN_baseline/conv2d/BiasAddBiasAdd#CNN_baseline/conv2d/Conv2D:output:02CNN_baseline/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЧЧ@2
CNN_baseline/conv2d/BiasAddЏ
CNN_baseline/conv2d/EluElu$CNN_baseline/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:         ЧЧ@2
CNN_baseline/conv2d/EluП
1CNN_baseline/batch_normalization_1/ReadVariableOpReadVariableOp:cnn_baseline_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype023
1CNN_baseline/batch_normalization_1/ReadVariableOpс
3CNN_baseline/batch_normalization_1/ReadVariableOp_1ReadVariableOp<cnn_baseline_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3CNN_baseline/batch_normalization_1/ReadVariableOp_1љ
BCNN_baseline/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpKcnn_baseline_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02D
BCNN_baseline/batch_normalization_1/FusedBatchNormV3/ReadVariableOpќ
DCNN_baseline/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMcnn_baseline_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02F
DCNN_baseline/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1й
3CNN_baseline/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3%CNN_baseline/conv2d/Elu:activations:09CNN_baseline/batch_normalization_1/ReadVariableOp:value:0;CNN_baseline/batch_normalization_1/ReadVariableOp_1:value:0JCNN_baseline/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0LCNN_baseline/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЧЧ@:@:@:@:@:*
epsilon%oЃ:*
is_training( 25
3CNN_baseline/batch_normalization_1/FusedBatchNormV3п
+CNN_baseline/conv2d_1/Conv2D/ReadVariableOpReadVariableOp4cnn_baseline_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02-
+CNN_baseline/conv2d_1/Conv2D/ReadVariableOpџ
CNN_baseline/conv2d_1/Conv2DConv2D7CNN_baseline/batch_normalization_1/FusedBatchNormV3:y:03CNN_baseline/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЩЩђ*
paddingVALID*
strides
2
CNN_baseline/conv2d_1/Conv2D¤
,CNN_baseline/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp5cnn_baseline_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,CNN_baseline/conv2d_1/BiasAdd/ReadVariableOpс
CNN_baseline/conv2d_1/BiasAddBiasAdd%CNN_baseline/conv2d_1/Conv2D:output:04CNN_baseline/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЩЩђ2
CNN_baseline/conv2d_1/BiasAddб
CNN_baseline/conv2d_1/EluElu&CNN_baseline/conv2d_1/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЩЩђ2
CNN_baseline/conv2d_1/Eluя
1CNN_baseline/batch_normalization_2/ReadVariableOpReadVariableOp:cnn_baseline_batch_normalization_2_readvariableop_resource*
_output_shapes	
:ђ*
dtype023
1CNN_baseline/batch_normalization_2/ReadVariableOpС
3CNN_baseline/batch_normalization_2/ReadVariableOp_1ReadVariableOp<cnn_baseline_batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype025
3CNN_baseline/batch_normalization_2/ReadVariableOp_1Љ
BCNN_baseline/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpKcnn_baseline_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02D
BCNN_baseline/batch_normalization_2/FusedBatchNormV3/ReadVariableOpЌ
DCNN_baseline/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMcnn_baseline_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02F
DCNN_baseline/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1─
3CNN_baseline/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3'CNN_baseline/conv2d_1/Elu:activations:09CNN_baseline/batch_normalization_2/ReadVariableOp:value:0;CNN_baseline/batch_normalization_2/ReadVariableOp_1:value:0JCNN_baseline/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0LCNN_baseline/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЩЩђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 25
3CNN_baseline/batch_normalization_2/FusedBatchNormV3┘
+CNN_baseline/conv2d_2/Conv2D/ReadVariableOpReadVariableOp4cnn_baseline_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02-
+CNN_baseline/conv2d_2/Conv2D/ReadVariableOpџ
CNN_baseline/conv2d_2/Conv2DConv2D7CNN_baseline/batch_normalization_2/FusedBatchNormV3:y:03CNN_baseline/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЭЭђ*
paddingVALID*
strides
2
CNN_baseline/conv2d_2/Conv2D¤
,CNN_baseline/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp5cnn_baseline_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,CNN_baseline/conv2d_2/BiasAdd/ReadVariableOpс
CNN_baseline/conv2d_2/BiasAddBiasAdd%CNN_baseline/conv2d_2/Conv2D:output:04CNN_baseline/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЭЭђ2
CNN_baseline/conv2d_2/BiasAddб
CNN_baseline/conv2d_2/EluElu&CNN_baseline/conv2d_2/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЭЭђ2
CNN_baseline/conv2d_2/Eluя
1CNN_baseline/batch_normalization_3/ReadVariableOpReadVariableOp:cnn_baseline_batch_normalization_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype023
1CNN_baseline/batch_normalization_3/ReadVariableOpС
3CNN_baseline/batch_normalization_3/ReadVariableOp_1ReadVariableOp<cnn_baseline_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype025
3CNN_baseline/batch_normalization_3/ReadVariableOp_1Љ
BCNN_baseline/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpKcnn_baseline_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02D
BCNN_baseline/batch_normalization_3/FusedBatchNormV3/ReadVariableOpЌ
DCNN_baseline/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMcnn_baseline_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02F
DCNN_baseline/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1─
3CNN_baseline/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3'CNN_baseline/conv2d_2/Elu:activations:09CNN_baseline/batch_normalization_3/ReadVariableOp:value:0;CNN_baseline/batch_normalization_3/ReadVariableOp_1:value:0JCNN_baseline/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0LCNN_baseline/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЭЭђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 25
3CNN_baseline/batch_normalization_3/FusedBatchNormV3┘
+CNN_baseline/conv2d_3/Conv2D/ReadVariableOpReadVariableOp4cnn_baseline_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02-
+CNN_baseline/conv2d_3/Conv2D/ReadVariableOpџ
CNN_baseline/conv2d_3/Conv2DConv2D7CNN_baseline/batch_normalization_3/FusedBatchNormV3:y:03CNN_baseline/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ШШђ*
paddingVALID*
strides
2
CNN_baseline/conv2d_3/Conv2D¤
,CNN_baseline/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp5cnn_baseline_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,CNN_baseline/conv2d_3/BiasAdd/ReadVariableOpс
CNN_baseline/conv2d_3/BiasAddBiasAdd%CNN_baseline/conv2d_3/Conv2D:output:04CNN_baseline/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ШШђ2
CNN_baseline/conv2d_3/BiasAddб
CNN_baseline/conv2d_3/EluElu&CNN_baseline/conv2d_3/BiasAdd:output:0*
T0*2
_output_shapes 
:         ШШђ2
CNN_baseline/conv2d_3/Eluя
1CNN_baseline/batch_normalization_4/ReadVariableOpReadVariableOp:cnn_baseline_batch_normalization_4_readvariableop_resource*
_output_shapes	
:ђ*
dtype023
1CNN_baseline/batch_normalization_4/ReadVariableOpС
3CNN_baseline/batch_normalization_4/ReadVariableOp_1ReadVariableOp<cnn_baseline_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype025
3CNN_baseline/batch_normalization_4/ReadVariableOp_1Љ
BCNN_baseline/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpKcnn_baseline_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02D
BCNN_baseline/batch_normalization_4/FusedBatchNormV3/ReadVariableOpЌ
DCNN_baseline/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMcnn_baseline_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02F
DCNN_baseline/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1─
3CNN_baseline/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3'CNN_baseline/conv2d_3/Elu:activations:09CNN_baseline/batch_normalization_4/ReadVariableOp:value:0;CNN_baseline/batch_normalization_4/ReadVariableOp_1:value:0JCNN_baseline/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0LCNN_baseline/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ШШђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 25
3CNN_baseline/batch_normalization_4/FusedBatchNormV3▒
#CNN_baseline/conv2d_transpose/ShapeShape7CNN_baseline/batch_normalization_4/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2%
#CNN_baseline/conv2d_transpose/Shape░
1CNN_baseline/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1CNN_baseline/conv2d_transpose/strided_slice/stack┤
3CNN_baseline/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3CNN_baseline/conv2d_transpose/strided_slice/stack_1┤
3CNN_baseline/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3CNN_baseline/conv2d_transpose/strided_slice/stack_2ќ
+CNN_baseline/conv2d_transpose/strided_sliceStridedSlice,CNN_baseline/conv2d_transpose/Shape:output:0:CNN_baseline/conv2d_transpose/strided_slice/stack:output:0<CNN_baseline/conv2d_transpose/strided_slice/stack_1:output:0<CNN_baseline/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+CNN_baseline/conv2d_transpose/strided_sliceЉ
%CNN_baseline/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value
B :Э2'
%CNN_baseline/conv2d_transpose/stack/1Љ
%CNN_baseline/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :Э2'
%CNN_baseline/conv2d_transpose/stack/2Љ
%CNN_baseline/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2'
%CNN_baseline/conv2d_transpose/stack/3к
#CNN_baseline/conv2d_transpose/stackPack4CNN_baseline/conv2d_transpose/strided_slice:output:0.CNN_baseline/conv2d_transpose/stack/1:output:0.CNN_baseline/conv2d_transpose/stack/2:output:0.CNN_baseline/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2%
#CNN_baseline/conv2d_transpose/stack┤
3CNN_baseline/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3CNN_baseline/conv2d_transpose/strided_slice_1/stackИ
5CNN_baseline/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5CNN_baseline/conv2d_transpose/strided_slice_1/stack_1И
5CNN_baseline/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5CNN_baseline/conv2d_transpose/strided_slice_1/stack_2а
-CNN_baseline/conv2d_transpose/strided_slice_1StridedSlice,CNN_baseline/conv2d_transpose/stack:output:0<CNN_baseline/conv2d_transpose/strided_slice_1/stack:output:0>CNN_baseline/conv2d_transpose/strided_slice_1/stack_1:output:0>CNN_baseline/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-CNN_baseline/conv2d_transpose/strided_slice_1Ј
=CNN_baseline/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpFcnn_baseline_conv2d_transpose_conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02?
=CNN_baseline/conv2d_transpose/conv2d_transpose/ReadVariableOpІ
.CNN_baseline/conv2d_transpose/conv2d_transposeConv2DBackpropInput,CNN_baseline/conv2d_transpose/stack:output:0ECNN_baseline/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:07CNN_baseline/batch_normalization_4/FusedBatchNormV3:y:0*
T0*2
_output_shapes 
:         ЭЭђ*
paddingVALID*
strides
20
.CNN_baseline/conv2d_transpose/conv2d_transposeу
4CNN_baseline/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp=cnn_baseline_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype026
4CNN_baseline/conv2d_transpose/BiasAdd/ReadVariableOpЇ
%CNN_baseline/conv2d_transpose/BiasAddBiasAdd7CNN_baseline/conv2d_transpose/conv2d_transpose:output:0<CNN_baseline/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЭЭђ2'
%CNN_baseline/conv2d_transpose/BiasAdd║
!CNN_baseline/conv2d_transpose/EluElu.CNN_baseline/conv2d_transpose/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЭЭђ2#
!CNN_baseline/conv2d_transpose/Eluя
1CNN_baseline/batch_normalization_5/ReadVariableOpReadVariableOp:cnn_baseline_batch_normalization_5_readvariableop_resource*
_output_shapes	
:ђ*
dtype023
1CNN_baseline/batch_normalization_5/ReadVariableOpС
3CNN_baseline/batch_normalization_5/ReadVariableOp_1ReadVariableOp<cnn_baseline_batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype025
3CNN_baseline/batch_normalization_5/ReadVariableOp_1Љ
BCNN_baseline/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpKcnn_baseline_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02D
BCNN_baseline/batch_normalization_5/FusedBatchNormV3/ReadVariableOpЌ
DCNN_baseline/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMcnn_baseline_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02F
DCNN_baseline/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1╠
3CNN_baseline/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3/CNN_baseline/conv2d_transpose/Elu:activations:09CNN_baseline/batch_normalization_5/ReadVariableOp:value:0;CNN_baseline/batch_normalization_5/ReadVariableOp_1:value:0JCNN_baseline/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0LCNN_baseline/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЭЭђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 25
3CNN_baseline/batch_normalization_5/FusedBatchNormV3х
%CNN_baseline/conv2d_transpose_1/ShapeShape7CNN_baseline/batch_normalization_5/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2'
%CNN_baseline/conv2d_transpose_1/Shape┤
3CNN_baseline/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3CNN_baseline/conv2d_transpose_1/strided_slice/stackИ
5CNN_baseline/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5CNN_baseline/conv2d_transpose_1/strided_slice/stack_1И
5CNN_baseline/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5CNN_baseline/conv2d_transpose_1/strided_slice/stack_2б
-CNN_baseline/conv2d_transpose_1/strided_sliceStridedSlice.CNN_baseline/conv2d_transpose_1/Shape:output:0<CNN_baseline/conv2d_transpose_1/strided_slice/stack:output:0>CNN_baseline/conv2d_transpose_1/strided_slice/stack_1:output:0>CNN_baseline/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-CNN_baseline/conv2d_transpose_1/strided_sliceЋ
'CNN_baseline/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value
B :Щ2)
'CNN_baseline/conv2d_transpose_1/stack/1Ћ
'CNN_baseline/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :Щ2)
'CNN_baseline/conv2d_transpose_1/stack/2Ћ
'CNN_baseline/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2)
'CNN_baseline/conv2d_transpose_1/stack/3м
%CNN_baseline/conv2d_transpose_1/stackPack6CNN_baseline/conv2d_transpose_1/strided_slice:output:00CNN_baseline/conv2d_transpose_1/stack/1:output:00CNN_baseline/conv2d_transpose_1/stack/2:output:00CNN_baseline/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%CNN_baseline/conv2d_transpose_1/stackИ
5CNN_baseline/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5CNN_baseline/conv2d_transpose_1/strided_slice_1/stack╝
7CNN_baseline/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7CNN_baseline/conv2d_transpose_1/strided_slice_1/stack_1╝
7CNN_baseline/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7CNN_baseline/conv2d_transpose_1/strided_slice_1/stack_2г
/CNN_baseline/conv2d_transpose_1/strided_slice_1StridedSlice.CNN_baseline/conv2d_transpose_1/stack:output:0>CNN_baseline/conv2d_transpose_1/strided_slice_1/stack:output:0@CNN_baseline/conv2d_transpose_1/strided_slice_1/stack_1:output:0@CNN_baseline/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/CNN_baseline/conv2d_transpose_1/strided_slice_1Ћ
?CNN_baseline/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpHcnn_baseline_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02A
?CNN_baseline/conv2d_transpose_1/conv2d_transpose/ReadVariableOpЊ
0CNN_baseline/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput.CNN_baseline/conv2d_transpose_1/stack:output:0GCNN_baseline/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:07CNN_baseline/batch_normalization_5/FusedBatchNormV3:y:0*
T0*2
_output_shapes 
:         ЩЩђ*
paddingVALID*
strides
22
0CNN_baseline/conv2d_transpose_1/conv2d_transposeь
6CNN_baseline/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp?cnn_baseline_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype028
6CNN_baseline/conv2d_transpose_1/BiasAdd/ReadVariableOpЋ
'CNN_baseline/conv2d_transpose_1/BiasAddBiasAdd9CNN_baseline/conv2d_transpose_1/conv2d_transpose:output:0>CNN_baseline/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЩЩђ2)
'CNN_baseline/conv2d_transpose_1/BiasAdd└
#CNN_baseline/conv2d_transpose_1/EluElu0CNN_baseline/conv2d_transpose_1/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЩЩђ2%
#CNN_baseline/conv2d_transpose_1/Eluя
1CNN_baseline/batch_normalization_6/ReadVariableOpReadVariableOp:cnn_baseline_batch_normalization_6_readvariableop_resource*
_output_shapes	
:ђ*
dtype023
1CNN_baseline/batch_normalization_6/ReadVariableOpС
3CNN_baseline/batch_normalization_6/ReadVariableOp_1ReadVariableOp<cnn_baseline_batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype025
3CNN_baseline/batch_normalization_6/ReadVariableOp_1Љ
BCNN_baseline/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpKcnn_baseline_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02D
BCNN_baseline/batch_normalization_6/FusedBatchNormV3/ReadVariableOpЌ
DCNN_baseline/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMcnn_baseline_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02F
DCNN_baseline/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1╬
3CNN_baseline/batch_normalization_6/FusedBatchNormV3FusedBatchNormV31CNN_baseline/conv2d_transpose_1/Elu:activations:09CNN_baseline/batch_normalization_6/ReadVariableOp:value:0;CNN_baseline/batch_normalization_6/ReadVariableOp_1:value:0JCNN_baseline/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0LCNN_baseline/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЩЩђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 25
3CNN_baseline/batch_normalization_6/FusedBatchNormV3х
%CNN_baseline/conv2d_transpose_2/ShapeShape7CNN_baseline/batch_normalization_6/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2'
%CNN_baseline/conv2d_transpose_2/Shape┤
3CNN_baseline/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3CNN_baseline/conv2d_transpose_2/strided_slice/stackИ
5CNN_baseline/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5CNN_baseline/conv2d_transpose_2/strided_slice/stack_1И
5CNN_baseline/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5CNN_baseline/conv2d_transpose_2/strided_slice/stack_2б
-CNN_baseline/conv2d_transpose_2/strided_sliceStridedSlice.CNN_baseline/conv2d_transpose_2/Shape:output:0<CNN_baseline/conv2d_transpose_2/strided_slice/stack:output:0>CNN_baseline/conv2d_transpose_2/strided_slice/stack_1:output:0>CNN_baseline/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-CNN_baseline/conv2d_transpose_2/strided_sliceЋ
'CNN_baseline/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value
B :Ч2)
'CNN_baseline/conv2d_transpose_2/stack/1Ћ
'CNN_baseline/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value
B :Ч2)
'CNN_baseline/conv2d_transpose_2/stack/2Ћ
'CNN_baseline/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2)
'CNN_baseline/conv2d_transpose_2/stack/3м
%CNN_baseline/conv2d_transpose_2/stackPack6CNN_baseline/conv2d_transpose_2/strided_slice:output:00CNN_baseline/conv2d_transpose_2/stack/1:output:00CNN_baseline/conv2d_transpose_2/stack/2:output:00CNN_baseline/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%CNN_baseline/conv2d_transpose_2/stackИ
5CNN_baseline/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5CNN_baseline/conv2d_transpose_2/strided_slice_1/stack╝
7CNN_baseline/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7CNN_baseline/conv2d_transpose_2/strided_slice_1/stack_1╝
7CNN_baseline/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7CNN_baseline/conv2d_transpose_2/strided_slice_1/stack_2г
/CNN_baseline/conv2d_transpose_2/strided_slice_1StridedSlice.CNN_baseline/conv2d_transpose_2/stack:output:0>CNN_baseline/conv2d_transpose_2/strided_slice_1/stack:output:0@CNN_baseline/conv2d_transpose_2/strided_slice_1/stack_1:output:0@CNN_baseline/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/CNN_baseline/conv2d_transpose_2/strided_slice_1Ћ
?CNN_baseline/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpHcnn_baseline_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02A
?CNN_baseline/conv2d_transpose_2/conv2d_transpose/ReadVariableOpЊ
0CNN_baseline/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput.CNN_baseline/conv2d_transpose_2/stack:output:0GCNN_baseline/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:07CNN_baseline/batch_normalization_6/FusedBatchNormV3:y:0*
T0*2
_output_shapes 
:         ЧЧђ*
paddingVALID*
strides
22
0CNN_baseline/conv2d_transpose_2/conv2d_transposeь
6CNN_baseline/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp?cnn_baseline_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype028
6CNN_baseline/conv2d_transpose_2/BiasAdd/ReadVariableOpЋ
'CNN_baseline/conv2d_transpose_2/BiasAddBiasAdd9CNN_baseline/conv2d_transpose_2/conv2d_transpose:output:0>CNN_baseline/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЧЧђ2)
'CNN_baseline/conv2d_transpose_2/BiasAdd└
#CNN_baseline/conv2d_transpose_2/EluElu0CNN_baseline/conv2d_transpose_2/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЧЧђ2%
#CNN_baseline/conv2d_transpose_2/Eluя
1CNN_baseline/batch_normalization_7/ReadVariableOpReadVariableOp:cnn_baseline_batch_normalization_7_readvariableop_resource*
_output_shapes	
:ђ*
dtype023
1CNN_baseline/batch_normalization_7/ReadVariableOpС
3CNN_baseline/batch_normalization_7/ReadVariableOp_1ReadVariableOp<cnn_baseline_batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype025
3CNN_baseline/batch_normalization_7/ReadVariableOp_1Љ
BCNN_baseline/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpKcnn_baseline_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02D
BCNN_baseline/batch_normalization_7/FusedBatchNormV3/ReadVariableOpЌ
DCNN_baseline/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMcnn_baseline_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02F
DCNN_baseline/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1╬
3CNN_baseline/batch_normalization_7/FusedBatchNormV3FusedBatchNormV31CNN_baseline/conv2d_transpose_2/Elu:activations:09CNN_baseline/batch_normalization_7/ReadVariableOp:value:0;CNN_baseline/batch_normalization_7/ReadVariableOp_1:value:0JCNN_baseline/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0LCNN_baseline/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЧЧђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 25
3CNN_baseline/batch_normalization_7/FusedBatchNormV3х
%CNN_baseline/conv2d_transpose_3/ShapeShape7CNN_baseline/batch_normalization_7/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2'
%CNN_baseline/conv2d_transpose_3/Shape┤
3CNN_baseline/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3CNN_baseline/conv2d_transpose_3/strided_slice/stackИ
5CNN_baseline/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5CNN_baseline/conv2d_transpose_3/strided_slice/stack_1И
5CNN_baseline/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5CNN_baseline/conv2d_transpose_3/strided_slice/stack_2б
-CNN_baseline/conv2d_transpose_3/strided_sliceStridedSlice.CNN_baseline/conv2d_transpose_3/Shape:output:0<CNN_baseline/conv2d_transpose_3/strided_slice/stack:output:0>CNN_baseline/conv2d_transpose_3/strided_slice/stack_1:output:0>CNN_baseline/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-CNN_baseline/conv2d_transpose_3/strided_sliceЋ
'CNN_baseline/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value
B :ђ2)
'CNN_baseline/conv2d_transpose_3/stack/1Ћ
'CNN_baseline/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value
B :ђ2)
'CNN_baseline/conv2d_transpose_3/stack/2ћ
'CNN_baseline/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2)
'CNN_baseline/conv2d_transpose_3/stack/3м
%CNN_baseline/conv2d_transpose_3/stackPack6CNN_baseline/conv2d_transpose_3/strided_slice:output:00CNN_baseline/conv2d_transpose_3/stack/1:output:00CNN_baseline/conv2d_transpose_3/stack/2:output:00CNN_baseline/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%CNN_baseline/conv2d_transpose_3/stackИ
5CNN_baseline/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5CNN_baseline/conv2d_transpose_3/strided_slice_1/stack╝
7CNN_baseline/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7CNN_baseline/conv2d_transpose_3/strided_slice_1/stack_1╝
7CNN_baseline/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7CNN_baseline/conv2d_transpose_3/strided_slice_1/stack_2г
/CNN_baseline/conv2d_transpose_3/strided_slice_1StridedSlice.CNN_baseline/conv2d_transpose_3/stack:output:0>CNN_baseline/conv2d_transpose_3/strided_slice_1/stack:output:0@CNN_baseline/conv2d_transpose_3/strided_slice_1/stack_1:output:0@CNN_baseline/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/CNN_baseline/conv2d_transpose_3/strided_slice_1ћ
?CNN_baseline/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpHcnn_baseline_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02A
?CNN_baseline/conv2d_transpose_3/conv2d_transpose/ReadVariableOpњ
0CNN_baseline/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput.CNN_baseline/conv2d_transpose_3/stack:output:0GCNN_baseline/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:07CNN_baseline/batch_normalization_7/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:         ђђ@*
paddingVALID*
strides
22
0CNN_baseline/conv2d_transpose_3/conv2d_transposeВ
6CNN_baseline/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp?cnn_baseline_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6CNN_baseline/conv2d_transpose_3/BiasAdd/ReadVariableOpћ
'CNN_baseline/conv2d_transpose_3/BiasAddBiasAdd9CNN_baseline/conv2d_transpose_3/conv2d_transpose:output:0>CNN_baseline/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ@2)
'CNN_baseline/conv2d_transpose_3/BiasAdd┐
#CNN_baseline/conv2d_transpose_3/EluElu0CNN_baseline/conv2d_transpose_3/BiasAdd:output:0*
T0*1
_output_shapes
:         ђђ@2%
#CNN_baseline/conv2d_transpose_3/EluП
1CNN_baseline/batch_normalization_8/ReadVariableOpReadVariableOp:cnn_baseline_batch_normalization_8_readvariableop_resource*
_output_shapes
:@*
dtype023
1CNN_baseline/batch_normalization_8/ReadVariableOpс
3CNN_baseline/batch_normalization_8/ReadVariableOp_1ReadVariableOp<cnn_baseline_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3CNN_baseline/batch_normalization_8/ReadVariableOp_1љ
BCNN_baseline/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpKcnn_baseline_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02D
BCNN_baseline/batch_normalization_8/FusedBatchNormV3/ReadVariableOpќ
DCNN_baseline/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMcnn_baseline_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02F
DCNN_baseline/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1╔
3CNN_baseline/batch_normalization_8/FusedBatchNormV3FusedBatchNormV31CNN_baseline/conv2d_transpose_3/Elu:activations:09CNN_baseline/batch_normalization_8/ReadVariableOp:value:0;CNN_baseline/batch_normalization_8/ReadVariableOp_1:value:0JCNN_baseline/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0LCNN_baseline/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ@:@:@:@:@:*
epsilon%oЃ:*
is_training( 25
3CNN_baseline/batch_normalization_8/FusedBatchNormV3х
%CNN_baseline/conv2d_transpose_4/ShapeShape7CNN_baseline/batch_normalization_8/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2'
%CNN_baseline/conv2d_transpose_4/Shape┤
3CNN_baseline/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3CNN_baseline/conv2d_transpose_4/strided_slice/stackИ
5CNN_baseline/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5CNN_baseline/conv2d_transpose_4/strided_slice/stack_1И
5CNN_baseline/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5CNN_baseline/conv2d_transpose_4/strided_slice/stack_2б
-CNN_baseline/conv2d_transpose_4/strided_sliceStridedSlice.CNN_baseline/conv2d_transpose_4/Shape:output:0<CNN_baseline/conv2d_transpose_4/strided_slice/stack:output:0>CNN_baseline/conv2d_transpose_4/strided_slice/stack_1:output:0>CNN_baseline/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-CNN_baseline/conv2d_transpose_4/strided_sliceЋ
'CNN_baseline/conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value
B :ђ2)
'CNN_baseline/conv2d_transpose_4/stack/1Ћ
'CNN_baseline/conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value
B :ђ2)
'CNN_baseline/conv2d_transpose_4/stack/2ћ
'CNN_baseline/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'CNN_baseline/conv2d_transpose_4/stack/3м
%CNN_baseline/conv2d_transpose_4/stackPack6CNN_baseline/conv2d_transpose_4/strided_slice:output:00CNN_baseline/conv2d_transpose_4/stack/1:output:00CNN_baseline/conv2d_transpose_4/stack/2:output:00CNN_baseline/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%CNN_baseline/conv2d_transpose_4/stackИ
5CNN_baseline/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5CNN_baseline/conv2d_transpose_4/strided_slice_1/stack╝
7CNN_baseline/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7CNN_baseline/conv2d_transpose_4/strided_slice_1/stack_1╝
7CNN_baseline/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7CNN_baseline/conv2d_transpose_4/strided_slice_1/stack_2г
/CNN_baseline/conv2d_transpose_4/strided_slice_1StridedSlice.CNN_baseline/conv2d_transpose_4/stack:output:0>CNN_baseline/conv2d_transpose_4/strided_slice_1/stack:output:0@CNN_baseline/conv2d_transpose_4/strided_slice_1/stack_1:output:0@CNN_baseline/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/CNN_baseline/conv2d_transpose_4/strided_slice_1Њ
?CNN_baseline/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpHcnn_baseline_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype02A
?CNN_baseline/conv2d_transpose_4/conv2d_transpose/ReadVariableOpњ
0CNN_baseline/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput.CNN_baseline/conv2d_transpose_4/stack:output:0GCNN_baseline/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:07CNN_baseline/batch_normalization_8/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:         ђђ*
paddingVALID*
strides
22
0CNN_baseline/conv2d_transpose_4/conv2d_transposeВ
6CNN_baseline/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp?cnn_baseline_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6CNN_baseline/conv2d_transpose_4/BiasAdd/ReadVariableOpћ
'CNN_baseline/conv2d_transpose_4/BiasAddBiasAdd9CNN_baseline/conv2d_transpose_4/conv2d_transpose:output:0>CNN_baseline/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ2)
'CNN_baseline/conv2d_transpose_4/BiasAddЋ
IdentityIdentity0CNN_baseline/conv2d_transpose_4/BiasAdd:output:0^NoOp*
T0*1
_output_shapes
:         ђђ2

Identity┐
NoOpNoOpA^CNN_baseline/batch_normalization/FusedBatchNormV3/ReadVariableOpC^CNN_baseline/batch_normalization/FusedBatchNormV3/ReadVariableOp_10^CNN_baseline/batch_normalization/ReadVariableOp2^CNN_baseline/batch_normalization/ReadVariableOp_1C^CNN_baseline/batch_normalization_1/FusedBatchNormV3/ReadVariableOpE^CNN_baseline/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12^CNN_baseline/batch_normalization_1/ReadVariableOp4^CNN_baseline/batch_normalization_1/ReadVariableOp_1C^CNN_baseline/batch_normalization_2/FusedBatchNormV3/ReadVariableOpE^CNN_baseline/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12^CNN_baseline/batch_normalization_2/ReadVariableOp4^CNN_baseline/batch_normalization_2/ReadVariableOp_1C^CNN_baseline/batch_normalization_3/FusedBatchNormV3/ReadVariableOpE^CNN_baseline/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12^CNN_baseline/batch_normalization_3/ReadVariableOp4^CNN_baseline/batch_normalization_3/ReadVariableOp_1C^CNN_baseline/batch_normalization_4/FusedBatchNormV3/ReadVariableOpE^CNN_baseline/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12^CNN_baseline/batch_normalization_4/ReadVariableOp4^CNN_baseline/batch_normalization_4/ReadVariableOp_1C^CNN_baseline/batch_normalization_5/FusedBatchNormV3/ReadVariableOpE^CNN_baseline/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12^CNN_baseline/batch_normalization_5/ReadVariableOp4^CNN_baseline/batch_normalization_5/ReadVariableOp_1C^CNN_baseline/batch_normalization_6/FusedBatchNormV3/ReadVariableOpE^CNN_baseline/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12^CNN_baseline/batch_normalization_6/ReadVariableOp4^CNN_baseline/batch_normalization_6/ReadVariableOp_1C^CNN_baseline/batch_normalization_7/FusedBatchNormV3/ReadVariableOpE^CNN_baseline/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12^CNN_baseline/batch_normalization_7/ReadVariableOp4^CNN_baseline/batch_normalization_7/ReadVariableOp_1C^CNN_baseline/batch_normalization_8/FusedBatchNormV3/ReadVariableOpE^CNN_baseline/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12^CNN_baseline/batch_normalization_8/ReadVariableOp4^CNN_baseline/batch_normalization_8/ReadVariableOp_1+^CNN_baseline/conv2d/BiasAdd/ReadVariableOp*^CNN_baseline/conv2d/Conv2D/ReadVariableOp-^CNN_baseline/conv2d_1/BiasAdd/ReadVariableOp,^CNN_baseline/conv2d_1/Conv2D/ReadVariableOp-^CNN_baseline/conv2d_2/BiasAdd/ReadVariableOp,^CNN_baseline/conv2d_2/Conv2D/ReadVariableOp-^CNN_baseline/conv2d_3/BiasAdd/ReadVariableOp,^CNN_baseline/conv2d_3/Conv2D/ReadVariableOp5^CNN_baseline/conv2d_transpose/BiasAdd/ReadVariableOp>^CNN_baseline/conv2d_transpose/conv2d_transpose/ReadVariableOp7^CNN_baseline/conv2d_transpose_1/BiasAdd/ReadVariableOp@^CNN_baseline/conv2d_transpose_1/conv2d_transpose/ReadVariableOp7^CNN_baseline/conv2d_transpose_2/BiasAdd/ReadVariableOp@^CNN_baseline/conv2d_transpose_2/conv2d_transpose/ReadVariableOp7^CNN_baseline/conv2d_transpose_3/BiasAdd/ReadVariableOp@^CNN_baseline/conv2d_transpose_3/conv2d_transpose/ReadVariableOp7^CNN_baseline/conv2d_transpose_4/BiasAdd/ReadVariableOp@^CNN_baseline/conv2d_transpose_4/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*џ
_input_shapesѕ
Ё:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2ё
@CNN_baseline/batch_normalization/FusedBatchNormV3/ReadVariableOp@CNN_baseline/batch_normalization/FusedBatchNormV3/ReadVariableOp2ѕ
BCNN_baseline/batch_normalization/FusedBatchNormV3/ReadVariableOp_1BCNN_baseline/batch_normalization/FusedBatchNormV3/ReadVariableOp_12b
/CNN_baseline/batch_normalization/ReadVariableOp/CNN_baseline/batch_normalization/ReadVariableOp2f
1CNN_baseline/batch_normalization/ReadVariableOp_11CNN_baseline/batch_normalization/ReadVariableOp_12ѕ
BCNN_baseline/batch_normalization_1/FusedBatchNormV3/ReadVariableOpBCNN_baseline/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2ї
DCNN_baseline/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1DCNN_baseline/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12f
1CNN_baseline/batch_normalization_1/ReadVariableOp1CNN_baseline/batch_normalization_1/ReadVariableOp2j
3CNN_baseline/batch_normalization_1/ReadVariableOp_13CNN_baseline/batch_normalization_1/ReadVariableOp_12ѕ
BCNN_baseline/batch_normalization_2/FusedBatchNormV3/ReadVariableOpBCNN_baseline/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2ї
DCNN_baseline/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1DCNN_baseline/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12f
1CNN_baseline/batch_normalization_2/ReadVariableOp1CNN_baseline/batch_normalization_2/ReadVariableOp2j
3CNN_baseline/batch_normalization_2/ReadVariableOp_13CNN_baseline/batch_normalization_2/ReadVariableOp_12ѕ
BCNN_baseline/batch_normalization_3/FusedBatchNormV3/ReadVariableOpBCNN_baseline/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2ї
DCNN_baseline/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1DCNN_baseline/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12f
1CNN_baseline/batch_normalization_3/ReadVariableOp1CNN_baseline/batch_normalization_3/ReadVariableOp2j
3CNN_baseline/batch_normalization_3/ReadVariableOp_13CNN_baseline/batch_normalization_3/ReadVariableOp_12ѕ
BCNN_baseline/batch_normalization_4/FusedBatchNormV3/ReadVariableOpBCNN_baseline/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2ї
DCNN_baseline/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1DCNN_baseline/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12f
1CNN_baseline/batch_normalization_4/ReadVariableOp1CNN_baseline/batch_normalization_4/ReadVariableOp2j
3CNN_baseline/batch_normalization_4/ReadVariableOp_13CNN_baseline/batch_normalization_4/ReadVariableOp_12ѕ
BCNN_baseline/batch_normalization_5/FusedBatchNormV3/ReadVariableOpBCNN_baseline/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2ї
DCNN_baseline/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1DCNN_baseline/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12f
1CNN_baseline/batch_normalization_5/ReadVariableOp1CNN_baseline/batch_normalization_5/ReadVariableOp2j
3CNN_baseline/batch_normalization_5/ReadVariableOp_13CNN_baseline/batch_normalization_5/ReadVariableOp_12ѕ
BCNN_baseline/batch_normalization_6/FusedBatchNormV3/ReadVariableOpBCNN_baseline/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2ї
DCNN_baseline/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1DCNN_baseline/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12f
1CNN_baseline/batch_normalization_6/ReadVariableOp1CNN_baseline/batch_normalization_6/ReadVariableOp2j
3CNN_baseline/batch_normalization_6/ReadVariableOp_13CNN_baseline/batch_normalization_6/ReadVariableOp_12ѕ
BCNN_baseline/batch_normalization_7/FusedBatchNormV3/ReadVariableOpBCNN_baseline/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2ї
DCNN_baseline/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1DCNN_baseline/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12f
1CNN_baseline/batch_normalization_7/ReadVariableOp1CNN_baseline/batch_normalization_7/ReadVariableOp2j
3CNN_baseline/batch_normalization_7/ReadVariableOp_13CNN_baseline/batch_normalization_7/ReadVariableOp_12ѕ
BCNN_baseline/batch_normalization_8/FusedBatchNormV3/ReadVariableOpBCNN_baseline/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2ї
DCNN_baseline/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1DCNN_baseline/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12f
1CNN_baseline/batch_normalization_8/ReadVariableOp1CNN_baseline/batch_normalization_8/ReadVariableOp2j
3CNN_baseline/batch_normalization_8/ReadVariableOp_13CNN_baseline/batch_normalization_8/ReadVariableOp_12X
*CNN_baseline/conv2d/BiasAdd/ReadVariableOp*CNN_baseline/conv2d/BiasAdd/ReadVariableOp2V
)CNN_baseline/conv2d/Conv2D/ReadVariableOp)CNN_baseline/conv2d/Conv2D/ReadVariableOp2\
,CNN_baseline/conv2d_1/BiasAdd/ReadVariableOp,CNN_baseline/conv2d_1/BiasAdd/ReadVariableOp2Z
+CNN_baseline/conv2d_1/Conv2D/ReadVariableOp+CNN_baseline/conv2d_1/Conv2D/ReadVariableOp2\
,CNN_baseline/conv2d_2/BiasAdd/ReadVariableOp,CNN_baseline/conv2d_2/BiasAdd/ReadVariableOp2Z
+CNN_baseline/conv2d_2/Conv2D/ReadVariableOp+CNN_baseline/conv2d_2/Conv2D/ReadVariableOp2\
,CNN_baseline/conv2d_3/BiasAdd/ReadVariableOp,CNN_baseline/conv2d_3/BiasAdd/ReadVariableOp2Z
+CNN_baseline/conv2d_3/Conv2D/ReadVariableOp+CNN_baseline/conv2d_3/Conv2D/ReadVariableOp2l
4CNN_baseline/conv2d_transpose/BiasAdd/ReadVariableOp4CNN_baseline/conv2d_transpose/BiasAdd/ReadVariableOp2~
=CNN_baseline/conv2d_transpose/conv2d_transpose/ReadVariableOp=CNN_baseline/conv2d_transpose/conv2d_transpose/ReadVariableOp2p
6CNN_baseline/conv2d_transpose_1/BiasAdd/ReadVariableOp6CNN_baseline/conv2d_transpose_1/BiasAdd/ReadVariableOp2ѓ
?CNN_baseline/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?CNN_baseline/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2p
6CNN_baseline/conv2d_transpose_2/BiasAdd/ReadVariableOp6CNN_baseline/conv2d_transpose_2/BiasAdd/ReadVariableOp2ѓ
?CNN_baseline/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?CNN_baseline/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2p
6CNN_baseline/conv2d_transpose_3/BiasAdd/ReadVariableOp6CNN_baseline/conv2d_transpose_3/BiasAdd/ReadVariableOp2ѓ
?CNN_baseline/conv2d_transpose_3/conv2d_transpose/ReadVariableOp?CNN_baseline/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2p
6CNN_baseline/conv2d_transpose_4/BiasAdd/ReadVariableOp6CNN_baseline/conv2d_transpose_4/BiasAdd/ReadVariableOp2ѓ
?CNN_baseline/conv2d_transpose_4/conv2d_transpose/ReadVariableOp?CNN_baseline/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:T P
-
_output_shapes
:         ђђ

_user_specified_nameu_vel
╦	
Н
6__inference_batch_normalization_7_layer_call_fn_165756

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCall╣
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1614262
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
╔	
Н
6__inference_batch_normalization_6_layer_call_fn_165565

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1612482
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
┘'
Џ
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_166126

inputsB
(conv2d_transpose_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2В
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3ѓ
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2В
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3│
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_transpose/ReadVariableOpы
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
2
conv2d_transposeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpц
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAddЁ
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+                           2

IdentityЅ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
▓
а
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_164949

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Л
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЩЩђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3z
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЩЩђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs
о
└
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_164823

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┌
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЧЧ@:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:         ЧЧ@2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         ЧЧ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:         ЧЧ@
 
_user_specified_nameinputs
ч
 
D__inference_conv2d_1_layer_call_and_return_conditional_losses_164843

inputs9
conv2d_readvariableop_resource:@ђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpќ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02
Conv2D/ReadVariableOpД
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЩЩђ*
paddingVALID*
strides
2
Conv2DЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpІ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЩЩђ2	
BiasAdd`
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:         ЩЩђ2
Eluw
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЧЧ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ЧЧ@
 
_user_specified_nameinputs
Ч
Е
1__inference_conv2d_transpose_layer_call_fn_165264

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1609062
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Ѕ	
Н
6__inference_batch_normalization_3_layer_call_fn_165039

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1627442
StatefulPartitionedCallє
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЭЭђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
╝
Е
1__inference_conv2d_transpose_layer_call_fn_165273

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1620952
StatefulPartitionedCallє
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ШШђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ШШђ
 
_user_specified_nameinputs
н
Й
O__inference_batch_normalization_layer_call_and_return_conditional_losses_164679

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┌
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ:::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:         ђђ2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         ђђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
Ќ
└
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_166035

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1Ѕ
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
с
ю
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_164769

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3Ѕ
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
з
а
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_165609

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1р
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3і
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
І	
Н
6__inference_batch_normalization_4_layer_call_fn_165170

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ШШђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1620622
StatefulPartitionedCallє
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ШШђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ШШђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ШШђ
 
_user_specified_nameinputs
І	
Н
6__inference_batch_normalization_5_layer_call_fn_165374

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1621182
StatefulPartitionedCallє
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЭЭђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
Њ
Ж
-__inference_CNN_baseline_layer_call_fn_163393	
u_vel
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@$
	unknown_9:@ђ

unknown_10:	ђ

unknown_11:	ђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:	ђ&

unknown_15:ђђ

unknown_16:	ђ

unknown_17:	ђ

unknown_18:	ђ

unknown_19:	ђ

unknown_20:	ђ&

unknown_21:ђђ

unknown_22:	ђ

unknown_23:	ђ

unknown_24:	ђ

unknown_25:	ђ

unknown_26:	ђ&

unknown_27:ђђ

unknown_28:	ђ

unknown_29:	ђ

unknown_30:	ђ

unknown_31:	ђ

unknown_32:	ђ&

unknown_33:ђђ

unknown_34:	ђ

unknown_35:	ђ

unknown_36:	ђ

unknown_37:	ђ

unknown_38:	ђ&

unknown_39:ђђ

unknown_40:	ђ

unknown_41:	ђ

unknown_42:	ђ

unknown_43:	ђ

unknown_44:	ђ%

unknown_45:@ђ

unknown_46:@

unknown_47:@

unknown_48:@

unknown_49:@

unknown_50:@$

unknown_51:@

unknown_52:
identityѕбStatefulPartitionedCall┴
StatefulPartitionedCallStatefulPartitionedCallu_velunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_52*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*F
_read_only_resource_inputs(
&$ #$%&)*+,/01256*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_CNN_baseline_layer_call_and_return_conditional_losses_1631692
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*џ
_input_shapesѕ
Ё:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
-
_output_shapes
:         ђђ

_user_specified_nameu_vel
╔	
Н
6__inference_batch_normalization_3_layer_call_fn_165013

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1606782
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
 
ђ
D__inference_conv2d_3_layer_call_and_return_conditional_losses_165131

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02
Conv2D/ReadVariableOpД
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ШШђ*
paddingVALID*
strides
2
Conv2DЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpІ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ШШђ2	
BiasAdd`
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:         ШШђ2
Eluw
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:         ШШђ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ЭЭђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
╔	
Н
6__inference_batch_normalization_4_layer_call_fn_165157

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1608042
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
╦	
Н
6__inference_batch_normalization_6_layer_call_fn_165552

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCall╣
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1612042
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
ќ
в
-__inference_CNN_baseline_layer_call_fn_164002

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@$
	unknown_9:@ђ

unknown_10:	ђ

unknown_11:	ђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:	ђ&

unknown_15:ђђ

unknown_16:	ђ

unknown_17:	ђ

unknown_18:	ђ

unknown_19:	ђ

unknown_20:	ђ&

unknown_21:ђђ

unknown_22:	ђ

unknown_23:	ђ

unknown_24:	ђ

unknown_25:	ђ

unknown_26:	ђ&

unknown_27:ђђ

unknown_28:	ђ

unknown_29:	ђ

unknown_30:	ђ

unknown_31:	ђ

unknown_32:	ђ&

unknown_33:ђђ

unknown_34:	ђ

unknown_35:	ђ

unknown_36:	ђ

unknown_37:	ђ

unknown_38:	ђ&

unknown_39:ђђ

unknown_40:	ђ

unknown_41:	ђ

unknown_42:	ђ

unknown_43:	ђ

unknown_44:	ђ%

unknown_45:@ђ

unknown_46:@

unknown_47:@

unknown_48:@

unknown_49:@

unknown_50:@$

unknown_51:@

unknown_52:
identityѕбStatefulPartitionedCall┬
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_52*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*F
_read_only_resource_inputs(
&$ #$%&)*+,/01256*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_CNN_baseline_layer_call_and_return_conditional_losses_1631692
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*џ
_input_shapesѕ
Ё:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:         ђђ
 
_user_specified_nameinputs
▓
а
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_162062

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Л
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ШШђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3z
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ШШђ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ШШђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:         ШШђ
 
_user_specified_nameinputs
о
└
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_162852

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┌
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЧЧ@:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:         ЧЧ@2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         ЧЧ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:         ЧЧ@
 
_user_specified_nameinputs
┴	
Л
6__inference_batch_normalization_1_layer_call_fn_164725

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1604262
StatefulPartitionedCallЋ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
г
А
)__inference_conv2d_2_layer_call_fn_164976

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1619952
StatefulPartitionedCallє
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ЩЩђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs
ђ	
Ф
3__inference_conv2d_transpose_2_layer_call_fn_165672

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1613502
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Т
─
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_165867

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▀
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЧЧђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1z
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЧЧђ2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЧЧђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:         ЧЧђ
 
_user_specified_nameinputs
Т
─
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_162531

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▀
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЧЧђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1z
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЧЧђ2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЧЧђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:         ЧЧђ
 
_user_specified_nameinputs
Д
─
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_160678

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1і
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
с
ю
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_166017

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3Ѕ
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Д
─
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_165423

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1і
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
з
а
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_160634

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1р
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3і
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
б╦
Т\
"__inference__traced_restore_166980
file_prefix8
*assignvariableop_batch_normalization_gamma:9
+assignvariableop_1_batch_normalization_beta:@
2assignvariableop_2_batch_normalization_moving_mean:D
6assignvariableop_3_batch_normalization_moving_variance::
 assignvariableop_4_conv2d_kernel:@,
assignvariableop_5_conv2d_bias:@<
.assignvariableop_6_batch_normalization_1_gamma:@;
-assignvariableop_7_batch_normalization_1_beta:@B
4assignvariableop_8_batch_normalization_1_moving_mean:@F
8assignvariableop_9_batch_normalization_1_moving_variance:@>
#assignvariableop_10_conv2d_1_kernel:@ђ0
!assignvariableop_11_conv2d_1_bias:	ђ>
/assignvariableop_12_batch_normalization_2_gamma:	ђ=
.assignvariableop_13_batch_normalization_2_beta:	ђD
5assignvariableop_14_batch_normalization_2_moving_mean:	ђH
9assignvariableop_15_batch_normalization_2_moving_variance:	ђ?
#assignvariableop_16_conv2d_2_kernel:ђђ0
!assignvariableop_17_conv2d_2_bias:	ђ>
/assignvariableop_18_batch_normalization_3_gamma:	ђ=
.assignvariableop_19_batch_normalization_3_beta:	ђD
5assignvariableop_20_batch_normalization_3_moving_mean:	ђH
9assignvariableop_21_batch_normalization_3_moving_variance:	ђ?
#assignvariableop_22_conv2d_3_kernel:ђђ0
!assignvariableop_23_conv2d_3_bias:	ђ>
/assignvariableop_24_batch_normalization_4_gamma:	ђ=
.assignvariableop_25_batch_normalization_4_beta:	ђD
5assignvariableop_26_batch_normalization_4_moving_mean:	ђH
9assignvariableop_27_batch_normalization_4_moving_variance:	ђG
+assignvariableop_28_conv2d_transpose_kernel:ђђ8
)assignvariableop_29_conv2d_transpose_bias:	ђ>
/assignvariableop_30_batch_normalization_5_gamma:	ђ=
.assignvariableop_31_batch_normalization_5_beta:	ђD
5assignvariableop_32_batch_normalization_5_moving_mean:	ђH
9assignvariableop_33_batch_normalization_5_moving_variance:	ђI
-assignvariableop_34_conv2d_transpose_1_kernel:ђђ:
+assignvariableop_35_conv2d_transpose_1_bias:	ђ>
/assignvariableop_36_batch_normalization_6_gamma:	ђ=
.assignvariableop_37_batch_normalization_6_beta:	ђD
5assignvariableop_38_batch_normalization_6_moving_mean:	ђH
9assignvariableop_39_batch_normalization_6_moving_variance:	ђI
-assignvariableop_40_conv2d_transpose_2_kernel:ђђ:
+assignvariableop_41_conv2d_transpose_2_bias:	ђ>
/assignvariableop_42_batch_normalization_7_gamma:	ђ=
.assignvariableop_43_batch_normalization_7_beta:	ђD
5assignvariableop_44_batch_normalization_7_moving_mean:	ђH
9assignvariableop_45_batch_normalization_7_moving_variance:	ђH
-assignvariableop_46_conv2d_transpose_3_kernel:@ђ9
+assignvariableop_47_conv2d_transpose_3_bias:@=
/assignvariableop_48_batch_normalization_8_gamma:@<
.assignvariableop_49_batch_normalization_8_beta:@C
5assignvariableop_50_batch_normalization_8_moving_mean:@G
9assignvariableop_51_batch_normalization_8_moving_variance:@G
-assignvariableop_52_conv2d_transpose_4_kernel:@9
+assignvariableop_53_conv2d_transpose_4_bias:'
assignvariableop_54_adam_iter:	 )
assignvariableop_55_adam_beta_1: )
assignvariableop_56_adam_beta_2: (
assignvariableop_57_adam_decay: 0
&assignvariableop_58_adam_learning_rate: #
assignvariableop_59_total: #
assignvariableop_60_count: B
4assignvariableop_61_adam_batch_normalization_gamma_m:A
3assignvariableop_62_adam_batch_normalization_beta_m:B
(assignvariableop_63_adam_conv2d_kernel_m:@4
&assignvariableop_64_adam_conv2d_bias_m:@D
6assignvariableop_65_adam_batch_normalization_1_gamma_m:@C
5assignvariableop_66_adam_batch_normalization_1_beta_m:@E
*assignvariableop_67_adam_conv2d_1_kernel_m:@ђ7
(assignvariableop_68_adam_conv2d_1_bias_m:	ђE
6assignvariableop_69_adam_batch_normalization_2_gamma_m:	ђD
5assignvariableop_70_adam_batch_normalization_2_beta_m:	ђF
*assignvariableop_71_adam_conv2d_2_kernel_m:ђђ7
(assignvariableop_72_adam_conv2d_2_bias_m:	ђE
6assignvariableop_73_adam_batch_normalization_3_gamma_m:	ђD
5assignvariableop_74_adam_batch_normalization_3_beta_m:	ђF
*assignvariableop_75_adam_conv2d_3_kernel_m:ђђ7
(assignvariableop_76_adam_conv2d_3_bias_m:	ђE
6assignvariableop_77_adam_batch_normalization_4_gamma_m:	ђD
5assignvariableop_78_adam_batch_normalization_4_beta_m:	ђN
2assignvariableop_79_adam_conv2d_transpose_kernel_m:ђђ?
0assignvariableop_80_adam_conv2d_transpose_bias_m:	ђE
6assignvariableop_81_adam_batch_normalization_5_gamma_m:	ђD
5assignvariableop_82_adam_batch_normalization_5_beta_m:	ђP
4assignvariableop_83_adam_conv2d_transpose_1_kernel_m:ђђA
2assignvariableop_84_adam_conv2d_transpose_1_bias_m:	ђE
6assignvariableop_85_adam_batch_normalization_6_gamma_m:	ђD
5assignvariableop_86_adam_batch_normalization_6_beta_m:	ђP
4assignvariableop_87_adam_conv2d_transpose_2_kernel_m:ђђA
2assignvariableop_88_adam_conv2d_transpose_2_bias_m:	ђE
6assignvariableop_89_adam_batch_normalization_7_gamma_m:	ђD
5assignvariableop_90_adam_batch_normalization_7_beta_m:	ђO
4assignvariableop_91_adam_conv2d_transpose_3_kernel_m:@ђ@
2assignvariableop_92_adam_conv2d_transpose_3_bias_m:@D
6assignvariableop_93_adam_batch_normalization_8_gamma_m:@C
5assignvariableop_94_adam_batch_normalization_8_beta_m:@N
4assignvariableop_95_adam_conv2d_transpose_4_kernel_m:@@
2assignvariableop_96_adam_conv2d_transpose_4_bias_m:B
4assignvariableop_97_adam_batch_normalization_gamma_v:A
3assignvariableop_98_adam_batch_normalization_beta_v:B
(assignvariableop_99_adam_conv2d_kernel_v:@5
'assignvariableop_100_adam_conv2d_bias_v:@E
7assignvariableop_101_adam_batch_normalization_1_gamma_v:@D
6assignvariableop_102_adam_batch_normalization_1_beta_v:@F
+assignvariableop_103_adam_conv2d_1_kernel_v:@ђ8
)assignvariableop_104_adam_conv2d_1_bias_v:	ђF
7assignvariableop_105_adam_batch_normalization_2_gamma_v:	ђE
6assignvariableop_106_adam_batch_normalization_2_beta_v:	ђG
+assignvariableop_107_adam_conv2d_2_kernel_v:ђђ8
)assignvariableop_108_adam_conv2d_2_bias_v:	ђF
7assignvariableop_109_adam_batch_normalization_3_gamma_v:	ђE
6assignvariableop_110_adam_batch_normalization_3_beta_v:	ђG
+assignvariableop_111_adam_conv2d_3_kernel_v:ђђ8
)assignvariableop_112_adam_conv2d_3_bias_v:	ђF
7assignvariableop_113_adam_batch_normalization_4_gamma_v:	ђE
6assignvariableop_114_adam_batch_normalization_4_beta_v:	ђO
3assignvariableop_115_adam_conv2d_transpose_kernel_v:ђђ@
1assignvariableop_116_adam_conv2d_transpose_bias_v:	ђF
7assignvariableop_117_adam_batch_normalization_5_gamma_v:	ђE
6assignvariableop_118_adam_batch_normalization_5_beta_v:	ђQ
5assignvariableop_119_adam_conv2d_transpose_1_kernel_v:ђђB
3assignvariableop_120_adam_conv2d_transpose_1_bias_v:	ђF
7assignvariableop_121_adam_batch_normalization_6_gamma_v:	ђE
6assignvariableop_122_adam_batch_normalization_6_beta_v:	ђQ
5assignvariableop_123_adam_conv2d_transpose_2_kernel_v:ђђB
3assignvariableop_124_adam_conv2d_transpose_2_bias_v:	ђF
7assignvariableop_125_adam_batch_normalization_7_gamma_v:	ђE
6assignvariableop_126_adam_batch_normalization_7_beta_v:	ђP
5assignvariableop_127_adam_conv2d_transpose_3_kernel_v:@ђA
3assignvariableop_128_adam_conv2d_transpose_3_bias_v:@E
7assignvariableop_129_adam_batch_normalization_8_gamma_v:@D
6assignvariableop_130_adam_batch_normalization_8_beta_v:@O
5assignvariableop_131_adam_conv2d_transpose_4_kernel_v:@A
3assignvariableop_132_adam_conv2d_transpose_4_bias_v:
identity_134ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_100бAssignVariableOp_101бAssignVariableOp_102бAssignVariableOp_103бAssignVariableOp_104бAssignVariableOp_105бAssignVariableOp_106бAssignVariableOp_107бAssignVariableOp_108бAssignVariableOp_109бAssignVariableOp_11бAssignVariableOp_110бAssignVariableOp_111бAssignVariableOp_112бAssignVariableOp_113бAssignVariableOp_114бAssignVariableOp_115бAssignVariableOp_116бAssignVariableOp_117бAssignVariableOp_118бAssignVariableOp_119бAssignVariableOp_12бAssignVariableOp_120бAssignVariableOp_121бAssignVariableOp_122бAssignVariableOp_123бAssignVariableOp_124бAssignVariableOp_125бAssignVariableOp_126бAssignVariableOp_127бAssignVariableOp_128бAssignVariableOp_129бAssignVariableOp_13бAssignVariableOp_130бAssignVariableOp_131бAssignVariableOp_132бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_40бAssignVariableOp_41бAssignVariableOp_42бAssignVariableOp_43бAssignVariableOp_44бAssignVariableOp_45бAssignVariableOp_46бAssignVariableOp_47бAssignVariableOp_48бAssignVariableOp_49бAssignVariableOp_5бAssignVariableOp_50бAssignVariableOp_51бAssignVariableOp_52бAssignVariableOp_53бAssignVariableOp_54бAssignVariableOp_55бAssignVariableOp_56бAssignVariableOp_57бAssignVariableOp_58бAssignVariableOp_59бAssignVariableOp_6бAssignVariableOp_60бAssignVariableOp_61бAssignVariableOp_62бAssignVariableOp_63бAssignVariableOp_64бAssignVariableOp_65бAssignVariableOp_66бAssignVariableOp_67бAssignVariableOp_68бAssignVariableOp_69бAssignVariableOp_7бAssignVariableOp_70бAssignVariableOp_71бAssignVariableOp_72бAssignVariableOp_73бAssignVariableOp_74бAssignVariableOp_75бAssignVariableOp_76бAssignVariableOp_77бAssignVariableOp_78бAssignVariableOp_79бAssignVariableOp_8бAssignVariableOp_80бAssignVariableOp_81бAssignVariableOp_82бAssignVariableOp_83бAssignVariableOp_84бAssignVariableOp_85бAssignVariableOp_86бAssignVariableOp_87бAssignVariableOp_88бAssignVariableOp_89бAssignVariableOp_9бAssignVariableOp_90бAssignVariableOp_91бAssignVariableOp_92бAssignVariableOp_93бAssignVariableOp_94бAssignVariableOp_95бAssignVariableOp_96бAssignVariableOp_97бAssignVariableOp_98бAssignVariableOp_99ЎK
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:є*
dtype0*цJ
valueџJBЌJєB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЪ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:є*
dtype0*б
valueўBЋєB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesл
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*«
_output_shapesЏ
ў::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Ќ
dtypesї
Ѕ2є	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЕ
AssignVariableOpAssignVariableOp*assignvariableop_batch_normalization_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1░
AssignVariableOp_1AssignVariableOp+assignvariableop_1_batch_normalization_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2и
AssignVariableOp_2AssignVariableOp2assignvariableop_2_batch_normalization_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3╗
AssignVariableOp_3AssignVariableOp6assignvariableop_3_batch_normalization_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ц
AssignVariableOp_4AssignVariableOp assignvariableop_4_conv2d_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Б
AssignVariableOp_5AssignVariableOpassignvariableop_5_conv2d_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6│
AssignVariableOp_6AssignVariableOp.assignvariableop_6_batch_normalization_1_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7▓
AssignVariableOp_7AssignVariableOp-assignvariableop_7_batch_normalization_1_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8╣
AssignVariableOp_8AssignVariableOp4assignvariableop_8_batch_normalization_1_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9й
AssignVariableOp_9AssignVariableOp8assignvariableop_9_batch_normalization_1_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ф
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv2d_1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Е
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv2d_1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12и
AssignVariableOp_12AssignVariableOp/assignvariableop_12_batch_normalization_2_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Х
AssignVariableOp_13AssignVariableOp.assignvariableop_13_batch_normalization_2_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14й
AssignVariableOp_14AssignVariableOp5assignvariableop_14_batch_normalization_2_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15┴
AssignVariableOp_15AssignVariableOp9assignvariableop_15_batch_normalization_2_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ф
AssignVariableOp_16AssignVariableOp#assignvariableop_16_conv2d_2_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Е
AssignVariableOp_17AssignVariableOp!assignvariableop_17_conv2d_2_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18и
AssignVariableOp_18AssignVariableOp/assignvariableop_18_batch_normalization_3_gammaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Х
AssignVariableOp_19AssignVariableOp.assignvariableop_19_batch_normalization_3_betaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20й
AssignVariableOp_20AssignVariableOp5assignvariableop_20_batch_normalization_3_moving_meanIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21┴
AssignVariableOp_21AssignVariableOp9assignvariableop_21_batch_normalization_3_moving_varianceIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Ф
AssignVariableOp_22AssignVariableOp#assignvariableop_22_conv2d_3_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Е
AssignVariableOp_23AssignVariableOp!assignvariableop_23_conv2d_3_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24и
AssignVariableOp_24AssignVariableOp/assignvariableop_24_batch_normalization_4_gammaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Х
AssignVariableOp_25AssignVariableOp.assignvariableop_25_batch_normalization_4_betaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26й
AssignVariableOp_26AssignVariableOp5assignvariableop_26_batch_normalization_4_moving_meanIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27┴
AssignVariableOp_27AssignVariableOp9assignvariableop_27_batch_normalization_4_moving_varianceIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28│
AssignVariableOp_28AssignVariableOp+assignvariableop_28_conv2d_transpose_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29▒
AssignVariableOp_29AssignVariableOp)assignvariableop_29_conv2d_transpose_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30и
AssignVariableOp_30AssignVariableOp/assignvariableop_30_batch_normalization_5_gammaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Х
AssignVariableOp_31AssignVariableOp.assignvariableop_31_batch_normalization_5_betaIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32й
AssignVariableOp_32AssignVariableOp5assignvariableop_32_batch_normalization_5_moving_meanIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33┴
AssignVariableOp_33AssignVariableOp9assignvariableop_33_batch_normalization_5_moving_varianceIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34х
AssignVariableOp_34AssignVariableOp-assignvariableop_34_conv2d_transpose_1_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35│
AssignVariableOp_35AssignVariableOp+assignvariableop_35_conv2d_transpose_1_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36и
AssignVariableOp_36AssignVariableOp/assignvariableop_36_batch_normalization_6_gammaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Х
AssignVariableOp_37AssignVariableOp.assignvariableop_37_batch_normalization_6_betaIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38й
AssignVariableOp_38AssignVariableOp5assignvariableop_38_batch_normalization_6_moving_meanIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39┴
AssignVariableOp_39AssignVariableOp9assignvariableop_39_batch_normalization_6_moving_varianceIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40х
AssignVariableOp_40AssignVariableOp-assignvariableop_40_conv2d_transpose_2_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41│
AssignVariableOp_41AssignVariableOp+assignvariableop_41_conv2d_transpose_2_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42и
AssignVariableOp_42AssignVariableOp/assignvariableop_42_batch_normalization_7_gammaIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43Х
AssignVariableOp_43AssignVariableOp.assignvariableop_43_batch_normalization_7_betaIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44й
AssignVariableOp_44AssignVariableOp5assignvariableop_44_batch_normalization_7_moving_meanIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45┴
AssignVariableOp_45AssignVariableOp9assignvariableop_45_batch_normalization_7_moving_varianceIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46х
AssignVariableOp_46AssignVariableOp-assignvariableop_46_conv2d_transpose_3_kernelIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47│
AssignVariableOp_47AssignVariableOp+assignvariableop_47_conv2d_transpose_3_biasIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48и
AssignVariableOp_48AssignVariableOp/assignvariableop_48_batch_normalization_8_gammaIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49Х
AssignVariableOp_49AssignVariableOp.assignvariableop_49_batch_normalization_8_betaIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50й
AssignVariableOp_50AssignVariableOp5assignvariableop_50_batch_normalization_8_moving_meanIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51┴
AssignVariableOp_51AssignVariableOp9assignvariableop_51_batch_normalization_8_moving_varianceIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52х
AssignVariableOp_52AssignVariableOp-assignvariableop_52_conv2d_transpose_4_kernelIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53│
AssignVariableOp_53AssignVariableOp+assignvariableop_53_conv2d_transpose_4_biasIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_54Ц
AssignVariableOp_54AssignVariableOpassignvariableop_54_adam_iterIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55Д
AssignVariableOp_55AssignVariableOpassignvariableop_55_adam_beta_1Identity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56Д
AssignVariableOp_56AssignVariableOpassignvariableop_56_adam_beta_2Identity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57д
AssignVariableOp_57AssignVariableOpassignvariableop_57_adam_decayIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58«
AssignVariableOp_58AssignVariableOp&assignvariableop_58_adam_learning_rateIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59А
AssignVariableOp_59AssignVariableOpassignvariableop_59_totalIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60А
AssignVariableOp_60AssignVariableOpassignvariableop_60_countIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61╝
AssignVariableOp_61AssignVariableOp4assignvariableop_61_adam_batch_normalization_gamma_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62╗
AssignVariableOp_62AssignVariableOp3assignvariableop_62_adam_batch_normalization_beta_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63░
AssignVariableOp_63AssignVariableOp(assignvariableop_63_adam_conv2d_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64«
AssignVariableOp_64AssignVariableOp&assignvariableop_64_adam_conv2d_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65Й
AssignVariableOp_65AssignVariableOp6assignvariableop_65_adam_batch_normalization_1_gamma_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66й
AssignVariableOp_66AssignVariableOp5assignvariableop_66_adam_batch_normalization_1_beta_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67▓
AssignVariableOp_67AssignVariableOp*assignvariableop_67_adam_conv2d_1_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68░
AssignVariableOp_68AssignVariableOp(assignvariableop_68_adam_conv2d_1_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69Й
AssignVariableOp_69AssignVariableOp6assignvariableop_69_adam_batch_normalization_2_gamma_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70й
AssignVariableOp_70AssignVariableOp5assignvariableop_70_adam_batch_normalization_2_beta_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71▓
AssignVariableOp_71AssignVariableOp*assignvariableop_71_adam_conv2d_2_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72░
AssignVariableOp_72AssignVariableOp(assignvariableop_72_adam_conv2d_2_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73Й
AssignVariableOp_73AssignVariableOp6assignvariableop_73_adam_batch_normalization_3_gamma_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74й
AssignVariableOp_74AssignVariableOp5assignvariableop_74_adam_batch_normalization_3_beta_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75▓
AssignVariableOp_75AssignVariableOp*assignvariableop_75_adam_conv2d_3_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76░
AssignVariableOp_76AssignVariableOp(assignvariableop_76_adam_conv2d_3_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77Й
AssignVariableOp_77AssignVariableOp6assignvariableop_77_adam_batch_normalization_4_gamma_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78й
AssignVariableOp_78AssignVariableOp5assignvariableop_78_adam_batch_normalization_4_beta_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79║
AssignVariableOp_79AssignVariableOp2assignvariableop_79_adam_conv2d_transpose_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80И
AssignVariableOp_80AssignVariableOp0assignvariableop_80_adam_conv2d_transpose_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81Й
AssignVariableOp_81AssignVariableOp6assignvariableop_81_adam_batch_normalization_5_gamma_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82й
AssignVariableOp_82AssignVariableOp5assignvariableop_82_adam_batch_normalization_5_beta_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83╝
AssignVariableOp_83AssignVariableOp4assignvariableop_83_adam_conv2d_transpose_1_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84║
AssignVariableOp_84AssignVariableOp2assignvariableop_84_adam_conv2d_transpose_1_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85Й
AssignVariableOp_85AssignVariableOp6assignvariableop_85_adam_batch_normalization_6_gamma_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86й
AssignVariableOp_86AssignVariableOp5assignvariableop_86_adam_batch_normalization_6_beta_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87╝
AssignVariableOp_87AssignVariableOp4assignvariableop_87_adam_conv2d_transpose_2_kernel_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88║
AssignVariableOp_88AssignVariableOp2assignvariableop_88_adam_conv2d_transpose_2_bias_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89Й
AssignVariableOp_89AssignVariableOp6assignvariableop_89_adam_batch_normalization_7_gamma_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90й
AssignVariableOp_90AssignVariableOp5assignvariableop_90_adam_batch_normalization_7_beta_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91╝
AssignVariableOp_91AssignVariableOp4assignvariableop_91_adam_conv2d_transpose_3_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92║
AssignVariableOp_92AssignVariableOp2assignvariableop_92_adam_conv2d_transpose_3_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93Й
AssignVariableOp_93AssignVariableOp6assignvariableop_93_adam_batch_normalization_8_gamma_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94й
AssignVariableOp_94AssignVariableOp5assignvariableop_94_adam_batch_normalization_8_beta_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95╝
AssignVariableOp_95AssignVariableOp4assignvariableop_95_adam_conv2d_transpose_4_kernel_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96║
AssignVariableOp_96AssignVariableOp2assignvariableop_96_adam_conv2d_transpose_4_bias_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97╝
AssignVariableOp_97AssignVariableOp4assignvariableop_97_adam_batch_normalization_gamma_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98╗
AssignVariableOp_98AssignVariableOp3assignvariableop_98_adam_batch_normalization_beta_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99░
AssignVariableOp_99AssignVariableOp(assignvariableop_99_adam_conv2d_kernel_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100▓
AssignVariableOp_100AssignVariableOp'assignvariableop_100_adam_conv2d_bias_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101┬
AssignVariableOp_101AssignVariableOp7assignvariableop_101_adam_batch_normalization_1_gamma_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102┴
AssignVariableOp_102AssignVariableOp6assignvariableop_102_adam_batch_normalization_1_beta_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103Х
AssignVariableOp_103AssignVariableOp+assignvariableop_103_adam_conv2d_1_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104┤
AssignVariableOp_104AssignVariableOp)assignvariableop_104_adam_conv2d_1_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105┬
AssignVariableOp_105AssignVariableOp7assignvariableop_105_adam_batch_normalization_2_gamma_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106┴
AssignVariableOp_106AssignVariableOp6assignvariableop_106_adam_batch_normalization_2_beta_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107Х
AssignVariableOp_107AssignVariableOp+assignvariableop_107_adam_conv2d_2_kernel_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108┤
AssignVariableOp_108AssignVariableOp)assignvariableop_108_adam_conv2d_2_bias_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109┬
AssignVariableOp_109AssignVariableOp7assignvariableop_109_adam_batch_normalization_3_gamma_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110┴
AssignVariableOp_110AssignVariableOp6assignvariableop_110_adam_batch_normalization_3_beta_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111Х
AssignVariableOp_111AssignVariableOp+assignvariableop_111_adam_conv2d_3_kernel_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112┤
AssignVariableOp_112AssignVariableOp)assignvariableop_112_adam_conv2d_3_bias_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113┬
AssignVariableOp_113AssignVariableOp7assignvariableop_113_adam_batch_normalization_4_gamma_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114┴
AssignVariableOp_114AssignVariableOp6assignvariableop_114_adam_batch_normalization_4_beta_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_114q
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:2
Identity_115Й
AssignVariableOp_115AssignVariableOp3assignvariableop_115_adam_conv2d_transpose_kernel_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_115q
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:2
Identity_116╝
AssignVariableOp_116AssignVariableOp1assignvariableop_116_adam_conv2d_transpose_bias_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_116q
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:2
Identity_117┬
AssignVariableOp_117AssignVariableOp7assignvariableop_117_adam_batch_normalization_5_gamma_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_117q
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:2
Identity_118┴
AssignVariableOp_118AssignVariableOp6assignvariableop_118_adam_batch_normalization_5_beta_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_118q
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:2
Identity_119└
AssignVariableOp_119AssignVariableOp5assignvariableop_119_adam_conv2d_transpose_1_kernel_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119q
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:2
Identity_120Й
AssignVariableOp_120AssignVariableOp3assignvariableop_120_adam_conv2d_transpose_1_bias_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_120q
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:2
Identity_121┬
AssignVariableOp_121AssignVariableOp7assignvariableop_121_adam_batch_normalization_6_gamma_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_121q
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:2
Identity_122┴
AssignVariableOp_122AssignVariableOp6assignvariableop_122_adam_batch_normalization_6_beta_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_122q
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:2
Identity_123└
AssignVariableOp_123AssignVariableOp5assignvariableop_123_adam_conv2d_transpose_2_kernel_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_123q
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:2
Identity_124Й
AssignVariableOp_124AssignVariableOp3assignvariableop_124_adam_conv2d_transpose_2_bias_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_124q
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:2
Identity_125┬
AssignVariableOp_125AssignVariableOp7assignvariableop_125_adam_batch_normalization_7_gamma_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_125q
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:2
Identity_126┴
AssignVariableOp_126AssignVariableOp6assignvariableop_126_adam_batch_normalization_7_beta_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_126q
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:2
Identity_127└
AssignVariableOp_127AssignVariableOp5assignvariableop_127_adam_conv2d_transpose_3_kernel_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_127q
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:2
Identity_128Й
AssignVariableOp_128AssignVariableOp3assignvariableop_128_adam_conv2d_transpose_3_bias_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_128q
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:2
Identity_129┬
AssignVariableOp_129AssignVariableOp7assignvariableop_129_adam_batch_normalization_8_gamma_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129q
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:2
Identity_130┴
AssignVariableOp_130AssignVariableOp6assignvariableop_130_adam_batch_normalization_8_beta_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_130q
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:2
Identity_131└
AssignVariableOp_131AssignVariableOp5assignvariableop_131_adam_conv2d_transpose_4_kernel_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_131q
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:2
Identity_132Й
AssignVariableOp_132AssignVariableOp3assignvariableop_132_adam_conv2d_transpose_4_bias_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1329
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp№
Identity_133Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_133i
Identity_134IdentityIdentity_133:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_134Н
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"%
identity_134Identity_134:output:0*А
_input_shapesЈ
ї: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
н
Й
O__inference_batch_normalization_layer_call_and_return_conditional_losses_162906

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┌
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ:::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:         ђђ2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         ђђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
▓
а
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_162174

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Л
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЩЩђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3z
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЩЩђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs
ы
ч
B__inference_conv2d_layer_call_and_return_conditional_losses_161907

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЧЧ@*
paddingVALID*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpі
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЧЧ@2	
BiasAdd_
EluEluBiasAdd:output:0*
T0*1
_output_shapes
:         ЧЧ@2
Eluv
IdentityIdentityElu:activations:0^NoOp*
T0*1
_output_shapes
:         ЧЧ@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ђђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
Е
а
)__inference_conv2d_1_layer_call_fn_164832

inputs"
unknown:@ђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЩЩђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_1619512
StatefulPartitionedCallє
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЧЧ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ЧЧ@
 
_user_specified_nameinputs
І	
Н
6__inference_batch_normalization_7_layer_call_fn_165782

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЧЧђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1622302
StatefulPartitionedCallє
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ЧЧђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЧЧђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ЧЧђ
 
_user_specified_nameinputs
Ћ
Й
O__inference_batch_normalization_layer_call_and_return_conditional_losses_160300

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1Ѕ
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Т
─
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_165255

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▀
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ШШђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1z
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ШШђ2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ШШђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:         ШШђ
 
_user_specified_nameinputs
 ~
к
H__inference_CNN_baseline_layer_call_and_return_conditional_losses_163169

inputs(
batch_normalization_163042:(
batch_normalization_163044:(
batch_normalization_163046:(
batch_normalization_163048:'
conv2d_163051:@
conv2d_163053:@*
batch_normalization_1_163056:@*
batch_normalization_1_163058:@*
batch_normalization_1_163060:@*
batch_normalization_1_163062:@*
conv2d_1_163065:@ђ
conv2d_1_163067:	ђ+
batch_normalization_2_163070:	ђ+
batch_normalization_2_163072:	ђ+
batch_normalization_2_163074:	ђ+
batch_normalization_2_163076:	ђ+
conv2d_2_163079:ђђ
conv2d_2_163081:	ђ+
batch_normalization_3_163084:	ђ+
batch_normalization_3_163086:	ђ+
batch_normalization_3_163088:	ђ+
batch_normalization_3_163090:	ђ+
conv2d_3_163093:ђђ
conv2d_3_163095:	ђ+
batch_normalization_4_163098:	ђ+
batch_normalization_4_163100:	ђ+
batch_normalization_4_163102:	ђ+
batch_normalization_4_163104:	ђ3
conv2d_transpose_163107:ђђ&
conv2d_transpose_163109:	ђ+
batch_normalization_5_163112:	ђ+
batch_normalization_5_163114:	ђ+
batch_normalization_5_163116:	ђ+
batch_normalization_5_163118:	ђ5
conv2d_transpose_1_163121:ђђ(
conv2d_transpose_1_163123:	ђ+
batch_normalization_6_163126:	ђ+
batch_normalization_6_163128:	ђ+
batch_normalization_6_163130:	ђ+
batch_normalization_6_163132:	ђ5
conv2d_transpose_2_163135:ђђ(
conv2d_transpose_2_163137:	ђ+
batch_normalization_7_163140:	ђ+
batch_normalization_7_163142:	ђ+
batch_normalization_7_163144:	ђ+
batch_normalization_7_163146:	ђ4
conv2d_transpose_3_163149:@ђ'
conv2d_transpose_3_163151:@*
batch_normalization_8_163154:@*
batch_normalization_8_163156:@*
batch_normalization_8_163158:@*
batch_normalization_8_163160:@3
conv2d_transpose_4_163163:@'
conv2d_transpose_4_163165:
identityѕб+batch_normalization/StatefulPartitionedCallб-batch_normalization_1/StatefulPartitionedCallб-batch_normalization_2/StatefulPartitionedCallб-batch_normalization_3/StatefulPartitionedCallб-batch_normalization_4/StatefulPartitionedCallб-batch_normalization_5/StatefulPartitionedCallб-batch_normalization_6/StatefulPartitionedCallб-batch_normalization_7/StatefulPartitionedCallб-batch_normalization_8/StatefulPartitionedCallбconv2d/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallб conv2d_2/StatefulPartitionedCallб conv2d_3/StatefulPartitionedCallб(conv2d_transpose/StatefulPartitionedCallб*conv2d_transpose_1/StatefulPartitionedCallб*conv2d_transpose_2/StatefulPartitionedCallб*conv2d_transpose_3/StatefulPartitionedCallб*conv2d_transpose_4/StatefulPartitionedCallя
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1618672
reshape/PartitionedCallг
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0batch_normalization_163042batch_normalization_163044batch_normalization_163046batch_normalization_163048*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1629062-
+batch_normalization/StatefulPartitionedCall┼
conv2d/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_163051conv2d_163053*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЧЧ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_1619072 
conv2d/StatefulPartitionedCall┴
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_1_163056batch_normalization_1_163058batch_normalization_1_163060batch_normalization_1_163062*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЧЧ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1628522/
-batch_normalization_1/StatefulPartitionedCallм
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_1_163065conv2d_1_163067*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЩЩђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_1619512"
 conv2d_1/StatefulPartitionedCall─
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_2_163070batch_normalization_2_163072batch_normalization_2_163074batch_normalization_2_163076*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЩЩђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1627982/
-batch_normalization_2/StatefulPartitionedCallм
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_2_163079conv2d_2_163081*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1619952"
 conv2d_2/StatefulPartitionedCall─
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_3_163084batch_normalization_3_163086batch_normalization_3_163088batch_normalization_3_163090*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1627442/
-batch_normalization_3/StatefulPartitionedCallм
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_3_163093conv2d_3_163095*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ШШђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_1620392"
 conv2d_3/StatefulPartitionedCall─
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_4_163098batch_normalization_4_163100batch_normalization_4_163102batch_normalization_4_163104*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ШШђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1626902/
-batch_normalization_4/StatefulPartitionedCallЩ
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0conv2d_transpose_163107conv2d_transpose_163109*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1620952*
(conv2d_transpose/StatefulPartitionedCall╠
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0batch_normalization_5_163112batch_normalization_5_163114batch_normalization_5_163116batch_normalization_5_163118*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1626372/
-batch_normalization_5/StatefulPartitionedCallё
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0conv2d_transpose_1_163121conv2d_transpose_1_163123*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЩЩђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1621512,
*conv2d_transpose_1/StatefulPartitionedCall╬
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0batch_normalization_6_163126batch_normalization_6_163128batch_normalization_6_163130batch_normalization_6_163132*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЩЩђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1625842/
-batch_normalization_6/StatefulPartitionedCallё
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0conv2d_transpose_2_163135conv2d_transpose_2_163137*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЧЧђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1622072,
*conv2d_transpose_2/StatefulPartitionedCall╬
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0batch_normalization_7_163140batch_normalization_7_163142batch_normalization_7_163144batch_normalization_7_163146*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЧЧђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1625312/
-batch_normalization_7/StatefulPartitionedCallЃ
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0conv2d_transpose_3_163149conv2d_transpose_3_163151*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1622632,
*conv2d_transpose_3/StatefulPartitionedCall═
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0batch_normalization_8_163154batch_normalization_8_163156batch_normalization_8_163158batch_normalization_8_163160*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1624782/
-batch_normalization_8/StatefulPartitionedCallЃ
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_transpose_4_163163conv2d_transpose_4_163165*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1623182,
*conv2d_transpose_4/StatefulPartitionedCallў
IdentityIdentity3conv2d_transpose_4/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ2

Identityт
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*џ
_input_shapesѕ
Ё:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
:         ђђ
 
_user_specified_nameinputs
Ч~
┼
H__inference_CNN_baseline_layer_call_and_return_conditional_losses_163655	
u_vel(
batch_normalization_163528:(
batch_normalization_163530:(
batch_normalization_163532:(
batch_normalization_163534:'
conv2d_163537:@
conv2d_163539:@*
batch_normalization_1_163542:@*
batch_normalization_1_163544:@*
batch_normalization_1_163546:@*
batch_normalization_1_163548:@*
conv2d_1_163551:@ђ
conv2d_1_163553:	ђ+
batch_normalization_2_163556:	ђ+
batch_normalization_2_163558:	ђ+
batch_normalization_2_163560:	ђ+
batch_normalization_2_163562:	ђ+
conv2d_2_163565:ђђ
conv2d_2_163567:	ђ+
batch_normalization_3_163570:	ђ+
batch_normalization_3_163572:	ђ+
batch_normalization_3_163574:	ђ+
batch_normalization_3_163576:	ђ+
conv2d_3_163579:ђђ
conv2d_3_163581:	ђ+
batch_normalization_4_163584:	ђ+
batch_normalization_4_163586:	ђ+
batch_normalization_4_163588:	ђ+
batch_normalization_4_163590:	ђ3
conv2d_transpose_163593:ђђ&
conv2d_transpose_163595:	ђ+
batch_normalization_5_163598:	ђ+
batch_normalization_5_163600:	ђ+
batch_normalization_5_163602:	ђ+
batch_normalization_5_163604:	ђ5
conv2d_transpose_1_163607:ђђ(
conv2d_transpose_1_163609:	ђ+
batch_normalization_6_163612:	ђ+
batch_normalization_6_163614:	ђ+
batch_normalization_6_163616:	ђ+
batch_normalization_6_163618:	ђ5
conv2d_transpose_2_163621:ђђ(
conv2d_transpose_2_163623:	ђ+
batch_normalization_7_163626:	ђ+
batch_normalization_7_163628:	ђ+
batch_normalization_7_163630:	ђ+
batch_normalization_7_163632:	ђ4
conv2d_transpose_3_163635:@ђ'
conv2d_transpose_3_163637:@*
batch_normalization_8_163640:@*
batch_normalization_8_163642:@*
batch_normalization_8_163644:@*
batch_normalization_8_163646:@3
conv2d_transpose_4_163649:@'
conv2d_transpose_4_163651:
identityѕб+batch_normalization/StatefulPartitionedCallб-batch_normalization_1/StatefulPartitionedCallб-batch_normalization_2/StatefulPartitionedCallб-batch_normalization_3/StatefulPartitionedCallб-batch_normalization_4/StatefulPartitionedCallб-batch_normalization_5/StatefulPartitionedCallб-batch_normalization_6/StatefulPartitionedCallб-batch_normalization_7/StatefulPartitionedCallб-batch_normalization_8/StatefulPartitionedCallбconv2d/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallб conv2d_2/StatefulPartitionedCallб conv2d_3/StatefulPartitionedCallб(conv2d_transpose/StatefulPartitionedCallб*conv2d_transpose_1/StatefulPartitionedCallб*conv2d_transpose_2/StatefulPartitionedCallб*conv2d_transpose_3/StatefulPartitionedCallб*conv2d_transpose_4/StatefulPartitionedCallП
reshape/PartitionedCallPartitionedCallu_vel*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1618672
reshape/PartitionedCallг
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0batch_normalization_163528batch_normalization_163530batch_normalization_163532batch_normalization_163534*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1629062-
+batch_normalization/StatefulPartitionedCall┼
conv2d/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_163537conv2d_163539*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЧЧ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_1619072 
conv2d/StatefulPartitionedCall┴
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_1_163542batch_normalization_1_163544batch_normalization_1_163546batch_normalization_1_163548*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЧЧ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1628522/
-batch_normalization_1/StatefulPartitionedCallм
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_1_163551conv2d_1_163553*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЩЩђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_1619512"
 conv2d_1/StatefulPartitionedCall─
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_2_163556batch_normalization_2_163558batch_normalization_2_163560batch_normalization_2_163562*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЩЩђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1627982/
-batch_normalization_2/StatefulPartitionedCallм
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_2_163565conv2d_2_163567*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1619952"
 conv2d_2/StatefulPartitionedCall─
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_3_163570batch_normalization_3_163572batch_normalization_3_163574batch_normalization_3_163576*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1627442/
-batch_normalization_3/StatefulPartitionedCallм
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_3_163579conv2d_3_163581*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ШШђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_1620392"
 conv2d_3/StatefulPartitionedCall─
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_4_163584batch_normalization_4_163586batch_normalization_4_163588batch_normalization_4_163590*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ШШђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1626902/
-batch_normalization_4/StatefulPartitionedCallЩ
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0conv2d_transpose_163593conv2d_transpose_163595*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1620952*
(conv2d_transpose/StatefulPartitionedCall╠
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0batch_normalization_5_163598batch_normalization_5_163600batch_normalization_5_163602batch_normalization_5_163604*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1626372/
-batch_normalization_5/StatefulPartitionedCallё
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0conv2d_transpose_1_163607conv2d_transpose_1_163609*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЩЩђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1621512,
*conv2d_transpose_1/StatefulPartitionedCall╬
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0batch_normalization_6_163612batch_normalization_6_163614batch_normalization_6_163616batch_normalization_6_163618*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЩЩђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1625842/
-batch_normalization_6/StatefulPartitionedCallё
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0conv2d_transpose_2_163621conv2d_transpose_2_163623*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЧЧђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1622072,
*conv2d_transpose_2/StatefulPartitionedCall╬
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0batch_normalization_7_163626batch_normalization_7_163628batch_normalization_7_163630batch_normalization_7_163632*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЧЧђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1625312/
-batch_normalization_7/StatefulPartitionedCallЃ
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0conv2d_transpose_3_163635conv2d_transpose_3_163637*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1622632,
*conv2d_transpose_3/StatefulPartitionedCall═
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0batch_normalization_8_163640batch_normalization_8_163642batch_normalization_8_163644batch_normalization_8_163646*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1624782/
-batch_normalization_8/StatefulPartitionedCallЃ
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_transpose_4_163649conv2d_transpose_4_163651*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1623182,
*conv2d_transpose_4/StatefulPartitionedCallў
IdentityIdentity3conv2d_transpose_4/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ2

Identityт
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*џ
_input_shapesѕ
Ё:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
:         ђђ

_user_specified_nameu_vel
┘'
Џ
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_161793

inputsB
(conv2d_transpose_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2В
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3ѓ
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2В
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3│
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_transpose/ReadVariableOpы
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
2
conv2d_transposeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpц
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAddЁ
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+                           2

IdentityЅ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Д
─
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_160552

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1і
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
ф
ю
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_162263

inputsC
(conv2d_transpose_readvariableop_resource:@ђ-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceU
stack/1Const*
_output_shapes
: *
dtype0*
value
B :ђ2	
stack/1U
stack/2Const*
_output_shapes
: *
dtype0*
value
B :ђ2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3њ
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1┤
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02!
conv2d_transpose/ReadVariableOpр
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:         ђђ@*
paddingVALID*
strides
2
conv2d_transposeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpћ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ@2	
BiasAdd_
EluEluBiasAdd:output:0*
T0*1
_output_shapes
:         ђђ@2
Eluv
IdentityIdentityElu:activations:0^NoOp*
T0*1
_output_shapes
:         ђђ@2

IdentityЅ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ЧЧђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Z V
2
_output_shapes 
:         ЧЧђ
 
_user_specified_nameinputs
Т
─
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_165111

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▀
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЭЭђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1z
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђ2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЭЭђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
з
а
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_165405

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1р
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3і
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Т
─
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_162690

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▀
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ШШђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1z
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ШШђ2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ШШђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:         ШШђ
 
_user_specified_nameinputs
І	
Н
6__inference_batch_normalization_2_layer_call_fn_164882

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЩЩђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1619742
StatefulPartitionedCallє
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЩЩђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs
╣
е
3__inference_conv2d_transpose_4_layer_call_fn_166089

inputs!
unknown:@
	unknown_0:
identityѕбStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1623182
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ђђ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ђђ@
 
_user_specified_nameinputs
з
а
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_160760

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1р
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3і
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
П
D
(__inference_reshape_layer_call_fn_164541

inputs
identity╬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1618672
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ђђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ђђ:U Q
-
_output_shapes
:         ђђ
 
_user_specified_nameinputs
б
ю
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_164805

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╠
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЧЧ@:@:@:@:@:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:         ЧЧ@2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         ЧЧ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:         ЧЧ@
 
_user_specified_nameinputs
Ѕ	
Н
6__inference_batch_normalization_6_layer_call_fn_165591

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЩЩђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1625842
StatefulPartitionedCallє
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЩЩђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs
Т
─
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_165459

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▀
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЭЭђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1z
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђ2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЭЭђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
Т
─
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_162584

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▀
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЩЩђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1z
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђ2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЩЩђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs
з
а
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_165201

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1р
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3і
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
▓
а
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_162230

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Л
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЧЧђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3z
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЧЧђ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЧЧђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:         ЧЧђ
 
_user_specified_nameinputs
б
ю
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_161930

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╠
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЧЧ@:@:@:@:@:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:         ЧЧ@2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         ЧЧ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:         ЧЧ@
 
_user_specified_nameinputs
─
Џ
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_162318

inputsB
(conv2d_transpose_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceU
stack/1Const*
_output_shapes
: *
dtype0*
value
B :ђ2	
stack/1U
stack/2Const*
_output_shapes
: *
dtype0*
value
B :ђ2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3њ
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1│
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_transpose/ReadVariableOpр
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:         ђђ*
paddingVALID*
strides
2
conv2d_transposeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpћ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:         ђђ2

IdentityЅ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ђђ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:         ђђ@
 
_user_specified_nameinputs
Ѓ	
Л
6__inference_batch_normalization_1_layer_call_fn_164738

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЧЧ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1619302
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЧЧ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         ЧЧ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ЧЧ@
 
_user_specified_nameinputs
─
Џ
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_166149

inputsB
(conv2d_transpose_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceU
stack/1Const*
_output_shapes
: *
dtype0*
value
B :ђ2	
stack/1U
stack/2Const*
_output_shapes
: *
dtype0*
value
B :ђ2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3њ
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1│
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_transpose/ReadVariableOpр
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:         ђђ*
paddingVALID*
strides
2
conv2d_transposeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpћ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:         ђђ2

IdentityЅ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ђђ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:         ђђ@
 
_user_specified_nameinputs
Ѓ	
Л
6__inference_batch_normalization_8_layer_call_fn_165986

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1622862
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         ђђ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ђђ@
 
_user_specified_nameinputs
Д
─
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_160804

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1і
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
о(
ю
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_165311

inputsD
(conv2d_transpose_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2В
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2	
stack/3ѓ
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2В
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3х
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_transpose/ReadVariableOpЫ
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,                           ђ*
paddingVALID*
strides
2
conv2d_transposeЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЦ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђ2	
BiasAddp
EluEluBiasAdd:output:0*
T0*B
_output_shapes0
.:,                           ђ2
EluЄ
IdentityIdentityElu:activations:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

IdentityЅ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Д
─
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_165627

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1і
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Т
─
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_162798

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▀
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЩЩђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1z
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђ2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЩЩђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs
▓
а
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_161974

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Л
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЩЩђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3z
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЩЩђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs
р
џ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_160256

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3Ѕ
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ш
р
$__inference_signature_wrapper_163776	
u_vel
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@$
	unknown_9:@ђ

unknown_10:	ђ

unknown_11:	ђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:	ђ&

unknown_15:ђђ

unknown_16:	ђ

unknown_17:	ђ

unknown_18:	ђ

unknown_19:	ђ

unknown_20:	ђ&

unknown_21:ђђ

unknown_22:	ђ

unknown_23:	ђ

unknown_24:	ђ

unknown_25:	ђ

unknown_26:	ђ&

unknown_27:ђђ

unknown_28:	ђ

unknown_29:	ђ

unknown_30:	ђ

unknown_31:	ђ

unknown_32:	ђ&

unknown_33:ђђ

unknown_34:	ђ

unknown_35:	ђ

unknown_36:	ђ

unknown_37:	ђ

unknown_38:	ђ&

unknown_39:ђђ

unknown_40:	ђ

unknown_41:	ђ

unknown_42:	ђ

unknown_43:	ђ

unknown_44:	ђ%

unknown_45:@ђ

unknown_46:@

unknown_47:@

unknown_48:@

unknown_49:@

unknown_50:@$

unknown_51:@

unknown_52:
identityѕбStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallu_velunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_52*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456*0
config_proto 

CPU

GPU2*0J 8ѓ **
f%R#
!__inference__wrapped_model_1602342
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*џ
_input_shapesѕ
Ё:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
-
_output_shapes
:         ђђ

_user_specified_nameu_vel
▓
а
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_162118

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Л
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЭЭђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3z
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЭЭђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
с
ю
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_160382

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3Ѕ
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╔	
Н
6__inference_batch_normalization_2_layer_call_fn_164869

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1605522
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Ђ	
Л
6__inference_batch_normalization_1_layer_call_fn_164751

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЧЧ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1628522
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЧЧ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         ЧЧ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ЧЧ@
 
_user_specified_nameinputs
╦	
Н
6__inference_batch_normalization_3_layer_call_fn_165000

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCall╣
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1606342
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
└
Ф
3__inference_conv2d_transpose_2_layer_call_fn_165681

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЧЧђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1622072
StatefulPartitionedCallє
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ЧЧђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ЩЩђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs
щ
е
3__inference_conv2d_transpose_4_layer_call_fn_166080

inputs!
unknown:@
	unknown_0:
identityѕбStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1617932
StatefulPartitionedCallЋ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           @: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╦	
Н
6__inference_batch_normalization_5_layer_call_fn_165348

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCall╣
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1609822
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
г
А
)__inference_conv2d_3_layer_call_fn_165120

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ШШђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_1620392
StatefulPartitionedCallє
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ШШђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ЭЭђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
о(
ю
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_160906

inputsD
(conv2d_transpose_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2В
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2	
stack/3ѓ
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2В
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3х
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_transpose/ReadVariableOpЫ
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,                           ђ*
paddingVALID*
strides
2
conv2d_transposeЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЦ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђ2	
BiasAddp
EluEluBiasAdd:output:0*
T0*B
_output_shapes0
.:,                           ђ2
EluЄ
IdentityIdentityElu:activations:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

IdentityЅ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
│
ъ
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_165539

inputsD
(conv2d_transpose_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceU
stack/1Const*
_output_shapes
: *
dtype0*
value
B :Щ2	
stack/1U
stack/2Const*
_output_shapes
: *
dtype0*
value
B :Щ2	
stack/2U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2	
stack/3њ
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1х
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_transpose/ReadVariableOpР
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*2
_output_shapes 
:         ЩЩђ*
paddingVALID*
strides
2
conv2d_transposeЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЋ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЩЩђ2	
BiasAdd`
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:         ЩЩђ2
Eluw
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђ2

IdentityЅ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ЭЭђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
¤(
ю
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_161572

inputsC
(conv2d_transpose_readvariableop_resource:@ђ-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2В
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3ѓ
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2В
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3┤
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02!
conv2d_transpose/ReadVariableOpы
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           @*
paddingVALID*
strides
2
conv2d_transposeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpц
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @2	
BiasAddo
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           @2
Eluє
IdentityIdentityElu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                           @2

IdentityЅ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
п(
ъ
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_165719

inputsD
(conv2d_transpose_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2В
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2	
stack/3ѓ
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2В
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3х
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_transpose/ReadVariableOpЫ
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,                           ђ*
paddingVALID*
strides
2
conv2d_transposeЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЦ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђ2	
BiasAddp
EluEluBiasAdd:output:0*
T0*B
_output_shapes0
.:,                           ђ2
EluЄ
IdentityIdentityElu:activations:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

IdentityЅ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Ћ
Й
O__inference_batch_normalization_layer_call_and_return_conditional_losses_164643

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1Ѕ
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Д
─
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_161248

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1і
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
РЅ
М?
__inference__traced_save_166571
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

identity_1ѕбMergeV2CheckpointsЈ
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1І
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameЊK
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:є*
dtype0*цJ
valueџJBЌJєB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЎ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:є*
dtype0*б
valueўBЋєB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesќ=
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop2savev2_conv2d_transpose_kernel_read_readvariableop0savev2_conv2d_transpose_bias_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop4savev2_conv2d_transpose_1_kernel_read_readvariableop2savev2_conv2d_transpose_1_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop4savev2_conv2d_transpose_2_kernel_read_readvariableop2savev2_conv2d_transpose_2_bias_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop4savev2_conv2d_transpose_3_kernel_read_readvariableop2savev2_conv2d_transpose_3_bias_read_readvariableop6savev2_batch_normalization_8_gamma_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableop4savev2_conv2d_transpose_4_kernel_read_readvariableop2savev2_conv2d_transpose_4_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop<savev2_adam_batch_normalization_2_beta_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop<savev2_adam_batch_normalization_3_beta_m_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_3_bias_m_read_readvariableop=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop<savev2_adam_batch_normalization_4_beta_m_read_readvariableop9savev2_adam_conv2d_transpose_kernel_m_read_readvariableop7savev2_adam_conv2d_transpose_bias_m_read_readvariableop=savev2_adam_batch_normalization_5_gamma_m_read_readvariableop<savev2_adam_batch_normalization_5_beta_m_read_readvariableop;savev2_adam_conv2d_transpose_1_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_1_bias_m_read_readvariableop=savev2_adam_batch_normalization_6_gamma_m_read_readvariableop<savev2_adam_batch_normalization_6_beta_m_read_readvariableop;savev2_adam_conv2d_transpose_2_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_2_bias_m_read_readvariableop=savev2_adam_batch_normalization_7_gamma_m_read_readvariableop<savev2_adam_batch_normalization_7_beta_m_read_readvariableop;savev2_adam_conv2d_transpose_3_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_3_bias_m_read_readvariableop=savev2_adam_batch_normalization_8_gamma_m_read_readvariableop<savev2_adam_batch_normalization_8_beta_m_read_readvariableop;savev2_adam_conv2d_transpose_4_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_4_bias_m_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop<savev2_adam_batch_normalization_2_beta_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop<savev2_adam_batch_normalization_3_beta_v_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop/savev2_adam_conv2d_3_bias_v_read_readvariableop=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop<savev2_adam_batch_normalization_4_beta_v_read_readvariableop9savev2_adam_conv2d_transpose_kernel_v_read_readvariableop7savev2_adam_conv2d_transpose_bias_v_read_readvariableop=savev2_adam_batch_normalization_5_gamma_v_read_readvariableop<savev2_adam_batch_normalization_5_beta_v_read_readvariableop;savev2_adam_conv2d_transpose_1_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_1_bias_v_read_readvariableop=savev2_adam_batch_normalization_6_gamma_v_read_readvariableop<savev2_adam_batch_normalization_6_beta_v_read_readvariableop;savev2_adam_conv2d_transpose_2_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_2_bias_v_read_readvariableop=savev2_adam_batch_normalization_7_gamma_v_read_readvariableop<savev2_adam_batch_normalization_7_beta_v_read_readvariableop;savev2_adam_conv2d_transpose_3_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_3_bias_v_read_readvariableop=savev2_adam_batch_normalization_8_gamma_v_read_readvariableop<savev2_adam_batch_normalization_8_beta_v_read_readvariableop;savev2_adam_conv2d_transpose_4_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_4_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Ќ
dtypesї
Ѕ2є	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*┼	
_input_shapes│	
░	: :::::@:@:@:@:@:@:@ђ:ђ:ђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:ђ:ђ:@ђ:@:@:@:@:@:@:: : : : : : : :::@:@:@:@:@ђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:@ђ:@:@:@:@::::@:@:@:@:@ђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:@ђ:@:@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:@: 
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
:@ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:.*
(
_output_shapes
:ђђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:.*
(
_output_shapes
:ђђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:.*
(
_output_shapes
:ђђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:! 

_output_shapes	
:ђ:!!

_output_shapes	
:ђ:!"

_output_shapes	
:ђ:.#*
(
_output_shapes
:ђђ:!$

_output_shapes	
:ђ:!%

_output_shapes	
:ђ:!&

_output_shapes	
:ђ:!'

_output_shapes	
:ђ:!(

_output_shapes	
:ђ:.)*
(
_output_shapes
:ђђ:!*

_output_shapes	
:ђ:!+

_output_shapes	
:ђ:!,

_output_shapes	
:ђ:!-

_output_shapes	
:ђ:!.

_output_shapes	
:ђ:-/)
'
_output_shapes
:@ђ: 0
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
:@: 6
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
:: ?

_output_shapes
::,@(
&
_output_shapes
:@: A
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
:@ђ:!E

_output_shapes	
:ђ:!F

_output_shapes	
:ђ:!G

_output_shapes	
:ђ:.H*
(
_output_shapes
:ђђ:!I

_output_shapes	
:ђ:!J

_output_shapes	
:ђ:!K

_output_shapes	
:ђ:.L*
(
_output_shapes
:ђђ:!M

_output_shapes	
:ђ:!N

_output_shapes	
:ђ:!O

_output_shapes	
:ђ:.P*
(
_output_shapes
:ђђ:!Q

_output_shapes	
:ђ:!R

_output_shapes	
:ђ:!S

_output_shapes	
:ђ:.T*
(
_output_shapes
:ђђ:!U

_output_shapes	
:ђ:!V

_output_shapes	
:ђ:!W

_output_shapes	
:ђ:.X*
(
_output_shapes
:ђђ:!Y

_output_shapes	
:ђ:!Z

_output_shapes	
:ђ:![

_output_shapes	
:ђ:-\)
'
_output_shapes
:@ђ: ]
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
:@: a

_output_shapes
:: b

_output_shapes
:: c

_output_shapes
::,d(
&
_output_shapes
:@: e
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
:@ђ:!i

_output_shapes	
:ђ:!j

_output_shapes	
:ђ:!k

_output_shapes	
:ђ:.l*
(
_output_shapes
:ђђ:!m

_output_shapes	
:ђ:!n

_output_shapes	
:ђ:!o

_output_shapes	
:ђ:.p*
(
_output_shapes
:ђђ:!q

_output_shapes	
:ђ:!r

_output_shapes	
:ђ:!s

_output_shapes	
:ђ:.t*
(
_output_shapes
:ђђ:!u

_output_shapes	
:ђ:!v

_output_shapes	
:ђ:!w

_output_shapes	
:ђ:.x*
(
_output_shapes
:ђђ:!y

_output_shapes	
:ђ:!z

_output_shapes	
:ђ:!{

_output_shapes	
:ђ:.|*
(
_output_shapes
:ђђ:!}

_output_shapes	
:ђ:!~

_output_shapes	
:ђ:!

_output_shapes	
:ђ:.ђ)
'
_output_shapes
:@ђ:!Ђ

_output_shapes
:@:!ѓ

_output_shapes
:@:!Ѓ

_output_shapes
:@:-ё(
&
_output_shapes
:@:!Ё

_output_shapes
::є

_output_shapes
: 
з
а
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_165813

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1р
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3і
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
а
џ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_161886

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╠
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ:::::*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:         ђђ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         ђђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
а
џ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_164661

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╠
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ:::::*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:         ђђ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         ђђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
 
ђ
D__inference_conv2d_2_layer_call_and_return_conditional_losses_164987

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02
Conv2D/ReadVariableOpД
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЭЭђ*
paddingVALID*
strides
2
Conv2DЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpІ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЭЭђ2	
BiasAdd`
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:         ЭЭђ2
Eluw
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ЩЩђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs
╦	
Н
6__inference_batch_normalization_4_layer_call_fn_165144

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCall╣
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1607602
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Ќ
└
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_160426

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1Ѕ
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Д
─
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_165831

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1і
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
└
Ф
3__inference_conv2d_transpose_1_layer_call_fn_165477

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЩЩђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1621512
StatefulPartitionedCallє
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ЭЭђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
╔	
Н
6__inference_batch_normalization_5_layer_call_fn_165361

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1610262
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Т
─
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_164967

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▀
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЩЩђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1z
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђ2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЩЩђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs
з
а
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_165057

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1р
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3і
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
о
└
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_166071

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┌
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ@:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:         ђђ@2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         ђђ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:         ђђ@
 
_user_specified_nameinputs
о
└
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_162478

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┌
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ@:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:         ђђ@2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         ђђ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:         ђђ@
 
_user_specified_nameinputs
 
ђ
D__inference_conv2d_2_layer_call_and_return_conditional_losses_161995

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02
Conv2D/ReadVariableOpД
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЭЭђ*
paddingVALID*
strides
2
Conv2DЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpІ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЭЭђ2	
BiasAdd`
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:         ЭЭђ2
Eluw
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ЩЩђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs
Ц
Ж
-__inference_CNN_baseline_layer_call_fn_162436	
u_vel
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@$
	unknown_9:@ђ

unknown_10:	ђ

unknown_11:	ђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:	ђ&

unknown_15:ђђ

unknown_16:	ђ

unknown_17:	ђ

unknown_18:	ђ

unknown_19:	ђ

unknown_20:	ђ&

unknown_21:ђђ

unknown_22:	ђ

unknown_23:	ђ

unknown_24:	ђ

unknown_25:	ђ

unknown_26:	ђ&

unknown_27:ђђ

unknown_28:	ђ

unknown_29:	ђ

unknown_30:	ђ

unknown_31:	ђ

unknown_32:	ђ&

unknown_33:ђђ

unknown_34:	ђ

unknown_35:	ђ

unknown_36:	ђ

unknown_37:	ђ

unknown_38:	ђ&

unknown_39:ђђ

unknown_40:	ђ

unknown_41:	ђ

unknown_42:	ђ

unknown_43:	ђ

unknown_44:	ђ%

unknown_45:@ђ

unknown_46:@

unknown_47:@

unknown_48:@

unknown_49:@

unknown_50:@$

unknown_51:@

unknown_52:
identityѕбStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallu_velunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_52*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_CNN_baseline_layer_call_and_return_conditional_losses_1623252
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*џ
_input_shapesѕ
Ё:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
-
_output_shapes
:         ђђ

_user_specified_nameu_vel
Ѕ	
Н
6__inference_batch_normalization_2_layer_call_fn_164895

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЩЩђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1627982
StatefulPartitionedCallє
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЩЩђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs
ђ	
Ф
3__inference_conv2d_transpose_1_layer_call_fn_165468

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1611282
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
б
ю
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_162286

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╠
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ@:@:@:@:@:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:         ђђ@2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         ђђ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:         ђђ@
 
_user_specified_nameinputs
Д
─
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_165219

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1і
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Ќ
└
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_164787

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1Ѕ
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Ђ	
Л
6__inference_batch_normalization_8_layer_call_fn_165999

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1624782
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         ђђ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ђђ@
 
_user_specified_nameinputs
п(
ъ
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_161350

inputsD
(conv2d_transpose_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2В
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2	
stack/3ѓ
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2В
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3х
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_transpose/ReadVariableOpЫ
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,                           ђ*
paddingVALID*
strides
2
conv2d_transposeЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЦ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђ2	
BiasAddp
EluEluBiasAdd:output:0*
T0*B
_output_shapes0
.:,                           ђ2
EluЄ
IdentityIdentityElu:activations:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

IdentityЅ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
ы
ч
B__inference_conv2d_layer_call_and_return_conditional_losses_164699

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЧЧ@*
paddingVALID*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpі
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЧЧ@2	
BiasAdd_
EluEluBiasAdd:output:0*
T0*1
_output_shapes
:         ЧЧ@2
Eluv
IdentityIdentityElu:activations:0^NoOp*
T0*1
_output_shapes
:         ЧЧ@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ђђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
з
а
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_160982

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1р
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3і
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
ч
 
D__inference_conv2d_1_layer_call_and_return_conditional_losses_161951

inputs9
conv2d_readvariableop_resource:@ђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpќ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02
Conv2D/ReadVariableOpД
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЩЩђ*
paddingVALID*
strides
2
Conv2DЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpІ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЩЩђ2	
BiasAdd`
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:         ЩЩђ2
Eluw
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЧЧ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ЧЧ@
 
_user_specified_nameinputs
б
ю
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_166053

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╠
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ@:@:@:@:@:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:         ђђ@2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         ђђ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:         ђђ@
 
_user_specified_nameinputs
├Ђ
Є2
H__inference_CNN_baseline_layer_call_and_return_conditional_losses_164269

inputs9
+batch_normalization_readvariableop_resource:;
-batch_normalization_readvariableop_1_resource:J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:?
%conv2d_conv2d_readvariableop_resource:@4
&conv2d_biasadd_readvariableop_resource:@;
-batch_normalization_1_readvariableop_resource:@=
/batch_normalization_1_readvariableop_1_resource:@L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@B
'conv2d_1_conv2d_readvariableop_resource:@ђ7
(conv2d_1_biasadd_readvariableop_resource:	ђ<
-batch_normalization_2_readvariableop_resource:	ђ>
/batch_normalization_2_readvariableop_1_resource:	ђM
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	ђO
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	ђC
'conv2d_2_conv2d_readvariableop_resource:ђђ7
(conv2d_2_biasadd_readvariableop_resource:	ђ<
-batch_normalization_3_readvariableop_resource:	ђ>
/batch_normalization_3_readvariableop_1_resource:	ђM
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	ђO
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	ђC
'conv2d_3_conv2d_readvariableop_resource:ђђ7
(conv2d_3_biasadd_readvariableop_resource:	ђ<
-batch_normalization_4_readvariableop_resource:	ђ>
/batch_normalization_4_readvariableop_1_resource:	ђM
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	ђO
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	ђU
9conv2d_transpose_conv2d_transpose_readvariableop_resource:ђђ?
0conv2d_transpose_biasadd_readvariableop_resource:	ђ<
-batch_normalization_5_readvariableop_resource:	ђ>
/batch_normalization_5_readvariableop_1_resource:	ђM
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:	ђO
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:	ђW
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:ђђA
2conv2d_transpose_1_biasadd_readvariableop_resource:	ђ<
-batch_normalization_6_readvariableop_resource:	ђ>
/batch_normalization_6_readvariableop_1_resource:	ђM
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource:	ђO
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:	ђW
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource:ђђA
2conv2d_transpose_2_biasadd_readvariableop_resource:	ђ<
-batch_normalization_7_readvariableop_resource:	ђ>
/batch_normalization_7_readvariableop_1_resource:	ђM
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:	ђO
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:	ђV
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource:@ђ@
2conv2d_transpose_3_biasadd_readvariableop_resource:@;
-batch_normalization_8_readvariableop_resource:@=
/batch_normalization_8_readvariableop_1_resource:@L
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:@U
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource:@@
2conv2d_transpose_4_biasadd_readvariableop_resource:
identityѕб3batch_normalization/FusedBatchNormV3/ReadVariableOpб5batch_normalization/FusedBatchNormV3/ReadVariableOp_1б"batch_normalization/ReadVariableOpб$batch_normalization/ReadVariableOp_1б5batch_normalization_1/FusedBatchNormV3/ReadVariableOpб7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_1/ReadVariableOpб&batch_normalization_1/ReadVariableOp_1б5batch_normalization_2/FusedBatchNormV3/ReadVariableOpб7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_2/ReadVariableOpб&batch_normalization_2/ReadVariableOp_1б5batch_normalization_3/FusedBatchNormV3/ReadVariableOpб7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_3/ReadVariableOpб&batch_normalization_3/ReadVariableOp_1б5batch_normalization_4/FusedBatchNormV3/ReadVariableOpб7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_4/ReadVariableOpб&batch_normalization_4/ReadVariableOp_1б5batch_normalization_5/FusedBatchNormV3/ReadVariableOpб7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_5/ReadVariableOpб&batch_normalization_5/ReadVariableOp_1б5batch_normalization_6/FusedBatchNormV3/ReadVariableOpб7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_6/ReadVariableOpб&batch_normalization_6/ReadVariableOp_1б5batch_normalization_7/FusedBatchNormV3/ReadVariableOpб7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_7/ReadVariableOpб&batch_normalization_7/ReadVariableOp_1б5batch_normalization_8/FusedBatchNormV3/ReadVariableOpб7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_8/ReadVariableOpб&batch_normalization_8/ReadVariableOp_1бconv2d/BiasAdd/ReadVariableOpбconv2d/Conv2D/ReadVariableOpбconv2d_1/BiasAdd/ReadVariableOpбconv2d_1/Conv2D/ReadVariableOpбconv2d_2/BiasAdd/ReadVariableOpбconv2d_2/Conv2D/ReadVariableOpбconv2d_3/BiasAdd/ReadVariableOpбconv2d_3/Conv2D/ReadVariableOpб'conv2d_transpose/BiasAdd/ReadVariableOpб0conv2d_transpose/conv2d_transpose/ReadVariableOpб)conv2d_transpose_1/BiasAdd/ReadVariableOpб2conv2d_transpose_1/conv2d_transpose/ReadVariableOpб)conv2d_transpose_2/BiasAdd/ReadVariableOpб2conv2d_transpose_2/conv2d_transpose/ReadVariableOpб)conv2d_transpose_3/BiasAdd/ReadVariableOpб2conv2d_transpose_3/conv2d_transpose/ReadVariableOpб)conv2d_transpose_4/BiasAdd/ReadVariableOpб2conv2d_transpose_4/conv2d_transpose/ReadVariableOpT
reshape/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape/Shapeё
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stackѕ
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1ѕ
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2њ
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_sliceu
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :ђ2
reshape/Reshape/shape/1u
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ђ2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3Ж
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shapeЉ
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*1
_output_shapes
:         ђђ2
reshape/Reshape░
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOpХ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1с
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpж
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1о
$batch_normalization/FusedBatchNormV3FusedBatchNormV3reshape/Reshape:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ:::::*
epsilon%oЃ:*
is_training( 2&
$batch_normalization/FusedBatchNormV3ф
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOpП
conv2d/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЧЧ@*
paddingVALID*
strides
2
conv2d/Conv2DА
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOpд
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЧЧ@2
conv2d/BiasAddt

conv2d/EluEluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:         ЧЧ@2

conv2d/EluХ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_1/ReadVariableOp╝
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_1/ReadVariableOp_1ж
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Р
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d/Elu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЧЧ@:@:@:@:@:*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3▒
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02 
conv2d_1/Conv2D/ReadVariableOpТ
conv2d_1/Conv2DConv2D*batch_normalization_1/FusedBatchNormV3:y:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЩЩђ*
paddingVALID*
strides
2
conv2d_1/Conv2Dе
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp»
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЩЩђ2
conv2d_1/BiasAdd{
conv2d_1/EluEluconv2d_1/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЩЩђ2
conv2d_1/Eluи
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:ђ*
dtype02&
$batch_normalization_2/ReadVariableOpй
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02(
&batch_normalization_2/ReadVariableOp_1Ж
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp­
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ж
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_1/Elu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЩЩђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3▓
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02 
conv2d_2/Conv2D/ReadVariableOpТ
conv2d_2/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЭЭђ*
paddingVALID*
strides
2
conv2d_2/Conv2Dе
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp»
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЭЭђ2
conv2d_2/BiasAdd{
conv2d_2/EluEluconv2d_2/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЭЭђ2
conv2d_2/Eluи
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02&
$batch_normalization_3/ReadVariableOpй
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02(
&batch_normalization_3/ReadVariableOp_1Ж
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp­
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ж
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_2/Elu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЭЭђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3▓
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02 
conv2d_3/Conv2D/ReadVariableOpТ
conv2d_3/Conv2DConv2D*batch_normalization_3/FusedBatchNormV3:y:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ШШђ*
paddingVALID*
strides
2
conv2d_3/Conv2Dе
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp»
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ШШђ2
conv2d_3/BiasAdd{
conv2d_3/EluEluconv2d_3/BiasAdd:output:0*
T0*2
_output_shapes 
:         ШШђ2
conv2d_3/Eluи
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:ђ*
dtype02&
$batch_normalization_4/ReadVariableOpй
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02(
&batch_normalization_4/ReadVariableOp_1Ж
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp­
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ж
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_3/Elu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ШШђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3і
conv2d_transpose/ShapeShape*batch_normalization_4/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose/Shapeќ
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stackџ
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1џ
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2╚
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slicew
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value
B :Э2
conv2d_transpose/stack/1w
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :Э2
conv2d_transpose/stack/2w
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2
conv2d_transpose/stack/3Э
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stackџ
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stackъ
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1ъ
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2м
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1У
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp╩
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_4/FusedBatchNormV3:y:0*
T0*2
_output_shapes 
:         ЭЭђ*
paddingVALID*
strides
2#
!conv2d_transpose/conv2d_transpose└
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp┘
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЭЭђ2
conv2d_transpose/BiasAddЊ
conv2d_transpose/EluElu!conv2d_transpose/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЭЭђ2
conv2d_transpose/Eluи
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:ђ*
dtype02&
$batch_normalization_5/ReadVariableOpй
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02(
&batch_normalization_5/ReadVariableOp_1Ж
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp­
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ы
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3"conv2d_transpose/Elu:activations:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЭЭђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3ј
conv2d_transpose_1/ShapeShape*batch_normalization_5/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shapeџ
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stackъ
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1ъ
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2н
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slice{
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value
B :Щ2
conv2d_transpose_1/stack/1{
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :Щ2
conv2d_transpose_1/stack/2{
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2
conv2d_transpose_1/stack/3ё
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stackъ
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_1/stackб
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1б
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2я
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1Ь
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpм
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_5/FusedBatchNormV3:y:0*
T0*2
_output_shapes 
:         ЩЩђ*
paddingVALID*
strides
2%
#conv2d_transpose_1/conv2d_transposeк
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOpр
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЩЩђ2
conv2d_transpose_1/BiasAddЎ
conv2d_transpose_1/EluElu#conv2d_transpose_1/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЩЩђ2
conv2d_transpose_1/Eluи
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes	
:ђ*
dtype02&
$batch_normalization_6/ReadVariableOpй
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02(
&batch_normalization_6/ReadVariableOp_1Ж
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp­
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1з
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_1/Elu:activations:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЩЩђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3ј
conv2d_transpose_2/ShapeShape*batch_normalization_6/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_2/Shapeџ
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_2/strided_slice/stackъ
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_1ъ
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_2н
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_2/strided_slice{
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value
B :Ч2
conv2d_transpose_2/stack/1{
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value
B :Ч2
conv2d_transpose_2/stack/2{
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2
conv2d_transpose_2/stack/3ё
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_2/stackъ
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_2/strided_slice_1/stackб
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_1б
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_2я
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_1Ь
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype024
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpм
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_6/FusedBatchNormV3:y:0*
T0*2
_output_shapes 
:         ЧЧђ*
paddingVALID*
strides
2%
#conv2d_transpose_2/conv2d_transposeк
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02+
)conv2d_transpose_2/BiasAdd/ReadVariableOpр
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЧЧђ2
conv2d_transpose_2/BiasAddЎ
conv2d_transpose_2/EluElu#conv2d_transpose_2/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЧЧђ2
conv2d_transpose_2/Eluи
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes	
:ђ*
dtype02&
$batch_normalization_7/ReadVariableOpй
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02(
&batch_normalization_7/ReadVariableOp_1Ж
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp­
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1з
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_2/Elu:activations:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЧЧђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3ј
conv2d_transpose_3/ShapeShape*batch_normalization_7/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_3/Shapeџ
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_3/strided_slice/stackъ
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_1ъ
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_2н
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_3/strided_slice{
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value
B :ђ2
conv2d_transpose_3/stack/1{
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value
B :ђ2
conv2d_transpose_3/stack/2z
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_3/stack/3ё
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_3/stackъ
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_3/strided_slice_1/stackб
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_1б
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_2я
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_3/strided_slice_1ь
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@ђ*
dtype024
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpЛ
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_7/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:         ђђ@*
paddingVALID*
strides
2%
#conv2d_transpose_3/conv2d_transpose┼
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)conv2d_transpose_3/BiasAdd/ReadVariableOpЯ
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ@2
conv2d_transpose_3/BiasAddў
conv2d_transpose_3/EluElu#conv2d_transpose_3/BiasAdd:output:0*
T0*1
_output_shapes
:         ђђ@2
conv2d_transpose_3/EluХ
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_8/ReadVariableOp╝
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_8/ReadVariableOp_1ж
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Ь
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_3/Elu:activations:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ@:@:@:@:@:*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_8/FusedBatchNormV3ј
conv2d_transpose_4/ShapeShape*batch_normalization_8/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_4/Shapeџ
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_4/strided_slice/stackъ
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_4/strided_slice/stack_1ъ
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_4/strided_slice/stack_2н
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_4/strided_slice{
conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value
B :ђ2
conv2d_transpose_4/stack/1{
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value
B :ђ2
conv2d_transpose_4/stack/2z
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_4/stack/3ё
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0#conv2d_transpose_4/stack/1:output:0#conv2d_transpose_4/stack/2:output:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_4/stackъ
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_4/strided_slice_1/stackб
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_4/strided_slice_1/stack_1б
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_4/strided_slice_1/stack_2я
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_4/strided_slice_1В
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype024
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpЛ
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_8/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:         ђђ*
paddingVALID*
strides
2%
#conv2d_transpose_4/conv2d_transpose┼
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_4/BiasAdd/ReadVariableOpЯ
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ2
conv2d_transpose_4/BiasAddѕ
IdentityIdentity#conv2d_transpose_4/BiasAdd:output:0^NoOp*
T0*1
_output_shapes
:         ђђ2

IdentityЂ
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*џ
_input_shapesѕ
Ё:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2j
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
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp:U Q
-
_output_shapes
:         ђђ
 
_user_specified_nameinputs
А
ю
'__inference_conv2d_layer_call_fn_164688

inputs!
unknown:@
	unknown_0:@
identityѕбStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЧЧ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_1619072
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЧЧ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ђђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
Ѕ	
Н
6__inference_batch_normalization_5_layer_call_fn_165387

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1626372
StatefulPartitionedCallє
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЭЭђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
Ќ
└
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_161692

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1Ѕ
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
▓
а
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_165441

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Л
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЭЭђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3z
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЭЭђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
з
а
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_161204

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1р
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3і
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
й	
¤
4__inference_batch_normalization_layer_call_fn_164581

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityѕбStatefulPartitionedCall┤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1603002
StatefulPartitionedCallЋ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╦	
Н
6__inference_batch_normalization_2_layer_call_fn_164856

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCall╣
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1605082
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
§
¤
4__inference_batch_normalization_layer_call_fn_164607

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityѕбStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1629062
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         ђђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
ф
ю
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_165947

inputsC
(conv2d_transpose_readvariableop_resource:@ђ-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceU
stack/1Const*
_output_shapes
: *
dtype0*
value
B :ђ2	
stack/1U
stack/2Const*
_output_shapes
: *
dtype0*
value
B :ђ2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3њ
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1┤
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02!
conv2d_transpose/ReadVariableOpр
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:         ђђ@*
paddingVALID*
strides
2
conv2d_transposeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpћ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ@2	
BiasAdd_
EluEluBiasAdd:output:0*
T0*1
_output_shapes
:         ђђ@2
Eluv
IdentityIdentityElu:activations:0^NoOp*
T0*1
_output_shapes
:         ђђ@2

IdentityЅ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ЧЧђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Z V
2
_output_shapes 
:         ЧЧђ
 
_user_specified_nameinputs
¤(
ю
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_165923

inputsC
(conv2d_transpose_readvariableop_resource:@ђ-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2В
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3ѓ
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2В
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3┤
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02!
conv2d_transpose/ReadVariableOpы
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           @*
paddingVALID*
strides
2
conv2d_transposeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpц
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @2	
BiasAddo
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           @2
Eluє
IdentityIdentityElu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                           @2

IdentityЅ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
ј
┼
H__inference_CNN_baseline_layer_call_and_return_conditional_losses_163524	
u_vel(
batch_normalization_163397:(
batch_normalization_163399:(
batch_normalization_163401:(
batch_normalization_163403:'
conv2d_163406:@
conv2d_163408:@*
batch_normalization_1_163411:@*
batch_normalization_1_163413:@*
batch_normalization_1_163415:@*
batch_normalization_1_163417:@*
conv2d_1_163420:@ђ
conv2d_1_163422:	ђ+
batch_normalization_2_163425:	ђ+
batch_normalization_2_163427:	ђ+
batch_normalization_2_163429:	ђ+
batch_normalization_2_163431:	ђ+
conv2d_2_163434:ђђ
conv2d_2_163436:	ђ+
batch_normalization_3_163439:	ђ+
batch_normalization_3_163441:	ђ+
batch_normalization_3_163443:	ђ+
batch_normalization_3_163445:	ђ+
conv2d_3_163448:ђђ
conv2d_3_163450:	ђ+
batch_normalization_4_163453:	ђ+
batch_normalization_4_163455:	ђ+
batch_normalization_4_163457:	ђ+
batch_normalization_4_163459:	ђ3
conv2d_transpose_163462:ђђ&
conv2d_transpose_163464:	ђ+
batch_normalization_5_163467:	ђ+
batch_normalization_5_163469:	ђ+
batch_normalization_5_163471:	ђ+
batch_normalization_5_163473:	ђ5
conv2d_transpose_1_163476:ђђ(
conv2d_transpose_1_163478:	ђ+
batch_normalization_6_163481:	ђ+
batch_normalization_6_163483:	ђ+
batch_normalization_6_163485:	ђ+
batch_normalization_6_163487:	ђ5
conv2d_transpose_2_163490:ђђ(
conv2d_transpose_2_163492:	ђ+
batch_normalization_7_163495:	ђ+
batch_normalization_7_163497:	ђ+
batch_normalization_7_163499:	ђ+
batch_normalization_7_163501:	ђ4
conv2d_transpose_3_163504:@ђ'
conv2d_transpose_3_163506:@*
batch_normalization_8_163509:@*
batch_normalization_8_163511:@*
batch_normalization_8_163513:@*
batch_normalization_8_163515:@3
conv2d_transpose_4_163518:@'
conv2d_transpose_4_163520:
identityѕб+batch_normalization/StatefulPartitionedCallб-batch_normalization_1/StatefulPartitionedCallб-batch_normalization_2/StatefulPartitionedCallб-batch_normalization_3/StatefulPartitionedCallб-batch_normalization_4/StatefulPartitionedCallб-batch_normalization_5/StatefulPartitionedCallб-batch_normalization_6/StatefulPartitionedCallб-batch_normalization_7/StatefulPartitionedCallб-batch_normalization_8/StatefulPartitionedCallбconv2d/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallб conv2d_2/StatefulPartitionedCallб conv2d_3/StatefulPartitionedCallб(conv2d_transpose/StatefulPartitionedCallб*conv2d_transpose_1/StatefulPartitionedCallб*conv2d_transpose_2/StatefulPartitionedCallб*conv2d_transpose_3/StatefulPartitionedCallб*conv2d_transpose_4/StatefulPartitionedCallП
reshape/PartitionedCallPartitionedCallu_vel*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1618672
reshape/PartitionedCall«
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0batch_normalization_163397batch_normalization_163399batch_normalization_163401batch_normalization_163403*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1618862-
+batch_normalization/StatefulPartitionedCall┼
conv2d/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_163406conv2d_163408*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЧЧ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_1619072 
conv2d/StatefulPartitionedCall├
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_1_163411batch_normalization_1_163413batch_normalization_1_163415batch_normalization_1_163417*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЧЧ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1619302/
-batch_normalization_1/StatefulPartitionedCallм
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_1_163420conv2d_1_163422*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЩЩђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_1619512"
 conv2d_1/StatefulPartitionedCallк
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_2_163425batch_normalization_2_163427batch_normalization_2_163429batch_normalization_2_163431*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЩЩђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1619742/
-batch_normalization_2/StatefulPartitionedCallм
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_2_163434conv2d_2_163436*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1619952"
 conv2d_2/StatefulPartitionedCallк
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_3_163439batch_normalization_3_163441batch_normalization_3_163443batch_normalization_3_163445*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1620182/
-batch_normalization_3/StatefulPartitionedCallм
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_3_163448conv2d_3_163450*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ШШђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_1620392"
 conv2d_3/StatefulPartitionedCallк
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_4_163453batch_normalization_4_163455batch_normalization_4_163457batch_normalization_4_163459*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ШШђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1620622/
-batch_normalization_4/StatefulPartitionedCallЩ
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0conv2d_transpose_163462conv2d_transpose_163464*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1620952*
(conv2d_transpose/StatefulPartitionedCall╬
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0batch_normalization_5_163467batch_normalization_5_163469batch_normalization_5_163471batch_normalization_5_163473*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1621182/
-batch_normalization_5/StatefulPartitionedCallё
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0conv2d_transpose_1_163476conv2d_transpose_1_163478*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЩЩђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1621512,
*conv2d_transpose_1/StatefulPartitionedCallл
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0batch_normalization_6_163481batch_normalization_6_163483batch_normalization_6_163485batch_normalization_6_163487*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЩЩђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1621742/
-batch_normalization_6/StatefulPartitionedCallё
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0conv2d_transpose_2_163490conv2d_transpose_2_163492*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЧЧђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1622072,
*conv2d_transpose_2/StatefulPartitionedCallл
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0batch_normalization_7_163495batch_normalization_7_163497batch_normalization_7_163499batch_normalization_7_163501*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЧЧђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1622302/
-batch_normalization_7/StatefulPartitionedCallЃ
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0conv2d_transpose_3_163504conv2d_transpose_3_163506*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1622632,
*conv2d_transpose_3/StatefulPartitionedCall¤
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0batch_normalization_8_163509batch_normalization_8_163511batch_normalization_8_163513batch_normalization_8_163515*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1622862/
-batch_normalization_8/StatefulPartitionedCallЃ
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_transpose_4_163518conv2d_transpose_4_163520*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1623182,
*conv2d_transpose_4/StatefulPartitionedCallў
IdentityIdentity3conv2d_transpose_4/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ2

Identityт
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*џ
_input_shapesѕ
Ё:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
:         ђђ

_user_specified_nameu_vel
▓
а
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_165093

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Л
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЭЭђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3z
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЭЭђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
▓
а
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_165237

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Л
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ШШђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3z
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ШШђ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ШШђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:         ШШђ
 
_user_specified_nameinputs
з
а
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_164913

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1р
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3і
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Т
─
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_162744

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▀
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЭЭђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1z
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђ2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЭЭђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
▓
а
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_165849

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Л
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЧЧђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3z
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЧЧђ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЧЧђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:         ЧЧђ
 
_user_specified_nameinputs
р
џ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_164625

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3Ѕ
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Д
─
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_161470

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1і
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
▀─
М7
H__inference_CNN_baseline_layer_call_and_return_conditional_losses_164536

inputs9
+batch_normalization_readvariableop_resource:;
-batch_normalization_readvariableop_1_resource:J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:?
%conv2d_conv2d_readvariableop_resource:@4
&conv2d_biasadd_readvariableop_resource:@;
-batch_normalization_1_readvariableop_resource:@=
/batch_normalization_1_readvariableop_1_resource:@L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@B
'conv2d_1_conv2d_readvariableop_resource:@ђ7
(conv2d_1_biasadd_readvariableop_resource:	ђ<
-batch_normalization_2_readvariableop_resource:	ђ>
/batch_normalization_2_readvariableop_1_resource:	ђM
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	ђO
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	ђC
'conv2d_2_conv2d_readvariableop_resource:ђђ7
(conv2d_2_biasadd_readvariableop_resource:	ђ<
-batch_normalization_3_readvariableop_resource:	ђ>
/batch_normalization_3_readvariableop_1_resource:	ђM
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	ђO
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	ђC
'conv2d_3_conv2d_readvariableop_resource:ђђ7
(conv2d_3_biasadd_readvariableop_resource:	ђ<
-batch_normalization_4_readvariableop_resource:	ђ>
/batch_normalization_4_readvariableop_1_resource:	ђM
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	ђO
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	ђU
9conv2d_transpose_conv2d_transpose_readvariableop_resource:ђђ?
0conv2d_transpose_biasadd_readvariableop_resource:	ђ<
-batch_normalization_5_readvariableop_resource:	ђ>
/batch_normalization_5_readvariableop_1_resource:	ђM
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:	ђO
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:	ђW
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:ђђA
2conv2d_transpose_1_biasadd_readvariableop_resource:	ђ<
-batch_normalization_6_readvariableop_resource:	ђ>
/batch_normalization_6_readvariableop_1_resource:	ђM
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource:	ђO
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:	ђW
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource:ђђA
2conv2d_transpose_2_biasadd_readvariableop_resource:	ђ<
-batch_normalization_7_readvariableop_resource:	ђ>
/batch_normalization_7_readvariableop_1_resource:	ђM
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:	ђO
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:	ђV
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource:@ђ@
2conv2d_transpose_3_biasadd_readvariableop_resource:@;
-batch_normalization_8_readvariableop_resource:@=
/batch_normalization_8_readvariableop_1_resource:@L
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:@U
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource:@@
2conv2d_transpose_4_biasadd_readvariableop_resource:
identityѕб"batch_normalization/AssignNewValueб$batch_normalization/AssignNewValue_1б3batch_normalization/FusedBatchNormV3/ReadVariableOpб5batch_normalization/FusedBatchNormV3/ReadVariableOp_1б"batch_normalization/ReadVariableOpб$batch_normalization/ReadVariableOp_1б$batch_normalization_1/AssignNewValueб&batch_normalization_1/AssignNewValue_1б5batch_normalization_1/FusedBatchNormV3/ReadVariableOpб7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_1/ReadVariableOpб&batch_normalization_1/ReadVariableOp_1б$batch_normalization_2/AssignNewValueб&batch_normalization_2/AssignNewValue_1б5batch_normalization_2/FusedBatchNormV3/ReadVariableOpб7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_2/ReadVariableOpб&batch_normalization_2/ReadVariableOp_1б$batch_normalization_3/AssignNewValueб&batch_normalization_3/AssignNewValue_1б5batch_normalization_3/FusedBatchNormV3/ReadVariableOpб7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_3/ReadVariableOpб&batch_normalization_3/ReadVariableOp_1б$batch_normalization_4/AssignNewValueб&batch_normalization_4/AssignNewValue_1б5batch_normalization_4/FusedBatchNormV3/ReadVariableOpб7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_4/ReadVariableOpб&batch_normalization_4/ReadVariableOp_1б$batch_normalization_5/AssignNewValueб&batch_normalization_5/AssignNewValue_1б5batch_normalization_5/FusedBatchNormV3/ReadVariableOpб7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_5/ReadVariableOpб&batch_normalization_5/ReadVariableOp_1б$batch_normalization_6/AssignNewValueб&batch_normalization_6/AssignNewValue_1б5batch_normalization_6/FusedBatchNormV3/ReadVariableOpб7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_6/ReadVariableOpб&batch_normalization_6/ReadVariableOp_1б$batch_normalization_7/AssignNewValueб&batch_normalization_7/AssignNewValue_1б5batch_normalization_7/FusedBatchNormV3/ReadVariableOpб7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_7/ReadVariableOpб&batch_normalization_7/ReadVariableOp_1б$batch_normalization_8/AssignNewValueб&batch_normalization_8/AssignNewValue_1б5batch_normalization_8/FusedBatchNormV3/ReadVariableOpб7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_8/ReadVariableOpб&batch_normalization_8/ReadVariableOp_1бconv2d/BiasAdd/ReadVariableOpбconv2d/Conv2D/ReadVariableOpбconv2d_1/BiasAdd/ReadVariableOpбconv2d_1/Conv2D/ReadVariableOpбconv2d_2/BiasAdd/ReadVariableOpбconv2d_2/Conv2D/ReadVariableOpбconv2d_3/BiasAdd/ReadVariableOpбconv2d_3/Conv2D/ReadVariableOpб'conv2d_transpose/BiasAdd/ReadVariableOpб0conv2d_transpose/conv2d_transpose/ReadVariableOpб)conv2d_transpose_1/BiasAdd/ReadVariableOpб2conv2d_transpose_1/conv2d_transpose/ReadVariableOpб)conv2d_transpose_2/BiasAdd/ReadVariableOpб2conv2d_transpose_2/conv2d_transpose/ReadVariableOpб)conv2d_transpose_3/BiasAdd/ReadVariableOpб2conv2d_transpose_3/conv2d_transpose/ReadVariableOpб)conv2d_transpose_4/BiasAdd/ReadVariableOpб2conv2d_transpose_4/conv2d_transpose/ReadVariableOpT
reshape/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape/Shapeё
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stackѕ
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1ѕ
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2њ
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_sliceu
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :ђ2
reshape/Reshape/shape/1u
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ђ2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3Ж
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shapeЉ
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*1
_output_shapes
:         ђђ2
reshape/Reshape░
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOpХ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1с
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpж
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1С
$batch_normalization/FusedBatchNormV3FusedBatchNormV3reshape/Reshape:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ:::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<2&
$batch_normalization/FusedBatchNormV3д
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue▓
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1ф
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOpП
conv2d/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЧЧ@*
paddingVALID*
strides
2
conv2d/Conv2DА
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOpд
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЧЧ@2
conv2d/BiasAddt

conv2d/EluEluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:         ЧЧ@2

conv2d/EluХ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_1/ReadVariableOp╝
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_1/ReadVariableOp_1ж
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1­
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d/Elu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЧЧ@:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2(
&batch_normalization_1/FusedBatchNormV3░
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue╝
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1▒
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02 
conv2d_1/Conv2D/ReadVariableOpТ
conv2d_1/Conv2DConv2D*batch_normalization_1/FusedBatchNormV3:y:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЩЩђ*
paddingVALID*
strides
2
conv2d_1/Conv2Dе
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp»
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЩЩђ2
conv2d_1/BiasAdd{
conv2d_1/EluEluconv2d_1/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЩЩђ2
conv2d_1/Eluи
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:ђ*
dtype02&
$batch_normalization_2/ReadVariableOpй
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02(
&batch_normalization_2/ReadVariableOp_1Ж
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp­
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1э
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_1/Elu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЩЩђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2(
&batch_normalization_2/FusedBatchNormV3░
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValue╝
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1▓
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02 
conv2d_2/Conv2D/ReadVariableOpТ
conv2d_2/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЭЭђ*
paddingVALID*
strides
2
conv2d_2/Conv2Dе
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp»
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЭЭђ2
conv2d_2/BiasAdd{
conv2d_2/EluEluconv2d_2/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЭЭђ2
conv2d_2/Eluи
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02&
$batch_normalization_3/ReadVariableOpй
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02(
&batch_normalization_3/ReadVariableOp_1Ж
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp­
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1э
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_2/Elu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЭЭђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2(
&batch_normalization_3/FusedBatchNormV3░
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValue╝
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1▓
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02 
conv2d_3/Conv2D/ReadVariableOpТ
conv2d_3/Conv2DConv2D*batch_normalization_3/FusedBatchNormV3:y:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ШШђ*
paddingVALID*
strides
2
conv2d_3/Conv2Dе
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp»
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ШШђ2
conv2d_3/BiasAdd{
conv2d_3/EluEluconv2d_3/BiasAdd:output:0*
T0*2
_output_shapes 
:         ШШђ2
conv2d_3/Eluи
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:ђ*
dtype02&
$batch_normalization_4/ReadVariableOpй
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02(
&batch_normalization_4/ReadVariableOp_1Ж
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp­
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1э
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_3/Elu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ШШђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2(
&batch_normalization_4/FusedBatchNormV3░
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_4/AssignNewValue╝
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_4/AssignNewValue_1і
conv2d_transpose/ShapeShape*batch_normalization_4/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose/Shapeќ
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stackџ
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1џ
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2╚
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slicew
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value
B :Э2
conv2d_transpose/stack/1w
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :Э2
conv2d_transpose/stack/2w
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2
conv2d_transpose/stack/3Э
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stackџ
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stackъ
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1ъ
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2м
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1У
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp╩
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_4/FusedBatchNormV3:y:0*
T0*2
_output_shapes 
:         ЭЭђ*
paddingVALID*
strides
2#
!conv2d_transpose/conv2d_transpose└
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp┘
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЭЭђ2
conv2d_transpose/BiasAddЊ
conv2d_transpose/EluElu!conv2d_transpose/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЭЭђ2
conv2d_transpose/Eluи
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:ђ*
dtype02&
$batch_normalization_5/ReadVariableOpй
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02(
&batch_normalization_5/ReadVariableOp_1Ж
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp­
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1 
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3"conv2d_transpose/Elu:activations:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЭЭђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2(
&batch_normalization_5/FusedBatchNormV3░
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_5/AssignNewValue╝
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_5/AssignNewValue_1ј
conv2d_transpose_1/ShapeShape*batch_normalization_5/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shapeџ
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stackъ
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1ъ
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2н
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slice{
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value
B :Щ2
conv2d_transpose_1/stack/1{
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :Щ2
conv2d_transpose_1/stack/2{
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2
conv2d_transpose_1/stack/3ё
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stackъ
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_1/stackб
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1б
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2я
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1Ь
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpм
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_5/FusedBatchNormV3:y:0*
T0*2
_output_shapes 
:         ЩЩђ*
paddingVALID*
strides
2%
#conv2d_transpose_1/conv2d_transposeк
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOpр
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЩЩђ2
conv2d_transpose_1/BiasAddЎ
conv2d_transpose_1/EluElu#conv2d_transpose_1/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЩЩђ2
conv2d_transpose_1/Eluи
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes	
:ђ*
dtype02&
$batch_normalization_6/ReadVariableOpй
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02(
&batch_normalization_6/ReadVariableOp_1Ж
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp­
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Ђ
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_1/Elu:activations:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЩЩђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2(
&batch_normalization_6/FusedBatchNormV3░
$batch_normalization_6/AssignNewValueAssignVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_6/AssignNewValue╝
&batch_normalization_6/AssignNewValue_1AssignVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_6/AssignNewValue_1ј
conv2d_transpose_2/ShapeShape*batch_normalization_6/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_2/Shapeџ
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_2/strided_slice/stackъ
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_1ъ
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_2н
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_2/strided_slice{
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value
B :Ч2
conv2d_transpose_2/stack/1{
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value
B :Ч2
conv2d_transpose_2/stack/2{
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2
conv2d_transpose_2/stack/3ё
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_2/stackъ
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_2/strided_slice_1/stackб
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_1б
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_2я
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_1Ь
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype024
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpм
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_6/FusedBatchNormV3:y:0*
T0*2
_output_shapes 
:         ЧЧђ*
paddingVALID*
strides
2%
#conv2d_transpose_2/conv2d_transposeк
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02+
)conv2d_transpose_2/BiasAdd/ReadVariableOpр
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЧЧђ2
conv2d_transpose_2/BiasAddЎ
conv2d_transpose_2/EluElu#conv2d_transpose_2/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЧЧђ2
conv2d_transpose_2/Eluи
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes	
:ђ*
dtype02&
$batch_normalization_7/ReadVariableOpй
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02(
&batch_normalization_7/ReadVariableOp_1Ж
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp­
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Ђ
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_2/Elu:activations:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЧЧђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2(
&batch_normalization_7/FusedBatchNormV3░
$batch_normalization_7/AssignNewValueAssignVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_7/AssignNewValue╝
&batch_normalization_7/AssignNewValue_1AssignVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_7/AssignNewValue_1ј
conv2d_transpose_3/ShapeShape*batch_normalization_7/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_3/Shapeџ
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_3/strided_slice/stackъ
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_1ъ
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_2н
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_3/strided_slice{
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value
B :ђ2
conv2d_transpose_3/stack/1{
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value
B :ђ2
conv2d_transpose_3/stack/2z
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_3/stack/3ё
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_3/stackъ
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_3/strided_slice_1/stackб
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_1б
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_2я
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_3/strided_slice_1ь
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@ђ*
dtype024
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpЛ
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_7/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:         ђђ@*
paddingVALID*
strides
2%
#conv2d_transpose_3/conv2d_transpose┼
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)conv2d_transpose_3/BiasAdd/ReadVariableOpЯ
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ@2
conv2d_transpose_3/BiasAddў
conv2d_transpose_3/EluElu#conv2d_transpose_3/BiasAdd:output:0*
T0*1
_output_shapes
:         ђђ@2
conv2d_transpose_3/EluХ
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_8/ReadVariableOp╝
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_8/ReadVariableOp_1ж
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Ч
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_3/Elu:activations:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ@:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2(
&batch_normalization_8/FusedBatchNormV3░
$batch_normalization_8/AssignNewValueAssignVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_8/AssignNewValue╝
&batch_normalization_8/AssignNewValue_1AssignVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_8/AssignNewValue_1ј
conv2d_transpose_4/ShapeShape*batch_normalization_8/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_4/Shapeџ
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_4/strided_slice/stackъ
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_4/strided_slice/stack_1ъ
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_4/strided_slice/stack_2н
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_4/strided_slice{
conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value
B :ђ2
conv2d_transpose_4/stack/1{
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value
B :ђ2
conv2d_transpose_4/stack/2z
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_4/stack/3ё
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0#conv2d_transpose_4/stack/1:output:0#conv2d_transpose_4/stack/2:output:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_4/stackъ
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_4/strided_slice_1/stackб
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_4/strided_slice_1/stack_1б
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_4/strided_slice_1/stack_2я
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_4/strided_slice_1В
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype024
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpЛ
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0*batch_normalization_8/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:         ђђ*
paddingVALID*
strides
2%
#conv2d_transpose_4/conv2d_transpose┼
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_4/BiasAdd/ReadVariableOpЯ
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ2
conv2d_transpose_4/BiasAddѕ
IdentityIdentity#conv2d_transpose_4/BiasAdd:output:0^NoOp*
T0*1
_output_shapes
:         ђђ2

Identity═
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*џ
_input_shapesѕ
Ё:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
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
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp:U Q
-
_output_shapes
:         ђђ
 
_user_specified_nameinputs
а
_
C__inference_reshape_layer_call_and_return_conditional_losses_161867

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicee
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :ђ2
Reshape/shape/1e
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ђ2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3║
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapey
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:         ђђ2	
Reshapen
IdentityIdentityReshape:output:0*
T0*1
_output_shapes
:         ђђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ђђ:U Q
-
_output_shapes
:         ђђ
 
_user_specified_nameinputs
е
в
-__inference_CNN_baseline_layer_call_fn_163889

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@$
	unknown_9:@ђ

unknown_10:	ђ

unknown_11:	ђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:	ђ&

unknown_15:ђђ

unknown_16:	ђ

unknown_17:	ђ

unknown_18:	ђ

unknown_19:	ђ

unknown_20:	ђ&

unknown_21:ђђ

unknown_22:	ђ

unknown_23:	ђ

unknown_24:	ђ

unknown_25:	ђ

unknown_26:	ђ&

unknown_27:ђђ

unknown_28:	ђ

unknown_29:	ђ

unknown_30:	ђ

unknown_31:	ђ

unknown_32:	ђ&

unknown_33:ђђ

unknown_34:	ђ

unknown_35:	ђ

unknown_36:	ђ

unknown_37:	ђ

unknown_38:	ђ&

unknown_39:ђђ

unknown_40:	ђ

unknown_41:	ђ

unknown_42:	ђ

unknown_43:	ђ

unknown_44:	ђ%

unknown_45:@ђ

unknown_46:@

unknown_47:@

unknown_48:@

unknown_49:@

unknown_50:@$

unknown_51:@

unknown_52:
identityѕбStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_52*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_CNN_baseline_layer_call_and_return_conditional_losses_1623252
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*џ
_input_shapesѕ
Ё:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:         ђђ
 
_user_specified_nameinputs
Т
─
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_165663

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▀
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЩЩђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1z
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђ2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЩЩђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs
╔	
Н
6__inference_batch_normalization_7_layer_call_fn_165769

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1614702
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
▓
а
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_162018

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Л
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЭЭђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3z
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЭЭђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
 
ђ
D__inference_conv2d_3_layer_call_and_return_conditional_losses_162039

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02
Conv2D/ReadVariableOpД
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ШШђ*
paddingVALID*
strides
2
Conv2DЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpІ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ШШђ2	
BiasAdd`
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:         ШШђ2
Eluw
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:         ШШђ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ЭЭђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
▒
ю
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_162095

inputsD
(conv2d_transpose_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceU
stack/1Const*
_output_shapes
: *
dtype0*
value
B :Э2	
stack/1U
stack/2Const*
_output_shapes
: *
dtype0*
value
B :Э2	
stack/2U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2	
stack/3њ
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1х
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_transpose/ReadVariableOpР
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*2
_output_shapes 
:         ЭЭђ*
paddingVALID*
strides
2
conv2d_transposeЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЋ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЭЭђ2	
BiasAdd`
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:         ЭЭђ2
Eluw
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђ2

IdentityЅ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ШШђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Z V
2
_output_shapes 
:         ШШђ
 
_user_specified_nameinputs
Д
─
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_164931

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1і
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Ѕ	
Н
6__inference_batch_normalization_4_layer_call_fn_165183

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ШШђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1626902
StatefulPartitionedCallє
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ШШђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ШШђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ШШђ
 
_user_specified_nameinputs
Т
─
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_162637

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▀
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЭЭђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1z
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђ2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЭЭђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
│
ъ
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_162151

inputsD
(conv2d_transpose_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceU
stack/1Const*
_output_shapes
: *
dtype0*
value
B :Щ2	
stack/1U
stack/2Const*
_output_shapes
: *
dtype0*
value
B :Щ2	
stack/2U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2	
stack/3њ
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1х
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_transpose/ReadVariableOpР
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*2
_output_shapes 
:         ЩЩђ*
paddingVALID*
strides
2
conv2d_transposeЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЋ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЩЩђ2	
BiasAdd`
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:         ЩЩђ2
Eluw
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђ2

IdentityЅ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ЭЭђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
┐	
¤
4__inference_batch_normalization_layer_call_fn_164568

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityѕбStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1602562
StatefulPartitionedCallЋ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
а
_
C__inference_reshape_layer_call_and_return_conditional_losses_164555

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicee
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :ђ2
Reshape/shape/1e
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ђ2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3║
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapey
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:         ђђ2	
Reshapen
IdentityIdentityReshape:output:0*
T0*1
_output_shapes
:         ђђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ђђ:U Q
-
_output_shapes
:         ђђ
 
_user_specified_nameinputs
├	
Л
6__inference_batch_normalization_8_layer_call_fn_165960

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1616482
StatefulPartitionedCallЋ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
│
ъ
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_162207

inputsD
(conv2d_transpose_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceU
stack/1Const*
_output_shapes
: *
dtype0*
value
B :Ч2	
stack/1U
stack/2Const*
_output_shapes
: *
dtype0*
value
B :Ч2	
stack/2U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2	
stack/3њ
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1х
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_transpose/ReadVariableOpР
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*2
_output_shapes 
:         ЧЧђ*
paddingVALID*
strides
2
conv2d_transposeЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЋ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЧЧђ2	
BiasAdd`
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:         ЧЧђ2
Eluw
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:         ЧЧђ2

IdentityЅ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ЩЩђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Z V
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs
І	
Н
6__inference_batch_normalization_3_layer_call_fn_165026

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1620182
StatefulPartitionedCallє
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЭЭђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
з
а
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_161426

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1р
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3і
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
с
ю
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_161648

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3Ѕ
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Љ
к
H__inference_CNN_baseline_layer_call_and_return_conditional_losses_162325

inputs(
batch_normalization_161887:(
batch_normalization_161889:(
batch_normalization_161891:(
batch_normalization_161893:'
conv2d_161908:@
conv2d_161910:@*
batch_normalization_1_161931:@*
batch_normalization_1_161933:@*
batch_normalization_1_161935:@*
batch_normalization_1_161937:@*
conv2d_1_161952:@ђ
conv2d_1_161954:	ђ+
batch_normalization_2_161975:	ђ+
batch_normalization_2_161977:	ђ+
batch_normalization_2_161979:	ђ+
batch_normalization_2_161981:	ђ+
conv2d_2_161996:ђђ
conv2d_2_161998:	ђ+
batch_normalization_3_162019:	ђ+
batch_normalization_3_162021:	ђ+
batch_normalization_3_162023:	ђ+
batch_normalization_3_162025:	ђ+
conv2d_3_162040:ђђ
conv2d_3_162042:	ђ+
batch_normalization_4_162063:	ђ+
batch_normalization_4_162065:	ђ+
batch_normalization_4_162067:	ђ+
batch_normalization_4_162069:	ђ3
conv2d_transpose_162096:ђђ&
conv2d_transpose_162098:	ђ+
batch_normalization_5_162119:	ђ+
batch_normalization_5_162121:	ђ+
batch_normalization_5_162123:	ђ+
batch_normalization_5_162125:	ђ5
conv2d_transpose_1_162152:ђђ(
conv2d_transpose_1_162154:	ђ+
batch_normalization_6_162175:	ђ+
batch_normalization_6_162177:	ђ+
batch_normalization_6_162179:	ђ+
batch_normalization_6_162181:	ђ5
conv2d_transpose_2_162208:ђђ(
conv2d_transpose_2_162210:	ђ+
batch_normalization_7_162231:	ђ+
batch_normalization_7_162233:	ђ+
batch_normalization_7_162235:	ђ+
batch_normalization_7_162237:	ђ4
conv2d_transpose_3_162264:@ђ'
conv2d_transpose_3_162266:@*
batch_normalization_8_162287:@*
batch_normalization_8_162289:@*
batch_normalization_8_162291:@*
batch_normalization_8_162293:@3
conv2d_transpose_4_162319:@'
conv2d_transpose_4_162321:
identityѕб+batch_normalization/StatefulPartitionedCallб-batch_normalization_1/StatefulPartitionedCallб-batch_normalization_2/StatefulPartitionedCallб-batch_normalization_3/StatefulPartitionedCallб-batch_normalization_4/StatefulPartitionedCallб-batch_normalization_5/StatefulPartitionedCallб-batch_normalization_6/StatefulPartitionedCallб-batch_normalization_7/StatefulPartitionedCallб-batch_normalization_8/StatefulPartitionedCallбconv2d/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallб conv2d_2/StatefulPartitionedCallб conv2d_3/StatefulPartitionedCallб(conv2d_transpose/StatefulPartitionedCallб*conv2d_transpose_1/StatefulPartitionedCallб*conv2d_transpose_2/StatefulPartitionedCallб*conv2d_transpose_3/StatefulPartitionedCallб*conv2d_transpose_4/StatefulPartitionedCallя
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1618672
reshape/PartitionedCall«
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0batch_normalization_161887batch_normalization_161889batch_normalization_161891batch_normalization_161893*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1618862-
+batch_normalization/StatefulPartitionedCall┼
conv2d/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_161908conv2d_161910*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЧЧ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_1619072 
conv2d/StatefulPartitionedCall├
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_1_161931batch_normalization_1_161933batch_normalization_1_161935batch_normalization_1_161937*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЧЧ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1619302/
-batch_normalization_1/StatefulPartitionedCallм
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_1_161952conv2d_1_161954*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЩЩђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_1619512"
 conv2d_1/StatefulPartitionedCallк
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_2_161975batch_normalization_2_161977batch_normalization_2_161979batch_normalization_2_161981*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЩЩђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1619742/
-batch_normalization_2/StatefulPartitionedCallм
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_2_161996conv2d_2_161998*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1619952"
 conv2d_2/StatefulPartitionedCallк
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_3_162019batch_normalization_3_162021batch_normalization_3_162023batch_normalization_3_162025*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1620182/
-batch_normalization_3/StatefulPartitionedCallм
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_3_162040conv2d_3_162042*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ШШђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_1620392"
 conv2d_3/StatefulPartitionedCallк
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_4_162063batch_normalization_4_162065batch_normalization_4_162067batch_normalization_4_162069*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ШШђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1620622/
-batch_normalization_4/StatefulPartitionedCallЩ
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0conv2d_transpose_162096conv2d_transpose_162098*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1620952*
(conv2d_transpose/StatefulPartitionedCall╬
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0batch_normalization_5_162119batch_normalization_5_162121batch_normalization_5_162123batch_normalization_5_162125*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1621182/
-batch_normalization_5/StatefulPartitionedCallё
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0conv2d_transpose_1_162152conv2d_transpose_1_162154*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЩЩђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1621512,
*conv2d_transpose_1/StatefulPartitionedCallл
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0batch_normalization_6_162175batch_normalization_6_162177batch_normalization_6_162179batch_normalization_6_162181*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЩЩђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1621742/
-batch_normalization_6/StatefulPartitionedCallё
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0conv2d_transpose_2_162208conv2d_transpose_2_162210*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЧЧђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1622072,
*conv2d_transpose_2/StatefulPartitionedCallл
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0batch_normalization_7_162231batch_normalization_7_162233batch_normalization_7_162235batch_normalization_7_162237*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЧЧђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1622302/
-batch_normalization_7/StatefulPartitionedCallЃ
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0conv2d_transpose_3_162264conv2d_transpose_3_162266*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1622632,
*conv2d_transpose_3/StatefulPartitionedCall¤
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0batch_normalization_8_162287batch_normalization_8_162289batch_normalization_8_162291batch_normalization_8_162293*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1622862/
-batch_normalization_8/StatefulPartitionedCallЃ
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_transpose_4_162319conv2d_transpose_4_162321*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1623182,
*conv2d_transpose_4/StatefulPartitionedCallў
IdentityIdentity3conv2d_transpose_4/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ2

Identityт
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*џ
_input_shapesѕ
Ё:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
:         ђђ
 
_user_specified_nameinputs
 
¤
4__inference_batch_normalization_layer_call_fn_164594

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityѕбStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1618862
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         ђђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
▒
ю
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_165335

inputsD
(conv2d_transpose_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceU
stack/1Const*
_output_shapes
: *
dtype0*
value
B :Э2	
stack/1U
stack/2Const*
_output_shapes
: *
dtype0*
value
B :Э2	
stack/2U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2	
stack/3њ
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1х
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_transpose/ReadVariableOpР
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*2
_output_shapes 
:         ЭЭђ*
paddingVALID*
strides
2
conv2d_transposeЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЋ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЭЭђ2	
BiasAdd`
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:         ЭЭђ2
Eluw
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђ2

IdentityЅ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ШШђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Z V
2
_output_shapes 
:         ШШђ
 
_user_specified_nameinputs
п(
ъ
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_161128

inputsD
(conv2d_transpose_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2В
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2	
stack/3ѓ
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2В
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3х
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_transpose/ReadVariableOpЫ
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,                           ђ*
paddingVALID*
strides
2
conv2d_transposeЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЦ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђ2	
BiasAddp
EluEluBiasAdd:output:0*
T0*B
_output_shapes0
.:,                           ђ2
EluЄ
IdentityIdentityElu:activations:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

IdentityЅ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
з
а
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_160508

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1р
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3і
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Д
─
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_165075

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1і
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
І	
Н
6__inference_batch_normalization_6_layer_call_fn_165578

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЩЩђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1621742
StatefulPartitionedCallє
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЩЩђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs
п(
ъ
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_165515

inputsD
(conv2d_transpose_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2В
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2	
stack/3ѓ
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2В
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3х
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_transpose/ReadVariableOpЫ
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,                           ђ*
paddingVALID*
strides
2
conv2d_transposeЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЦ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђ2	
BiasAddp
EluEluBiasAdd:output:0*
T0*B
_output_shapes0
.:,                           ђ2
EluЄ
IdentityIdentityElu:activations:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

IdentityЅ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
▓
а
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_165645

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Л
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЩЩђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3z
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђ2

IdentityИ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЩЩђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs
│
ъ
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_165743

inputsD
(conv2d_transpose_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceU
stack/1Const*
_output_shapes
: *
dtype0*
value
B :Ч2	
stack/1U
stack/2Const*
_output_shapes
: *
dtype0*
value
B :Ч2	
stack/2U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2	
stack/3њ
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1х
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_transpose/ReadVariableOpР
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*2
_output_shapes 
:         ЧЧђ*
paddingVALID*
strides
2
conv2d_transposeЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЋ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЧЧђ2	
BiasAdd`
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:         ЧЧђ2
Eluw
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:         ЧЧђ2

IdentityЅ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ЩЩђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Z V
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs
┴	
Л
6__inference_batch_normalization_8_layer_call_fn_165973

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1616922
StatefulPartitionedCallЋ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Д
─
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_161026

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1і
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Ѕ	
Н
6__inference_batch_normalization_7_layer_call_fn_165795

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЧЧђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1625312
StatefulPartitionedCallє
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ЧЧђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЧЧђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ЧЧђ
 
_user_specified_nameinputs
Ч
Е
3__inference_conv2d_transpose_3_layer_call_fn_165876

inputs"
unknown:@ђ
	unknown_0:@
identityѕбStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1615722
StatefulPartitionedCallЋ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
╝
Е
3__inference_conv2d_transpose_3_layer_call_fn_165885

inputs"
unknown:@ђ
	unknown_0:@
identityѕбStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1622632
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ЧЧђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ЧЧђ
 
_user_specified_nameinputs"еL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*┴
serving_defaultГ
=
u_vel4
serving_default_u_vel:0         ђђP
conv2d_transpose_4:
StatefulPartitionedCall:0         ђђtensorflow/serving/predict:▒Ы
л
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
layer_with_weights-8
layer-10
layer_with_weights-9
layer-11
layer_with_weights-10
layer-12
layer_with_weights-11
layer-13
layer_with_weights-12
layer-14
layer_with_weights-13
layer-15
layer_with_weights-14
layer-16
layer_with_weights-15
layer-17
layer_with_weights-16
layer-18
layer_with_weights-17
layer-19
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
▄_default_save_signature
П__call__
+я&call_and_return_all_conditional_losses"
_tf_keras_network
"
_tf_keras_input_layer
Д
regularization_losses
	variables
trainable_variables
	keras_api
▀__call__
+Я&call_and_return_all_conditional_losses"
_tf_keras_layer
В
axis
	 gamma
!beta
"moving_mean
#moving_variance
$regularization_losses
%	variables
&trainable_variables
'	keras_api
р__call__
+Р&call_and_return_all_conditional_losses"
_tf_keras_layer
й

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
с__call__
+С&call_and_return_all_conditional_losses"
_tf_keras_layer
В
.axis
	/gamma
0beta
1moving_mean
2moving_variance
3regularization_losses
4	variables
5trainable_variables
6	keras_api
т__call__
+Т&call_and_return_all_conditional_losses"
_tf_keras_layer
й

7kernel
8bias
9regularization_losses
:	variables
;trainable_variables
<	keras_api
у__call__
+У&call_and_return_all_conditional_losses"
_tf_keras_layer
В
=axis
	>gamma
?beta
@moving_mean
Amoving_variance
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
ж__call__
+Ж&call_and_return_all_conditional_losses"
_tf_keras_layer
й

Fkernel
Gbias
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
в__call__
+В&call_and_return_all_conditional_losses"
_tf_keras_layer
В
Laxis
	Mgamma
Nbeta
Omoving_mean
Pmoving_variance
Qregularization_losses
R	variables
Strainable_variables
T	keras_api
ь__call__
+Ь&call_and_return_all_conditional_losses"
_tf_keras_layer
й

Ukernel
Vbias
Wregularization_losses
X	variables
Ytrainable_variables
Z	keras_api
№__call__
+­&call_and_return_all_conditional_losses"
_tf_keras_layer
В
[axis
	\gamma
]beta
^moving_mean
_moving_variance
`regularization_losses
a	variables
btrainable_variables
c	keras_api
ы__call__
+Ы&call_and_return_all_conditional_losses"
_tf_keras_layer
й

dkernel
ebias
fregularization_losses
g	variables
htrainable_variables
i	keras_api
з__call__
+З&call_and_return_all_conditional_losses"
_tf_keras_layer
В
jaxis
	kgamma
lbeta
mmoving_mean
nmoving_variance
oregularization_losses
p	variables
qtrainable_variables
r	keras_api
ш__call__
+Ш&call_and_return_all_conditional_losses"
_tf_keras_layer
й

skernel
tbias
uregularization_losses
v	variables
wtrainable_variables
x	keras_api
э__call__
+Э&call_and_return_all_conditional_losses"
_tf_keras_layer
Ь
yaxis
	zgamma
{beta
|moving_mean
}moving_variance
~regularization_losses
	variables
ђtrainable_variables
Ђ	keras_api
щ__call__
+Щ&call_and_return_all_conditional_losses"
_tf_keras_layer
├
ѓkernel
	Ѓbias
ёregularization_losses
Ё	variables
єtrainable_variables
Є	keras_api
ч__call__
+Ч&call_and_return_all_conditional_losses"
_tf_keras_layer
ш
	ѕaxis

Ѕgamma
	іbeta
Іmoving_mean
їmoving_variance
Їregularization_losses
ј	variables
Јtrainable_variables
љ	keras_api
§__call__
+■&call_and_return_all_conditional_losses"
_tf_keras_layer
├
Љkernel
	њbias
Њregularization_losses
ћ	variables
Ћtrainable_variables
ќ	keras_api
 __call__
+ђ&call_and_return_all_conditional_losses"
_tf_keras_layer
ш
	Ќaxis

ўgamma
	Ўbeta
џmoving_mean
Џmoving_variance
юregularization_losses
Ю	variables
ъtrainable_variables
Ъ	keras_api
Ђ__call__
+ѓ&call_and_return_all_conditional_losses"
_tf_keras_layer
├
аkernel
	Аbias
бregularization_losses
Б	variables
цtrainable_variables
Ц	keras_api
Ѓ__call__
+ё&call_and_return_all_conditional_losses"
_tf_keras_layer
╝
	дiter
Дbeta_1
еbeta_2

Еdecay
фlearning_rate mћ!mЋ(mќ)mЌ/mў0mЎ7mџ8mЏ>mю?mЮFmъGmЪMmаNmАUmбVmБ\mц]mЦdmдemДkmеlmЕsmфtmФzmг{mГ	ѓm«	Ѓm»	Ѕm░	іm▒	Љm▓	њm│	ўm┤	Ўmх	аmХ	Аmи vИ!v╣(v║)v╗/v╝0vй7vЙ8v┐>v└?v┴Fv┬Gv├Mv─Nv┼UvкVvК\v╚]v╔dv╩ev╦kv╠lv═sv╬tv¤zvл{vЛ	ѓvм	ЃvМ	Ѕvн	іvН	Љvо	њvО	ўvп	Ўv┘	аv┌	Аv█"
	optimizer
 "
trackable_list_wrapper
н
 0
!1
"2
#3
(4
)5
/6
07
18
29
710
811
>12
?13
@14
A15
F16
G17
M18
N19
O20
P21
U22
V23
\24
]25
^26
_27
d28
e29
k30
l31
m32
n33
s34
t35
z36
{37
|38
}39
ѓ40
Ѓ41
Ѕ42
і43
І44
ї45
Љ46
њ47
ў48
Ў49
џ50
Џ51
а52
А53"
trackable_list_wrapper
└
 0
!1
(2
)3
/4
05
76
87
>8
?9
F10
G11
M12
N13
U14
V15
\16
]17
d18
e19
k20
l21
s22
t23
z24
{25
ѓ26
Ѓ27
Ѕ28
і29
Љ30
њ31
ў32
Ў33
а34
А35"
trackable_list_wrapper
М
regularization_losses
	variables
trainable_variables
Фnon_trainable_variables
гlayers
Гlayer_metrics
«metrics
 »layer_regularization_losses
П__call__
▄_default_save_signature
+я&call_and_return_all_conditional_losses
'я"call_and_return_conditional_losses"
_generic_user_object
-
Ёserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
regularization_losses
	variables
trainable_variables
░non_trainable_variables
▒layers
▓layer_metrics
│metrics
 ┤layer_regularization_losses
▀__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%2batch_normalization/gamma
&:$2batch_normalization/beta
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
 "
trackable_list_wrapper
<
 0
!1
"2
#3"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
х
$regularization_losses
%	variables
&trainable_variables
хnon_trainable_variables
Хlayers
иlayer_metrics
Иmetrics
 ╣layer_regularization_losses
р__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
':%@2conv2d/kernel
:@2conv2d/bias
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
х
*regularization_losses
+	variables
,trainable_variables
║non_trainable_variables
╗layers
╝layer_metrics
йmetrics
 Йlayer_regularization_losses
с__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_1/gamma
(:&@2batch_normalization_1/beta
1:/@ (2!batch_normalization_1/moving_mean
5:3@ (2%batch_normalization_1/moving_variance
 "
trackable_list_wrapper
<
/0
01
12
23"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
х
3regularization_losses
4	variables
5trainable_variables
┐non_trainable_variables
└layers
┴layer_metrics
┬metrics
 ├layer_regularization_losses
т__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
*:(@ђ2conv2d_1/kernel
:ђ2conv2d_1/bias
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
х
9regularization_losses
:	variables
;trainable_variables
─non_trainable_variables
┼layers
кlayer_metrics
Кmetrics
 ╚layer_regularization_losses
у__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(ђ2batch_normalization_2/gamma
):'ђ2batch_normalization_2/beta
2:0ђ (2!batch_normalization_2/moving_mean
6:4ђ (2%batch_normalization_2/moving_variance
 "
trackable_list_wrapper
<
>0
?1
@2
A3"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
х
Bregularization_losses
C	variables
Dtrainable_variables
╔non_trainable_variables
╩layers
╦layer_metrics
╠metrics
 ═layer_regularization_losses
ж__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
+:)ђђ2conv2d_2/kernel
:ђ2conv2d_2/bias
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
х
Hregularization_losses
I	variables
Jtrainable_variables
╬non_trainable_variables
¤layers
лlayer_metrics
Лmetrics
 мlayer_regularization_losses
в__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(ђ2batch_normalization_3/gamma
):'ђ2batch_normalization_3/beta
2:0ђ (2!batch_normalization_3/moving_mean
6:4ђ (2%batch_normalization_3/moving_variance
 "
trackable_list_wrapper
<
M0
N1
O2
P3"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
х
Qregularization_losses
R	variables
Strainable_variables
Мnon_trainable_variables
нlayers
Нlayer_metrics
оmetrics
 Оlayer_regularization_losses
ь__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
+:)ђђ2conv2d_3/kernel
:ђ2conv2d_3/bias
 "
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
х
Wregularization_losses
X	variables
Ytrainable_variables
пnon_trainable_variables
┘layers
┌layer_metrics
█metrics
 ▄layer_regularization_losses
№__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(ђ2batch_normalization_4/gamma
):'ђ2batch_normalization_4/beta
2:0ђ (2!batch_normalization_4/moving_mean
6:4ђ (2%batch_normalization_4/moving_variance
 "
trackable_list_wrapper
<
\0
]1
^2
_3"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
х
`regularization_losses
a	variables
btrainable_variables
Пnon_trainable_variables
яlayers
▀layer_metrics
Яmetrics
 рlayer_regularization_losses
ы__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
3:1ђђ2conv2d_transpose/kernel
$:"ђ2conv2d_transpose/bias
 "
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
х
fregularization_losses
g	variables
htrainable_variables
Рnon_trainable_variables
сlayers
Сlayer_metrics
тmetrics
 Тlayer_regularization_losses
з__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(ђ2batch_normalization_5/gamma
):'ђ2batch_normalization_5/beta
2:0ђ (2!batch_normalization_5/moving_mean
6:4ђ (2%batch_normalization_5/moving_variance
 "
trackable_list_wrapper
<
k0
l1
m2
n3"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
х
oregularization_losses
p	variables
qtrainable_variables
уnon_trainable_variables
Уlayers
жlayer_metrics
Жmetrics
 вlayer_regularization_losses
ш__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
5:3ђђ2conv2d_transpose_1/kernel
&:$ђ2conv2d_transpose_1/bias
 "
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
х
uregularization_losses
v	variables
wtrainable_variables
Вnon_trainable_variables
ьlayers
Ьlayer_metrics
№metrics
 ­layer_regularization_losses
э__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(ђ2batch_normalization_6/gamma
):'ђ2batch_normalization_6/beta
2:0ђ (2!batch_normalization_6/moving_mean
6:4ђ (2%batch_normalization_6/moving_variance
 "
trackable_list_wrapper
<
z0
{1
|2
}3"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
Х
~regularization_losses
	variables
ђtrainable_variables
ыnon_trainable_variables
Ыlayers
зlayer_metrics
Зmetrics
 шlayer_regularization_losses
щ__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
5:3ђђ2conv2d_transpose_2/kernel
&:$ђ2conv2d_transpose_2/bias
 "
trackable_list_wrapper
0
ѓ0
Ѓ1"
trackable_list_wrapper
0
ѓ0
Ѓ1"
trackable_list_wrapper
И
ёregularization_losses
Ё	variables
єtrainable_variables
Шnon_trainable_variables
эlayers
Эlayer_metrics
щmetrics
 Щlayer_regularization_losses
ч__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(ђ2batch_normalization_7/gamma
):'ђ2batch_normalization_7/beta
2:0ђ (2!batch_normalization_7/moving_mean
6:4ђ (2%batch_normalization_7/moving_variance
 "
trackable_list_wrapper
@
Ѕ0
і1
І2
ї3"
trackable_list_wrapper
0
Ѕ0
і1"
trackable_list_wrapper
И
Їregularization_losses
ј	variables
Јtrainable_variables
чnon_trainable_variables
Чlayers
§layer_metrics
■metrics
  layer_regularization_losses
§__call__
+■&call_and_return_all_conditional_losses
'■"call_and_return_conditional_losses"
_generic_user_object
4:2@ђ2conv2d_transpose_3/kernel
%:#@2conv2d_transpose_3/bias
 "
trackable_list_wrapper
0
Љ0
њ1"
trackable_list_wrapper
0
Љ0
њ1"
trackable_list_wrapper
И
Њregularization_losses
ћ	variables
Ћtrainable_variables
ђnon_trainable_variables
Ђlayers
ѓlayer_metrics
Ѓmetrics
 ёlayer_regularization_losses
 __call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_8/gamma
(:&@2batch_normalization_8/beta
1:/@ (2!batch_normalization_8/moving_mean
5:3@ (2%batch_normalization_8/moving_variance
 "
trackable_list_wrapper
@
ў0
Ў1
џ2
Џ3"
trackable_list_wrapper
0
ў0
Ў1"
trackable_list_wrapper
И
юregularization_losses
Ю	variables
ъtrainable_variables
Ёnon_trainable_variables
єlayers
Єlayer_metrics
ѕmetrics
 Ѕlayer_regularization_losses
Ђ__call__
+ѓ&call_and_return_all_conditional_losses
'ѓ"call_and_return_conditional_losses"
_generic_user_object
3:1@2conv2d_transpose_4/kernel
%:#2conv2d_transpose_4/bias
 "
trackable_list_wrapper
0
а0
А1"
trackable_list_wrapper
0
а0
А1"
trackable_list_wrapper
И
бregularization_losses
Б	variables
цtrainable_variables
іnon_trainable_variables
Іlayers
їlayer_metrics
Їmetrics
 јlayer_regularization_losses
Ѓ__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ф
"0
#1
12
23
@4
A5
O6
P7
^8
_9
m10
n11
|12
}13
І14
ї15
џ16
Џ17"
trackable_list_wrapper
Х
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
19"
trackable_list_wrapper
 "
trackable_dict_wrapper
(
Ј0"
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
.
"0
#1"
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
.
10
21"
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
.
@0
A1"
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
.
O0
P1"
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
.
^0
_1"
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
.
m0
n1"
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
.
|0
}1"
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
0
І0
ї1"
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
0
џ0
Џ1"
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
R

љtotal

Љcount
њ	variables
Њ	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
љ0
Љ1"
trackable_list_wrapper
.
њ	variables"
_generic_user_object
,:*2 Adam/batch_normalization/gamma/m
+:)2Adam/batch_normalization/beta/m
,:*@2Adam/conv2d/kernel/m
:@2Adam/conv2d/bias/m
.:,@2"Adam/batch_normalization_1/gamma/m
-:+@2!Adam/batch_normalization_1/beta/m
/:-@ђ2Adam/conv2d_1/kernel/m
!:ђ2Adam/conv2d_1/bias/m
/:-ђ2"Adam/batch_normalization_2/gamma/m
.:,ђ2!Adam/batch_normalization_2/beta/m
0:.ђђ2Adam/conv2d_2/kernel/m
!:ђ2Adam/conv2d_2/bias/m
/:-ђ2"Adam/batch_normalization_3/gamma/m
.:,ђ2!Adam/batch_normalization_3/beta/m
0:.ђђ2Adam/conv2d_3/kernel/m
!:ђ2Adam/conv2d_3/bias/m
/:-ђ2"Adam/batch_normalization_4/gamma/m
.:,ђ2!Adam/batch_normalization_4/beta/m
8:6ђђ2Adam/conv2d_transpose/kernel/m
):'ђ2Adam/conv2d_transpose/bias/m
/:-ђ2"Adam/batch_normalization_5/gamma/m
.:,ђ2!Adam/batch_normalization_5/beta/m
::8ђђ2 Adam/conv2d_transpose_1/kernel/m
+:)ђ2Adam/conv2d_transpose_1/bias/m
/:-ђ2"Adam/batch_normalization_6/gamma/m
.:,ђ2!Adam/batch_normalization_6/beta/m
::8ђђ2 Adam/conv2d_transpose_2/kernel/m
+:)ђ2Adam/conv2d_transpose_2/bias/m
/:-ђ2"Adam/batch_normalization_7/gamma/m
.:,ђ2!Adam/batch_normalization_7/beta/m
9:7@ђ2 Adam/conv2d_transpose_3/kernel/m
*:(@2Adam/conv2d_transpose_3/bias/m
.:,@2"Adam/batch_normalization_8/gamma/m
-:+@2!Adam/batch_normalization_8/beta/m
8:6@2 Adam/conv2d_transpose_4/kernel/m
*:(2Adam/conv2d_transpose_4/bias/m
,:*2 Adam/batch_normalization/gamma/v
+:)2Adam/batch_normalization/beta/v
,:*@2Adam/conv2d/kernel/v
:@2Adam/conv2d/bias/v
.:,@2"Adam/batch_normalization_1/gamma/v
-:+@2!Adam/batch_normalization_1/beta/v
/:-@ђ2Adam/conv2d_1/kernel/v
!:ђ2Adam/conv2d_1/bias/v
/:-ђ2"Adam/batch_normalization_2/gamma/v
.:,ђ2!Adam/batch_normalization_2/beta/v
0:.ђђ2Adam/conv2d_2/kernel/v
!:ђ2Adam/conv2d_2/bias/v
/:-ђ2"Adam/batch_normalization_3/gamma/v
.:,ђ2!Adam/batch_normalization_3/beta/v
0:.ђђ2Adam/conv2d_3/kernel/v
!:ђ2Adam/conv2d_3/bias/v
/:-ђ2"Adam/batch_normalization_4/gamma/v
.:,ђ2!Adam/batch_normalization_4/beta/v
8:6ђђ2Adam/conv2d_transpose/kernel/v
):'ђ2Adam/conv2d_transpose/bias/v
/:-ђ2"Adam/batch_normalization_5/gamma/v
.:,ђ2!Adam/batch_normalization_5/beta/v
::8ђђ2 Adam/conv2d_transpose_1/kernel/v
+:)ђ2Adam/conv2d_transpose_1/bias/v
/:-ђ2"Adam/batch_normalization_6/gamma/v
.:,ђ2!Adam/batch_normalization_6/beta/v
::8ђђ2 Adam/conv2d_transpose_2/kernel/v
+:)ђ2Adam/conv2d_transpose_2/bias/v
/:-ђ2"Adam/batch_normalization_7/gamma/v
.:,ђ2!Adam/batch_normalization_7/beta/v
9:7@ђ2 Adam/conv2d_transpose_3/kernel/v
*:(@2Adam/conv2d_transpose_3/bias/v
.:,@2"Adam/batch_normalization_8/gamma/v
-:+@2!Adam/batch_normalization_8/beta/v
8:6@2 Adam/conv2d_transpose_4/kernel/v
*:(2Adam/conv2d_transpose_4/bias/v
╩BК
!__inference__wrapped_model_160234u_vel"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ѓ2 
-__inference_CNN_baseline_layer_call_fn_162436
-__inference_CNN_baseline_layer_call_fn_163889
-__inference_CNN_baseline_layer_call_fn_164002
-__inference_CNN_baseline_layer_call_fn_163393└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ь2в
H__inference_CNN_baseline_layer_call_and_return_conditional_losses_164269
H__inference_CNN_baseline_layer_call_and_return_conditional_losses_164536
H__inference_CNN_baseline_layer_call_and_return_conditional_losses_163524
H__inference_CNN_baseline_layer_call_and_return_conditional_losses_163655└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
м2¤
(__inference_reshape_layer_call_fn_164541б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ь2Ж
C__inference_reshape_layer_call_and_return_conditional_losses_164555б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
њ2Ј
4__inference_batch_normalization_layer_call_fn_164568
4__inference_batch_normalization_layer_call_fn_164581
4__inference_batch_normalization_layer_call_fn_164594
4__inference_batch_normalization_layer_call_fn_164607┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
■2ч
O__inference_batch_normalization_layer_call_and_return_conditional_losses_164625
O__inference_batch_normalization_layer_call_and_return_conditional_losses_164643
O__inference_batch_normalization_layer_call_and_return_conditional_losses_164661
O__inference_batch_normalization_layer_call_and_return_conditional_losses_164679┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Л2╬
'__inference_conv2d_layer_call_fn_164688б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
В2ж
B__inference_conv2d_layer_call_and_return_conditional_losses_164699б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
џ2Ќ
6__inference_batch_normalization_1_layer_call_fn_164712
6__inference_batch_normalization_1_layer_call_fn_164725
6__inference_batch_normalization_1_layer_call_fn_164738
6__inference_batch_normalization_1_layer_call_fn_164751┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
є2Ѓ
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_164769
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_164787
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_164805
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_164823┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
М2л
)__inference_conv2d_1_layer_call_fn_164832б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_conv2d_1_layer_call_and_return_conditional_losses_164843б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
џ2Ќ
6__inference_batch_normalization_2_layer_call_fn_164856
6__inference_batch_normalization_2_layer_call_fn_164869
6__inference_batch_normalization_2_layer_call_fn_164882
6__inference_batch_normalization_2_layer_call_fn_164895┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
є2Ѓ
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_164913
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_164931
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_164949
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_164967┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
М2л
)__inference_conv2d_2_layer_call_fn_164976б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_conv2d_2_layer_call_and_return_conditional_losses_164987б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
џ2Ќ
6__inference_batch_normalization_3_layer_call_fn_165000
6__inference_batch_normalization_3_layer_call_fn_165013
6__inference_batch_normalization_3_layer_call_fn_165026
6__inference_batch_normalization_3_layer_call_fn_165039┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
є2Ѓ
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_165057
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_165075
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_165093
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_165111┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
М2л
)__inference_conv2d_3_layer_call_fn_165120б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_conv2d_3_layer_call_and_return_conditional_losses_165131б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
џ2Ќ
6__inference_batch_normalization_4_layer_call_fn_165144
6__inference_batch_normalization_4_layer_call_fn_165157
6__inference_batch_normalization_4_layer_call_fn_165170
6__inference_batch_normalization_4_layer_call_fn_165183┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
є2Ѓ
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_165201
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_165219
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_165237
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_165255┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ј2І
1__inference_conv2d_transpose_layer_call_fn_165264
1__inference_conv2d_transpose_layer_call_fn_165273б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
─2┴
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_165311
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_165335б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
џ2Ќ
6__inference_batch_normalization_5_layer_call_fn_165348
6__inference_batch_normalization_5_layer_call_fn_165361
6__inference_batch_normalization_5_layer_call_fn_165374
6__inference_batch_normalization_5_layer_call_fn_165387┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
є2Ѓ
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_165405
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_165423
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_165441
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_165459┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
њ2Ј
3__inference_conv2d_transpose_1_layer_call_fn_165468
3__inference_conv2d_transpose_1_layer_call_fn_165477б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╚2┼
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_165515
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_165539б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
џ2Ќ
6__inference_batch_normalization_6_layer_call_fn_165552
6__inference_batch_normalization_6_layer_call_fn_165565
6__inference_batch_normalization_6_layer_call_fn_165578
6__inference_batch_normalization_6_layer_call_fn_165591┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
є2Ѓ
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_165609
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_165627
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_165645
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_165663┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
њ2Ј
3__inference_conv2d_transpose_2_layer_call_fn_165672
3__inference_conv2d_transpose_2_layer_call_fn_165681б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╚2┼
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_165719
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_165743б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
џ2Ќ
6__inference_batch_normalization_7_layer_call_fn_165756
6__inference_batch_normalization_7_layer_call_fn_165769
6__inference_batch_normalization_7_layer_call_fn_165782
6__inference_batch_normalization_7_layer_call_fn_165795┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
є2Ѓ
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_165813
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_165831
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_165849
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_165867┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
њ2Ј
3__inference_conv2d_transpose_3_layer_call_fn_165876
3__inference_conv2d_transpose_3_layer_call_fn_165885б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╚2┼
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_165923
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_165947б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
џ2Ќ
6__inference_batch_normalization_8_layer_call_fn_165960
6__inference_batch_normalization_8_layer_call_fn_165973
6__inference_batch_normalization_8_layer_call_fn_165986
6__inference_batch_normalization_8_layer_call_fn_165999┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
є2Ѓ
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_166017
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_166035
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_166053
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_166071┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
њ2Ј
3__inference_conv2d_transpose_4_layer_call_fn_166080
3__inference_conv2d_transpose_4_layer_call_fn_166089б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╚2┼
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_166126
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_166149б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╔Bк
$__inference_signature_wrapper_163776u_vel"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 ѓ
H__inference_CNN_baseline_layer_call_and_return_conditional_losses_163524хD !"#()/01278>?@AFGMNOPUV\]^_deklmnstz{|}ѓЃЅіІїЉњўЎџЏаА<б9
2б/
%і"
u_vel         ђђ
p 

 
ф "/б,
%і"
0         ђђ
џ ѓ
H__inference_CNN_baseline_layer_call_and_return_conditional_losses_163655хD !"#()/01278>?@AFGMNOPUV\]^_deklmnstz{|}ѓЃЅіІїЉњўЎџЏаА<б9
2б/
%і"
u_vel         ђђ
p

 
ф "/б,
%і"
0         ђђ
џ Ѓ
H__inference_CNN_baseline_layer_call_and_return_conditional_losses_164269ХD !"#()/01278>?@AFGMNOPUV\]^_deklmnstz{|}ѓЃЅіІїЉњўЎџЏаА=б:
3б0
&і#
inputs         ђђ
p 

 
ф "/б,
%і"
0         ђђ
џ Ѓ
H__inference_CNN_baseline_layer_call_and_return_conditional_losses_164536ХD !"#()/01278>?@AFGMNOPUV\]^_deklmnstz{|}ѓЃЅіІїЉњўЎџЏаА=б:
3б0
&і#
inputs         ђђ
p

 
ф "/б,
%і"
0         ђђ
џ ┌
-__inference_CNN_baseline_layer_call_fn_162436еD !"#()/01278>?@AFGMNOPUV\]^_deklmnstz{|}ѓЃЅіІїЉњўЎџЏаА<б9
2б/
%і"
u_vel         ђђ
p 

 
ф ""і         ђђ┌
-__inference_CNN_baseline_layer_call_fn_163393еD !"#()/01278>?@AFGMNOPUV\]^_deklmnstz{|}ѓЃЅіІїЉњўЎџЏаА<б9
2б/
%і"
u_vel         ђђ
p

 
ф ""і         ђђ█
-__inference_CNN_baseline_layer_call_fn_163889ЕD !"#()/01278>?@AFGMNOPUV\]^_deklmnstz{|}ѓЃЅіІїЉњўЎџЏаА=б:
3б0
&і#
inputs         ђђ
p 

 
ф ""і         ђђ█
-__inference_CNN_baseline_layer_call_fn_164002ЕD !"#()/01278>?@AFGMNOPUV\]^_deklmnstz{|}ѓЃЅіІїЉњўЎџЏаА=б:
3б0
&і#
inputs         ђђ
p

 
ф ""і         ђђш
!__inference__wrapped_model_160234¤D !"#()/01278>?@AFGMNOPUV\]^_deklmnstz{|}ѓЃЅіІїЉњўЎџЏаА4б1
*б'
%і"
u_vel         ђђ
ф "QфN
L
conv2d_transpose_46і3
conv2d_transpose_4         ђђВ
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_164769ќ/012MбJ
Cб@
:і7
inputs+                           @
p 
ф "?б<
5і2
0+                           @
џ В
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_164787ќ/012MбJ
Cб@
:і7
inputs+                           @
p
ф "?б<
5і2
0+                           @
џ ╦
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_164805v/012=б:
3б0
*і'
inputs         ЧЧ@
p 
ф "/б,
%і"
0         ЧЧ@
џ ╦
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_164823v/012=б:
3б0
*і'
inputs         ЧЧ@
p
ф "/б,
%і"
0         ЧЧ@
џ ─
6__inference_batch_normalization_1_layer_call_fn_164712Ѕ/012MбJ
Cб@
:і7
inputs+                           @
p 
ф "2і/+                           @─
6__inference_batch_normalization_1_layer_call_fn_164725Ѕ/012MбJ
Cб@
:і7
inputs+                           @
p
ф "2і/+                           @Б
6__inference_batch_normalization_1_layer_call_fn_164738i/012=б:
3б0
*і'
inputs         ЧЧ@
p 
ф ""і         ЧЧ@Б
6__inference_batch_normalization_1_layer_call_fn_164751i/012=б:
3б0
*і'
inputs         ЧЧ@
p
ф ""і         ЧЧ@Ь
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_164913ў>?@ANбK
DбA
;і8
inputs,                           ђ
p 
ф "@б=
6і3
0,                           ђ
џ Ь
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_164931ў>?@ANбK
DбA
;і8
inputs,                           ђ
p
ф "@б=
6і3
0,                           ђ
џ ═
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_164949x>?@A>б;
4б1
+і(
inputs         ЩЩђ
p 
ф "0б-
&і#
0         ЩЩђ
џ ═
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_164967x>?@A>б;
4б1
+і(
inputs         ЩЩђ
p
ф "0б-
&і#
0         ЩЩђ
џ к
6__inference_batch_normalization_2_layer_call_fn_164856І>?@ANбK
DбA
;і8
inputs,                           ђ
p 
ф "3і0,                           ђк
6__inference_batch_normalization_2_layer_call_fn_164869І>?@ANбK
DбA
;і8
inputs,                           ђ
p
ф "3і0,                           ђЦ
6__inference_batch_normalization_2_layer_call_fn_164882k>?@A>б;
4б1
+і(
inputs         ЩЩђ
p 
ф "#і          ЩЩђЦ
6__inference_batch_normalization_2_layer_call_fn_164895k>?@A>б;
4б1
+і(
inputs         ЩЩђ
p
ф "#і          ЩЩђЬ
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_165057ўMNOPNбK
DбA
;і8
inputs,                           ђ
p 
ф "@б=
6і3
0,                           ђ
џ Ь
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_165075ўMNOPNбK
DбA
;і8
inputs,                           ђ
p
ф "@б=
6і3
0,                           ђ
џ ═
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_165093xMNOP>б;
4б1
+і(
inputs         ЭЭђ
p 
ф "0б-
&і#
0         ЭЭђ
џ ═
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_165111xMNOP>б;
4б1
+і(
inputs         ЭЭђ
p
ф "0б-
&і#
0         ЭЭђ
џ к
6__inference_batch_normalization_3_layer_call_fn_165000ІMNOPNбK
DбA
;і8
inputs,                           ђ
p 
ф "3і0,                           ђк
6__inference_batch_normalization_3_layer_call_fn_165013ІMNOPNбK
DбA
;і8
inputs,                           ђ
p
ф "3і0,                           ђЦ
6__inference_batch_normalization_3_layer_call_fn_165026kMNOP>б;
4б1
+і(
inputs         ЭЭђ
p 
ф "#і          ЭЭђЦ
6__inference_batch_normalization_3_layer_call_fn_165039kMNOP>б;
4б1
+і(
inputs         ЭЭђ
p
ф "#і          ЭЭђЬ
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_165201ў\]^_NбK
DбA
;і8
inputs,                           ђ
p 
ф "@б=
6і3
0,                           ђ
џ Ь
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_165219ў\]^_NбK
DбA
;і8
inputs,                           ђ
p
ф "@б=
6і3
0,                           ђ
џ ═
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_165237x\]^_>б;
4б1
+і(
inputs         ШШђ
p 
ф "0б-
&і#
0         ШШђ
џ ═
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_165255x\]^_>б;
4б1
+і(
inputs         ШШђ
p
ф "0б-
&і#
0         ШШђ
џ к
6__inference_batch_normalization_4_layer_call_fn_165144І\]^_NбK
DбA
;і8
inputs,                           ђ
p 
ф "3і0,                           ђк
6__inference_batch_normalization_4_layer_call_fn_165157І\]^_NбK
DбA
;і8
inputs,                           ђ
p
ф "3і0,                           ђЦ
6__inference_batch_normalization_4_layer_call_fn_165170k\]^_>б;
4б1
+і(
inputs         ШШђ
p 
ф "#і          ШШђЦ
6__inference_batch_normalization_4_layer_call_fn_165183k\]^_>б;
4б1
+і(
inputs         ШШђ
p
ф "#і          ШШђЬ
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_165405ўklmnNбK
DбA
;і8
inputs,                           ђ
p 
ф "@б=
6і3
0,                           ђ
џ Ь
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_165423ўklmnNбK
DбA
;і8
inputs,                           ђ
p
ф "@б=
6і3
0,                           ђ
џ ═
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_165441xklmn>б;
4б1
+і(
inputs         ЭЭђ
p 
ф "0б-
&і#
0         ЭЭђ
џ ═
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_165459xklmn>б;
4б1
+і(
inputs         ЭЭђ
p
ф "0б-
&і#
0         ЭЭђ
џ к
6__inference_batch_normalization_5_layer_call_fn_165348ІklmnNбK
DбA
;і8
inputs,                           ђ
p 
ф "3і0,                           ђк
6__inference_batch_normalization_5_layer_call_fn_165361ІklmnNбK
DбA
;і8
inputs,                           ђ
p
ф "3і0,                           ђЦ
6__inference_batch_normalization_5_layer_call_fn_165374kklmn>б;
4б1
+і(
inputs         ЭЭђ
p 
ф "#і          ЭЭђЦ
6__inference_batch_normalization_5_layer_call_fn_165387kklmn>б;
4б1
+і(
inputs         ЭЭђ
p
ф "#і          ЭЭђЬ
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_165609ўz{|}NбK
DбA
;і8
inputs,                           ђ
p 
ф "@б=
6і3
0,                           ђ
џ Ь
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_165627ўz{|}NбK
DбA
;і8
inputs,                           ђ
p
ф "@б=
6і3
0,                           ђ
џ ═
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_165645xz{|}>б;
4б1
+і(
inputs         ЩЩђ
p 
ф "0б-
&і#
0         ЩЩђ
џ ═
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_165663xz{|}>б;
4б1
+і(
inputs         ЩЩђ
p
ф "0б-
&і#
0         ЩЩђ
џ к
6__inference_batch_normalization_6_layer_call_fn_165552Іz{|}NбK
DбA
;і8
inputs,                           ђ
p 
ф "3і0,                           ђк
6__inference_batch_normalization_6_layer_call_fn_165565Іz{|}NбK
DбA
;і8
inputs,                           ђ
p
ф "3і0,                           ђЦ
6__inference_batch_normalization_6_layer_call_fn_165578kz{|}>б;
4б1
+і(
inputs         ЩЩђ
p 
ф "#і          ЩЩђЦ
6__inference_batch_normalization_6_layer_call_fn_165591kz{|}>б;
4б1
+і(
inputs         ЩЩђ
p
ф "#і          ЩЩђЫ
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_165813юЅіІїNбK
DбA
;і8
inputs,                           ђ
p 
ф "@б=
6і3
0,                           ђ
џ Ы
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_165831юЅіІїNбK
DбA
;і8
inputs,                           ђ
p
ф "@б=
6і3
0,                           ђ
џ Л
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_165849|ЅіІї>б;
4б1
+і(
inputs         ЧЧђ
p 
ф "0б-
&і#
0         ЧЧђ
џ Л
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_165867|ЅіІї>б;
4б1
+і(
inputs         ЧЧђ
p
ф "0б-
&і#
0         ЧЧђ
џ ╩
6__inference_batch_normalization_7_layer_call_fn_165756ЈЅіІїNбK
DбA
;і8
inputs,                           ђ
p 
ф "3і0,                           ђ╩
6__inference_batch_normalization_7_layer_call_fn_165769ЈЅіІїNбK
DбA
;і8
inputs,                           ђ
p
ф "3і0,                           ђЕ
6__inference_batch_normalization_7_layer_call_fn_165782oЅіІї>б;
4б1
+і(
inputs         ЧЧђ
p 
ф "#і          ЧЧђЕ
6__inference_batch_normalization_7_layer_call_fn_165795oЅіІї>б;
4б1
+і(
inputs         ЧЧђ
p
ф "#і          ЧЧђ­
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_166017џўЎџЏMбJ
Cб@
:і7
inputs+                           @
p 
ф "?б<
5і2
0+                           @
џ ­
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_166035џўЎџЏMбJ
Cб@
:і7
inputs+                           @
p
ф "?б<
5і2
0+                           @
џ ¤
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_166053zўЎџЏ=б:
3б0
*і'
inputs         ђђ@
p 
ф "/б,
%і"
0         ђђ@
џ ¤
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_166071zўЎџЏ=б:
3б0
*і'
inputs         ђђ@
p
ф "/б,
%і"
0         ђђ@
џ ╚
6__inference_batch_normalization_8_layer_call_fn_165960ЇўЎџЏMбJ
Cб@
:і7
inputs+                           @
p 
ф "2і/+                           @╚
6__inference_batch_normalization_8_layer_call_fn_165973ЇўЎџЏMбJ
Cб@
:і7
inputs+                           @
p
ф "2і/+                           @Д
6__inference_batch_normalization_8_layer_call_fn_165986mўЎџЏ=б:
3б0
*і'
inputs         ђђ@
p 
ф ""і         ђђ@Д
6__inference_batch_normalization_8_layer_call_fn_165999mўЎџЏ=б:
3б0
*і'
inputs         ђђ@
p
ф ""і         ђђ@Ж
O__inference_batch_normalization_layer_call_and_return_conditional_losses_164625ќ !"#MбJ
Cб@
:і7
inputs+                           
p 
ф "?б<
5і2
0+                           
џ Ж
O__inference_batch_normalization_layer_call_and_return_conditional_losses_164643ќ !"#MбJ
Cб@
:і7
inputs+                           
p
ф "?б<
5і2
0+                           
џ ╔
O__inference_batch_normalization_layer_call_and_return_conditional_losses_164661v !"#=б:
3б0
*і'
inputs         ђђ
p 
ф "/б,
%і"
0         ђђ
џ ╔
O__inference_batch_normalization_layer_call_and_return_conditional_losses_164679v !"#=б:
3б0
*і'
inputs         ђђ
p
ф "/б,
%і"
0         ђђ
џ ┬
4__inference_batch_normalization_layer_call_fn_164568Ѕ !"#MбJ
Cб@
:і7
inputs+                           
p 
ф "2і/+                           ┬
4__inference_batch_normalization_layer_call_fn_164581Ѕ !"#MбJ
Cб@
:і7
inputs+                           
p
ф "2і/+                           А
4__inference_batch_normalization_layer_call_fn_164594i !"#=б:
3б0
*і'
inputs         ђђ
p 
ф ""і         ђђА
4__inference_batch_normalization_layer_call_fn_164607i !"#=б:
3б0
*і'
inputs         ђђ
p
ф ""і         ђђ╣
D__inference_conv2d_1_layer_call_and_return_conditional_losses_164843q789б6
/б,
*і'
inputs         ЧЧ@
ф "0б-
&і#
0         ЩЩђ
џ Љ
)__inference_conv2d_1_layer_call_fn_164832d789б6
/б,
*і'
inputs         ЧЧ@
ф "#і          ЩЩђ║
D__inference_conv2d_2_layer_call_and_return_conditional_losses_164987rFG:б7
0б-
+і(
inputs         ЩЩђ
ф "0б-
&і#
0         ЭЭђ
џ њ
)__inference_conv2d_2_layer_call_fn_164976eFG:б7
0б-
+і(
inputs         ЩЩђ
ф "#і          ЭЭђ║
D__inference_conv2d_3_layer_call_and_return_conditional_losses_165131rUV:б7
0б-
+і(
inputs         ЭЭђ
ф "0б-
&і#
0         ШШђ
џ њ
)__inference_conv2d_3_layer_call_fn_165120eUV:б7
0б-
+і(
inputs         ЭЭђ
ф "#і          ШШђХ
B__inference_conv2d_layer_call_and_return_conditional_losses_164699p()9б6
/б,
*і'
inputs         ђђ
ф "/б,
%і"
0         ЧЧ@
џ ј
'__inference_conv2d_layer_call_fn_164688c()9б6
/б,
*і'
inputs         ђђ
ф ""і         ЧЧ@т
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_165515њstJбG
@б=
;і8
inputs,                           ђ
ф "@б=
6і3
0,                           ђ
џ ─
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_165539rst:б7
0б-
+і(
inputs         ЭЭђ
ф "0б-
&і#
0         ЩЩђ
џ й
3__inference_conv2d_transpose_1_layer_call_fn_165468ЁstJбG
@б=
;і8
inputs,                           ђ
ф "3і0,                           ђю
3__inference_conv2d_transpose_1_layer_call_fn_165477est:б7
0б-
+і(
inputs         ЭЭђ
ф "#і          ЩЩђу
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_165719ћѓЃJбG
@б=
;і8
inputs,                           ђ
ф "@б=
6і3
0,                           ђ
џ к
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_165743tѓЃ:б7
0б-
+і(
inputs         ЩЩђ
ф "0б-
&і#
0         ЧЧђ
џ ┐
3__inference_conv2d_transpose_2_layer_call_fn_165672ЄѓЃJбG
@б=
;і8
inputs,                           ђ
ф "3і0,                           ђъ
3__inference_conv2d_transpose_2_layer_call_fn_165681gѓЃ:б7
0б-
+і(
inputs         ЩЩђ
ф "#і          ЧЧђТ
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_165923ЊЉњJбG
@б=
;і8
inputs,                           ђ
ф "?б<
5і2
0+                           @
џ ┼
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_165947sЉњ:б7
0б-
+і(
inputs         ЧЧђ
ф "/б,
%і"
0         ђђ@
џ Й
3__inference_conv2d_transpose_3_layer_call_fn_165876єЉњJбG
@б=
;і8
inputs,                           ђ
ф "2і/+                           @Ю
3__inference_conv2d_transpose_3_layer_call_fn_165885fЉњ:б7
0б-
+і(
inputs         ЧЧђ
ф ""і         ђђ@т
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_166126њаАIбF
?б<
:і7
inputs+                           @
ф "?б<
5і2
0+                           
џ ─
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_166149rаА9б6
/б,
*і'
inputs         ђђ@
ф "/б,
%і"
0         ђђ
џ й
3__inference_conv2d_transpose_4_layer_call_fn_166080ЁаАIбF
?б<
:і7
inputs+                           @
ф "2і/+                           ю
3__inference_conv2d_transpose_4_layer_call_fn_166089eаА9б6
/б,
*і'
inputs         ђђ@
ф ""і         ђђс
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_165311њdeJбG
@б=
;і8
inputs,                           ђ
ф "@б=
6і3
0,                           ђ
џ ┬
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_165335rde:б7
0б-
+і(
inputs         ШШђ
ф "0б-
&і#
0         ЭЭђ
џ ╗
1__inference_conv2d_transpose_layer_call_fn_165264ЁdeJбG
@б=
;і8
inputs,                           ђ
ф "3і0,                           ђџ
1__inference_conv2d_transpose_layer_call_fn_165273ede:б7
0б-
+і(
inputs         ШШђ
ф "#і          ЭЭђ»
C__inference_reshape_layer_call_and_return_conditional_losses_164555h5б2
+б(
&і#
inputs         ђђ
ф "/б,
%і"
0         ђђ
џ Є
(__inference_reshape_layer_call_fn_164541[5б2
+б(
&і#
inputs         ђђ
ф ""і         ђђЂ
$__inference_signature_wrapper_163776пD !"#()/01278>?@AFGMNOPUV\]^_deklmnstz{|}ѓЃЅіІїЉњўЎџЏаА=б:
б 
3ф0
.
u_vel%і"
u_vel         ђђ"QфN
L
conv2d_transpose_46і3
conv2d_transpose_4         ђђ