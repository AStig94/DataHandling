│т5
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
 ѕ"serve*2.6.02v2.6.0-rc2-32-g919f693420e8ЋЌ1
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
ЊО
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*═о
value┬оBЙо BХо
Њ
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer_with_weights-10
layer-16
layer_with_weights-11
layer-17
layer_with_weights-12
layer-18
layer_with_weights-13
layer-19
layer_with_weights-14
layer-20
layer_with_weights-15
layer-21
layer_with_weights-16
layer-22
layer_with_weights-17
layer-23
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
R
	variables
 regularization_losses
!trainable_variables
"	keras_api
Ќ
#axis
	$gamma
%beta
&moving_mean
'moving_variance
(	variables
)regularization_losses
*trainable_variables
+	keras_api
R
,	variables
-regularization_losses
.trainable_variables
/	keras_api
h

0kernel
1bias
2	variables
3regularization_losses
4trainable_variables
5	keras_api
Ќ
6axis
	7gamma
8beta
9moving_mean
:moving_variance
;	variables
<regularization_losses
=trainable_variables
>	keras_api
R
?	variables
@regularization_losses
Atrainable_variables
B	keras_api
h

Ckernel
Dbias
E	variables
Fregularization_losses
Gtrainable_variables
H	keras_api
Ќ
Iaxis
	Jgamma
Kbeta
Lmoving_mean
Mmoving_variance
N	variables
Oregularization_losses
Ptrainable_variables
Q	keras_api
R
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
h

Vkernel
Wbias
X	variables
Yregularization_losses
Ztrainable_variables
[	keras_api
Ќ
\axis
	]gamma
^beta
_moving_mean
`moving_variance
a	variables
bregularization_losses
ctrainable_variables
d	keras_api
R
e	variables
fregularization_losses
gtrainable_variables
h	keras_api
h

ikernel
jbias
k	variables
lregularization_losses
mtrainable_variables
n	keras_api
Ќ
oaxis
	pgamma
qbeta
rmoving_mean
smoving_variance
t	variables
uregularization_losses
vtrainable_variables
w	keras_api
h

xkernel
ybias
z	variables
{regularization_losses
|trainable_variables
}	keras_api
ъ
~axis
	gamma
	ђbeta
Ђmoving_mean
ѓmoving_variance
Ѓ	variables
ёregularization_losses
Ёtrainable_variables
є	keras_api
n
Єkernel
	ѕbias
Ѕ	variables
іregularization_losses
Іtrainable_variables
ї	keras_api
а
	Їaxis

јgamma
	Јbeta
љmoving_mean
Љmoving_variance
њ	variables
Њregularization_losses
ћtrainable_variables
Ћ	keras_api
n
ќkernel
	Ќbias
ў	variables
Ўregularization_losses
џtrainable_variables
Џ	keras_api
а
	юaxis

Юgamma
	ъbeta
Ъmoving_mean
аmoving_variance
А	variables
бregularization_losses
Бtrainable_variables
ц	keras_api
n
Цkernel
	дbias
Д	variables
еregularization_losses
Еtrainable_variables
ф	keras_api
а
	Фaxis

гgamma
	Гbeta
«moving_mean
»moving_variance
░	variables
▒regularization_losses
▓trainable_variables
│	keras_api
n
┤kernel
	хbias
Х	variables
иregularization_losses
Иtrainable_variables
╣	keras_api
│
	║iter
╗beta_1
╝beta_2

йdecay
Йlearning_rate$m╝%mй0mЙ1m┐7m└8m┴Cm┬Dm├Jm─Km┼VmкWmК]m╚^m╔im╩jm╦pm╠qm═xm╬ym¤mл	ђmЛ	Єmм	ѕmМ	јmн	ЈmН	ќmо	ЌmО	Юmп	ъm┘	Цm┌	дm█	гm▄	ГmП	┤mя	хm▀$vЯ%vр0vР1vс7vС8vтCvТDvуJvУKvжVvЖWvв]vВ^vьivЬjv№pv­qvыxvЫyvзvЗ	ђvш	ЄvШ	ѕvэ	јvЭ	Јvщ	ќvЩ	Ќvч	ЮvЧ	ъv§	Цv■	дv 	гvђ	ГvЂ	┤vѓ	хvЃ
й
$0
%1
&2
'3
04
15
76
87
98
:9
C10
D11
J12
K13
L14
M15
V16
W17
]18
^19
_20
`21
i22
j23
p24
q25
r26
s27
x28
y29
30
ђ31
Ђ32
ѓ33
Є34
ѕ35
ј36
Ј37
љ38
Љ39
ќ40
Ќ41
Ю42
ъ43
Ъ44
а45
Ц46
д47
г48
Г49
«50
»51
┤52
х53
 
Ц
$0
%1
02
13
74
85
C6
D7
J8
K9
V10
W11
]12
^13
i14
j15
p16
q17
x18
y19
20
ђ21
Є22
ѕ23
ј24
Ј25
ќ26
Ќ27
Ю28
ъ29
Ц30
д31
г32
Г33
┤34
х35
▓
┐layers
└metrics
	variables
┴layer_metrics
regularization_losses
 ┬layer_regularization_losses
trainable_variables
├non_trainable_variables
 
 
 
 
▓
─layers
┼metrics
	variables
кlayer_metrics
 regularization_losses
 Кlayer_regularization_losses
!trainable_variables
╚non_trainable_variables
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

$0
%1
&2
'3
 

$0
%1
▓
╔layers
╩metrics
(	variables
╦layer_metrics
)regularization_losses
 ╠layer_regularization_losses
*trainable_variables
═non_trainable_variables
 
 
 
▓
╬layers
¤metrics
,	variables
лlayer_metrics
-regularization_losses
 Лlayer_regularization_losses
.trainable_variables
мnon_trainable_variables
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

00
11
 

00
11
▓
Мlayers
нmetrics
2	variables
Нlayer_metrics
3regularization_losses
 оlayer_regularization_losses
4trainable_variables
Оnon_trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

70
81
92
:3
 

70
81
▓
пlayers
┘metrics
;	variables
┌layer_metrics
<regularization_losses
 █layer_regularization_losses
=trainable_variables
▄non_trainable_variables
 
 
 
▓
Пlayers
яmetrics
?	variables
▀layer_metrics
@regularization_losses
 Яlayer_regularization_losses
Atrainable_variables
рnon_trainable_variables
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

C0
D1
 

C0
D1
▓
Рlayers
сmetrics
E	variables
Сlayer_metrics
Fregularization_losses
 тlayer_regularization_losses
Gtrainable_variables
Тnon_trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

J0
K1
L2
M3
 

J0
K1
▓
уlayers
Уmetrics
N	variables
жlayer_metrics
Oregularization_losses
 Жlayer_regularization_losses
Ptrainable_variables
вnon_trainable_variables
 
 
 
▓
Вlayers
ьmetrics
R	variables
Ьlayer_metrics
Sregularization_losses
 №layer_regularization_losses
Ttrainable_variables
­non_trainable_variables
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

V0
W1
 

V0
W1
▓
ыlayers
Ыmetrics
X	variables
зlayer_metrics
Yregularization_losses
 Зlayer_regularization_losses
Ztrainable_variables
шnon_trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

]0
^1
_2
`3
 

]0
^1
▓
Шlayers
эmetrics
a	variables
Эlayer_metrics
bregularization_losses
 щlayer_regularization_losses
ctrainable_variables
Щnon_trainable_variables
 
 
 
▓
чlayers
Чmetrics
e	variables
§layer_metrics
fregularization_losses
 ■layer_regularization_losses
gtrainable_variables
 non_trainable_variables
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

i0
j1
 

i0
j1
▓
ђlayers
Ђmetrics
k	variables
ѓlayer_metrics
lregularization_losses
 Ѓlayer_regularization_losses
mtrainable_variables
ёnon_trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

p0
q1
r2
s3
 

p0
q1
▓
Ёlayers
єmetrics
t	variables
Єlayer_metrics
uregularization_losses
 ѕlayer_regularization_losses
vtrainable_variables
Ѕnon_trainable_variables
ca
VARIABLE_VALUEconv2d_transpose/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_transpose/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

x0
y1
 

x0
y1
▓
іlayers
Іmetrics
z	variables
їlayer_metrics
{regularization_losses
 Їlayer_regularization_losses
|trainable_variables
јnon_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_5/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_5/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_5/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_5/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
ђ1
Ђ2
ѓ3
 

0
ђ1
х
Јlayers
љmetrics
Ѓ	variables
Љlayer_metrics
ёregularization_losses
 њlayer_regularization_losses
Ёtrainable_variables
Њnon_trainable_variables
fd
VARIABLE_VALUEconv2d_transpose_1/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_1/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

Є0
ѕ1
 

Є0
ѕ1
х
ћlayers
Ћmetrics
Ѕ	variables
ќlayer_metrics
іregularization_losses
 Ќlayer_regularization_losses
Іtrainable_variables
ўnon_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_6/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_6/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_6/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_6/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
ј0
Ј1
љ2
Љ3
 

ј0
Ј1
х
Ўlayers
џmetrics
њ	variables
Џlayer_metrics
Њregularization_losses
 юlayer_regularization_losses
ћtrainable_variables
Юnon_trainable_variables
fd
VARIABLE_VALUEconv2d_transpose_2/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_2/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE

ќ0
Ќ1
 

ќ0
Ќ1
х
ъlayers
Ъmetrics
ў	variables
аlayer_metrics
Ўregularization_losses
 Аlayer_regularization_losses
џtrainable_variables
бnon_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_7/gamma6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_7/beta5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_7/moving_mean<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_7/moving_variance@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
Ю0
ъ1
Ъ2
а3
 

Ю0
ъ1
х
Бlayers
цmetrics
А	variables
Цlayer_metrics
бregularization_losses
 дlayer_regularization_losses
Бtrainable_variables
Дnon_trainable_variables
fd
VARIABLE_VALUEconv2d_transpose_3/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_3/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE

Ц0
д1
 

Ц0
д1
х
еlayers
Еmetrics
Д	variables
фlayer_metrics
еregularization_losses
 Фlayer_regularization_losses
Еtrainable_variables
гnon_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_8/gamma6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_8/beta5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_8/moving_mean<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_8/moving_variance@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
г0
Г1
«2
»3
 

г0
Г1
х
Гlayers
«metrics
░	variables
»layer_metrics
▒regularization_losses
 ░layer_regularization_losses
▓trainable_variables
▒non_trainable_variables
fd
VARIABLE_VALUEconv2d_transpose_4/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_4/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE

┤0
х1
 

┤0
х1
х
▓layers
│metrics
Х	variables
┤layer_metrics
иregularization_losses
 хlayer_regularization_losses
Иtrainable_variables
Хnon_trainable_variables
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
19
20
21
22
23

и0
 
 
ј
&0
'1
92
:3
L4
M5
_6
`7
r8
s9
Ђ10
ѓ11
љ12
Љ13
Ъ14
а15
«16
»17
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
&0
'1
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
90
:1
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
L0
M1
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
_0
`1
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
r0
s1
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
Ђ0
ѓ1
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
љ0
Љ1
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
Ъ0
а1
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
«0
»1
 
 
 
 
 
8

Иtotal

╣count
║	variables
╗	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

И0
╣1

║	variables
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
$__inference_signature_wrapper_236349
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
__inference__traced_save_239288
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
"__inference__traced_restore_239697ае,
А
ю
'__inference_conv2d_layer_call_fn_237324

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
B__inference_conv2d_layer_call_and_return_conditional_losses_2343552
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
§
¤
4__inference_batch_normalization_layer_call_fn_237216

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
O__inference_batch_normalization_layer_call_and_return_conditional_losses_2354672
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
Ѓ	
Л
6__inference_batch_normalization_1_layer_call_fn_237374

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
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2343782
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
с
ю
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_238734

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
Щ╣
┐8
!__inference__wrapped_model_232675	
u_velA
3dropout_batch_normalization_readvariableop_resource:C
5dropout_batch_normalization_readvariableop_1_resource:R
Ddropout_batch_normalization_fusedbatchnormv3_readvariableop_resource:T
Fdropout_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:G
-dropout_conv2d_conv2d_readvariableop_resource:@<
.dropout_conv2d_biasadd_readvariableop_resource:@C
5dropout_batch_normalization_1_readvariableop_resource:@E
7dropout_batch_normalization_1_readvariableop_1_resource:@T
Fdropout_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@V
Hdropout_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@J
/dropout_conv2d_1_conv2d_readvariableop_resource:@ђ?
0dropout_conv2d_1_biasadd_readvariableop_resource:	ђD
5dropout_batch_normalization_2_readvariableop_resource:	ђF
7dropout_batch_normalization_2_readvariableop_1_resource:	ђU
Fdropout_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	ђW
Hdropout_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	ђK
/dropout_conv2d_2_conv2d_readvariableop_resource:ђђ?
0dropout_conv2d_2_biasadd_readvariableop_resource:	ђD
5dropout_batch_normalization_3_readvariableop_resource:	ђF
7dropout_batch_normalization_3_readvariableop_1_resource:	ђU
Fdropout_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	ђW
Hdropout_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	ђK
/dropout_conv2d_3_conv2d_readvariableop_resource:ђђ?
0dropout_conv2d_3_biasadd_readvariableop_resource:	ђD
5dropout_batch_normalization_4_readvariableop_resource:	ђF
7dropout_batch_normalization_4_readvariableop_1_resource:	ђU
Fdropout_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	ђW
Hdropout_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	ђ]
Adropout_conv2d_transpose_conv2d_transpose_readvariableop_resource:ђђG
8dropout_conv2d_transpose_biasadd_readvariableop_resource:	ђD
5dropout_batch_normalization_5_readvariableop_resource:	ђF
7dropout_batch_normalization_5_readvariableop_1_resource:	ђU
Fdropout_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:	ђW
Hdropout_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:	ђ_
Cdropout_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:ђђI
:dropout_conv2d_transpose_1_biasadd_readvariableop_resource:	ђD
5dropout_batch_normalization_6_readvariableop_resource:	ђF
7dropout_batch_normalization_6_readvariableop_1_resource:	ђU
Fdropout_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:	ђW
Hdropout_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:	ђ_
Cdropout_conv2d_transpose_2_conv2d_transpose_readvariableop_resource:ђђI
:dropout_conv2d_transpose_2_biasadd_readvariableop_resource:	ђD
5dropout_batch_normalization_7_readvariableop_resource:	ђF
7dropout_batch_normalization_7_readvariableop_1_resource:	ђU
Fdropout_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:	ђW
Hdropout_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:	ђ^
Cdropout_conv2d_transpose_3_conv2d_transpose_readvariableop_resource:@ђH
:dropout_conv2d_transpose_3_biasadd_readvariableop_resource:@C
5dropout_batch_normalization_8_readvariableop_resource:@E
7dropout_batch_normalization_8_readvariableop_1_resource:@T
Fdropout_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:@V
Hdropout_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:@]
Cdropout_conv2d_transpose_4_conv2d_transpose_readvariableop_resource:@H
:dropout_conv2d_transpose_4_biasadd_readvariableop_resource:
identityѕб;dropout/batch_normalization/FusedBatchNormV3/ReadVariableOpб=dropout/batch_normalization/FusedBatchNormV3/ReadVariableOp_1б*dropout/batch_normalization/ReadVariableOpб,dropout/batch_normalization/ReadVariableOp_1б=dropout/batch_normalization_1/FusedBatchNormV3/ReadVariableOpб?dropout/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1б,dropout/batch_normalization_1/ReadVariableOpб.dropout/batch_normalization_1/ReadVariableOp_1б=dropout/batch_normalization_2/FusedBatchNormV3/ReadVariableOpб?dropout/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1б,dropout/batch_normalization_2/ReadVariableOpб.dropout/batch_normalization_2/ReadVariableOp_1б=dropout/batch_normalization_3/FusedBatchNormV3/ReadVariableOpб?dropout/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1б,dropout/batch_normalization_3/ReadVariableOpб.dropout/batch_normalization_3/ReadVariableOp_1б=dropout/batch_normalization_4/FusedBatchNormV3/ReadVariableOpб?dropout/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б,dropout/batch_normalization_4/ReadVariableOpб.dropout/batch_normalization_4/ReadVariableOp_1б=dropout/batch_normalization_5/FusedBatchNormV3/ReadVariableOpб?dropout/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б,dropout/batch_normalization_5/ReadVariableOpб.dropout/batch_normalization_5/ReadVariableOp_1б=dropout/batch_normalization_6/FusedBatchNormV3/ReadVariableOpб?dropout/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1б,dropout/batch_normalization_6/ReadVariableOpб.dropout/batch_normalization_6/ReadVariableOp_1б=dropout/batch_normalization_7/FusedBatchNormV3/ReadVariableOpб?dropout/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1б,dropout/batch_normalization_7/ReadVariableOpб.dropout/batch_normalization_7/ReadVariableOp_1б=dropout/batch_normalization_8/FusedBatchNormV3/ReadVariableOpб?dropout/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1б,dropout/batch_normalization_8/ReadVariableOpб.dropout/batch_normalization_8/ReadVariableOp_1б%dropout/conv2d/BiasAdd/ReadVariableOpб$dropout/conv2d/Conv2D/ReadVariableOpб'dropout/conv2d_1/BiasAdd/ReadVariableOpб&dropout/conv2d_1/Conv2D/ReadVariableOpб'dropout/conv2d_2/BiasAdd/ReadVariableOpб&dropout/conv2d_2/Conv2D/ReadVariableOpб'dropout/conv2d_3/BiasAdd/ReadVariableOpб&dropout/conv2d_3/Conv2D/ReadVariableOpб/dropout/conv2d_transpose/BiasAdd/ReadVariableOpб8dropout/conv2d_transpose/conv2d_transpose/ReadVariableOpб1dropout/conv2d_transpose_1/BiasAdd/ReadVariableOpб:dropout/conv2d_transpose_1/conv2d_transpose/ReadVariableOpб1dropout/conv2d_transpose_2/BiasAdd/ReadVariableOpб:dropout/conv2d_transpose_2/conv2d_transpose/ReadVariableOpб1dropout/conv2d_transpose_3/BiasAdd/ReadVariableOpб:dropout/conv2d_transpose_3/conv2d_transpose/ReadVariableOpб1dropout/conv2d_transpose_4/BiasAdd/ReadVariableOpб:dropout/conv2d_transpose_4/conv2d_transpose/ReadVariableOpc
dropout/reshape/ShapeShapeu_vel*
T0*
_output_shapes
:2
dropout/reshape/Shapeћ
#dropout/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#dropout/reshape/strided_slice/stackў
%dropout/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%dropout/reshape/strided_slice/stack_1ў
%dropout/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%dropout/reshape/strided_slice/stack_2┬
dropout/reshape/strided_sliceStridedSlicedropout/reshape/Shape:output:0,dropout/reshape/strided_slice/stack:output:0.dropout/reshape/strided_slice/stack_1:output:0.dropout/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dropout/reshape/strided_sliceЁ
dropout/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :ђ2!
dropout/reshape/Reshape/shape/1Ё
dropout/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ђ2!
dropout/reshape/Reshape/shape/2ё
dropout/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2!
dropout/reshape/Reshape/shape/3џ
dropout/reshape/Reshape/shapePack&dropout/reshape/strided_slice:output:0(dropout/reshape/Reshape/shape/1:output:0(dropout/reshape/Reshape/shape/2:output:0(dropout/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
dropout/reshape/Reshape/shapeе
dropout/reshape/ReshapeReshapeu_vel&dropout/reshape/Reshape/shape:output:0*
T0*1
_output_shapes
:         ђђ2
dropout/reshape/Reshape╚
*dropout/batch_normalization/ReadVariableOpReadVariableOp3dropout_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02,
*dropout/batch_normalization/ReadVariableOp╬
,dropout/batch_normalization/ReadVariableOp_1ReadVariableOp5dropout_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,dropout/batch_normalization/ReadVariableOp_1ч
;dropout/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpDdropout_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02=
;dropout/batch_normalization/FusedBatchNormV3/ReadVariableOpЂ
=dropout/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFdropout_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02?
=dropout/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ј
,dropout/batch_normalization/FusedBatchNormV3FusedBatchNormV3 dropout/reshape/Reshape:output:02dropout/batch_normalization/ReadVariableOp:value:04dropout/batch_normalization/ReadVariableOp_1:value:0Cdropout/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Edropout/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ:::::*
epsilon%oЃ:*
is_training( 2.
,dropout/batch_normalization/FusedBatchNormV3«
dropout/dropout/IdentityIdentity0dropout/batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:         ђђ2
dropout/dropout/Identity┬
$dropout/conv2d/Conv2D/ReadVariableOpReadVariableOp-dropout_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02&
$dropout/conv2d/Conv2D/ReadVariableOpЬ
dropout/conv2d/Conv2DConv2D!dropout/dropout/Identity:output:0,dropout/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЧЧ@*
paddingVALID*
strides
2
dropout/conv2d/Conv2D╣
%dropout/conv2d/BiasAdd/ReadVariableOpReadVariableOp.dropout_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%dropout/conv2d/BiasAdd/ReadVariableOpк
dropout/conv2d/BiasAddBiasAdddropout/conv2d/Conv2D:output:0-dropout/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЧЧ@2
dropout/conv2d/BiasAddї
dropout/conv2d/EluEludropout/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:         ЧЧ@2
dropout/conv2d/Elu╬
,dropout/batch_normalization_1/ReadVariableOpReadVariableOp5dropout_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype02.
,dropout/batch_normalization_1/ReadVariableOpн
.dropout/batch_normalization_1/ReadVariableOp_1ReadVariableOp7dropout_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype020
.dropout/batch_normalization_1/ReadVariableOp_1Ђ
=dropout/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpFdropout_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02?
=dropout/batch_normalization_1/FusedBatchNormV3/ReadVariableOpЄ
?dropout/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHdropout_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02A
?dropout/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1џ
.dropout/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3 dropout/conv2d/Elu:activations:04dropout/batch_normalization_1/ReadVariableOp:value:06dropout/batch_normalization_1/ReadVariableOp_1:value:0Edropout/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Gdropout/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЧЧ@:@:@:@:@:*
epsilon%oЃ:*
is_training( 20
.dropout/batch_normalization_1/FusedBatchNormV3┤
dropout/dropout_1/IdentityIdentity2dropout/batch_normalization_1/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:         ЧЧ@2
dropout/dropout_1/Identity╔
&dropout/conv2d_1/Conv2D/ReadVariableOpReadVariableOp/dropout_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02(
&dropout/conv2d_1/Conv2D/ReadVariableOpэ
dropout/conv2d_1/Conv2DConv2D#dropout/dropout_1/Identity:output:0.dropout/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЩЩђ*
paddingVALID*
strides
2
dropout/conv2d_1/Conv2D└
'dropout/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp0dropout_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02)
'dropout/conv2d_1/BiasAdd/ReadVariableOp¤
dropout/conv2d_1/BiasAddBiasAdd dropout/conv2d_1/Conv2D:output:0/dropout/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЩЩђ2
dropout/conv2d_1/BiasAddЊ
dropout/conv2d_1/EluElu!dropout/conv2d_1/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЩЩђ2
dropout/conv2d_1/Elu¤
,dropout/batch_normalization_2/ReadVariableOpReadVariableOp5dropout_batch_normalization_2_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,dropout/batch_normalization_2/ReadVariableOpН
.dropout/batch_normalization_2/ReadVariableOp_1ReadVariableOp7dropout_batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype020
.dropout/batch_normalization_2/ReadVariableOp_1ѓ
=dropout/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpFdropout_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02?
=dropout/batch_normalization_2/FusedBatchNormV3/ReadVariableOpѕ
?dropout/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHdropout_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02A
?dropout/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1А
.dropout/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3"dropout/conv2d_1/Elu:activations:04dropout/batch_normalization_2/ReadVariableOp:value:06dropout/batch_normalization_2/ReadVariableOp_1:value:0Edropout/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Gdropout/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЩЩђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 20
.dropout/batch_normalization_2/FusedBatchNormV3х
dropout/dropout_2/IdentityIdentity2dropout/batch_normalization_2/FusedBatchNormV3:y:0*
T0*2
_output_shapes 
:         ЩЩђ2
dropout/dropout_2/Identity╩
&dropout/conv2d_2/Conv2D/ReadVariableOpReadVariableOp/dropout_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02(
&dropout/conv2d_2/Conv2D/ReadVariableOpэ
dropout/conv2d_2/Conv2DConv2D#dropout/dropout_2/Identity:output:0.dropout/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЭЭђ*
paddingVALID*
strides
2
dropout/conv2d_2/Conv2D└
'dropout/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp0dropout_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02)
'dropout/conv2d_2/BiasAdd/ReadVariableOp¤
dropout/conv2d_2/BiasAddBiasAdd dropout/conv2d_2/Conv2D:output:0/dropout/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЭЭђ2
dropout/conv2d_2/BiasAddЊ
dropout/conv2d_2/EluElu!dropout/conv2d_2/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЭЭђ2
dropout/conv2d_2/Elu¤
,dropout/batch_normalization_3/ReadVariableOpReadVariableOp5dropout_batch_normalization_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,dropout/batch_normalization_3/ReadVariableOpН
.dropout/batch_normalization_3/ReadVariableOp_1ReadVariableOp7dropout_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype020
.dropout/batch_normalization_3/ReadVariableOp_1ѓ
=dropout/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpFdropout_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02?
=dropout/batch_normalization_3/FusedBatchNormV3/ReadVariableOpѕ
?dropout/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHdropout_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02A
?dropout/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1А
.dropout/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3"dropout/conv2d_2/Elu:activations:04dropout/batch_normalization_3/ReadVariableOp:value:06dropout/batch_normalization_3/ReadVariableOp_1:value:0Edropout/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Gdropout/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЭЭђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 20
.dropout/batch_normalization_3/FusedBatchNormV3х
dropout/dropout_3/IdentityIdentity2dropout/batch_normalization_3/FusedBatchNormV3:y:0*
T0*2
_output_shapes 
:         ЭЭђ2
dropout/dropout_3/Identity╩
&dropout/conv2d_3/Conv2D/ReadVariableOpReadVariableOp/dropout_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02(
&dropout/conv2d_3/Conv2D/ReadVariableOpэ
dropout/conv2d_3/Conv2DConv2D#dropout/dropout_3/Identity:output:0.dropout/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ШШђ*
paddingVALID*
strides
2
dropout/conv2d_3/Conv2D└
'dropout/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp0dropout_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02)
'dropout/conv2d_3/BiasAdd/ReadVariableOp¤
dropout/conv2d_3/BiasAddBiasAdd dropout/conv2d_3/Conv2D:output:0/dropout/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ШШђ2
dropout/conv2d_3/BiasAddЊ
dropout/conv2d_3/EluElu!dropout/conv2d_3/BiasAdd:output:0*
T0*2
_output_shapes 
:         ШШђ2
dropout/conv2d_3/Elu¤
,dropout/batch_normalization_4/ReadVariableOpReadVariableOp5dropout_batch_normalization_4_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,dropout/batch_normalization_4/ReadVariableOpН
.dropout/batch_normalization_4/ReadVariableOp_1ReadVariableOp7dropout_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype020
.dropout/batch_normalization_4/ReadVariableOp_1ѓ
=dropout/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpFdropout_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02?
=dropout/batch_normalization_4/FusedBatchNormV3/ReadVariableOpѕ
?dropout/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHdropout_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02A
?dropout/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1А
.dropout/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3"dropout/conv2d_3/Elu:activations:04dropout/batch_normalization_4/ReadVariableOp:value:06dropout/batch_normalization_4/ReadVariableOp_1:value:0Edropout/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Gdropout/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ШШђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 20
.dropout/batch_normalization_4/FusedBatchNormV3б
dropout/conv2d_transpose/ShapeShape2dropout/batch_normalization_4/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2 
dropout/conv2d_transpose/Shapeд
,dropout/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,dropout/conv2d_transpose/strided_slice/stackф
.dropout/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.dropout/conv2d_transpose/strided_slice/stack_1ф
.dropout/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.dropout/conv2d_transpose/strided_slice/stack_2Э
&dropout/conv2d_transpose/strided_sliceStridedSlice'dropout/conv2d_transpose/Shape:output:05dropout/conv2d_transpose/strided_slice/stack:output:07dropout/conv2d_transpose/strided_slice/stack_1:output:07dropout/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&dropout/conv2d_transpose/strided_sliceЄ
 dropout/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value
B :Э2"
 dropout/conv2d_transpose/stack/1Є
 dropout/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :Э2"
 dropout/conv2d_transpose/stack/2Є
 dropout/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2"
 dropout/conv2d_transpose/stack/3е
dropout/conv2d_transpose/stackPack/dropout/conv2d_transpose/strided_slice:output:0)dropout/conv2d_transpose/stack/1:output:0)dropout/conv2d_transpose/stack/2:output:0)dropout/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2 
dropout/conv2d_transpose/stackф
.dropout/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.dropout/conv2d_transpose/strided_slice_1/stack«
0dropout/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0dropout/conv2d_transpose/strided_slice_1/stack_1«
0dropout/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0dropout/conv2d_transpose/strided_slice_1/stack_2ѓ
(dropout/conv2d_transpose/strided_slice_1StridedSlice'dropout/conv2d_transpose/stack:output:07dropout/conv2d_transpose/strided_slice_1/stack:output:09dropout/conv2d_transpose/strided_slice_1/stack_1:output:09dropout/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(dropout/conv2d_transpose/strided_slice_1ђ
8dropout/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpAdropout_conv2d_transpose_conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02:
8dropout/conv2d_transpose/conv2d_transpose/ReadVariableOpЫ
)dropout/conv2d_transpose/conv2d_transposeConv2DBackpropInput'dropout/conv2d_transpose/stack:output:0@dropout/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:02dropout/batch_normalization_4/FusedBatchNormV3:y:0*
T0*2
_output_shapes 
:         ЭЭђ*
paddingVALID*
strides
2+
)dropout/conv2d_transpose/conv2d_transposeп
/dropout/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp8dropout_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype021
/dropout/conv2d_transpose/BiasAdd/ReadVariableOpщ
 dropout/conv2d_transpose/BiasAddBiasAdd2dropout/conv2d_transpose/conv2d_transpose:output:07dropout/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЭЭђ2"
 dropout/conv2d_transpose/BiasAddФ
dropout/conv2d_transpose/EluElu)dropout/conv2d_transpose/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЭЭђ2
dropout/conv2d_transpose/Elu¤
,dropout/batch_normalization_5/ReadVariableOpReadVariableOp5dropout_batch_normalization_5_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,dropout/batch_normalization_5/ReadVariableOpН
.dropout/batch_normalization_5/ReadVariableOp_1ReadVariableOp7dropout_batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype020
.dropout/batch_normalization_5/ReadVariableOp_1ѓ
=dropout/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpFdropout_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02?
=dropout/batch_normalization_5/FusedBatchNormV3/ReadVariableOpѕ
?dropout/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHdropout_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02A
?dropout/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Е
.dropout/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3*dropout/conv2d_transpose/Elu:activations:04dropout/batch_normalization_5/ReadVariableOp:value:06dropout/batch_normalization_5/ReadVariableOp_1:value:0Edropout/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Gdropout/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЭЭђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 20
.dropout/batch_normalization_5/FusedBatchNormV3д
 dropout/conv2d_transpose_1/ShapeShape2dropout/batch_normalization_5/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2"
 dropout/conv2d_transpose_1/Shapeф
.dropout/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.dropout/conv2d_transpose_1/strided_slice/stack«
0dropout/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0dropout/conv2d_transpose_1/strided_slice/stack_1«
0dropout/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0dropout/conv2d_transpose_1/strided_slice/stack_2ё
(dropout/conv2d_transpose_1/strided_sliceStridedSlice)dropout/conv2d_transpose_1/Shape:output:07dropout/conv2d_transpose_1/strided_slice/stack:output:09dropout/conv2d_transpose_1/strided_slice/stack_1:output:09dropout/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(dropout/conv2d_transpose_1/strided_sliceІ
"dropout/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value
B :Щ2$
"dropout/conv2d_transpose_1/stack/1І
"dropout/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :Щ2$
"dropout/conv2d_transpose_1/stack/2І
"dropout/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2$
"dropout/conv2d_transpose_1/stack/3┤
 dropout/conv2d_transpose_1/stackPack1dropout/conv2d_transpose_1/strided_slice:output:0+dropout/conv2d_transpose_1/stack/1:output:0+dropout/conv2d_transpose_1/stack/2:output:0+dropout/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 dropout/conv2d_transpose_1/stack«
0dropout/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dropout/conv2d_transpose_1/strided_slice_1/stack▓
2dropout/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dropout/conv2d_transpose_1/strided_slice_1/stack_1▓
2dropout/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dropout/conv2d_transpose_1/strided_slice_1/stack_2ј
*dropout/conv2d_transpose_1/strided_slice_1StridedSlice)dropout/conv2d_transpose_1/stack:output:09dropout/conv2d_transpose_1/strided_slice_1/stack:output:0;dropout/conv2d_transpose_1/strided_slice_1/stack_1:output:0;dropout/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dropout/conv2d_transpose_1/strided_slice_1є
:dropout/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpCdropout_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02<
:dropout/conv2d_transpose_1/conv2d_transpose/ReadVariableOpЩ
+dropout/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput)dropout/conv2d_transpose_1/stack:output:0Bdropout/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:02dropout/batch_normalization_5/FusedBatchNormV3:y:0*
T0*2
_output_shapes 
:         ЩЩђ*
paddingVALID*
strides
2-
+dropout/conv2d_transpose_1/conv2d_transposeя
1dropout/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp:dropout_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype023
1dropout/conv2d_transpose_1/BiasAdd/ReadVariableOpЂ
"dropout/conv2d_transpose_1/BiasAddBiasAdd4dropout/conv2d_transpose_1/conv2d_transpose:output:09dropout/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЩЩђ2$
"dropout/conv2d_transpose_1/BiasAdd▒
dropout/conv2d_transpose_1/EluElu+dropout/conv2d_transpose_1/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЩЩђ2 
dropout/conv2d_transpose_1/Elu¤
,dropout/batch_normalization_6/ReadVariableOpReadVariableOp5dropout_batch_normalization_6_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,dropout/batch_normalization_6/ReadVariableOpН
.dropout/batch_normalization_6/ReadVariableOp_1ReadVariableOp7dropout_batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype020
.dropout/batch_normalization_6/ReadVariableOp_1ѓ
=dropout/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpFdropout_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02?
=dropout/batch_normalization_6/FusedBatchNormV3/ReadVariableOpѕ
?dropout/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHdropout_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02A
?dropout/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Ф
.dropout/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3,dropout/conv2d_transpose_1/Elu:activations:04dropout/batch_normalization_6/ReadVariableOp:value:06dropout/batch_normalization_6/ReadVariableOp_1:value:0Edropout/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Gdropout/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЩЩђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 20
.dropout/batch_normalization_6/FusedBatchNormV3д
 dropout/conv2d_transpose_2/ShapeShape2dropout/batch_normalization_6/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2"
 dropout/conv2d_transpose_2/Shapeф
.dropout/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.dropout/conv2d_transpose_2/strided_slice/stack«
0dropout/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0dropout/conv2d_transpose_2/strided_slice/stack_1«
0dropout/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0dropout/conv2d_transpose_2/strided_slice/stack_2ё
(dropout/conv2d_transpose_2/strided_sliceStridedSlice)dropout/conv2d_transpose_2/Shape:output:07dropout/conv2d_transpose_2/strided_slice/stack:output:09dropout/conv2d_transpose_2/strided_slice/stack_1:output:09dropout/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(dropout/conv2d_transpose_2/strided_sliceІ
"dropout/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value
B :Ч2$
"dropout/conv2d_transpose_2/stack/1І
"dropout/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value
B :Ч2$
"dropout/conv2d_transpose_2/stack/2І
"dropout/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2$
"dropout/conv2d_transpose_2/stack/3┤
 dropout/conv2d_transpose_2/stackPack1dropout/conv2d_transpose_2/strided_slice:output:0+dropout/conv2d_transpose_2/stack/1:output:0+dropout/conv2d_transpose_2/stack/2:output:0+dropout/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 dropout/conv2d_transpose_2/stack«
0dropout/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dropout/conv2d_transpose_2/strided_slice_1/stack▓
2dropout/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dropout/conv2d_transpose_2/strided_slice_1/stack_1▓
2dropout/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dropout/conv2d_transpose_2/strided_slice_1/stack_2ј
*dropout/conv2d_transpose_2/strided_slice_1StridedSlice)dropout/conv2d_transpose_2/stack:output:09dropout/conv2d_transpose_2/strided_slice_1/stack:output:0;dropout/conv2d_transpose_2/strided_slice_1/stack_1:output:0;dropout/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dropout/conv2d_transpose_2/strided_slice_1є
:dropout/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpCdropout_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02<
:dropout/conv2d_transpose_2/conv2d_transpose/ReadVariableOpЩ
+dropout/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput)dropout/conv2d_transpose_2/stack:output:0Bdropout/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:02dropout/batch_normalization_6/FusedBatchNormV3:y:0*
T0*2
_output_shapes 
:         ЧЧђ*
paddingVALID*
strides
2-
+dropout/conv2d_transpose_2/conv2d_transposeя
1dropout/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp:dropout_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype023
1dropout/conv2d_transpose_2/BiasAdd/ReadVariableOpЂ
"dropout/conv2d_transpose_2/BiasAddBiasAdd4dropout/conv2d_transpose_2/conv2d_transpose:output:09dropout/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЧЧђ2$
"dropout/conv2d_transpose_2/BiasAdd▒
dropout/conv2d_transpose_2/EluElu+dropout/conv2d_transpose_2/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЧЧђ2 
dropout/conv2d_transpose_2/Elu¤
,dropout/batch_normalization_7/ReadVariableOpReadVariableOp5dropout_batch_normalization_7_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,dropout/batch_normalization_7/ReadVariableOpН
.dropout/batch_normalization_7/ReadVariableOp_1ReadVariableOp7dropout_batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype020
.dropout/batch_normalization_7/ReadVariableOp_1ѓ
=dropout/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpFdropout_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02?
=dropout/batch_normalization_7/FusedBatchNormV3/ReadVariableOpѕ
?dropout/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHdropout_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02A
?dropout/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Ф
.dropout/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3,dropout/conv2d_transpose_2/Elu:activations:04dropout/batch_normalization_7/ReadVariableOp:value:06dropout/batch_normalization_7/ReadVariableOp_1:value:0Edropout/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Gdropout/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЧЧђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 20
.dropout/batch_normalization_7/FusedBatchNormV3д
 dropout/conv2d_transpose_3/ShapeShape2dropout/batch_normalization_7/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2"
 dropout/conv2d_transpose_3/Shapeф
.dropout/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.dropout/conv2d_transpose_3/strided_slice/stack«
0dropout/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0dropout/conv2d_transpose_3/strided_slice/stack_1«
0dropout/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0dropout/conv2d_transpose_3/strided_slice/stack_2ё
(dropout/conv2d_transpose_3/strided_sliceStridedSlice)dropout/conv2d_transpose_3/Shape:output:07dropout/conv2d_transpose_3/strided_slice/stack:output:09dropout/conv2d_transpose_3/strided_slice/stack_1:output:09dropout/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(dropout/conv2d_transpose_3/strided_sliceІ
"dropout/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value
B :ђ2$
"dropout/conv2d_transpose_3/stack/1І
"dropout/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value
B :ђ2$
"dropout/conv2d_transpose_3/stack/2і
"dropout/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2$
"dropout/conv2d_transpose_3/stack/3┤
 dropout/conv2d_transpose_3/stackPack1dropout/conv2d_transpose_3/strided_slice:output:0+dropout/conv2d_transpose_3/stack/1:output:0+dropout/conv2d_transpose_3/stack/2:output:0+dropout/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 dropout/conv2d_transpose_3/stack«
0dropout/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dropout/conv2d_transpose_3/strided_slice_1/stack▓
2dropout/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dropout/conv2d_transpose_3/strided_slice_1/stack_1▓
2dropout/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dropout/conv2d_transpose_3/strided_slice_1/stack_2ј
*dropout/conv2d_transpose_3/strided_slice_1StridedSlice)dropout/conv2d_transpose_3/stack:output:09dropout/conv2d_transpose_3/strided_slice_1/stack:output:0;dropout/conv2d_transpose_3/strided_slice_1/stack_1:output:0;dropout/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dropout/conv2d_transpose_3/strided_slice_1Ё
:dropout/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpCdropout_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02<
:dropout/conv2d_transpose_3/conv2d_transpose/ReadVariableOpщ
+dropout/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput)dropout/conv2d_transpose_3/stack:output:0Bdropout/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:02dropout/batch_normalization_7/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:         ђђ@*
paddingVALID*
strides
2-
+dropout/conv2d_transpose_3/conv2d_transposeП
1dropout/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp:dropout_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1dropout/conv2d_transpose_3/BiasAdd/ReadVariableOpђ
"dropout/conv2d_transpose_3/BiasAddBiasAdd4dropout/conv2d_transpose_3/conv2d_transpose:output:09dropout/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ@2$
"dropout/conv2d_transpose_3/BiasAdd░
dropout/conv2d_transpose_3/EluElu+dropout/conv2d_transpose_3/BiasAdd:output:0*
T0*1
_output_shapes
:         ђђ@2 
dropout/conv2d_transpose_3/Elu╬
,dropout/batch_normalization_8/ReadVariableOpReadVariableOp5dropout_batch_normalization_8_readvariableop_resource*
_output_shapes
:@*
dtype02.
,dropout/batch_normalization_8/ReadVariableOpн
.dropout/batch_normalization_8/ReadVariableOp_1ReadVariableOp7dropout_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:@*
dtype020
.dropout/batch_normalization_8/ReadVariableOp_1Ђ
=dropout/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpFdropout_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02?
=dropout/batch_normalization_8/FusedBatchNormV3/ReadVariableOpЄ
?dropout/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHdropout_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02A
?dropout/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1д
.dropout/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3,dropout/conv2d_transpose_3/Elu:activations:04dropout/batch_normalization_8/ReadVariableOp:value:06dropout/batch_normalization_8/ReadVariableOp_1:value:0Edropout/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Gdropout/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ@:@:@:@:@:*
epsilon%oЃ:*
is_training( 20
.dropout/batch_normalization_8/FusedBatchNormV3д
 dropout/conv2d_transpose_4/ShapeShape2dropout/batch_normalization_8/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2"
 dropout/conv2d_transpose_4/Shapeф
.dropout/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.dropout/conv2d_transpose_4/strided_slice/stack«
0dropout/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0dropout/conv2d_transpose_4/strided_slice/stack_1«
0dropout/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0dropout/conv2d_transpose_4/strided_slice/stack_2ё
(dropout/conv2d_transpose_4/strided_sliceStridedSlice)dropout/conv2d_transpose_4/Shape:output:07dropout/conv2d_transpose_4/strided_slice/stack:output:09dropout/conv2d_transpose_4/strided_slice/stack_1:output:09dropout/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(dropout/conv2d_transpose_4/strided_sliceІ
"dropout/conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value
B :ђ2$
"dropout/conv2d_transpose_4/stack/1І
"dropout/conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value
B :ђ2$
"dropout/conv2d_transpose_4/stack/2і
"dropout/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"dropout/conv2d_transpose_4/stack/3┤
 dropout/conv2d_transpose_4/stackPack1dropout/conv2d_transpose_4/strided_slice:output:0+dropout/conv2d_transpose_4/stack/1:output:0+dropout/conv2d_transpose_4/stack/2:output:0+dropout/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 dropout/conv2d_transpose_4/stack«
0dropout/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dropout/conv2d_transpose_4/strided_slice_1/stack▓
2dropout/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dropout/conv2d_transpose_4/strided_slice_1/stack_1▓
2dropout/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dropout/conv2d_transpose_4/strided_slice_1/stack_2ј
*dropout/conv2d_transpose_4/strided_slice_1StridedSlice)dropout/conv2d_transpose_4/stack:output:09dropout/conv2d_transpose_4/strided_slice_1/stack:output:0;dropout/conv2d_transpose_4/strided_slice_1/stack_1:output:0;dropout/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dropout/conv2d_transpose_4/strided_slice_1ё
:dropout/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpCdropout_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype02<
:dropout/conv2d_transpose_4/conv2d_transpose/ReadVariableOpщ
+dropout/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput)dropout/conv2d_transpose_4/stack:output:0Bdropout/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:02dropout/batch_normalization_8/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:         ђђ*
paddingVALID*
strides
2-
+dropout/conv2d_transpose_4/conv2d_transposeП
1dropout/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp:dropout_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1dropout/conv2d_transpose_4/BiasAdd/ReadVariableOpђ
"dropout/conv2d_transpose_4/BiasAddBiasAdd4dropout/conv2d_transpose_4/conv2d_transpose:output:09dropout/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ2$
"dropout/conv2d_transpose_4/BiasAddљ
IdentityIdentity+dropout/conv2d_transpose_4/BiasAdd:output:0^NoOp*
T0*1
_output_shapes
:         ђђ2

Identity▒
NoOpNoOp<^dropout/batch_normalization/FusedBatchNormV3/ReadVariableOp>^dropout/batch_normalization/FusedBatchNormV3/ReadVariableOp_1+^dropout/batch_normalization/ReadVariableOp-^dropout/batch_normalization/ReadVariableOp_1>^dropout/batch_normalization_1/FusedBatchNormV3/ReadVariableOp@^dropout/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1-^dropout/batch_normalization_1/ReadVariableOp/^dropout/batch_normalization_1/ReadVariableOp_1>^dropout/batch_normalization_2/FusedBatchNormV3/ReadVariableOp@^dropout/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1-^dropout/batch_normalization_2/ReadVariableOp/^dropout/batch_normalization_2/ReadVariableOp_1>^dropout/batch_normalization_3/FusedBatchNormV3/ReadVariableOp@^dropout/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1-^dropout/batch_normalization_3/ReadVariableOp/^dropout/batch_normalization_3/ReadVariableOp_1>^dropout/batch_normalization_4/FusedBatchNormV3/ReadVariableOp@^dropout/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1-^dropout/batch_normalization_4/ReadVariableOp/^dropout/batch_normalization_4/ReadVariableOp_1>^dropout/batch_normalization_5/FusedBatchNormV3/ReadVariableOp@^dropout/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1-^dropout/batch_normalization_5/ReadVariableOp/^dropout/batch_normalization_5/ReadVariableOp_1>^dropout/batch_normalization_6/FusedBatchNormV3/ReadVariableOp@^dropout/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1-^dropout/batch_normalization_6/ReadVariableOp/^dropout/batch_normalization_6/ReadVariableOp_1>^dropout/batch_normalization_7/FusedBatchNormV3/ReadVariableOp@^dropout/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1-^dropout/batch_normalization_7/ReadVariableOp/^dropout/batch_normalization_7/ReadVariableOp_1>^dropout/batch_normalization_8/FusedBatchNormV3/ReadVariableOp@^dropout/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1-^dropout/batch_normalization_8/ReadVariableOp/^dropout/batch_normalization_8/ReadVariableOp_1&^dropout/conv2d/BiasAdd/ReadVariableOp%^dropout/conv2d/Conv2D/ReadVariableOp(^dropout/conv2d_1/BiasAdd/ReadVariableOp'^dropout/conv2d_1/Conv2D/ReadVariableOp(^dropout/conv2d_2/BiasAdd/ReadVariableOp'^dropout/conv2d_2/Conv2D/ReadVariableOp(^dropout/conv2d_3/BiasAdd/ReadVariableOp'^dropout/conv2d_3/Conv2D/ReadVariableOp0^dropout/conv2d_transpose/BiasAdd/ReadVariableOp9^dropout/conv2d_transpose/conv2d_transpose/ReadVariableOp2^dropout/conv2d_transpose_1/BiasAdd/ReadVariableOp;^dropout/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2^dropout/conv2d_transpose_2/BiasAdd/ReadVariableOp;^dropout/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2^dropout/conv2d_transpose_3/BiasAdd/ReadVariableOp;^dropout/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2^dropout/conv2d_transpose_4/BiasAdd/ReadVariableOp;^dropout/conv2d_transpose_4/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*џ
_input_shapesѕ
Ё:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2z
;dropout/batch_normalization/FusedBatchNormV3/ReadVariableOp;dropout/batch_normalization/FusedBatchNormV3/ReadVariableOp2~
=dropout/batch_normalization/FusedBatchNormV3/ReadVariableOp_1=dropout/batch_normalization/FusedBatchNormV3/ReadVariableOp_12X
*dropout/batch_normalization/ReadVariableOp*dropout/batch_normalization/ReadVariableOp2\
,dropout/batch_normalization/ReadVariableOp_1,dropout/batch_normalization/ReadVariableOp_12~
=dropout/batch_normalization_1/FusedBatchNormV3/ReadVariableOp=dropout/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2ѓ
?dropout/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?dropout/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12\
,dropout/batch_normalization_1/ReadVariableOp,dropout/batch_normalization_1/ReadVariableOp2`
.dropout/batch_normalization_1/ReadVariableOp_1.dropout/batch_normalization_1/ReadVariableOp_12~
=dropout/batch_normalization_2/FusedBatchNormV3/ReadVariableOp=dropout/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2ѓ
?dropout/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?dropout/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12\
,dropout/batch_normalization_2/ReadVariableOp,dropout/batch_normalization_2/ReadVariableOp2`
.dropout/batch_normalization_2/ReadVariableOp_1.dropout/batch_normalization_2/ReadVariableOp_12~
=dropout/batch_normalization_3/FusedBatchNormV3/ReadVariableOp=dropout/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2ѓ
?dropout/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?dropout/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12\
,dropout/batch_normalization_3/ReadVariableOp,dropout/batch_normalization_3/ReadVariableOp2`
.dropout/batch_normalization_3/ReadVariableOp_1.dropout/batch_normalization_3/ReadVariableOp_12~
=dropout/batch_normalization_4/FusedBatchNormV3/ReadVariableOp=dropout/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2ѓ
?dropout/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?dropout/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12\
,dropout/batch_normalization_4/ReadVariableOp,dropout/batch_normalization_4/ReadVariableOp2`
.dropout/batch_normalization_4/ReadVariableOp_1.dropout/batch_normalization_4/ReadVariableOp_12~
=dropout/batch_normalization_5/FusedBatchNormV3/ReadVariableOp=dropout/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2ѓ
?dropout/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?dropout/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12\
,dropout/batch_normalization_5/ReadVariableOp,dropout/batch_normalization_5/ReadVariableOp2`
.dropout/batch_normalization_5/ReadVariableOp_1.dropout/batch_normalization_5/ReadVariableOp_12~
=dropout/batch_normalization_6/FusedBatchNormV3/ReadVariableOp=dropout/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2ѓ
?dropout/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?dropout/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12\
,dropout/batch_normalization_6/ReadVariableOp,dropout/batch_normalization_6/ReadVariableOp2`
.dropout/batch_normalization_6/ReadVariableOp_1.dropout/batch_normalization_6/ReadVariableOp_12~
=dropout/batch_normalization_7/FusedBatchNormV3/ReadVariableOp=dropout/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2ѓ
?dropout/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?dropout/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12\
,dropout/batch_normalization_7/ReadVariableOp,dropout/batch_normalization_7/ReadVariableOp2`
.dropout/batch_normalization_7/ReadVariableOp_1.dropout/batch_normalization_7/ReadVariableOp_12~
=dropout/batch_normalization_8/FusedBatchNormV3/ReadVariableOp=dropout/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2ѓ
?dropout/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?dropout/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12\
,dropout/batch_normalization_8/ReadVariableOp,dropout/batch_normalization_8/ReadVariableOp2`
.dropout/batch_normalization_8/ReadVariableOp_1.dropout/batch_normalization_8/ReadVariableOp_12N
%dropout/conv2d/BiasAdd/ReadVariableOp%dropout/conv2d/BiasAdd/ReadVariableOp2L
$dropout/conv2d/Conv2D/ReadVariableOp$dropout/conv2d/Conv2D/ReadVariableOp2R
'dropout/conv2d_1/BiasAdd/ReadVariableOp'dropout/conv2d_1/BiasAdd/ReadVariableOp2P
&dropout/conv2d_1/Conv2D/ReadVariableOp&dropout/conv2d_1/Conv2D/ReadVariableOp2R
'dropout/conv2d_2/BiasAdd/ReadVariableOp'dropout/conv2d_2/BiasAdd/ReadVariableOp2P
&dropout/conv2d_2/Conv2D/ReadVariableOp&dropout/conv2d_2/Conv2D/ReadVariableOp2R
'dropout/conv2d_3/BiasAdd/ReadVariableOp'dropout/conv2d_3/BiasAdd/ReadVariableOp2P
&dropout/conv2d_3/Conv2D/ReadVariableOp&dropout/conv2d_3/Conv2D/ReadVariableOp2b
/dropout/conv2d_transpose/BiasAdd/ReadVariableOp/dropout/conv2d_transpose/BiasAdd/ReadVariableOp2t
8dropout/conv2d_transpose/conv2d_transpose/ReadVariableOp8dropout/conv2d_transpose/conv2d_transpose/ReadVariableOp2f
1dropout/conv2d_transpose_1/BiasAdd/ReadVariableOp1dropout/conv2d_transpose_1/BiasAdd/ReadVariableOp2x
:dropout/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:dropout/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2f
1dropout/conv2d_transpose_2/BiasAdd/ReadVariableOp1dropout/conv2d_transpose_2/BiasAdd/ReadVariableOp2x
:dropout/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:dropout/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2f
1dropout/conv2d_transpose_3/BiasAdd/ReadVariableOp1dropout/conv2d_transpose_3/BiasAdd/ReadVariableOp2x
:dropout/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:dropout/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2f
1dropout/conv2d_transpose_4/BiasAdd/ReadVariableOp1dropout/conv2d_transpose_4/BiasAdd/ReadVariableOp2x
:dropout/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:dropout/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:T P
-
_output_shapes
:         ђђ

_user_specified_nameu_vel
л
c
*__inference_dropout_3_layer_call_fn_237811

inputs
identityѕбStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2352002
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
_construction_contextkEagerRuntime*1
_input_shapes 
:         ЭЭђ22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
щ
е
3__inference_conv2d_transpose_4_layer_call_fn_238797

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
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_2342342
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
Ѕ	
Н
6__inference_batch_normalization_6_layer_call_fn_238308

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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2350532
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
╚
a
(__inference_dropout_layer_call_fn_237298

inputs
identityѕбStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputs*
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
C__inference_dropout_layer_call_and_return_conditional_losses_2354312
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
_construction_contextkEagerRuntime*0
_input_shapes
:         ђђ22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
с
ю
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_232823

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
І	
Н
6__inference_batch_normalization_6_layer_call_fn_238295

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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2346432
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
Ч
Е
1__inference_conv2d_transpose_layer_call_fn_237981

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
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_2333472
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
Џ
т
(__inference_dropout_layer_call_fn_234905	
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
identityѕбStatefulPartitionedCall╬
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
GPU2*0J 8ѓ *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_2347942
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
џ
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_237474

inputs

identity_1d
IdentityIdentityinputs*
T0*1
_output_shapes
:         ЧЧ@2

Identitys

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:         ЧЧ@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЧЧ@:Y U
1
_output_shapes
:         ЧЧ@
 
_user_specified_nameinputs
з
а
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_237918

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
b
C__inference_dropout_layer_call_and_return_conditional_losses_237315

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUН?2
dropout/Const}
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:         ђђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЙ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*1
_output_shapes
:         ђђ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2
dropout/GreaterEqual/y╚
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:         ђђ2
dropout/GreaterEqualЅ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:         ђђ2
dropout/Castё
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:         ђђ2
dropout/Mul_1o
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:         ђђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ђђ:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
б
ю
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_234378

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
╝
Е
3__inference_conv2d_transpose_3_layer_call_fn_238602

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
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_2347322
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
 
_user_specified_nameinputs
╝
Е
1__inference_conv2d_transpose_layer_call_fn_237990

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
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_2345642
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
▓
а
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_237783

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
─
Џ
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_238866

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
ђ	
Ф
3__inference_conv2d_transpose_1_layer_call_fn_238185

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
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_2335692
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
¤(
ю
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_238640

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
з
а
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_233645

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
Д
─
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_238344

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
ъ
Т
(__inference_dropout_layer_call_fn_236462

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
identityѕбStatefulPartitionedCall¤
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
GPU2*0J 8ѓ *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_2347942
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
о
└
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_235390

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
╔	
Н
6__inference_batch_normalization_2_layer_call_fn_237532

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
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2329932
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
└
Ф
3__inference_conv2d_transpose_1_layer_call_fn_238194

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
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_2346202
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
│
ъ
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_238256

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
з
а
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_238326

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
┴	
Л
6__inference_batch_normalization_1_layer_call_fn_237361

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
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2328672
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
 
ђ
D__inference_conv2d_3_layer_call_and_return_conditional_losses_237848

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
Ќ
└
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_234133

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
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_233467

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
│
ъ
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_234620

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
ЪЉ
╬
C__inference_dropout_layer_call_and_return_conditional_losses_236228	
u_vel(
batch_normalization_236097:(
batch_normalization_236099:(
batch_normalization_236101:(
batch_normalization_236103:'
conv2d_236107:@
conv2d_236109:@*
batch_normalization_1_236112:@*
batch_normalization_1_236114:@*
batch_normalization_1_236116:@*
batch_normalization_1_236118:@*
conv2d_1_236122:@ђ
conv2d_1_236124:	ђ+
batch_normalization_2_236127:	ђ+
batch_normalization_2_236129:	ђ+
batch_normalization_2_236131:	ђ+
batch_normalization_2_236133:	ђ+
conv2d_2_236137:ђђ
conv2d_2_236139:	ђ+
batch_normalization_3_236142:	ђ+
batch_normalization_3_236144:	ђ+
batch_normalization_3_236146:	ђ+
batch_normalization_3_236148:	ђ+
conv2d_3_236152:ђђ
conv2d_3_236154:	ђ+
batch_normalization_4_236157:	ђ+
batch_normalization_4_236159:	ђ+
batch_normalization_4_236161:	ђ+
batch_normalization_4_236163:	ђ3
conv2d_transpose_236166:ђђ&
conv2d_transpose_236168:	ђ+
batch_normalization_5_236171:	ђ+
batch_normalization_5_236173:	ђ+
batch_normalization_5_236175:	ђ+
batch_normalization_5_236177:	ђ5
conv2d_transpose_1_236180:ђђ(
conv2d_transpose_1_236182:	ђ+
batch_normalization_6_236185:	ђ+
batch_normalization_6_236187:	ђ+
batch_normalization_6_236189:	ђ+
batch_normalization_6_236191:	ђ5
conv2d_transpose_2_236194:ђђ(
conv2d_transpose_2_236196:	ђ+
batch_normalization_7_236199:	ђ+
batch_normalization_7_236201:	ђ+
batch_normalization_7_236203:	ђ+
batch_normalization_7_236205:	ђ4
conv2d_transpose_3_236208:@ђ'
conv2d_transpose_3_236210:@*
batch_normalization_8_236213:@*
batch_normalization_8_236215:@*
batch_normalization_8_236217:@*
batch_normalization_8_236219:@3
conv2d_transpose_4_236222:@'
conv2d_transpose_4_236224:
identityѕб+batch_normalization/StatefulPartitionedCallб-batch_normalization_1/StatefulPartitionedCallб-batch_normalization_2/StatefulPartitionedCallб-batch_normalization_3/StatefulPartitionedCallб-batch_normalization_4/StatefulPartitionedCallб-batch_normalization_5/StatefulPartitionedCallб-batch_normalization_6/StatefulPartitionedCallб-batch_normalization_7/StatefulPartitionedCallб-batch_normalization_8/StatefulPartitionedCallбconv2d/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallб conv2d_2/StatefulPartitionedCallб conv2d_3/StatefulPartitionedCallб(conv2d_transpose/StatefulPartitionedCallб*conv2d_transpose_1/StatefulPartitionedCallб*conv2d_transpose_2/StatefulPartitionedCallб*conv2d_transpose_3/StatefulPartitionedCallб*conv2d_transpose_4/StatefulPartitionedCallбdropout/StatefulPartitionedCallб!dropout_1/StatefulPartitionedCallб!dropout_2/StatefulPartitionedCallб!dropout_3/StatefulPartitionedCallП
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
C__inference_reshape_layer_call_and_return_conditional_losses_2343082
reshape/PartitionedCallг
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0batch_normalization_236097batch_normalization_236099batch_normalization_236101batch_normalization_236103*
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
O__inference_batch_normalization_layer_call_and_return_conditional_losses_2354672-
+batch_normalization/StatefulPartitionedCallц
dropout/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
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
C__inference_dropout_layer_call_and_return_conditional_losses_2354312!
dropout/StatefulPartitionedCall╣
conv2d/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv2d_236107conv2d_236109*
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
B__inference_conv2d_layer_call_and_return_conditional_losses_2343552 
conv2d/StatefulPartitionedCall┴
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_1_236112batch_normalization_1_236114batch_normalization_1_236116batch_normalization_1_236118*
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
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2353902/
-batch_normalization_1/StatefulPartitionedCall╬
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЧЧ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_2353542#
!dropout_1/StatefulPartitionedCallк
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv2d_1_236122conv2d_1_236124*
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
D__inference_conv2d_1_layer_call_and_return_conditional_losses_2344062"
 conv2d_1/StatefulPartitionedCall─
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_2_236127batch_normalization_2_236129batch_normalization_2_236131batch_normalization_2_236133*
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
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2353132/
-batch_normalization_2/StatefulPartitionedCallЛ
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЩЩђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2352772#
!dropout_2/StatefulPartitionedCallк
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0conv2d_2_236137conv2d_2_236139*
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
D__inference_conv2d_2_layer_call_and_return_conditional_losses_2344572"
 conv2d_2/StatefulPartitionedCall─
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_3_236142batch_normalization_3_236144batch_normalization_3_236146batch_normalization_3_236148*
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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2352362/
-batch_normalization_3/StatefulPartitionedCallЛ
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2352002#
!dropout_3/StatefulPartitionedCallк
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0conv2d_3_236152conv2d_3_236154*
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
D__inference_conv2d_3_layer_call_and_return_conditional_losses_2345082"
 conv2d_3/StatefulPartitionedCall─
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_4_236157batch_normalization_4_236159batch_normalization_4_236161batch_normalization_4_236163*
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
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2351592/
-batch_normalization_4/StatefulPartitionedCallЩ
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0conv2d_transpose_236166conv2d_transpose_236168*
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
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_2345642*
(conv2d_transpose/StatefulPartitionedCall╠
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0batch_normalization_5_236171batch_normalization_5_236173batch_normalization_5_236175batch_normalization_5_236177*
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
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2351062/
-batch_normalization_5/StatefulPartitionedCallё
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0conv2d_transpose_1_236180conv2d_transpose_1_236182*
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
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_2346202,
*conv2d_transpose_1/StatefulPartitionedCall╬
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0batch_normalization_6_236185batch_normalization_6_236187batch_normalization_6_236189batch_normalization_6_236191*
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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2350532/
-batch_normalization_6/StatefulPartitionedCallё
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0conv2d_transpose_2_236194conv2d_transpose_2_236196*
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
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_2346762,
*conv2d_transpose_2/StatefulPartitionedCall╬
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0batch_normalization_7_236199batch_normalization_7_236201batch_normalization_7_236203batch_normalization_7_236205*
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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2350002/
-batch_normalization_7/StatefulPartitionedCallЃ
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0conv2d_transpose_3_236208conv2d_transpose_3_236210*
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
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_2347322,
*conv2d_transpose_3/StatefulPartitionedCall═
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0batch_normalization_8_236213batch_normalization_8_236215batch_normalization_8_236217batch_normalization_8_236219*
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
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2349472/
-batch_normalization_8/StatefulPartitionedCallЃ
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_transpose_4_236222conv2d_transpose_4_236224*
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
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_2347872,
*conv2d_transpose_4/StatefulPartitionedCallў
IdentityIdentity3conv2d_transpose_4/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ2

Identityз
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*"
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
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:T P
-
_output_shapes
:         ђђ

_user_specified_nameu_vel
Ё
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_237657

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUН?2
dropout/Const~
dropout/MulMulinputsdropout/Const:output:0*
T0*2
_output_shapes 
:         ЩЩђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┐
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*2
_output_shapes 
:         ЩЩђ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2
dropout/GreaterEqual/y╔
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*2
_output_shapes 
:         ЩЩђ2
dropout/GreaterEqualі
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*2
_output_shapes 
:         ЩЩђ2
dropout/CastЁ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*2
_output_shapes 
:         ЩЩђ2
dropout/Mul_1p
IdentityIdentitydropout/Mul_1:z:0*
T0*2
_output_shapes 
:         ЩЩђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ЩЩђ:Z V
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs
Т
─
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_235000

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
ш
р
$__inference_signature_wrapper_236349	
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
!__inference__wrapped_model_2326752
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
о(
ю
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_233347

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
Т
─
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_238584

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
▓
а
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_237612

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
ї
Т
(__inference_dropout_layer_call_fn_236575

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
identityѕбStatefulPartitionedCallй
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
GPU2*0J 8ѓ *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_2357342
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
л
c
*__inference_dropout_2_layer_call_fn_237640

inputs
identityѕбStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЩЩђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2352772
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
_construction_contextkEagerRuntime*1
_input_shapes 
:         ЩЩђ22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs
Ч
Е
3__inference_conv2d_transpose_3_layer_call_fn_238593

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
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_2340132
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
Ќ
└
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_237423

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
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_233245

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
▓
а
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_234429

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
І	
Н
6__inference_batch_normalization_3_layer_call_fn_237716

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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2344802
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
├	
Л
6__inference_batch_normalization_1_layer_call_fn_237348

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
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2328232
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
)__inference_conv2d_2_layer_call_fn_237666

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
D__inference_conv2d_2_layer_call_and_return_conditional_losses_2344572
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
ф
ю
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_238664

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
╦	
Н
6__inference_batch_normalization_2_layer_call_fn_237519

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
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2329492
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
о
└
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_238788

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
╦	
Н
6__inference_batch_normalization_3_layer_call_fn_237690

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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2330752
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
▓
а
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_234699

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
▓
а
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_234480

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
Ћ
Й
O__inference_batch_normalization_layer_call_and_return_conditional_losses_237252

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
ч
 
D__inference_conv2d_1_layer_call_and_return_conditional_losses_234406

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
Т
─
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_235106

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
╠
c
*__inference_dropout_1_layer_call_fn_237469

inputs
identityѕбStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЧЧ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_2353542
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
_construction_contextkEagerRuntime*0
_input_shapes
:         ЧЧ@22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ЧЧ@
 
_user_specified_nameinputs
Д
─
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_237765

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
┘'
Џ
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_234234

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
ы
ч
B__inference_conv2d_layer_call_and_return_conditional_losses_237335

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
І	
Н
6__inference_batch_normalization_7_layer_call_fn_238499

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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2346992
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
╔	
Н
6__inference_batch_normalization_7_layer_call_fn_238486

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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2339112
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
т
D
(__inference_dropout_layer_call_fn_237293

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
C__inference_dropout_layer_call_and_return_conditional_losses_2343422
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ђђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ђђ:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
▓
а
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_237954

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
▓
а
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_234531

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
Ѕ	
Н
6__inference_batch_normalization_2_layer_call_fn_237558

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
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2353132
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
▓
а
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_234587

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
¤(
ю
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_234013

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
б╦
Т\
"__inference__traced_restore_239697
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
ы
ч
B__inference_conv2d_layer_call_and_return_conditional_losses_234355

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
ъ
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_234495

inputs

identity_1e
IdentityIdentityinputs*
T0*2
_output_shapes 
:         ЭЭђ2

Identityt

Identity_1IdentityIdentity:output:0*
T0*2
_output_shapes 
:         ЭЭђ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ЭЭђ:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
з
а
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_233867

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
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_232993

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
Ё
d
E__inference_dropout_3_layer_call_and_return_conditional_losses_237828

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUН?2
dropout/Const~
dropout/MulMulinputsdropout/Const:output:0*
T0*2
_output_shapes 
:         ЭЭђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┐
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*2
_output_shapes 
:         ЭЭђ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2
dropout/GreaterEqual/y╔
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*2
_output_shapes 
:         ЭЭђ2
dropout/GreaterEqualі
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*2
_output_shapes 
:         ЭЭђ2
dropout/CastЁ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*2
_output_shapes 
:         ЭЭђ2
dropout/Mul_1p
IdentityIdentitydropout/Mul_1:z:0*
T0*2
_output_shapes 
:         ЭЭђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ЭЭђ:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
╔	
Н
6__inference_batch_normalization_4_layer_call_fn_237874

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
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2332452
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
§
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_235354

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUН?2
dropout/Const}
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:         ЧЧ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЙ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*1
_output_shapes
:         ЧЧ@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2
dropout/GreaterEqual/y╚
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:         ЧЧ@2
dropout/GreaterEqualЅ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:         ЧЧ@2
dropout/Castё
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:         ЧЧ@2
dropout/Mul_1o
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:         ЧЧ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЧЧ@:Y U
1
_output_shapes
:         ЧЧ@
 
_user_specified_nameinputs
╦	
Н
6__inference_batch_normalization_5_layer_call_fn_238065

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
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2334232
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
о(
ю
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_238028

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
Т
─
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_235236

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
╦	
Н
6__inference_batch_normalization_6_layer_call_fn_238269

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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2336452
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
▓
а
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_234643

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
ъ
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_237816

inputs

identity_1e
IdentityIdentityinputs*
T0*2
_output_shapes 
:         ЭЭђ2

Identityt

Identity_1IdentityIdentity:output:0*
T0*2
_output_shapes 
:         ЭЭђ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ЭЭђ:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
І	
Н
6__inference_batch_normalization_2_layer_call_fn_237545

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
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2344292
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
│
ъ
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_234676

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
ЈІ
└
C__inference_dropout_layer_call_and_return_conditional_losses_236093	
u_vel(
batch_normalization_235962:(
batch_normalization_235964:(
batch_normalization_235966:(
batch_normalization_235968:'
conv2d_235972:@
conv2d_235974:@*
batch_normalization_1_235977:@*
batch_normalization_1_235979:@*
batch_normalization_1_235981:@*
batch_normalization_1_235983:@*
conv2d_1_235987:@ђ
conv2d_1_235989:	ђ+
batch_normalization_2_235992:	ђ+
batch_normalization_2_235994:	ђ+
batch_normalization_2_235996:	ђ+
batch_normalization_2_235998:	ђ+
conv2d_2_236002:ђђ
conv2d_2_236004:	ђ+
batch_normalization_3_236007:	ђ+
batch_normalization_3_236009:	ђ+
batch_normalization_3_236011:	ђ+
batch_normalization_3_236013:	ђ+
conv2d_3_236017:ђђ
conv2d_3_236019:	ђ+
batch_normalization_4_236022:	ђ+
batch_normalization_4_236024:	ђ+
batch_normalization_4_236026:	ђ+
batch_normalization_4_236028:	ђ3
conv2d_transpose_236031:ђђ&
conv2d_transpose_236033:	ђ+
batch_normalization_5_236036:	ђ+
batch_normalization_5_236038:	ђ+
batch_normalization_5_236040:	ђ+
batch_normalization_5_236042:	ђ5
conv2d_transpose_1_236045:ђђ(
conv2d_transpose_1_236047:	ђ+
batch_normalization_6_236050:	ђ+
batch_normalization_6_236052:	ђ+
batch_normalization_6_236054:	ђ+
batch_normalization_6_236056:	ђ5
conv2d_transpose_2_236059:ђђ(
conv2d_transpose_2_236061:	ђ+
batch_normalization_7_236064:	ђ+
batch_normalization_7_236066:	ђ+
batch_normalization_7_236068:	ђ+
batch_normalization_7_236070:	ђ4
conv2d_transpose_3_236073:@ђ'
conv2d_transpose_3_236075:@*
batch_normalization_8_236078:@*
batch_normalization_8_236080:@*
batch_normalization_8_236082:@*
batch_normalization_8_236084:@3
conv2d_transpose_4_236087:@'
conv2d_transpose_4_236089:
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
C__inference_reshape_layer_call_and_return_conditional_losses_2343082
reshape/PartitionedCall«
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0batch_normalization_235962batch_normalization_235964batch_normalization_235966batch_normalization_235968*
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
O__inference_batch_normalization_layer_call_and_return_conditional_losses_2343272-
+batch_normalization/StatefulPartitionedCallї
dropout/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
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
C__inference_dropout_layer_call_and_return_conditional_losses_2343422
dropout/PartitionedCall▒
conv2d/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv2d_235972conv2d_235974*
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
B__inference_conv2d_layer_call_and_return_conditional_losses_2343552 
conv2d/StatefulPartitionedCall├
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_1_235977batch_normalization_1_235979batch_normalization_1_235981batch_normalization_1_235983*
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
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2343782/
-batch_normalization_1/StatefulPartitionedCallћ
dropout_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЧЧ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_2343932
dropout_1/PartitionedCallЙ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv2d_1_235987conv2d_1_235989*
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
D__inference_conv2d_1_layer_call_and_return_conditional_losses_2344062"
 conv2d_1/StatefulPartitionedCallк
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_2_235992batch_normalization_2_235994batch_normalization_2_235996batch_normalization_2_235998*
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
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2344292/
-batch_normalization_2/StatefulPartitionedCallЋ
dropout_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЩЩђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2344442
dropout_2/PartitionedCallЙ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0conv2d_2_236002conv2d_2_236004*
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
D__inference_conv2d_2_layer_call_and_return_conditional_losses_2344572"
 conv2d_2/StatefulPartitionedCallк
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_3_236007batch_normalization_3_236009batch_normalization_3_236011batch_normalization_3_236013*
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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2344802/
-batch_normalization_3/StatefulPartitionedCallЋ
dropout_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2344952
dropout_3/PartitionedCallЙ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0conv2d_3_236017conv2d_3_236019*
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
D__inference_conv2d_3_layer_call_and_return_conditional_losses_2345082"
 conv2d_3/StatefulPartitionedCallк
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_4_236022batch_normalization_4_236024batch_normalization_4_236026batch_normalization_4_236028*
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
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2345312/
-batch_normalization_4/StatefulPartitionedCallЩ
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0conv2d_transpose_236031conv2d_transpose_236033*
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
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_2345642*
(conv2d_transpose/StatefulPartitionedCall╬
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0batch_normalization_5_236036batch_normalization_5_236038batch_normalization_5_236040batch_normalization_5_236042*
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
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2345872/
-batch_normalization_5/StatefulPartitionedCallё
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0conv2d_transpose_1_236045conv2d_transpose_1_236047*
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
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_2346202,
*conv2d_transpose_1/StatefulPartitionedCallл
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0batch_normalization_6_236050batch_normalization_6_236052batch_normalization_6_236054batch_normalization_6_236056*
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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2346432/
-batch_normalization_6/StatefulPartitionedCallё
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0conv2d_transpose_2_236059conv2d_transpose_2_236061*
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
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_2346762,
*conv2d_transpose_2/StatefulPartitionedCallл
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0batch_normalization_7_236064batch_normalization_7_236066batch_normalization_7_236068batch_normalization_7_236070*
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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2346992/
-batch_normalization_7/StatefulPartitionedCallЃ
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0conv2d_transpose_3_236073conv2d_transpose_3_236075*
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
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_2347322,
*conv2d_transpose_3/StatefulPartitionedCall¤
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0batch_normalization_8_236078batch_normalization_8_236080batch_normalization_8_236082batch_normalization_8_236084*
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
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2347552/
-batch_normalization_8/StatefulPartitionedCallЃ
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_transpose_4_236087conv2d_transpose_4_236089*
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
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_2347872,
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
а
_
C__inference_reshape_layer_call_and_return_conditional_losses_234308

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
з
а
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_237576

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
р
џ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_232697

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
Т
─
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_237801

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
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_238158

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
Ѕ	
Н
6__inference_batch_normalization_5_layer_call_fn_238104

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
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2351062
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
џ
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_234393

inputs

identity_1d
IdentityIdentityinputs*
T0*1
_output_shapes
:         ЧЧ@2

Identitys

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:         ЧЧ@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЧЧ@:Y U
1
_output_shapes
:         ЧЧ@
 
_user_specified_nameinputs
Ѕ
т
(__inference_dropout_layer_call_fn_235958	
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
identityѕбStatefulPartitionedCall╝
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
GPU2*0J 8ѓ *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_2357342
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
Т
─
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_235159

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
з
а
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_238122

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
з
а
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_233075

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
ь
F
*__inference_dropout_2_layer_call_fn_237635

inputs
identityЛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЩЩђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2344442
PartitionedCallw
IdentityIdentityPartitionedCall:output:0*
T0*2
_output_shapes 
:         ЩЩђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ЩЩђ:Z V
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs
І	
Н
6__inference_batch_normalization_4_layer_call_fn_237887

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
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2345312
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
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_235313

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
а
џ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_237270

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
╦	
Н
6__inference_batch_normalization_4_layer_call_fn_237861

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
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2332012
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
6__inference_batch_normalization_7_layer_call_fn_238473

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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2338672
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
з
а
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_233423

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
Ё
d
E__inference_dropout_3_layer_call_and_return_conditional_losses_235200

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUН?2
dropout/Const~
dropout/MulMulinputsdropout/Const:output:0*
T0*2
_output_shapes 
:         ЭЭђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┐
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*2
_output_shapes 
:         ЭЭђ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2
dropout/GreaterEqual/y╔
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*2
_output_shapes 
:         ЭЭђ2
dropout/GreaterEqualі
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*2
_output_shapes 
:         ЭЭђ2
dropout/CastЁ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*2
_output_shapes 
:         ЭЭђ2
dropout/Mul_1p
IdentityIdentitydropout/Mul_1:z:0*
T0*2
_output_shapes 
:         ЭЭђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ЭЭђ:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
Т
─
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_237972

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
п(
ъ
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_238232

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
ж
F
*__inference_dropout_1_layer_call_fn_237464

inputs
identityл
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЧЧ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_2343932
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ЧЧ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЧЧ@:Y U
1
_output_shapes
:         ЧЧ@
 
_user_specified_nameinputs
└
Ф
3__inference_conv2d_transpose_2_layer_call_fn_238398

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
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_2346762
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
ъ
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_237645

inputs

identity_1e
IdentityIdentityinputs*
T0*2
_output_shapes 
:         ЩЩђ2

Identityt

Identity_1IdentityIdentity:output:0*
T0*2
_output_shapes 
:         ЩЩђ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ЩЩђ:Z V
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs
п(
ъ
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_238436

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
з
а
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_238530

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
├	
Л
6__inference_batch_normalization_8_layer_call_fn_238677

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
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2340892
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
з
а
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_233201

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
Д
─
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_238548

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
с
ю
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_237405

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
Ѕ	
Н
6__inference_batch_normalization_3_layer_call_fn_237729

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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2352362
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
╔	
Н
6__inference_batch_normalization_3_layer_call_fn_237703

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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2331192
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
Ђ	
Л
6__inference_batch_normalization_8_layer_call_fn_238716

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
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2349472
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
Т
─
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_235053

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
▒
ю
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_234564

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
ь
F
*__inference_dropout_3_layer_call_fn_237806

inputs
identityЛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2344952
PartitionedCallw
IdentityIdentityPartitionedCall:output:0*
T0*2
_output_shapes 
:         ЭЭђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ЭЭђ:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
┴	
Л
6__inference_batch_normalization_8_layer_call_fn_238690

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
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2341332
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
╔	
Н
6__inference_batch_normalization_6_layer_call_fn_238282

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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2336892
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
з
а
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_232949

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
┐	
¤
4__inference_batch_normalization_layer_call_fn_237177

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
O__inference_batch_normalization_layer_call_and_return_conditional_losses_2326972
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
Ќ
└
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_232867

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
Е
а
)__inference_conv2d_1_layer_call_fn_237495

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
D__inference_conv2d_1_layer_call_and_return_conditional_losses_2344062
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
Д
─
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_238140

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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_237747

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
Ѕ	
Н
6__inference_batch_normalization_4_layer_call_fn_237900

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
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2351592
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
а
џ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_234327

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
н
Й
O__inference_batch_normalization_layer_call_and_return_conditional_losses_237288

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
ЩЁ
ѓ2
C__inference_dropout_layer_call_and_return_conditional_losses_236846

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
$batch_normalization/FusedBatchNormV3ќ
dropout/IdentityIdentity(batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:         ђђ2
dropout/Identityф
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOp╬
conv2d/Conv2DConv2Ddropout/Identity:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
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
&batch_normalization_1/FusedBatchNormV3ю
dropout_1/IdentityIdentity*batch_normalization_1/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:         ЧЧ@2
dropout_1/Identity▒
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02 
conv2d_1/Conv2D/ReadVariableOpО
conv2d_1/Conv2DConv2Ddropout_1/Identity:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
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
&batch_normalization_2/FusedBatchNormV3Ю
dropout_2/IdentityIdentity*batch_normalization_2/FusedBatchNormV3:y:0*
T0*2
_output_shapes 
:         ЩЩђ2
dropout_2/Identity▓
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02 
conv2d_2/Conv2D/ReadVariableOpО
conv2d_2/Conv2DConv2Ddropout_2/Identity:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
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
&batch_normalization_3/FusedBatchNormV3Ю
dropout_3/IdentityIdentity*batch_normalization_3/FusedBatchNormV3:y:0*
T0*2
_output_shapes 
:         ЭЭђ2
dropout_3/Identity▓
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02 
conv2d_3/Conv2D/ReadVariableOpО
conv2d_3/Conv2DConv2Ddropout_3/Identity:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
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
п(
ъ
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_233791

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
с
ю
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_234089

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
╣
е
3__inference_conv2d_transpose_4_layer_call_fn_238806

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
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_2347872
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
г
А
)__inference_conv2d_3_layer_call_fn_237837

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
D__inference_conv2d_3_layer_call_and_return_conditional_losses_2345082
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
▓
а
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_238362

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
╔	
Н
6__inference_batch_normalization_5_layer_call_fn_238078

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
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2334672
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
І	
Н
6__inference_batch_normalization_5_layer_call_fn_238091

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
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2345872
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
н
Й
O__inference_batch_normalization_layer_call_and_return_conditional_losses_235467

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
ў
a
C__inference_dropout_layer_call_and_return_conditional_losses_234342

inputs

identity_1d
IdentityIdentityinputs*
T0*1
_output_shapes
:         ђђ2

Identitys

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:         ђђ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ђђ:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
б
ю
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_238770

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
б
ю
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_237441

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
┘'
Џ
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_238843

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
ф
ю
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_234732

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
Ѕ	
Н
6__inference_batch_normalization_7_layer_call_fn_238512

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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2350002
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
─
Џ
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_234787

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
о
└
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_234947

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
Ё
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_235277

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUН?2
dropout/Const~
dropout/MulMulinputsdropout/Const:output:0*
T0*2
_output_shapes 
:         ЩЩђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┐
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*2
_output_shapes 
:         ЩЩђ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2
dropout/GreaterEqual/y╔
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*2
_output_shapes 
:         ЩЩђ2
dropout/GreaterEqualі
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*2
_output_shapes 
:         ЩЩђ2
dropout/CastЁ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*2
_output_shapes 
:         ЩЩђ2
dropout/Mul_1p
IdentityIdentitydropout/Mul_1:z:0*
T0*2
_output_shapes 
:         ЩЩђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ЩЩђ:Z V
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs
й	
¤
4__inference_batch_normalization_layer_call_fn_237190

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
O__inference_batch_normalization_layer_call_and_return_conditional_losses_2327412
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
 
ђ
D__inference_conv2d_3_layer_call_and_return_conditional_losses_234508

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
 
¤
4__inference_batch_normalization_layer_call_fn_237203

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
O__inference_batch_normalization_layer_call_and_return_conditional_losses_2343272
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
Ћ
Й
O__inference_batch_normalization_layer_call_and_return_conditional_losses_232741

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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_233119

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
Ѓ	
Л
6__inference_batch_normalization_8_layer_call_fn_238703

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
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2347552
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
р
џ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_237234

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
б
ю
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_234755

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
▒
ю
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_238052

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
Щ№
╬7
C__inference_dropout_layer_call_and_return_conditional_losses_237145

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
$batch_normalization/AssignNewValue_1s
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUН?2
dropout/dropout/Constи
dropout/dropout/MulMul(batch_normalization/FusedBatchNormV3:y:0dropout/dropout/Const:output:0*
T0*1
_output_shapes
:         ђђ2
dropout/dropout/Mulє
dropout/dropout/ShapeShape(batch_normalization/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
dropout/dropout/Shapeо
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*1
_output_shapes
:         ђђ*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЁ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2 
dropout/dropout/GreaterEqual/yУ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:         ђђ2
dropout/dropout/GreaterEqualА
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:         ђђ2
dropout/dropout/Castц
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:         ђђ2
dropout/dropout/Mul_1ф
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOp╬
conv2d/Conv2DConv2Ddropout/dropout/Mul_1:z:0$conv2d/Conv2D/ReadVariableOp:value:0*
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
&batch_normalization_1/AssignNewValue_1w
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUН?2
dropout_1/dropout/Const┐
dropout_1/dropout/MulMul*batch_normalization_1/FusedBatchNormV3:y:0 dropout_1/dropout/Const:output:0*
T0*1
_output_shapes
:         ЧЧ@2
dropout_1/dropout/Mulї
dropout_1/dropout/ShapeShape*batch_normalization_1/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape▄
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*1
_output_shapes
:         ЧЧ@*
dtype020
.dropout_1/dropout/random_uniform/RandomUniformЅ
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2"
 dropout_1/dropout/GreaterEqual/y­
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:         ЧЧ@2 
dropout_1/dropout/GreaterEqualД
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:         ЧЧ@2
dropout_1/dropout/Castг
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*1
_output_shapes
:         ЧЧ@2
dropout_1/dropout/Mul_1▒
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02 
conv2d_1/Conv2D/ReadVariableOpО
conv2d_1/Conv2DConv2Ddropout_1/dropout/Mul_1:z:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
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
&batch_normalization_2/AssignNewValue_1w
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUН?2
dropout_2/dropout/Const└
dropout_2/dropout/MulMul*batch_normalization_2/FusedBatchNormV3:y:0 dropout_2/dropout/Const:output:0*
T0*2
_output_shapes 
:         ЩЩђ2
dropout_2/dropout/Mulї
dropout_2/dropout/ShapeShape*batch_normalization_2/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
dropout_2/dropout/ShapeП
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*2
_output_shapes 
:         ЩЩђ*
dtype020
.dropout_2/dropout/random_uniform/RandomUniformЅ
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2"
 dropout_2/dropout/GreaterEqual/yы
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*2
_output_shapes 
:         ЩЩђ2 
dropout_2/dropout/GreaterEqualе
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*2
_output_shapes 
:         ЩЩђ2
dropout_2/dropout/CastГ
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*2
_output_shapes 
:         ЩЩђ2
dropout_2/dropout/Mul_1▓
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02 
conv2d_2/Conv2D/ReadVariableOpО
conv2d_2/Conv2DConv2Ddropout_2/dropout/Mul_1:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
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
&batch_normalization_3/AssignNewValue_1w
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUН?2
dropout_3/dropout/Const└
dropout_3/dropout/MulMul*batch_normalization_3/FusedBatchNormV3:y:0 dropout_3/dropout/Const:output:0*
T0*2
_output_shapes 
:         ЭЭђ2
dropout_3/dropout/Mulї
dropout_3/dropout/ShapeShape*batch_normalization_3/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
dropout_3/dropout/ShapeП
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*2
_output_shapes 
:         ЭЭђ*
dtype020
.dropout_3/dropout/random_uniform/RandomUniformЅ
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2"
 dropout_3/dropout/GreaterEqual/yы
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*2
_output_shapes 
:         ЭЭђ2 
dropout_3/dropout/GreaterEqualе
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*2
_output_shapes 
:         ЭЭђ2
dropout_3/dropout/CastГ
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*2
_output_shapes 
:         ЭЭђ2
dropout_3/dropout/Mul_1▓
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02 
conv2d_3/Conv2D/ReadVariableOpО
conv2d_3/Conv2DConv2Ddropout_3/dropout/Mul_1:z:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
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
│
ъ
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_238460

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
бЉ
¤
C__inference_dropout_layer_call_and_return_conditional_losses_235734

inputs(
batch_normalization_235603:(
batch_normalization_235605:(
batch_normalization_235607:(
batch_normalization_235609:'
conv2d_235613:@
conv2d_235615:@*
batch_normalization_1_235618:@*
batch_normalization_1_235620:@*
batch_normalization_1_235622:@*
batch_normalization_1_235624:@*
conv2d_1_235628:@ђ
conv2d_1_235630:	ђ+
batch_normalization_2_235633:	ђ+
batch_normalization_2_235635:	ђ+
batch_normalization_2_235637:	ђ+
batch_normalization_2_235639:	ђ+
conv2d_2_235643:ђђ
conv2d_2_235645:	ђ+
batch_normalization_3_235648:	ђ+
batch_normalization_3_235650:	ђ+
batch_normalization_3_235652:	ђ+
batch_normalization_3_235654:	ђ+
conv2d_3_235658:ђђ
conv2d_3_235660:	ђ+
batch_normalization_4_235663:	ђ+
batch_normalization_4_235665:	ђ+
batch_normalization_4_235667:	ђ+
batch_normalization_4_235669:	ђ3
conv2d_transpose_235672:ђђ&
conv2d_transpose_235674:	ђ+
batch_normalization_5_235677:	ђ+
batch_normalization_5_235679:	ђ+
batch_normalization_5_235681:	ђ+
batch_normalization_5_235683:	ђ5
conv2d_transpose_1_235686:ђђ(
conv2d_transpose_1_235688:	ђ+
batch_normalization_6_235691:	ђ+
batch_normalization_6_235693:	ђ+
batch_normalization_6_235695:	ђ+
batch_normalization_6_235697:	ђ5
conv2d_transpose_2_235700:ђђ(
conv2d_transpose_2_235702:	ђ+
batch_normalization_7_235705:	ђ+
batch_normalization_7_235707:	ђ+
batch_normalization_7_235709:	ђ+
batch_normalization_7_235711:	ђ4
conv2d_transpose_3_235714:@ђ'
conv2d_transpose_3_235716:@*
batch_normalization_8_235719:@*
batch_normalization_8_235721:@*
batch_normalization_8_235723:@*
batch_normalization_8_235725:@3
conv2d_transpose_4_235728:@'
conv2d_transpose_4_235730:
identityѕб+batch_normalization/StatefulPartitionedCallб-batch_normalization_1/StatefulPartitionedCallб-batch_normalization_2/StatefulPartitionedCallб-batch_normalization_3/StatefulPartitionedCallб-batch_normalization_4/StatefulPartitionedCallб-batch_normalization_5/StatefulPartitionedCallб-batch_normalization_6/StatefulPartitionedCallб-batch_normalization_7/StatefulPartitionedCallб-batch_normalization_8/StatefulPartitionedCallбconv2d/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallб conv2d_2/StatefulPartitionedCallб conv2d_3/StatefulPartitionedCallб(conv2d_transpose/StatefulPartitionedCallб*conv2d_transpose_1/StatefulPartitionedCallб*conv2d_transpose_2/StatefulPartitionedCallб*conv2d_transpose_3/StatefulPartitionedCallб*conv2d_transpose_4/StatefulPartitionedCallбdropout/StatefulPartitionedCallб!dropout_1/StatefulPartitionedCallб!dropout_2/StatefulPartitionedCallб!dropout_3/StatefulPartitionedCallя
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
C__inference_reshape_layer_call_and_return_conditional_losses_2343082
reshape/PartitionedCallг
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0batch_normalization_235603batch_normalization_235605batch_normalization_235607batch_normalization_235609*
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
O__inference_batch_normalization_layer_call_and_return_conditional_losses_2354672-
+batch_normalization/StatefulPartitionedCallц
dropout/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
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
C__inference_dropout_layer_call_and_return_conditional_losses_2354312!
dropout/StatefulPartitionedCall╣
conv2d/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv2d_235613conv2d_235615*
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
B__inference_conv2d_layer_call_and_return_conditional_losses_2343552 
conv2d/StatefulPartitionedCall┴
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_1_235618batch_normalization_1_235620batch_normalization_1_235622batch_normalization_1_235624*
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
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2353902/
-batch_normalization_1/StatefulPartitionedCall╬
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЧЧ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_2353542#
!dropout_1/StatefulPartitionedCallк
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv2d_1_235628conv2d_1_235630*
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
D__inference_conv2d_1_layer_call_and_return_conditional_losses_2344062"
 conv2d_1/StatefulPartitionedCall─
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_2_235633batch_normalization_2_235635batch_normalization_2_235637batch_normalization_2_235639*
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
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2353132/
-batch_normalization_2/StatefulPartitionedCallЛ
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЩЩђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2352772#
!dropout_2/StatefulPartitionedCallк
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0conv2d_2_235643conv2d_2_235645*
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
D__inference_conv2d_2_layer_call_and_return_conditional_losses_2344572"
 conv2d_2/StatefulPartitionedCall─
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_3_235648batch_normalization_3_235650batch_normalization_3_235652batch_normalization_3_235654*
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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2352362/
-batch_normalization_3/StatefulPartitionedCallЛ
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2352002#
!dropout_3/StatefulPartitionedCallк
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0conv2d_3_235658conv2d_3_235660*
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
D__inference_conv2d_3_layer_call_and_return_conditional_losses_2345082"
 conv2d_3/StatefulPartitionedCall─
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_4_235663batch_normalization_4_235665batch_normalization_4_235667batch_normalization_4_235669*
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
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2351592/
-batch_normalization_4/StatefulPartitionedCallЩ
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0conv2d_transpose_235672conv2d_transpose_235674*
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
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_2345642*
(conv2d_transpose/StatefulPartitionedCall╠
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0batch_normalization_5_235677batch_normalization_5_235679batch_normalization_5_235681batch_normalization_5_235683*
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
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2351062/
-batch_normalization_5/StatefulPartitionedCallё
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0conv2d_transpose_1_235686conv2d_transpose_1_235688*
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
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_2346202,
*conv2d_transpose_1/StatefulPartitionedCall╬
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0batch_normalization_6_235691batch_normalization_6_235693batch_normalization_6_235695batch_normalization_6_235697*
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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2350532/
-batch_normalization_6/StatefulPartitionedCallё
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0conv2d_transpose_2_235700conv2d_transpose_2_235702*
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
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_2346762,
*conv2d_transpose_2/StatefulPartitionedCall╬
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0batch_normalization_7_235705batch_normalization_7_235707batch_normalization_7_235709batch_normalization_7_235711*
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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2350002/
-batch_normalization_7/StatefulPartitionedCallЃ
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0conv2d_transpose_3_235714conv2d_transpose_3_235716*
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
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_2347322,
*conv2d_transpose_3/StatefulPartitionedCall═
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0batch_normalization_8_235719batch_normalization_8_235721batch_normalization_8_235723batch_normalization_8_235725*
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
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2349472/
-batch_normalization_8/StatefulPartitionedCallЃ
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_transpose_4_235728conv2d_transpose_4_235730*
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
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_2347872,
*conv2d_transpose_4/StatefulPartitionedCallў
IdentityIdentity3conv2d_transpose_4/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ2

Identityз
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*"
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
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:U Q
-
_output_shapes
:         ђђ
 
_user_specified_nameinputs
 
ђ
D__inference_conv2d_2_layer_call_and_return_conditional_losses_237677

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
Ќ
└
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_238752

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
ъ
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_234444

inputs

identity_1e
IdentityIdentityinputs*
T0*2
_output_shapes 
:         ЩЩђ2

Identityt

Identity_1IdentityIdentity:output:0*
T0*2
_output_shapes 
:         ЩЩђ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ЩЩђ:Z V
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs
ў
a
C__inference_dropout_layer_call_and_return_conditional_losses_237303

inputs

identity_1d
IdentityIdentityinputs*
T0*1
_output_shapes
:         ђђ2

Identitys

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:         ђђ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ђђ:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
 
ђ
D__inference_conv2d_2_layer_call_and_return_conditional_losses_234457

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
▓
а
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_238566

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
§
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_237486

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUН?2
dropout/Const}
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:         ЧЧ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЙ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*1
_output_shapes
:         ЧЧ@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2
dropout/GreaterEqual/y╚
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:         ЧЧ@2
dropout/GreaterEqualЅ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:         ЧЧ@2
dropout/Castё
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:         ЧЧ@2
dropout/Mul_1o
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:         ЧЧ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЧЧ@:Y U
1
_output_shapes
:         ЧЧ@
 
_user_specified_nameinputs
Д
─
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_237936

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
ђ	
Ф
3__inference_conv2d_transpose_2_layer_call_fn_238389

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
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_2337912
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
о
└
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_237459

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
Д
─
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_233911

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
ч
b
C__inference_dropout_layer_call_and_return_conditional_losses_235431

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUН?2
dropout/Const}
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:         ђђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЙ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*1
_output_shapes
:         ђђ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2
dropout/GreaterEqual/y╚
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:         ђђ2
dropout/GreaterEqualЅ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:         ђђ2
dropout/Castё
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:         ђђ2
dropout/Mul_1o
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:         ђђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ђђ:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
П
D
(__inference_reshape_layer_call_fn_237150

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
C__inference_reshape_layer_call_and_return_conditional_losses_2343082
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
ч
 
D__inference_conv2d_1_layer_call_and_return_conditional_losses_237506

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
п(
ъ
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_233569

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
Т
─
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_238176

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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_238380

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
Ђ	
Л
6__inference_batch_normalization_1_layer_call_fn_237387

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
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2353902
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
РЅ
М?
__inference__traced_save_239288
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
Д
─
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_237594

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
Д
─
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_233689

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
а
_
C__inference_reshape_layer_call_and_return_conditional_losses_237164

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
њІ
┴
C__inference_dropout_layer_call_and_return_conditional_losses_234794

inputs(
batch_normalization_234328:(
batch_normalization_234330:(
batch_normalization_234332:(
batch_normalization_234334:'
conv2d_234356:@
conv2d_234358:@*
batch_normalization_1_234379:@*
batch_normalization_1_234381:@*
batch_normalization_1_234383:@*
batch_normalization_1_234385:@*
conv2d_1_234407:@ђ
conv2d_1_234409:	ђ+
batch_normalization_2_234430:	ђ+
batch_normalization_2_234432:	ђ+
batch_normalization_2_234434:	ђ+
batch_normalization_2_234436:	ђ+
conv2d_2_234458:ђђ
conv2d_2_234460:	ђ+
batch_normalization_3_234481:	ђ+
batch_normalization_3_234483:	ђ+
batch_normalization_3_234485:	ђ+
batch_normalization_3_234487:	ђ+
conv2d_3_234509:ђђ
conv2d_3_234511:	ђ+
batch_normalization_4_234532:	ђ+
batch_normalization_4_234534:	ђ+
batch_normalization_4_234536:	ђ+
batch_normalization_4_234538:	ђ3
conv2d_transpose_234565:ђђ&
conv2d_transpose_234567:	ђ+
batch_normalization_5_234588:	ђ+
batch_normalization_5_234590:	ђ+
batch_normalization_5_234592:	ђ+
batch_normalization_5_234594:	ђ5
conv2d_transpose_1_234621:ђђ(
conv2d_transpose_1_234623:	ђ+
batch_normalization_6_234644:	ђ+
batch_normalization_6_234646:	ђ+
batch_normalization_6_234648:	ђ+
batch_normalization_6_234650:	ђ5
conv2d_transpose_2_234677:ђђ(
conv2d_transpose_2_234679:	ђ+
batch_normalization_7_234700:	ђ+
batch_normalization_7_234702:	ђ+
batch_normalization_7_234704:	ђ+
batch_normalization_7_234706:	ђ4
conv2d_transpose_3_234733:@ђ'
conv2d_transpose_3_234735:@*
batch_normalization_8_234756:@*
batch_normalization_8_234758:@*
batch_normalization_8_234760:@*
batch_normalization_8_234762:@3
conv2d_transpose_4_234788:@'
conv2d_transpose_4_234790:
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
C__inference_reshape_layer_call_and_return_conditional_losses_2343082
reshape/PartitionedCall«
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0batch_normalization_234328batch_normalization_234330batch_normalization_234332batch_normalization_234334*
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
O__inference_batch_normalization_layer_call_and_return_conditional_losses_2343272-
+batch_normalization/StatefulPartitionedCallї
dropout/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
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
C__inference_dropout_layer_call_and_return_conditional_losses_2343422
dropout/PartitionedCall▒
conv2d/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv2d_234356conv2d_234358*
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
B__inference_conv2d_layer_call_and_return_conditional_losses_2343552 
conv2d/StatefulPartitionedCall├
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_1_234379batch_normalization_1_234381batch_normalization_1_234383batch_normalization_1_234385*
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
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2343782/
-batch_normalization_1/StatefulPartitionedCallћ
dropout_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЧЧ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_2343932
dropout_1/PartitionedCallЙ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv2d_1_234407conv2d_1_234409*
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
D__inference_conv2d_1_layer_call_and_return_conditional_losses_2344062"
 conv2d_1/StatefulPartitionedCallк
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_2_234430batch_normalization_2_234432batch_normalization_2_234434batch_normalization_2_234436*
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
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2344292/
-batch_normalization_2/StatefulPartitionedCallЋ
dropout_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЩЩђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2344442
dropout_2/PartitionedCallЙ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0conv2d_2_234458conv2d_2_234460*
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
D__inference_conv2d_2_layer_call_and_return_conditional_losses_2344572"
 conv2d_2/StatefulPartitionedCallк
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_3_234481batch_normalization_3_234483batch_normalization_3_234485batch_normalization_3_234487*
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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2344802/
-batch_normalization_3/StatefulPartitionedCallЋ
dropout_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2344952
dropout_3/PartitionedCallЙ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0conv2d_3_234509conv2d_3_234511*
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
D__inference_conv2d_3_layer_call_and_return_conditional_losses_2345082"
 conv2d_3/StatefulPartitionedCallк
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_4_234532batch_normalization_4_234534batch_normalization_4_234536batch_normalization_4_234538*
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
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2345312/
-batch_normalization_4/StatefulPartitionedCallЩ
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0conv2d_transpose_234565conv2d_transpose_234567*
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
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_2345642*
(conv2d_transpose/StatefulPartitionedCall╬
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0batch_normalization_5_234588batch_normalization_5_234590batch_normalization_5_234592batch_normalization_5_234594*
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
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2345872/
-batch_normalization_5/StatefulPartitionedCallё
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0conv2d_transpose_1_234621conv2d_transpose_1_234623*
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
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_2346202,
*conv2d_transpose_1/StatefulPartitionedCallл
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0batch_normalization_6_234644batch_normalization_6_234646batch_normalization_6_234648batch_normalization_6_234650*
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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2346432/
-batch_normalization_6/StatefulPartitionedCallё
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0conv2d_transpose_2_234677conv2d_transpose_2_234679*
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
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_2346762,
*conv2d_transpose_2/StatefulPartitionedCallл
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0batch_normalization_7_234700batch_normalization_7_234702batch_normalization_7_234704batch_normalization_7_234706*
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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2346992/
-batch_normalization_7/StatefulPartitionedCallЃ
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0conv2d_transpose_3_234733conv2d_transpose_3_234735*
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
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_2347322,
*conv2d_transpose_3/StatefulPartitionedCall¤
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0batch_normalization_8_234756batch_normalization_8_234758batch_normalization_8_234760batch_normalization_8_234762*
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
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2347552/
-batch_normalization_8/StatefulPartitionedCallЃ
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_transpose_4_234788conv2d_transpose_4_234790*
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
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_2347872,
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
Т
─
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_237630

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
StatefulPartitionedCall:0         ђђtensorflow/serving/predict:т│
ѕ
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer_with_weights-10
layer-16
layer_with_weights-11
layer-17
layer_with_weights-12
layer-18
layer_with_weights-13
layer-19
layer_with_weights-14
layer-20
layer_with_weights-15
layer-21
layer_with_weights-16
layer-22
layer_with_weights-17
layer-23
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
ё_default_save_signature
Ё__call__
+є&call_and_return_all_conditional_losses"
_tf_keras_network
"
_tf_keras_input_layer
Д
	variables
 regularization_losses
!trainable_variables
"	keras_api
Є__call__
+ѕ&call_and_return_all_conditional_losses"
_tf_keras_layer
В
#axis
	$gamma
%beta
&moving_mean
'moving_variance
(	variables
)regularization_losses
*trainable_variables
+	keras_api
Ѕ__call__
+і&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
,	variables
-regularization_losses
.trainable_variables
/	keras_api
І__call__
+ї&call_and_return_all_conditional_losses"
_tf_keras_layer
й

0kernel
1bias
2	variables
3regularization_losses
4trainable_variables
5	keras_api
Ї__call__
+ј&call_and_return_all_conditional_losses"
_tf_keras_layer
В
6axis
	7gamma
8beta
9moving_mean
:moving_variance
;	variables
<regularization_losses
=trainable_variables
>	keras_api
Ј__call__
+љ&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
?	variables
@regularization_losses
Atrainable_variables
B	keras_api
Љ__call__
+њ&call_and_return_all_conditional_losses"
_tf_keras_layer
й

Ckernel
Dbias
E	variables
Fregularization_losses
Gtrainable_variables
H	keras_api
Њ__call__
+ћ&call_and_return_all_conditional_losses"
_tf_keras_layer
В
Iaxis
	Jgamma
Kbeta
Lmoving_mean
Mmoving_variance
N	variables
Oregularization_losses
Ptrainable_variables
Q	keras_api
Ћ__call__
+ќ&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
Ќ__call__
+ў&call_and_return_all_conditional_losses"
_tf_keras_layer
й

Vkernel
Wbias
X	variables
Yregularization_losses
Ztrainable_variables
[	keras_api
Ў__call__
+џ&call_and_return_all_conditional_losses"
_tf_keras_layer
В
\axis
	]gamma
^beta
_moving_mean
`moving_variance
a	variables
bregularization_losses
ctrainable_variables
d	keras_api
Џ__call__
+ю&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
e	variables
fregularization_losses
gtrainable_variables
h	keras_api
Ю__call__
+ъ&call_and_return_all_conditional_losses"
_tf_keras_layer
й

ikernel
jbias
k	variables
lregularization_losses
mtrainable_variables
n	keras_api
Ъ__call__
+а&call_and_return_all_conditional_losses"
_tf_keras_layer
В
oaxis
	pgamma
qbeta
rmoving_mean
smoving_variance
t	variables
uregularization_losses
vtrainable_variables
w	keras_api
А__call__
+б&call_and_return_all_conditional_losses"
_tf_keras_layer
й

xkernel
ybias
z	variables
{regularization_losses
|trainable_variables
}	keras_api
Б__call__
+ц&call_and_return_all_conditional_losses"
_tf_keras_layer
з
~axis
	gamma
	ђbeta
Ђmoving_mean
ѓmoving_variance
Ѓ	variables
ёregularization_losses
Ёtrainable_variables
є	keras_api
Ц__call__
+д&call_and_return_all_conditional_losses"
_tf_keras_layer
├
Єkernel
	ѕbias
Ѕ	variables
іregularization_losses
Іtrainable_variables
ї	keras_api
Д__call__
+е&call_and_return_all_conditional_losses"
_tf_keras_layer
ш
	Їaxis

јgamma
	Јbeta
љmoving_mean
Љmoving_variance
њ	variables
Њregularization_losses
ћtrainable_variables
Ћ	keras_api
Е__call__
+ф&call_and_return_all_conditional_losses"
_tf_keras_layer
├
ќkernel
	Ќbias
ў	variables
Ўregularization_losses
џtrainable_variables
Џ	keras_api
Ф__call__
+г&call_and_return_all_conditional_losses"
_tf_keras_layer
ш
	юaxis

Юgamma
	ъbeta
Ъmoving_mean
аmoving_variance
А	variables
бregularization_losses
Бtrainable_variables
ц	keras_api
Г__call__
+«&call_and_return_all_conditional_losses"
_tf_keras_layer
├
Цkernel
	дbias
Д	variables
еregularization_losses
Еtrainable_variables
ф	keras_api
»__call__
+░&call_and_return_all_conditional_losses"
_tf_keras_layer
ш
	Фaxis

гgamma
	Гbeta
«moving_mean
»moving_variance
░	variables
▒regularization_losses
▓trainable_variables
│	keras_api
▒__call__
+▓&call_and_return_all_conditional_losses"
_tf_keras_layer
├
┤kernel
	хbias
Х	variables
иregularization_losses
Иtrainable_variables
╣	keras_api
│__call__
+┤&call_and_return_all_conditional_losses"
_tf_keras_layer
к
	║iter
╗beta_1
╝beta_2

йdecay
Йlearning_rate$m╝%mй0mЙ1m┐7m└8m┴Cm┬Dm├Jm─Km┼VmкWmК]m╚^m╔im╩jm╦pm╠qm═xm╬ym¤mл	ђmЛ	Єmм	ѕmМ	јmн	ЈmН	ќmо	ЌmО	Юmп	ъm┘	Цm┌	дm█	гm▄	ГmП	┤mя	хm▀$vЯ%vр0vР1vс7vС8vтCvТDvуJvУKvжVvЖWvв]vВ^vьivЬjv№pv­qvыxvЫyvзvЗ	ђvш	ЄvШ	ѕvэ	јvЭ	Јvщ	ќvЩ	Ќvч	ЮvЧ	ъv§	Цv■	дv 	гvђ	ГvЂ	┤vѓ	хvЃ"
	optimizer
П
$0
%1
&2
'3
04
15
76
87
98
:9
C10
D11
J12
K13
L14
M15
V16
W17
]18
^19
_20
`21
i22
j23
p24
q25
r26
s27
x28
y29
30
ђ31
Ђ32
ѓ33
Є34
ѕ35
ј36
Ј37
љ38
Љ39
ќ40
Ќ41
Ю42
ъ43
Ъ44
а45
Ц46
д47
г48
Г49
«50
»51
┤52
х53"
trackable_list_wrapper
 "
trackable_list_wrapper
┼
$0
%1
02
13
74
85
C6
D7
J8
K9
V10
W11
]12
^13
i14
j15
p16
q17
x18
y19
20
ђ21
Є22
ѕ23
ј24
Ј25
ќ26
Ќ27
Ю28
ъ29
Ц30
д31
г32
Г33
┤34
х35"
trackable_list_wrapper
М
┐layers
└metrics
	variables
┴layer_metrics
regularization_losses
 ┬layer_regularization_losses
trainable_variables
├non_trainable_variables
Ё__call__
ё_default_save_signature
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
-
хserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
─layers
┼metrics
	variables
кlayer_metrics
 regularization_losses
 Кlayer_regularization_losses
!trainable_variables
╚non_trainable_variables
Є__call__
+ѕ&call_and_return_all_conditional_losses
'ѕ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%2batch_normalization/gamma
&:$2batch_normalization/beta
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
<
$0
%1
&2
'3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
х
╔layers
╩metrics
(	variables
╦layer_metrics
)regularization_losses
 ╠layer_regularization_losses
*trainable_variables
═non_trainable_variables
Ѕ__call__
+і&call_and_return_all_conditional_losses
'і"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
╬layers
¤metrics
,	variables
лlayer_metrics
-regularization_losses
 Лlayer_regularization_losses
.trainable_variables
мnon_trainable_variables
І__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
':%@2conv2d/kernel
:@2conv2d/bias
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
х
Мlayers
нmetrics
2	variables
Нlayer_metrics
3regularization_losses
 оlayer_regularization_losses
4trainable_variables
Оnon_trainable_variables
Ї__call__
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_1/gamma
(:&@2batch_normalization_1/beta
1:/@ (2!batch_normalization_1/moving_mean
5:3@ (2%batch_normalization_1/moving_variance
<
70
81
92
:3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
х
пlayers
┘metrics
;	variables
┌layer_metrics
<regularization_losses
 █layer_regularization_losses
=trainable_variables
▄non_trainable_variables
Ј__call__
+љ&call_and_return_all_conditional_losses
'љ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Пlayers
яmetrics
?	variables
▀layer_metrics
@regularization_losses
 Яlayer_regularization_losses
Atrainable_variables
рnon_trainable_variables
Љ__call__
+њ&call_and_return_all_conditional_losses
'њ"call_and_return_conditional_losses"
_generic_user_object
*:(@ђ2conv2d_1/kernel
:ђ2conv2d_1/bias
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
х
Рlayers
сmetrics
E	variables
Сlayer_metrics
Fregularization_losses
 тlayer_regularization_losses
Gtrainable_variables
Тnon_trainable_variables
Њ__call__
+ћ&call_and_return_all_conditional_losses
'ћ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(ђ2batch_normalization_2/gamma
):'ђ2batch_normalization_2/beta
2:0ђ (2!batch_normalization_2/moving_mean
6:4ђ (2%batch_normalization_2/moving_variance
<
J0
K1
L2
M3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
х
уlayers
Уmetrics
N	variables
жlayer_metrics
Oregularization_losses
 Жlayer_regularization_losses
Ptrainable_variables
вnon_trainable_variables
Ћ__call__
+ќ&call_and_return_all_conditional_losses
'ќ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Вlayers
ьmetrics
R	variables
Ьlayer_metrics
Sregularization_losses
 №layer_regularization_losses
Ttrainable_variables
­non_trainable_variables
Ќ__call__
+ў&call_and_return_all_conditional_losses
'ў"call_and_return_conditional_losses"
_generic_user_object
+:)ђђ2conv2d_2/kernel
:ђ2conv2d_2/bias
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
х
ыlayers
Ыmetrics
X	variables
зlayer_metrics
Yregularization_losses
 Зlayer_regularization_losses
Ztrainable_variables
шnon_trainable_variables
Ў__call__
+џ&call_and_return_all_conditional_losses
'џ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(ђ2batch_normalization_3/gamma
):'ђ2batch_normalization_3/beta
2:0ђ (2!batch_normalization_3/moving_mean
6:4ђ (2%batch_normalization_3/moving_variance
<
]0
^1
_2
`3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
х
Шlayers
эmetrics
a	variables
Эlayer_metrics
bregularization_losses
 щlayer_regularization_losses
ctrainable_variables
Щnon_trainable_variables
Џ__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
чlayers
Чmetrics
e	variables
§layer_metrics
fregularization_losses
 ■layer_regularization_losses
gtrainable_variables
 non_trainable_variables
Ю__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
+:)ђђ2conv2d_3/kernel
:ђ2conv2d_3/bias
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
х
ђlayers
Ђmetrics
k	variables
ѓlayer_metrics
lregularization_losses
 Ѓlayer_regularization_losses
mtrainable_variables
ёnon_trainable_variables
Ъ__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(ђ2batch_normalization_4/gamma
):'ђ2batch_normalization_4/beta
2:0ђ (2!batch_normalization_4/moving_mean
6:4ђ (2%batch_normalization_4/moving_variance
<
p0
q1
r2
s3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
х
Ёlayers
єmetrics
t	variables
Єlayer_metrics
uregularization_losses
 ѕlayer_regularization_losses
vtrainable_variables
Ѕnon_trainable_variables
А__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
3:1ђђ2conv2d_transpose/kernel
$:"ђ2conv2d_transpose/bias
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
х
іlayers
Іmetrics
z	variables
їlayer_metrics
{regularization_losses
 Їlayer_regularization_losses
|trainable_variables
јnon_trainable_variables
Б__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(ђ2batch_normalization_5/gamma
):'ђ2batch_normalization_5/beta
2:0ђ (2!batch_normalization_5/moving_mean
6:4ђ (2%batch_normalization_5/moving_variance
?
0
ђ1
Ђ2
ѓ3"
trackable_list_wrapper
 "
trackable_list_wrapper
/
0
ђ1"
trackable_list_wrapper
И
Јlayers
љmetrics
Ѓ	variables
Љlayer_metrics
ёregularization_losses
 њlayer_regularization_losses
Ёtrainable_variables
Њnon_trainable_variables
Ц__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
_generic_user_object
5:3ђђ2conv2d_transpose_1/kernel
&:$ђ2conv2d_transpose_1/bias
0
Є0
ѕ1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Є0
ѕ1"
trackable_list_wrapper
И
ћlayers
Ћmetrics
Ѕ	variables
ќlayer_metrics
іregularization_losses
 Ќlayer_regularization_losses
Іtrainable_variables
ўnon_trainable_variables
Д__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(ђ2batch_normalization_6/gamma
):'ђ2batch_normalization_6/beta
2:0ђ (2!batch_normalization_6/moving_mean
6:4ђ (2%batch_normalization_6/moving_variance
@
ј0
Ј1
љ2
Љ3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
ј0
Ј1"
trackable_list_wrapper
И
Ўlayers
џmetrics
њ	variables
Џlayer_metrics
Њregularization_losses
 юlayer_regularization_losses
ћtrainable_variables
Юnon_trainable_variables
Е__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
5:3ђђ2conv2d_transpose_2/kernel
&:$ђ2conv2d_transpose_2/bias
0
ќ0
Ќ1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
ќ0
Ќ1"
trackable_list_wrapper
И
ъlayers
Ъmetrics
ў	variables
аlayer_metrics
Ўregularization_losses
 Аlayer_regularization_losses
џtrainable_variables
бnon_trainable_variables
Ф__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(ђ2batch_normalization_7/gamma
):'ђ2batch_normalization_7/beta
2:0ђ (2!batch_normalization_7/moving_mean
6:4ђ (2%batch_normalization_7/moving_variance
@
Ю0
ъ1
Ъ2
а3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ю0
ъ1"
trackable_list_wrapper
И
Бlayers
цmetrics
А	variables
Цlayer_metrics
бregularization_losses
 дlayer_regularization_losses
Бtrainable_variables
Дnon_trainable_variables
Г__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
4:2@ђ2conv2d_transpose_3/kernel
%:#@2conv2d_transpose_3/bias
0
Ц0
д1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ц0
д1"
trackable_list_wrapper
И
еlayers
Еmetrics
Д	variables
фlayer_metrics
еregularization_losses
 Фlayer_regularization_losses
Еtrainable_variables
гnon_trainable_variables
»__call__
+░&call_and_return_all_conditional_losses
'░"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_8/gamma
(:&@2batch_normalization_8/beta
1:/@ (2!batch_normalization_8/moving_mean
5:3@ (2%batch_normalization_8/moving_variance
@
г0
Г1
«2
»3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
г0
Г1"
trackable_list_wrapper
И
Гlayers
«metrics
░	variables
»layer_metrics
▒regularization_losses
 ░layer_regularization_losses
▓trainable_variables
▒non_trainable_variables
▒__call__
+▓&call_and_return_all_conditional_losses
'▓"call_and_return_conditional_losses"
_generic_user_object
3:1@2conv2d_transpose_4/kernel
%:#2conv2d_transpose_4/bias
0
┤0
х1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
┤0
х1"
trackable_list_wrapper
И
▓layers
│metrics
Х	variables
┤layer_metrics
иregularization_losses
 хlayer_regularization_losses
Иtrainable_variables
Хnon_trainable_variables
│__call__
+┤&call_and_return_all_conditional_losses
'┤"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
о
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
23"
trackable_list_wrapper
(
и0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
«
&0
'1
92
:3
L4
M5
_6
`7
r8
s9
Ђ10
ѓ11
љ12
Љ13
Ъ14
а15
«16
»17"
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
.
&0
'1"
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
.
90
:1"
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
.
L0
M1"
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
.
_0
`1"
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
.
r0
s1"
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
0
Ђ0
ѓ1"
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
0
љ0
Љ1"
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
0
Ъ0
а1"
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
0
«0
»1"
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

Иtotal

╣count
║	variables
╗	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
И0
╣1"
trackable_list_wrapper
.
║	variables"
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
!__inference__wrapped_model_232675u_vel"ў
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
Ь2в
(__inference_dropout_layer_call_fn_234905
(__inference_dropout_layer_call_fn_236462
(__inference_dropout_layer_call_fn_236575
(__inference_dropout_layer_call_fn_235958└
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
┌2О
C__inference_dropout_layer_call_and_return_conditional_losses_236846
C__inference_dropout_layer_call_and_return_conditional_losses_237145
C__inference_dropout_layer_call_and_return_conditional_losses_236093
C__inference_dropout_layer_call_and_return_conditional_losses_236228└
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
(__inference_reshape_layer_call_fn_237150б
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
C__inference_reshape_layer_call_and_return_conditional_losses_237164б
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
4__inference_batch_normalization_layer_call_fn_237177
4__inference_batch_normalization_layer_call_fn_237190
4__inference_batch_normalization_layer_call_fn_237203
4__inference_batch_normalization_layer_call_fn_237216┤
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
O__inference_batch_normalization_layer_call_and_return_conditional_losses_237234
O__inference_batch_normalization_layer_call_and_return_conditional_losses_237252
O__inference_batch_normalization_layer_call_and_return_conditional_losses_237270
O__inference_batch_normalization_layer_call_and_return_conditional_losses_237288┤
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
(__inference_dropout_layer_call_fn_237293
(__inference_dropout_layer_call_fn_237298┤
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
─2┴
C__inference_dropout_layer_call_and_return_conditional_losses_237303
C__inference_dropout_layer_call_and_return_conditional_losses_237315┤
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
'__inference_conv2d_layer_call_fn_237324б
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
B__inference_conv2d_layer_call_and_return_conditional_losses_237335б
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
6__inference_batch_normalization_1_layer_call_fn_237348
6__inference_batch_normalization_1_layer_call_fn_237361
6__inference_batch_normalization_1_layer_call_fn_237374
6__inference_batch_normalization_1_layer_call_fn_237387┤
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
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_237405
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_237423
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_237441
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_237459┤
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
*__inference_dropout_1_layer_call_fn_237464
*__inference_dropout_1_layer_call_fn_237469┤
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
╚2┼
E__inference_dropout_1_layer_call_and_return_conditional_losses_237474
E__inference_dropout_1_layer_call_and_return_conditional_losses_237486┤
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
)__inference_conv2d_1_layer_call_fn_237495б
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
D__inference_conv2d_1_layer_call_and_return_conditional_losses_237506б
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
6__inference_batch_normalization_2_layer_call_fn_237519
6__inference_batch_normalization_2_layer_call_fn_237532
6__inference_batch_normalization_2_layer_call_fn_237545
6__inference_batch_normalization_2_layer_call_fn_237558┤
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
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_237576
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_237594
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_237612
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_237630┤
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
*__inference_dropout_2_layer_call_fn_237635
*__inference_dropout_2_layer_call_fn_237640┤
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
╚2┼
E__inference_dropout_2_layer_call_and_return_conditional_losses_237645
E__inference_dropout_2_layer_call_and_return_conditional_losses_237657┤
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
)__inference_conv2d_2_layer_call_fn_237666б
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
D__inference_conv2d_2_layer_call_and_return_conditional_losses_237677б
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
6__inference_batch_normalization_3_layer_call_fn_237690
6__inference_batch_normalization_3_layer_call_fn_237703
6__inference_batch_normalization_3_layer_call_fn_237716
6__inference_batch_normalization_3_layer_call_fn_237729┤
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
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_237747
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_237765
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_237783
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_237801┤
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
*__inference_dropout_3_layer_call_fn_237806
*__inference_dropout_3_layer_call_fn_237811┤
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
╚2┼
E__inference_dropout_3_layer_call_and_return_conditional_losses_237816
E__inference_dropout_3_layer_call_and_return_conditional_losses_237828┤
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
)__inference_conv2d_3_layer_call_fn_237837б
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
D__inference_conv2d_3_layer_call_and_return_conditional_losses_237848б
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
6__inference_batch_normalization_4_layer_call_fn_237861
6__inference_batch_normalization_4_layer_call_fn_237874
6__inference_batch_normalization_4_layer_call_fn_237887
6__inference_batch_normalization_4_layer_call_fn_237900┤
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
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_237918
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_237936
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_237954
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_237972┤
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
1__inference_conv2d_transpose_layer_call_fn_237981
1__inference_conv2d_transpose_layer_call_fn_237990б
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
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_238028
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_238052б
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
6__inference_batch_normalization_5_layer_call_fn_238065
6__inference_batch_normalization_5_layer_call_fn_238078
6__inference_batch_normalization_5_layer_call_fn_238091
6__inference_batch_normalization_5_layer_call_fn_238104┤
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
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_238122
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_238140
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_238158
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_238176┤
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
3__inference_conv2d_transpose_1_layer_call_fn_238185
3__inference_conv2d_transpose_1_layer_call_fn_238194б
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
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_238232
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_238256б
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
6__inference_batch_normalization_6_layer_call_fn_238269
6__inference_batch_normalization_6_layer_call_fn_238282
6__inference_batch_normalization_6_layer_call_fn_238295
6__inference_batch_normalization_6_layer_call_fn_238308┤
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
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_238326
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_238344
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_238362
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_238380┤
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
3__inference_conv2d_transpose_2_layer_call_fn_238389
3__inference_conv2d_transpose_2_layer_call_fn_238398б
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
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_238436
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_238460б
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
6__inference_batch_normalization_7_layer_call_fn_238473
6__inference_batch_normalization_7_layer_call_fn_238486
6__inference_batch_normalization_7_layer_call_fn_238499
6__inference_batch_normalization_7_layer_call_fn_238512┤
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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_238530
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_238548
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_238566
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_238584┤
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
3__inference_conv2d_transpose_3_layer_call_fn_238593
3__inference_conv2d_transpose_3_layer_call_fn_238602б
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
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_238640
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_238664б
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
6__inference_batch_normalization_8_layer_call_fn_238677
6__inference_batch_normalization_8_layer_call_fn_238690
6__inference_batch_normalization_8_layer_call_fn_238703
6__inference_batch_normalization_8_layer_call_fn_238716┤
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
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_238734
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_238752
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_238770
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_238788┤
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
3__inference_conv2d_transpose_4_layer_call_fn_238797
3__inference_conv2d_transpose_4_layer_call_fn_238806б
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
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_238843
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_238866б
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
$__inference_signature_wrapper_236349u_vel"ћ
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
 ■
!__inference__wrapped_model_232675пM$%&'01789:CDJKLMVW]^_`ijpqrsxyђЂѓЄѕјЈљЉќЌЮъЪаЦдгГ«»┤х4б1
*б'
%і"
u_vel         ђђ
ф "QфN
L
conv2d_transpose_46і3
conv2d_transpose_4         ђђВ
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_237405ќ789:MбJ
Cб@
:і7
inputs+                           @
p 
ф "?б<
5і2
0+                           @
џ В
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_237423ќ789:MбJ
Cб@
:і7
inputs+                           @
p
ф "?б<
5і2
0+                           @
џ ╦
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_237441v789:=б:
3б0
*і'
inputs         ЧЧ@
p 
ф "/б,
%і"
0         ЧЧ@
џ ╦
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_237459v789:=б:
3б0
*і'
inputs         ЧЧ@
p
ф "/б,
%і"
0         ЧЧ@
џ ─
6__inference_batch_normalization_1_layer_call_fn_237348Ѕ789:MбJ
Cб@
:і7
inputs+                           @
p 
ф "2і/+                           @─
6__inference_batch_normalization_1_layer_call_fn_237361Ѕ789:MбJ
Cб@
:і7
inputs+                           @
p
ф "2і/+                           @Б
6__inference_batch_normalization_1_layer_call_fn_237374i789:=б:
3б0
*і'
inputs         ЧЧ@
p 
ф ""і         ЧЧ@Б
6__inference_batch_normalization_1_layer_call_fn_237387i789:=б:
3б0
*і'
inputs         ЧЧ@
p
ф ""і         ЧЧ@Ь
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_237576ўJKLMNбK
DбA
;і8
inputs,                           ђ
p 
ф "@б=
6і3
0,                           ђ
џ Ь
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_237594ўJKLMNбK
DбA
;і8
inputs,                           ђ
p
ф "@б=
6і3
0,                           ђ
џ ═
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_237612xJKLM>б;
4б1
+і(
inputs         ЩЩђ
p 
ф "0б-
&і#
0         ЩЩђ
џ ═
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_237630xJKLM>б;
4б1
+і(
inputs         ЩЩђ
p
ф "0б-
&і#
0         ЩЩђ
џ к
6__inference_batch_normalization_2_layer_call_fn_237519ІJKLMNбK
DбA
;і8
inputs,                           ђ
p 
ф "3і0,                           ђк
6__inference_batch_normalization_2_layer_call_fn_237532ІJKLMNбK
DбA
;і8
inputs,                           ђ
p
ф "3і0,                           ђЦ
6__inference_batch_normalization_2_layer_call_fn_237545kJKLM>б;
4б1
+і(
inputs         ЩЩђ
p 
ф "#і          ЩЩђЦ
6__inference_batch_normalization_2_layer_call_fn_237558kJKLM>б;
4б1
+і(
inputs         ЩЩђ
p
ф "#і          ЩЩђЬ
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_237747ў]^_`NбK
DбA
;і8
inputs,                           ђ
p 
ф "@б=
6і3
0,                           ђ
џ Ь
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_237765ў]^_`NбK
DбA
;і8
inputs,                           ђ
p
ф "@б=
6і3
0,                           ђ
џ ═
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_237783x]^_`>б;
4б1
+і(
inputs         ЭЭђ
p 
ф "0б-
&і#
0         ЭЭђ
џ ═
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_237801x]^_`>б;
4б1
+і(
inputs         ЭЭђ
p
ф "0б-
&і#
0         ЭЭђ
џ к
6__inference_batch_normalization_3_layer_call_fn_237690І]^_`NбK
DбA
;і8
inputs,                           ђ
p 
ф "3і0,                           ђк
6__inference_batch_normalization_3_layer_call_fn_237703І]^_`NбK
DбA
;і8
inputs,                           ђ
p
ф "3і0,                           ђЦ
6__inference_batch_normalization_3_layer_call_fn_237716k]^_`>б;
4б1
+і(
inputs         ЭЭђ
p 
ф "#і          ЭЭђЦ
6__inference_batch_normalization_3_layer_call_fn_237729k]^_`>б;
4б1
+і(
inputs         ЭЭђ
p
ф "#і          ЭЭђЬ
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_237918ўpqrsNбK
DбA
;і8
inputs,                           ђ
p 
ф "@б=
6і3
0,                           ђ
џ Ь
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_237936ўpqrsNбK
DбA
;і8
inputs,                           ђ
p
ф "@б=
6і3
0,                           ђ
џ ═
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_237954xpqrs>б;
4б1
+і(
inputs         ШШђ
p 
ф "0б-
&і#
0         ШШђ
џ ═
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_237972xpqrs>б;
4б1
+і(
inputs         ШШђ
p
ф "0б-
&і#
0         ШШђ
џ к
6__inference_batch_normalization_4_layer_call_fn_237861ІpqrsNбK
DбA
;і8
inputs,                           ђ
p 
ф "3і0,                           ђк
6__inference_batch_normalization_4_layer_call_fn_237874ІpqrsNбK
DбA
;і8
inputs,                           ђ
p
ф "3і0,                           ђЦ
6__inference_batch_normalization_4_layer_call_fn_237887kpqrs>б;
4б1
+і(
inputs         ШШђ
p 
ф "#і          ШШђЦ
6__inference_batch_normalization_4_layer_call_fn_237900kpqrs>б;
4б1
+і(
inputs         ШШђ
p
ф "#і          ШШђы
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_238122ЏђЂѓNбK
DбA
;і8
inputs,                           ђ
p 
ф "@б=
6і3
0,                           ђ
џ ы
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_238140ЏђЂѓNбK
DбA
;і8
inputs,                           ђ
p
ф "@б=
6і3
0,                           ђ
џ л
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_238158{ђЂѓ>б;
4б1
+і(
inputs         ЭЭђ
p 
ф "0б-
&і#
0         ЭЭђ
џ л
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_238176{ђЂѓ>б;
4б1
+і(
inputs         ЭЭђ
p
ф "0б-
&і#
0         ЭЭђ
џ ╔
6__inference_batch_normalization_5_layer_call_fn_238065јђЂѓNбK
DбA
;і8
inputs,                           ђ
p 
ф "3і0,                           ђ╔
6__inference_batch_normalization_5_layer_call_fn_238078јђЂѓNбK
DбA
;і8
inputs,                           ђ
p
ф "3і0,                           ђе
6__inference_batch_normalization_5_layer_call_fn_238091nђЂѓ>б;
4б1
+і(
inputs         ЭЭђ
p 
ф "#і          ЭЭђе
6__inference_batch_normalization_5_layer_call_fn_238104nђЂѓ>б;
4б1
+і(
inputs         ЭЭђ
p
ф "#і          ЭЭђЫ
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_238326юјЈљЉNбK
DбA
;і8
inputs,                           ђ
p 
ф "@б=
6і3
0,                           ђ
џ Ы
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_238344юјЈљЉNбK
DбA
;і8
inputs,                           ђ
p
ф "@б=
6і3
0,                           ђ
џ Л
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_238362|јЈљЉ>б;
4б1
+і(
inputs         ЩЩђ
p 
ф "0б-
&і#
0         ЩЩђ
џ Л
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_238380|јЈљЉ>б;
4б1
+і(
inputs         ЩЩђ
p
ф "0б-
&і#
0         ЩЩђ
џ ╩
6__inference_batch_normalization_6_layer_call_fn_238269ЈјЈљЉNбK
DбA
;і8
inputs,                           ђ
p 
ф "3і0,                           ђ╩
6__inference_batch_normalization_6_layer_call_fn_238282ЈјЈљЉNбK
DбA
;і8
inputs,                           ђ
p
ф "3і0,                           ђЕ
6__inference_batch_normalization_6_layer_call_fn_238295oјЈљЉ>б;
4б1
+і(
inputs         ЩЩђ
p 
ф "#і          ЩЩђЕ
6__inference_batch_normalization_6_layer_call_fn_238308oјЈљЉ>б;
4б1
+і(
inputs         ЩЩђ
p
ф "#і          ЩЩђЫ
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_238530юЮъЪаNбK
DбA
;і8
inputs,                           ђ
p 
ф "@б=
6і3
0,                           ђ
џ Ы
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_238548юЮъЪаNбK
DбA
;і8
inputs,                           ђ
p
ф "@б=
6і3
0,                           ђ
џ Л
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_238566|ЮъЪа>б;
4б1
+і(
inputs         ЧЧђ
p 
ф "0б-
&і#
0         ЧЧђ
џ Л
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_238584|ЮъЪа>б;
4б1
+і(
inputs         ЧЧђ
p
ф "0б-
&і#
0         ЧЧђ
џ ╩
6__inference_batch_normalization_7_layer_call_fn_238473ЈЮъЪаNбK
DбA
;і8
inputs,                           ђ
p 
ф "3і0,                           ђ╩
6__inference_batch_normalization_7_layer_call_fn_238486ЈЮъЪаNбK
DбA
;і8
inputs,                           ђ
p
ф "3і0,                           ђЕ
6__inference_batch_normalization_7_layer_call_fn_238499oЮъЪа>б;
4б1
+і(
inputs         ЧЧђ
p 
ф "#і          ЧЧђЕ
6__inference_batch_normalization_7_layer_call_fn_238512oЮъЪа>б;
4б1
+і(
inputs         ЧЧђ
p
ф "#і          ЧЧђ­
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_238734џгГ«»MбJ
Cб@
:і7
inputs+                           @
p 
ф "?б<
5і2
0+                           @
џ ­
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_238752џгГ«»MбJ
Cб@
:і7
inputs+                           @
p
ф "?б<
5і2
0+                           @
џ ¤
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_238770zгГ«»=б:
3б0
*і'
inputs         ђђ@
p 
ф "/б,
%і"
0         ђђ@
џ ¤
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_238788zгГ«»=б:
3б0
*і'
inputs         ђђ@
p
ф "/б,
%і"
0         ђђ@
џ ╚
6__inference_batch_normalization_8_layer_call_fn_238677ЇгГ«»MбJ
Cб@
:і7
inputs+                           @
p 
ф "2і/+                           @╚
6__inference_batch_normalization_8_layer_call_fn_238690ЇгГ«»MбJ
Cб@
:і7
inputs+                           @
p
ф "2і/+                           @Д
6__inference_batch_normalization_8_layer_call_fn_238703mгГ«»=б:
3б0
*і'
inputs         ђђ@
p 
ф ""і         ђђ@Д
6__inference_batch_normalization_8_layer_call_fn_238716mгГ«»=б:
3б0
*і'
inputs         ђђ@
p
ф ""і         ђђ@Ж
O__inference_batch_normalization_layer_call_and_return_conditional_losses_237234ќ$%&'MбJ
Cб@
:і7
inputs+                           
p 
ф "?б<
5і2
0+                           
џ Ж
O__inference_batch_normalization_layer_call_and_return_conditional_losses_237252ќ$%&'MбJ
Cб@
:і7
inputs+                           
p
ф "?б<
5і2
0+                           
џ ╔
O__inference_batch_normalization_layer_call_and_return_conditional_losses_237270v$%&'=б:
3б0
*і'
inputs         ђђ
p 
ф "/б,
%і"
0         ђђ
џ ╔
O__inference_batch_normalization_layer_call_and_return_conditional_losses_237288v$%&'=б:
3б0
*і'
inputs         ђђ
p
ф "/б,
%і"
0         ђђ
џ ┬
4__inference_batch_normalization_layer_call_fn_237177Ѕ$%&'MбJ
Cб@
:і7
inputs+                           
p 
ф "2і/+                           ┬
4__inference_batch_normalization_layer_call_fn_237190Ѕ$%&'MбJ
Cб@
:і7
inputs+                           
p
ф "2і/+                           А
4__inference_batch_normalization_layer_call_fn_237203i$%&'=б:
3б0
*і'
inputs         ђђ
p 
ф ""і         ђђА
4__inference_batch_normalization_layer_call_fn_237216i$%&'=б:
3б0
*і'
inputs         ђђ
p
ф ""і         ђђ╣
D__inference_conv2d_1_layer_call_and_return_conditional_losses_237506qCD9б6
/б,
*і'
inputs         ЧЧ@
ф "0б-
&і#
0         ЩЩђ
џ Љ
)__inference_conv2d_1_layer_call_fn_237495dCD9б6
/б,
*і'
inputs         ЧЧ@
ф "#і          ЩЩђ║
D__inference_conv2d_2_layer_call_and_return_conditional_losses_237677rVW:б7
0б-
+і(
inputs         ЩЩђ
ф "0б-
&і#
0         ЭЭђ
џ њ
)__inference_conv2d_2_layer_call_fn_237666eVW:б7
0б-
+і(
inputs         ЩЩђ
ф "#і          ЭЭђ║
D__inference_conv2d_3_layer_call_and_return_conditional_losses_237848rij:б7
0б-
+і(
inputs         ЭЭђ
ф "0б-
&і#
0         ШШђ
џ њ
)__inference_conv2d_3_layer_call_fn_237837eij:б7
0б-
+і(
inputs         ЭЭђ
ф "#і          ШШђХ
B__inference_conv2d_layer_call_and_return_conditional_losses_237335p019б6
/б,
*і'
inputs         ђђ
ф "/б,
%і"
0         ЧЧ@
џ ј
'__inference_conv2d_layer_call_fn_237324c019б6
/б,
*і'
inputs         ђђ
ф ""і         ЧЧ@у
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_238232ћЄѕJбG
@б=
;і8
inputs,                           ђ
ф "@б=
6і3
0,                           ђ
џ к
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_238256tЄѕ:б7
0б-
+і(
inputs         ЭЭђ
ф "0б-
&і#
0         ЩЩђ
џ ┐
3__inference_conv2d_transpose_1_layer_call_fn_238185ЄЄѕJбG
@б=
;і8
inputs,                           ђ
ф "3і0,                           ђъ
3__inference_conv2d_transpose_1_layer_call_fn_238194gЄѕ:б7
0б-
+і(
inputs         ЭЭђ
ф "#і          ЩЩђу
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_238436ћќЌJбG
@б=
;і8
inputs,                           ђ
ф "@б=
6і3
0,                           ђ
џ к
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_238460tќЌ:б7
0б-
+і(
inputs         ЩЩђ
ф "0б-
&і#
0         ЧЧђ
џ ┐
3__inference_conv2d_transpose_2_layer_call_fn_238389ЄќЌJбG
@б=
;і8
inputs,                           ђ
ф "3і0,                           ђъ
3__inference_conv2d_transpose_2_layer_call_fn_238398gќЌ:б7
0б-
+і(
inputs         ЩЩђ
ф "#і          ЧЧђТ
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_238640ЊЦдJбG
@б=
;і8
inputs,                           ђ
ф "?б<
5і2
0+                           @
џ ┼
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_238664sЦд:б7
0б-
+і(
inputs         ЧЧђ
ф "/б,
%і"
0         ђђ@
џ Й
3__inference_conv2d_transpose_3_layer_call_fn_238593єЦдJбG
@б=
;і8
inputs,                           ђ
ф "2і/+                           @Ю
3__inference_conv2d_transpose_3_layer_call_fn_238602fЦд:б7
0б-
+і(
inputs         ЧЧђ
ф ""і         ђђ@т
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_238843њ┤хIбF
?б<
:і7
inputs+                           @
ф "?б<
5і2
0+                           
џ ─
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_238866r┤х9б6
/б,
*і'
inputs         ђђ@
ф "/б,
%і"
0         ђђ
џ й
3__inference_conv2d_transpose_4_layer_call_fn_238797Ё┤хIбF
?б<
:і7
inputs+                           @
ф "2і/+                           ю
3__inference_conv2d_transpose_4_layer_call_fn_238806e┤х9б6
/б,
*і'
inputs         ђђ@
ф ""і         ђђс
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_238028њxyJбG
@б=
;і8
inputs,                           ђ
ф "@б=
6і3
0,                           ђ
џ ┬
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_238052rxy:б7
0б-
+і(
inputs         ШШђ
ф "0б-
&і#
0         ЭЭђ
џ ╗
1__inference_conv2d_transpose_layer_call_fn_237981ЁxyJбG
@б=
;і8
inputs,                           ђ
ф "3і0,                           ђџ
1__inference_conv2d_transpose_layer_call_fn_237990exy:б7
0б-
+і(
inputs         ШШђ
ф "#і          ЭЭђ╣
E__inference_dropout_1_layer_call_and_return_conditional_losses_237474p=б:
3б0
*і'
inputs         ЧЧ@
p 
ф "/б,
%і"
0         ЧЧ@
џ ╣
E__inference_dropout_1_layer_call_and_return_conditional_losses_237486p=б:
3б0
*і'
inputs         ЧЧ@
p
ф "/б,
%і"
0         ЧЧ@
џ Љ
*__inference_dropout_1_layer_call_fn_237464c=б:
3б0
*і'
inputs         ЧЧ@
p 
ф ""і         ЧЧ@Љ
*__inference_dropout_1_layer_call_fn_237469c=б:
3б0
*і'
inputs         ЧЧ@
p
ф ""і         ЧЧ@╗
E__inference_dropout_2_layer_call_and_return_conditional_losses_237645r>б;
4б1
+і(
inputs         ЩЩђ
p 
ф "0б-
&і#
0         ЩЩђ
џ ╗
E__inference_dropout_2_layer_call_and_return_conditional_losses_237657r>б;
4б1
+і(
inputs         ЩЩђ
p
ф "0б-
&і#
0         ЩЩђ
џ Њ
*__inference_dropout_2_layer_call_fn_237635e>б;
4б1
+і(
inputs         ЩЩђ
p 
ф "#і          ЩЩђЊ
*__inference_dropout_2_layer_call_fn_237640e>б;
4б1
+і(
inputs         ЩЩђ
p
ф "#і          ЩЩђ╗
E__inference_dropout_3_layer_call_and_return_conditional_losses_237816r>б;
4б1
+і(
inputs         ЭЭђ
p 
ф "0б-
&і#
0         ЭЭђ
џ ╗
E__inference_dropout_3_layer_call_and_return_conditional_losses_237828r>б;
4б1
+і(
inputs         ЭЭђ
p
ф "0б-
&і#
0         ЭЭђ
џ Њ
*__inference_dropout_3_layer_call_fn_237806e>б;
4б1
+і(
inputs         ЭЭђ
p 
ф "#і          ЭЭђЊ
*__inference_dropout_3_layer_call_fn_237811e>б;
4б1
+і(
inputs         ЭЭђ
p
ф "#і          ЭЭђє
C__inference_dropout_layer_call_and_return_conditional_losses_236093ЙM$%&'01789:CDJKLMVW]^_`ijpqrsxyђЂѓЄѕјЈљЉќЌЮъЪаЦдгГ«»┤х<б9
2б/
%і"
u_vel         ђђ
p 

 
ф "/б,
%і"
0         ђђ
џ є
C__inference_dropout_layer_call_and_return_conditional_losses_236228ЙM$%&'01789:CDJKLMVW]^_`ijpqrsxyђЂѓЄѕјЈљЉќЌЮъЪаЦдгГ«»┤х<б9
2б/
%і"
u_vel         ђђ
p

 
ф "/б,
%і"
0         ђђ
џ Є
C__inference_dropout_layer_call_and_return_conditional_losses_236846┐M$%&'01789:CDJKLMVW]^_`ijpqrsxyђЂѓЄѕјЈљЉќЌЮъЪаЦдгГ«»┤х=б:
3б0
&і#
inputs         ђђ
p 

 
ф "/б,
%і"
0         ђђ
џ Є
C__inference_dropout_layer_call_and_return_conditional_losses_237145┐M$%&'01789:CDJKLMVW]^_`ijpqrsxyђЂѓЄѕјЈљЉќЌЮъЪаЦдгГ«»┤х=б:
3б0
&і#
inputs         ђђ
p

 
ф "/б,
%і"
0         ђђ
џ и
C__inference_dropout_layer_call_and_return_conditional_losses_237303p=б:
3б0
*і'
inputs         ђђ
p 
ф "/б,
%і"
0         ђђ
џ и
C__inference_dropout_layer_call_and_return_conditional_losses_237315p=б:
3б0
*і'
inputs         ђђ
p
ф "/б,
%і"
0         ђђ
џ я
(__inference_dropout_layer_call_fn_234905▒M$%&'01789:CDJKLMVW]^_`ijpqrsxyђЂѓЄѕјЈљЉќЌЮъЪаЦдгГ«»┤х<б9
2б/
%і"
u_vel         ђђ
p 

 
ф ""і         ђђя
(__inference_dropout_layer_call_fn_235958▒M$%&'01789:CDJKLMVW]^_`ijpqrsxyђЂѓЄѕјЈљЉќЌЮъЪаЦдгГ«»┤х<б9
2б/
%і"
u_vel         ђђ
p

 
ф ""і         ђђ▀
(__inference_dropout_layer_call_fn_236462▓M$%&'01789:CDJKLMVW]^_`ijpqrsxyђЂѓЄѕјЈљЉќЌЮъЪаЦдгГ«»┤х=б:
3б0
&і#
inputs         ђђ
p 

 
ф ""і         ђђ▀
(__inference_dropout_layer_call_fn_236575▓M$%&'01789:CDJKLMVW]^_`ijpqrsxyђЂѓЄѕјЈљЉќЌЮъЪаЦдгГ«»┤х=б:
3б0
&і#
inputs         ђђ
p

 
ф ""і         ђђЈ
(__inference_dropout_layer_call_fn_237293c=б:
3б0
*і'
inputs         ђђ
p 
ф ""і         ђђЈ
(__inference_dropout_layer_call_fn_237298c=б:
3б0
*і'
inputs         ђђ
p
ф ""і         ђђ»
C__inference_reshape_layer_call_and_return_conditional_losses_237164h5б2
+б(
&і#
inputs         ђђ
ф "/б,
%і"
0         ђђ
џ Є
(__inference_reshape_layer_call_fn_237150[5б2
+б(
&і#
inputs         ђђ
ф ""і         ђђі
$__inference_signature_wrapper_236349рM$%&'01789:CDJKLMVW]^_`ijpqrsxyђЂѓЄѕјЈљЉќЌЮъЪаЦдгГ«»┤х=б:
б 
3ф0
.
u_vel%і"
u_vel         ђђ"QфN
L
conv2d_transpose_46і3
conv2d_transpose_4         ђђ