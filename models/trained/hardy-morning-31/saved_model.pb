╩┼4
ещ
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
┴
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
executor_typestring ѕе
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
 ѕ"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28└▒/
і
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma
Ѓ
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0
ѕ
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta
Ђ
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0
ќ
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean
Ј
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0
ъ
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance
Ќ
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
shape:ђђ*(
shared_nameconv2d_transpose/kernel
Ї
+conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose/kernel*(
_output_shapes
:ђђ*
dtype0
Ѓ
conv2d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameconv2d_transpose/bias
|
)conv2d_transpose/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose/bias*
_output_shapes	
:ђ*
dtype0
Ј
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*,
shared_namebatch_normalization_5/gamma
ѕ
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes	
:ђ*
dtype0
Ї
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*+
shared_namebatch_normalization_5/beta
є
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes	
:ђ*
dtype0
Џ
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!batch_normalization_5/moving_mean
ћ
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes	
:ђ*
dtype0
Б
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*6
shared_name'%batch_normalization_5/moving_variance
ю
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes	
:ђ*
dtype0
ў
conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ**
shared_nameconv2d_transpose_1/kernel
Љ
-conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/kernel*(
_output_shapes
:ђђ*
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
shape:ђђ**
shared_nameconv2d_transpose_2/kernel
Љ
-conv2d_transpose_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/kernel*(
_output_shapes
:ђђ*
dtype0
Є
conv2d_transpose_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*(
shared_nameconv2d_transpose_2/bias
ђ
+conv2d_transpose_2/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/bias*
_output_shapes	
:ђ*
dtype0
Ј
batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*,
shared_namebatch_normalization_7/gamma
ѕ
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes	
:ђ*
dtype0
Ї
batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*+
shared_namebatch_normalization_7/beta
є
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes	
:ђ*
dtype0
Џ
!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!batch_normalization_7/moving_mean
ћ
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes	
:ђ*
dtype0
Б
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*6
shared_name'%batch_normalization_7/moving_variance
ю
9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes	
:ђ*
dtype0
Ќ
conv2d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@└**
shared_nameconv2d_transpose_3/kernel
љ
-conv2d_transpose_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/kernel*'
_output_shapes
:@└*
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
shape:D**
shared_nameconv2d_transpose_4/kernel
Ј
-conv2d_transpose_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_4/kernel*&
_output_shapes
:D*
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
shape:*1
shared_name" Adam/batch_normalization/gamma/m
Љ
4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes
:*
dtype0
ќ
Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/batch_normalization/beta/m
Ј
3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes
:*
dtype0
ї
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d/kernel/m
Ё
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
shape:ђђ*/
shared_name Adam/conv2d_transpose/kernel/m
Џ
2Adam/conv2d_transpose/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/kernel/m*(
_output_shapes
:ђђ*
dtype0
Љ
Adam/conv2d_transpose/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*-
shared_nameAdam/conv2d_transpose/bias/m
і
0Adam/conv2d_transpose/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/bias/m*
_output_shapes	
:ђ*
dtype0
Ю
"Adam/batch_normalization_5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*3
shared_name$"Adam/batch_normalization_5/gamma/m
ќ
6Adam/batch_normalization_5/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_5/gamma/m*
_output_shapes	
:ђ*
dtype0
Џ
!Adam/batch_normalization_5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!Adam/batch_normalization_5/beta/m
ћ
5Adam/batch_normalization_5/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_5/beta/m*
_output_shapes	
:ђ*
dtype0
д
 Adam/conv2d_transpose_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*1
shared_name" Adam/conv2d_transpose_1/kernel/m
Ъ
4Adam/conv2d_transpose_1/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_1/kernel/m*(
_output_shapes
:ђђ*
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
shape:ђђ*1
shared_name" Adam/conv2d_transpose_2/kernel/m
Ъ
4Adam/conv2d_transpose_2/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_2/kernel/m*(
_output_shapes
:ђђ*
dtype0
Ћ
Adam/conv2d_transpose_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*/
shared_name Adam/conv2d_transpose_2/bias/m
ј
2Adam/conv2d_transpose_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_2/bias/m*
_output_shapes	
:ђ*
dtype0
Ю
"Adam/batch_normalization_7/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*3
shared_name$"Adam/batch_normalization_7/gamma/m
ќ
6Adam/batch_normalization_7/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/m*
_output_shapes	
:ђ*
dtype0
Џ
!Adam/batch_normalization_7/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!Adam/batch_normalization_7/beta/m
ћ
5Adam/batch_normalization_7/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/m*
_output_shapes	
:ђ*
dtype0
Ц
 Adam/conv2d_transpose_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@└*1
shared_name" Adam/conv2d_transpose_3/kernel/m
ъ
4Adam/conv2d_transpose_3/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_3/kernel/m*'
_output_shapes
:@└*
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
shape:D*1
shared_name" Adam/conv2d_transpose_4/kernel/m
Ю
4Adam/conv2d_transpose_4/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_4/kernel/m*&
_output_shapes
:D*
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
shape:*1
shared_name" Adam/batch_normalization/gamma/v
Љ
4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes
:*
dtype0
ќ
Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/batch_normalization/beta/v
Ј
3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes
:*
dtype0
ї
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d/kernel/v
Ё
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
shape:ђђ*/
shared_name Adam/conv2d_transpose/kernel/v
Џ
2Adam/conv2d_transpose/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/kernel/v*(
_output_shapes
:ђђ*
dtype0
Љ
Adam/conv2d_transpose/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*-
shared_nameAdam/conv2d_transpose/bias/v
і
0Adam/conv2d_transpose/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/bias/v*
_output_shapes	
:ђ*
dtype0
Ю
"Adam/batch_normalization_5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*3
shared_name$"Adam/batch_normalization_5/gamma/v
ќ
6Adam/batch_normalization_5/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_5/gamma/v*
_output_shapes	
:ђ*
dtype0
Џ
!Adam/batch_normalization_5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!Adam/batch_normalization_5/beta/v
ћ
5Adam/batch_normalization_5/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_5/beta/v*
_output_shapes	
:ђ*
dtype0
д
 Adam/conv2d_transpose_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*1
shared_name" Adam/conv2d_transpose_1/kernel/v
Ъ
4Adam/conv2d_transpose_1/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_1/kernel/v*(
_output_shapes
:ђђ*
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
shape:ђђ*1
shared_name" Adam/conv2d_transpose_2/kernel/v
Ъ
4Adam/conv2d_transpose_2/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_2/kernel/v*(
_output_shapes
:ђђ*
dtype0
Ћ
Adam/conv2d_transpose_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*/
shared_name Adam/conv2d_transpose_2/bias/v
ј
2Adam/conv2d_transpose_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_2/bias/v*
_output_shapes	
:ђ*
dtype0
Ю
"Adam/batch_normalization_7/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*3
shared_name$"Adam/batch_normalization_7/gamma/v
ќ
6Adam/batch_normalization_7/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/v*
_output_shapes	
:ђ*
dtype0
Џ
!Adam/batch_normalization_7/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!Adam/batch_normalization_7/beta/v
ћ
5Adam/batch_normalization_7/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/v*
_output_shapes	
:ђ*
dtype0
Ц
 Adam/conv2d_transpose_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@└*1
shared_name" Adam/conv2d_transpose_3/kernel/v
ъ
4Adam/conv2d_transpose_3/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_3/kernel/v*'
_output_shapes
:@└*
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
shape:D*1
shared_name" Adam/conv2d_transpose_4/kernel/v
Ю
4Adam/conv2d_transpose_4/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_4/kernel/v*&
_output_shapes
:D*
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
єС
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*└с
valueхсB▒с BЕс
Ѓ
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-0

layer-9
layer_with_weights-1
layer-10
layer_with_weights-2
layer-11
layer_with_weights-3
layer-12
layer_with_weights-4
layer-13
layer_with_weights-5
layer-14
layer_with_weights-6
layer-15
layer_with_weights-7
layer-16
layer_with_weights-8
layer-17
layer-18
layer_with_weights-9
layer-19
layer_with_weights-10
layer-20
layer-21
layer_with_weights-11
layer-22
layer_with_weights-12
layer-23
layer-24
layer_with_weights-13
layer-25
layer_with_weights-14
layer-26
layer-27
layer_with_weights-15
layer-28
layer_with_weights-16
layer-29
layer-30
 layer_with_weights-17
 layer-31
!	optimizer
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&
signatures
 
 
 
 
R
'	variables
(trainable_variables
)regularization_losses
*	keras_api
R
+	variables
,trainable_variables
-regularization_losses
.	keras_api
R
/	variables
0trainable_variables
1regularization_losses
2	keras_api
R
3	variables
4trainable_variables
5regularization_losses
6	keras_api
R
7	variables
8trainable_variables
9regularization_losses
:	keras_api
Ќ
;axis
	<gamma
=beta
>moving_mean
?moving_variance
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
h

Dkernel
Ebias
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
Ќ
Jaxis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
h

Skernel
Tbias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Ќ
Yaxis
	Zgamma
[beta
\moving_mean
]moving_variance
^	variables
_trainable_variables
`regularization_losses
a	keras_api
h

bkernel
cbias
d	variables
etrainable_variables
fregularization_losses
g	keras_api
Ќ
haxis
	igamma
jbeta
kmoving_mean
lmoving_variance
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
h

qkernel
rbias
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
Ќ
waxis
	xgamma
ybeta
zmoving_mean
{moving_variance
|	variables
}trainable_variables
~regularization_losses
	keras_api
V
ђ	variables
Ђtrainable_variables
ѓregularization_losses
Ѓ	keras_api
n
ёkernel
	Ёbias
є	variables
Єtrainable_variables
ѕregularization_losses
Ѕ	keras_api
а
	іaxis

Іgamma
	їbeta
Їmoving_mean
јmoving_variance
Ј	variables
љtrainable_variables
Љregularization_losses
њ	keras_api
V
Њ	variables
ћtrainable_variables
Ћregularization_losses
ќ	keras_api
n
Ќkernel
	ўbias
Ў	variables
џtrainable_variables
Џregularization_losses
ю	keras_api
а
	Юaxis

ъgamma
	Ъbeta
аmoving_mean
Аmoving_variance
б	variables
Бtrainable_variables
цregularization_losses
Ц	keras_api
V
д	variables
Дtrainable_variables
еregularization_losses
Е	keras_api
n
фkernel
	Фbias
г	variables
Гtrainable_variables
«regularization_losses
»	keras_api
а
	░axis

▒gamma
	▓beta
│moving_mean
┤moving_variance
х	variables
Хtrainable_variables
иregularization_losses
И	keras_api
V
╣	variables
║trainable_variables
╗regularization_losses
╝	keras_api
n
йkernel
	Йbias
┐	variables
└trainable_variables
┴regularization_losses
┬	keras_api
а
	├axis

─gamma
	┼beta
кmoving_mean
Кmoving_variance
╚	variables
╔trainable_variables
╩regularization_losses
╦	keras_api
V
╠	variables
═trainable_variables
╬regularization_losses
¤	keras_api
n
лkernel
	Лbias
м	variables
Мtrainable_variables
нregularization_losses
Н	keras_api
╣
	оiter
Оbeta_1
пbeta_2

┘decay
┌learning_rate<mы=mЫDmзEmЗKmшLmШSmэTmЭZmщ[mЩbmчcmЧim§jm■qm rmђxmЂymѓ	ёmЃ	Ёmё	ІmЁ	їmє	ЌmЄ	ўmѕ	ъmЅ	Ъmі	фmІ	Фmї	▒mЇ	▓mј	йmЈ	Йmљ	─mЉ	┼mњ	лmЊ	Лmћ<vЋ=vќDvЌEvўKvЎLvџSvЏTvюZvЮ[vъbvЪcvаivАjvбqvБrvцxvЦyvд	ёvД	Ёvе	ІvЕ	їvф	ЌvФ	ўvг	ъvГ	Ъv«	фv»	Фv░	▒v▒	▓v▓	йv│	Йv┤	─vх	┼vХ	лvи	ЛvИ
└
<0
=1
>2
?3
D4
E5
K6
L7
M8
N9
S10
T11
Z12
[13
\14
]15
b16
c17
i18
j19
k20
l21
q22
r23
x24
y25
z26
{27
ё28
Ё29
І30
ї31
Ї32
ј33
Ќ34
ў35
ъ36
Ъ37
а38
А39
ф40
Ф41
▒42
▓43
│44
┤45
й46
Й47
─48
┼49
к50
К51
л52
Л53
е
<0
=1
D2
E3
K4
L5
S6
T7
Z8
[9
b10
c11
i12
j13
q14
r15
x16
y17
ё18
Ё19
І20
ї21
Ќ22
ў23
ъ24
Ъ25
ф26
Ф27
▒28
▓29
й30
Й31
─32
┼33
л34
Л35
 
▓
█non_trainable_variables
▄layers
Пmetrics
 яlayer_regularization_losses
▀layer_metrics
"	variables
#trainable_variables
$regularization_losses
 
 
 
 
▓
Яnon_trainable_variables
рlayers
Рmetrics
 сlayer_regularization_losses
Сlayer_metrics
'	variables
(trainable_variables
)regularization_losses
 
 
 
▓
тnon_trainable_variables
Тlayers
уmetrics
 Уlayer_regularization_losses
жlayer_metrics
+	variables
,trainable_variables
-regularization_losses
 
 
 
▓
Жnon_trainable_variables
вlayers
Вmetrics
 ьlayer_regularization_losses
Ьlayer_metrics
/	variables
0trainable_variables
1regularization_losses
 
 
 
▓
№non_trainable_variables
­layers
ыmetrics
 Ыlayer_regularization_losses
зlayer_metrics
3	variables
4trainable_variables
5regularization_losses
 
 
 
▓
Зnon_trainable_variables
шlayers
Шmetrics
 эlayer_regularization_losses
Эlayer_metrics
7	variables
8trainable_variables
9regularization_losses
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

<0
=1
>2
?3

<0
=1
 
▓
щnon_trainable_variables
Щlayers
чmetrics
 Чlayer_regularization_losses
§layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

D0
E1

D0
E1
 
▓
■non_trainable_variables
 layers
ђmetrics
 Ђlayer_regularization_losses
ѓlayer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

K0
L1
M2
N3

K0
L1
 
▓
Ѓnon_trainable_variables
ёlayers
Ёmetrics
 єlayer_regularization_losses
Єlayer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

S0
T1

S0
T1
 
▓
ѕnon_trainable_variables
Ѕlayers
іmetrics
 Іlayer_regularization_losses
їlayer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

Z0
[1
\2
]3

Z0
[1
 
▓
Їnon_trainable_variables
јlayers
Јmetrics
 љlayer_regularization_losses
Љlayer_metrics
^	variables
_trainable_variables
`regularization_losses
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

b0
c1

b0
c1
 
▓
њnon_trainable_variables
Њlayers
ћmetrics
 Ћlayer_regularization_losses
ќlayer_metrics
d	variables
etrainable_variables
fregularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

i0
j1
k2
l3

i0
j1
 
▓
Ќnon_trainable_variables
ўlayers
Ўmetrics
 џlayer_regularization_losses
Џlayer_metrics
m	variables
ntrainable_variables
oregularization_losses
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

q0
r1

q0
r1
 
▓
юnon_trainable_variables
Юlayers
ъmetrics
 Ъlayer_regularization_losses
аlayer_metrics
s	variables
ttrainable_variables
uregularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

x0
y1
z2
{3

x0
y1
 
▓
Аnon_trainable_variables
бlayers
Бmetrics
 цlayer_regularization_losses
Цlayer_metrics
|	variables
}trainable_variables
~regularization_losses
 
 
 
х
дnon_trainable_variables
Дlayers
еmetrics
 Еlayer_regularization_losses
фlayer_metrics
ђ	variables
Ђtrainable_variables
ѓregularization_losses
ca
VARIABLE_VALUEconv2d_transpose/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_transpose/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

ё0
Ё1

ё0
Ё1
 
х
Фnon_trainable_variables
гlayers
Гmetrics
 «layer_regularization_losses
»layer_metrics
є	variables
Єtrainable_variables
ѕregularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_5/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_5/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_5/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_5/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
І0
ї1
Ї2
ј3

І0
ї1
 
х
░non_trainable_variables
▒layers
▓metrics
 │layer_regularization_losses
┤layer_metrics
Ј	variables
љtrainable_variables
Љregularization_losses
 
 
 
х
хnon_trainable_variables
Хlayers
иmetrics
 Иlayer_regularization_losses
╣layer_metrics
Њ	variables
ћtrainable_variables
Ћregularization_losses
fd
VARIABLE_VALUEconv2d_transpose_1/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_1/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

Ќ0
ў1

Ќ0
ў1
 
х
║non_trainable_variables
╗layers
╝metrics
 йlayer_regularization_losses
Йlayer_metrics
Ў	variables
џtrainable_variables
Џregularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_6/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_6/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_6/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_6/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
ъ0
Ъ1
а2
А3

ъ0
Ъ1
 
х
┐non_trainable_variables
└layers
┴metrics
 ┬layer_regularization_losses
├layer_metrics
б	variables
Бtrainable_variables
цregularization_losses
 
 
 
х
─non_trainable_variables
┼layers
кmetrics
 Кlayer_regularization_losses
╚layer_metrics
д	variables
Дtrainable_variables
еregularization_losses
fd
VARIABLE_VALUEconv2d_transpose_2/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_2/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE

ф0
Ф1

ф0
Ф1
 
х
╔non_trainable_variables
╩layers
╦metrics
 ╠layer_regularization_losses
═layer_metrics
г	variables
Гtrainable_variables
«regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_7/gamma6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_7/beta5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_7/moving_mean<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_7/moving_variance@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
▒0
▓1
│2
┤3

▒0
▓1
 
х
╬non_trainable_variables
¤layers
лmetrics
 Лlayer_regularization_losses
мlayer_metrics
х	variables
Хtrainable_variables
иregularization_losses
 
 
 
х
Мnon_trainable_variables
нlayers
Нmetrics
 оlayer_regularization_losses
Оlayer_metrics
╣	variables
║trainable_variables
╗regularization_losses
fd
VARIABLE_VALUEconv2d_transpose_3/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_3/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE

й0
Й1

й0
Й1
 
х
пnon_trainable_variables
┘layers
┌metrics
 █layer_regularization_losses
▄layer_metrics
┐	variables
└trainable_variables
┴regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_8/gamma6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_8/beta5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_8/moving_mean<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_8/moving_variance@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
─0
┼1
к2
К3

─0
┼1
 
х
Пnon_trainable_variables
яlayers
▀metrics
 Яlayer_regularization_losses
рlayer_metrics
╚	variables
╔trainable_variables
╩regularization_losses
 
 
 
х
Рnon_trainable_variables
сlayers
Сmetrics
 тlayer_regularization_losses
Тlayer_metrics
╠	variables
═trainable_variables
╬regularization_losses
fd
VARIABLE_VALUEconv2d_transpose_4/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_4/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE

л0
Л1

л0
Л1
 
х
уnon_trainable_variables
Уlayers
жmetrics
 Жlayer_regularization_losses
вlayer_metrics
м	variables
Мtrainable_variables
нregularization_losses
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
ј
>0
?1
M2
N3
\4
]5
k6
l7
z8
{9
Ї10
ј11
а12
А13
│14
┤15
к16
К17
Ш
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

В0
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
>0
?1
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
M0
N1
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
\0
]1
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
k0
l1
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
z0
{1
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
Ї0
ј1
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
а0
А1
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
│0
┤1
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
к0
К1
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

ьtotal

Ьcount
№	variables
­	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

ь0
Ь1

№	variables
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
serving_default_pr0.2Placeholder*-
_output_shapes
:         ђђ*
dtype0*"
shape:         ђђ
ё
serving_default_u_velPlaceholder*-
_output_shapes
:         ђђ*
dtype0*"
shape:         ђђ
ё
serving_default_v_velPlaceholder*-
_output_shapes
:         ђђ*
dtype0*"
shape:         ђђ
ё
serving_default_w_velPlaceholder*-
_output_shapes
:         ђђ*
dtype0*"
shape:         ђђ
Ђ
StatefulPartitionedCallStatefulPartitionedCallserving_default_pr0.2serving_default_u_velserving_default_v_velserving_default_w_velbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d/kernelconv2d/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_3/kernelconv2d_3/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceconv2d_transpose/kernelconv2d_transpose/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_transpose_1/kernelconv2d_transpose_1/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv2d_transpose_2/kernelconv2d_transpose_2/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_transpose_3/kernelconv2d_transpose_3/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv2d_transpose_4/kernelconv2d_transpose_4/bias*E
Tin>
<2:*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456789*0
config_proto 

CPU

GPU2*0J 8ѓ *.
f)R'
%__inference_signature_wrapper_1032760
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
п6
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
GPU2*0J 8ѓ *)
f$R"
 __inference__traced_save_1035793
№!
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
GPU2*0J 8ѓ *,
f'R%
#__inference__traced_restore_1036202Э▒*
█
┴
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1029907

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
╠
Е
4__inference_conv2d_transpose_4_layer_call_fn_1035299

inputs!
unknown:D
	unknown_0:
identityѕбStatefulPartitionedCallЂ
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
GPU2*0J 8ѓ *X
fSRQ
O__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1030603Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           D: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           D
 
_user_specified_nameinputs
Ї
Ю
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1030759

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0И
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЧЧ@:@:@:@:@:*
epsilon%oЃ:*
is_training( m
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:         ЧЧ@░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
ы#
Ю
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1030491

inputsC
(conv2d_transpose_readvariableop_resource:@└-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
valueB:┘
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
valueB:┘
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
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЉ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@└*
dtype0П
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           @*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ў
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @h
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           @z
IdentityIdentityElu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                           @Ђ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           └: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,                           └
 
_user_specified_nameinputs
Ђ
Љ
'__inference_model_layer_call_fn_1032992
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

unknown_27:ђђ

unknown_28:	ђ

unknown_29:	ђ

unknown_30:	ђ

unknown_31:	ђ

unknown_32:	ђ&

unknown_33:ђђ

unknown_34:	ђ

unknown_35:	ђ

unknown_36:	ђ

unknown_37:	ђ

unknown_38:	ђ&

unknown_39:ђђ

unknown_40:	ђ

unknown_41:	ђ

unknown_42:	ђ

unknown_43:	ђ

unknown_44:	ђ%

unknown_45:@└

unknown_46:@

unknown_47:@

unknown_48:@

unknown_49:@

unknown_50:@$

unknown_51:D

unknown_52:
identityѕбStatefulPartitionedCall─
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
:         ђђ*F
_read_only_resource_inputs(
&$	
 !"#&'(),-./234589*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1032123y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesМ
л:         ђђ:         ђђ:         ђђ:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
-
_output_shapes
:         ђђ
"
_user_specified_name
inputs/0:WS
-
_output_shapes
:         ђђ
"
_user_specified_name
inputs/1:WS
-
_output_shapes
:         ђђ
"
_user_specified_name
inputs/2:WS
-
_output_shapes
:         ђђ
"
_user_specified_name
inputs/3
П
о
7__inference_batch_normalization_6_layer_call_fn_1034758

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЈ
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1031021z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЩЩђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs
ї
Ч
C__inference_conv2d_layer_call_and_return_conditional_losses_1030736

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0ю
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЧЧ@*
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
:         ЧЧ@X
EluEluBiasAdd:output:0*
T0*1
_output_shapes
:         ЧЧ@j
IdentityIdentityElu:activations:0^NoOp*
T0*1
_output_shapes
:         ЧЧ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ђђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
Ч
t
J__inference_concatenate_4_layer_call_and_return_conditional_losses_1031103

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ђ
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*2
_output_shapes 
:         ЧЧ└b
IdentityIdentityconcat:output:0*
T0*2
_output_shapes 
:         ЧЧ└"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         ЧЧ@:         ЧЧђ:Y U
1
_output_shapes
:         ЧЧ@
 
_user_specified_nameinputs:ZV
2
_output_shapes 
:         ЧЧђ
 
_user_specified_nameinputs
▀
b
F__inference_reshape_2_layer_call_and_return_conditional_losses_1030669

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
valueB:Л
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
B :ђR
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ђQ
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Е
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:         ђђb
IdentityIdentityReshape:output:0*
T0*1
_output_shapes
:         ђђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ђђ:U Q
-
_output_shapes
:         ђђ
 
_user_specified_nameinputs
є
v
J__inference_concatenate_2_layer_call_and_return_conditional_losses_1034639
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ѓ
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*2
_output_shapes 
:         ЭЭђb
IdentityIdentityconcat:output:0*
T0*2
_output_shapes 
:         ЭЭђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:         ЭЭђ:         ЭЭђ:\ X
2
_output_shapes 
:         ЭЭђ
"
_user_specified_name
inputs/0:\X
2
_output_shapes 
:         ЭЭђ
"
_user_specified_name
inputs/1
Єў
а
B__inference_model_layer_call_and_return_conditional_losses_1031199

inputs
inputs_1
inputs_2
inputs_3)
batch_normalization_1030716:)
batch_normalization_1030718:)
batch_normalization_1030720:)
batch_normalization_1030722:(
conv2d_1030737:@
conv2d_1030739:@+
batch_normalization_1_1030760:@+
batch_normalization_1_1030762:@+
batch_normalization_1_1030764:@+
batch_normalization_1_1030766:@+
conv2d_1_1030781:@ђ
conv2d_1_1030783:	ђ,
batch_normalization_2_1030804:	ђ,
batch_normalization_2_1030806:	ђ,
batch_normalization_2_1030808:	ђ,
batch_normalization_2_1030810:	ђ,
conv2d_2_1030825:ђђ
conv2d_2_1030827:	ђ,
batch_normalization_3_1030848:	ђ,
batch_normalization_3_1030850:	ђ,
batch_normalization_3_1030852:	ђ,
batch_normalization_3_1030854:	ђ,
conv2d_3_1030869:ђђ
conv2d_3_1030871:	ђ,
batch_normalization_4_1030892:	ђ,
batch_normalization_4_1030894:	ђ,
batch_normalization_4_1030896:	ђ,
batch_normalization_4_1030898:	ђ4
conv2d_transpose_1030934:ђђ'
conv2d_transpose_1030936:	ђ,
batch_normalization_5_1030957:	ђ,
batch_normalization_5_1030959:	ђ,
batch_normalization_5_1030961:	ђ,
batch_normalization_5_1030963:	ђ6
conv2d_transpose_1_1030999:ђђ)
conv2d_transpose_1_1031001:	ђ,
batch_normalization_6_1031022:	ђ,
batch_normalization_6_1031024:	ђ,
batch_normalization_6_1031026:	ђ,
batch_normalization_6_1031028:	ђ6
conv2d_transpose_2_1031064:ђђ)
conv2d_transpose_2_1031066:	ђ,
batch_normalization_7_1031087:	ђ,
batch_normalization_7_1031089:	ђ,
batch_normalization_7_1031091:	ђ,
batch_normalization_7_1031093:	ђ5
conv2d_transpose_3_1031129:@└(
conv2d_transpose_3_1031131:@+
batch_normalization_8_1031152:@+
batch_normalization_8_1031154:@+
batch_normalization_8_1031156:@+
batch_normalization_8_1031158:@4
conv2d_transpose_4_1031193:D(
conv2d_transpose_4_1031195:
identityѕб+batch_normalization/StatefulPartitionedCallб-batch_normalization_1/StatefulPartitionedCallб-batch_normalization_2/StatefulPartitionedCallб-batch_normalization_3/StatefulPartitionedCallб-batch_normalization_4/StatefulPartitionedCallб-batch_normalization_5/StatefulPartitionedCallб-batch_normalization_6/StatefulPartitionedCallб-batch_normalization_7/StatefulPartitionedCallб-batch_normalization_8/StatefulPartitionedCallбconv2d/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallб conv2d_2/StatefulPartitionedCallб conv2d_3/StatefulPartitionedCallб(conv2d_transpose/StatefulPartitionedCallб*conv2d_transpose_1/StatefulPartitionedCallб*conv2d_transpose_2/StatefulPartitionedCallб*conv2d_transpose_3/StatefulPartitionedCallб*conv2d_transpose_4/StatefulPartitionedCall─
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
GPU2*0J 8ѓ *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_1030637╩
reshape_1/PartitionedCallPartitionedCallinputs_1*
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
GPU2*0J 8ѓ *O
fJRH
F__inference_reshape_1_layer_call_and_return_conditional_losses_1030653╩
reshape_2/PartitionedCallPartitionedCallinputs_2*
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
GPU2*0J 8ѓ *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_1030669╩
reshape_3/PartitionedCallPartitionedCallinputs_3*
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
GPU2*0J 8ѓ *O
fJRH
F__inference_reshape_3_layer_call_and_return_conditional_losses_1030685Н
concatenate/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0"reshape_1/PartitionedCall:output:0"reshape_2/PartitionedCall:output:0"reshape_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_1030696ѕ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0batch_normalization_1030716batch_normalization_1030718batch_normalization_1030720batch_normalization_1030722*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1030715д
conv2d/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_1030737conv2d_1030739*
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
GPU2*0J 8ѓ *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_1030736Ќ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_1_1030760batch_normalization_1_1030762batch_normalization_1_1030764batch_normalization_1_1030766*
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1030759▒
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_1_1030781conv2d_1_1030783*
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
GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1030780џ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_2_1030804batch_normalization_2_1030806batch_normalization_2_1030808batch_normalization_2_1030810*
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1030803▒
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_2_1030825conv2d_2_1030827*
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
GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1030824џ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_3_1030848batch_normalization_3_1030850batch_normalization_3_1030852batch_normalization_3_1030854*
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1030847▒
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_3_1030869conv2d_3_1030871*
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
GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1030868џ
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_4_1030892batch_normalization_4_1030894batch_normalization_4_1030896batch_normalization_4_1030898*
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1030891Г
concatenate_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:06batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ШШђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1030908┴
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0conv2d_transpose_1030934conv2d_transpose_1030936*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1030933б
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0batch_normalization_5_1030957batch_normalization_5_1030959batch_normalization_5_1030961batch_normalization_5_1030963*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1030956Г
concatenate_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:06batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_1030973╔
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0conv2d_transpose_1_1030999conv2d_transpose_1_1031001*
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
GPU2*0J 8ѓ *X
fSRQ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1030998ц
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0batch_normalization_6_1031022batch_normalization_6_1031024batch_normalization_6_1031026batch_normalization_6_1031028*
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1031021Г
concatenate_3/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:06batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЩЩђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_concatenate_3_layer_call_and_return_conditional_losses_1031038╔
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0conv2d_transpose_2_1031064conv2d_transpose_2_1031066*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЧЧђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1031063ц
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0batch_normalization_7_1031087batch_normalization_7_1031089batch_normalization_7_1031091batch_normalization_7_1031093*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЧЧђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1031086Ф
concatenate_4/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:06batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЧЧ└* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_concatenate_4_layer_call_and_return_conditional_losses_1031103╚
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv2d_transpose_3_1031129conv2d_transpose_3_1031131*
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
GPU2*0J 8ѓ *X
fSRQ
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1031128Б
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0batch_normalization_8_1031152batch_normalization_8_1031154batch_normalization_8_1031156batch_normalization_8_1031158*
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1031151Д
concatenate_5/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:06batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђD* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_concatenate_5_layer_call_and_return_conditional_losses_1031168╚
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0conv2d_transpose_4_1031193conv2d_transpose_4_1031195*
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
GPU2*0J 8ѓ *X
fSRQ
O__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1031192ї
IdentityIdentity3conv2d_transpose_4/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђП
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesМ
л:         ђђ:         ђђ:         ђђ:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
 
_user_specified_nameinputs:UQ
-
_output_shapes
:         ђђ
 
_user_specified_nameinputs:UQ
-
_output_shapes
:         ђђ
 
_user_specified_nameinputs:UQ
-
_output_shapes
:         ђђ
 
_user_specified_nameinputs
▀
b
F__inference_reshape_1_layer_call_and_return_conditional_losses_1033654

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
valueB:Л
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
B :ђR
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ђQ
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Е
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:         ђђb
IdentityIdentityReshape:output:0*
T0*1
_output_shapes
:         ђђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ђђ:U Q
-
_output_shapes
:         ђђ
 
_user_specified_nameinputs
Щ#
Ъ
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1030378

inputsD
(conv2d_transpose_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
valueB:┘
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
valueB:┘
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
B :ђy
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
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskњ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0я
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,                           ђ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0џ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђi
EluEluBiasAdd:output:0*
T0*B
_output_shapes0
.:,                           ђ{
IdentityIdentityElu:activations:0^NoOp*
T0*B
_output_shapes0
.:,                           ђЂ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
І
Џ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1033815

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0И
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ:::::*
epsilon%oЃ:*
is_training( m
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:         ђђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         ђђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
└
E
)__inference_reshape_layer_call_fn_1033621

inputs
identity╝
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
GPU2*0J 8ѓ *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_1030637j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ђђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ђђ:U Q
-
_output_shapes
:         ђђ
 
_user_specified_nameinputs
П
о
7__inference_batch_normalization_3_layer_call_fn_1034180

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЈ
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1030847z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЭЭђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
ю	
о
7__inference_batch_normalization_2_layer_call_fn_1034023

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЮ
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1029971і
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
ъ	
о
7__inference_batch_normalization_3_layer_call_fn_1034154

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЪ
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1030004і
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
М
м
7__inference_batch_normalization_8_layer_call_fn_1035205

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallї
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1031359y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         ђђ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ђђ@
 
_user_specified_nameinputs
Ю
А
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1030847

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0й
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЭЭђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
Э#
Ю
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1030152

inputsD
(conv2d_transpose_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
valueB:┘
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
valueB:┘
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
B :ђy
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
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskњ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0я
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,                           ђ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0џ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђi
EluEluBiasAdd:output:0*
T0*B
_output_shapes0
.:,                           ђ{
IdentityIdentityElu:activations:0^NoOp*
T0*B
_output_shapes0
.:,                           ђЂ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
в
┼
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1030212

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
Ю
А
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1034391

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0й
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ШШђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ШШђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
Џ
Ђ
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1030868

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0Ю
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ШШђ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ђ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ШШђY
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:         ШШђk
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:         ШШђw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
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
№
Ё
'__inference_model_layer_call_fn_1031310	
u_vel	
v_vel	
w_vel	
pr0_2
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

unknown_27:ђђ

unknown_28:	ђ

unknown_29:	ђ

unknown_30:	ђ

unknown_31:	ђ

unknown_32:	ђ&

unknown_33:ђђ

unknown_34:	ђ

unknown_35:	ђ

unknown_36:	ђ

unknown_37:	ђ

unknown_38:	ђ&

unknown_39:ђђ

unknown_40:	ђ

unknown_41:	ђ

unknown_42:	ђ

unknown_43:	ђ

unknown_44:	ђ%

unknown_45:@└

unknown_46:@

unknown_47:@

unknown_48:@

unknown_49:@

unknown_50:@$

unknown_51:D

unknown_52:
identityѕбStatefulPartitionedCall╩
StatefulPartitionedCallStatefulPartitionedCallu_velv_velw_velpr0_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:         ђђ*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456789*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1031199y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesМ
л:         ђђ:         ђђ:         ђђ:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
-
_output_shapes
:         ђђ

_user_specified_nameu_vel:TP
-
_output_shapes
:         ђђ

_user_specified_namev_vel:TP
-
_output_shapes
:         ђђ

_user_specified_namew_vel:TP
-
_output_shapes
:         ђђ

_user_specified_namepr0.2
М
г
4__inference_conv2d_transpose_1_layer_call_fn_1034648

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallѓ
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
GPU2*0J 8ѓ *X
fSRQ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1030265і
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
╦
Џ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1033779

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
з
[
/__inference_concatenate_3_layer_call_fn_1034849
inputs_0
inputs_1
identityл
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЩЩђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_concatenate_3_layer_call_and_return_conditional_losses_1031038k
IdentityIdentityPartitionedCall:output:0*
T0*2
_output_shapes 
:         ЩЩђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:         ЩЩђ:         ЩЩђ:\ X
2
_output_shapes 
:         ЩЩђ
"
_user_specified_name
inputs/0:\X
2
_output_shapes 
:         ЩЩђ
"
_user_specified_name
inputs/1
в
┼
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1035024

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
─
G
+__inference_reshape_1_layer_call_fn_1033640

inputs
identityЙ
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
GPU2*0J 8ѓ *O
fJRH
F__inference_reshape_1_layer_call_and_return_conditional_losses_1030653j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ђђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ђђ:U Q
-
_output_shapes
:         ђђ
 
_user_specified_nameinputs
П
о
7__inference_batch_normalization_4_layer_call_fn_1034324

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЈ
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1030891z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ШШђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ШШђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ШШђ
 
_user_specified_nameinputs
¤
л
5__inference_batch_normalization_layer_call_fn_1033761

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityѕбStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1031815y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         ђђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
ю	
о
7__inference_batch_normalization_5_layer_call_fn_1034528

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЮ
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1030212і
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
ў
ј
H__inference_concatenate_layer_call_and_return_conditional_losses_1030696

inputs
inputs_1
inputs_2
inputs_3
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Њ
concatConcatV2inputsinputs_1inputs_2inputs_3concat/axis:output:0*
N*
T0*1
_output_shapes
:         ђђa
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:         ђђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Є
_input_shapesv
t:         ђђ:         ђђ:         ђђ:         ђђ:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs:YU
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs:YU
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs:YU
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
Ќ
ђ
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1030780

inputs9
conv2d_readvariableop_resource:@ђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype0Ю
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЩЩђ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ђ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЩЩђY
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:         ЩЩђk
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
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
Ю
А
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1030891

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0й
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ШШђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ШШђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
Ю
А
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1034247

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0й
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЭЭђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
¤
ф
4__inference_conv2d_transpose_3_layer_call_fn_1035082

inputs"
unknown:@└
	unknown_0:@
identityѕбStatefulPartitionedCallЂ
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
GPU2*0J 8ѓ *X
fSRQ
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1030491Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           └: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           └
 
_user_specified_nameinputs
═
Ъ
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1034936

inputsD
(conv2d_transpose_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
B :ЧJ
stack/2Const*
_output_shapes
: *
dtype0*
value
B :ЧJ
stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђЅ
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
valueB:┘
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskњ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0╬
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*2
_output_shapes 
:         ЧЧђ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0і
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЧЧђY
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:         ЧЧђk
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:         ЧЧђЂ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ЩЩђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Z V
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs
Ф
┼
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1034409

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0╦
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ШШђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ШШђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
в
┼
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1034085

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
¤
ф
2__inference_conv2d_transpose_layer_call_fn_1034431

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallђ
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
GPU2*0J 8ѓ *V
fQRO
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1030152і
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
ё
v
J__inference_concatenate_4_layer_call_and_return_conditional_losses_1035073
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ѓ
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*2
_output_shapes 
:         ЧЧ└b
IdentityIdentityconcat:output:0*
T0*2
_output_shapes 
:         ЧЧ└"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         ЧЧ@:         ЧЧђ:[ W
1
_output_shapes
:         ЧЧ@
"
_user_specified_name
inputs/0:\X
2
_output_shapes 
:         ЧЧђ
"
_user_specified_name
inputs/1
ю	
о
7__inference_batch_normalization_7_layer_call_fn_1034962

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЮ
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1030438і
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
─
Ю
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1035153

inputsC
(conv2d_transpose_readvariableop_resource:@└-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
B :ђJ
stack/2Const*
_output_shapes
: *
dtype0*
value
B :ђI
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@Ѕ
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
valueB:┘
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЉ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@└*
dtype0═
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:         ђђ@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ѕ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ@X
EluEluBiasAdd:output:0*
T0*1
_output_shapes
:         ђђ@j
IdentityIdentityElu:activations:0^NoOp*
T0*1
_output_shapes
:         ђђ@Ђ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ЧЧ└: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Z V
2
_output_shapes 
:         ЧЧ└
 
_user_specified_nameinputs
М
г
4__inference_conv2d_transpose_2_layer_call_fn_1034865

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallѓ
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
GPU2*0J 8ѓ *X
fSRQ
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1030378і
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
М
м
7__inference_batch_normalization_1_layer_call_fn_1033905

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallї
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1031761y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЧЧ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         ЧЧ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ЧЧ@
 
_user_specified_nameinputs
Џ
┴
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1033977

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЧЧ@:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0m
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:         ЧЧ@н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
─б
у\
#__inference__traced_restore_1036202
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
+assignvariableop_28_conv2d_transpose_kernel:ђђ8
)assignvariableop_29_conv2d_transpose_bias:	ђ>
/assignvariableop_30_batch_normalization_5_gamma:	ђ=
.assignvariableop_31_batch_normalization_5_beta:	ђD
5assignvariableop_32_batch_normalization_5_moving_mean:	ђH
9assignvariableop_33_batch_normalization_5_moving_variance:	ђI
-assignvariableop_34_conv2d_transpose_1_kernel:ђђ:
+assignvariableop_35_conv2d_transpose_1_bias:	ђ>
/assignvariableop_36_batch_normalization_6_gamma:	ђ=
.assignvariableop_37_batch_normalization_6_beta:	ђD
5assignvariableop_38_batch_normalization_6_moving_mean:	ђH
9assignvariableop_39_batch_normalization_6_moving_variance:	ђI
-assignvariableop_40_conv2d_transpose_2_kernel:ђђ:
+assignvariableop_41_conv2d_transpose_2_bias:	ђ>
/assignvariableop_42_batch_normalization_7_gamma:	ђ=
.assignvariableop_43_batch_normalization_7_beta:	ђD
5assignvariableop_44_batch_normalization_7_moving_mean:	ђH
9assignvariableop_45_batch_normalization_7_moving_variance:	ђH
-assignvariableop_46_conv2d_transpose_3_kernel:@└9
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
2assignvariableop_79_adam_conv2d_transpose_kernel_m:ђђ?
0assignvariableop_80_adam_conv2d_transpose_bias_m:	ђE
6assignvariableop_81_adam_batch_normalization_5_gamma_m:	ђD
5assignvariableop_82_adam_batch_normalization_5_beta_m:	ђP
4assignvariableop_83_adam_conv2d_transpose_1_kernel_m:ђђA
2assignvariableop_84_adam_conv2d_transpose_1_bias_m:	ђE
6assignvariableop_85_adam_batch_normalization_6_gamma_m:	ђD
5assignvariableop_86_adam_batch_normalization_6_beta_m:	ђP
4assignvariableop_87_adam_conv2d_transpose_2_kernel_m:ђђA
2assignvariableop_88_adam_conv2d_transpose_2_bias_m:	ђE
6assignvariableop_89_adam_batch_normalization_7_gamma_m:	ђD
5assignvariableop_90_adam_batch_normalization_7_beta_m:	ђO
4assignvariableop_91_adam_conv2d_transpose_3_kernel_m:@└@
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
3assignvariableop_115_adam_conv2d_transpose_kernel_v:ђђ@
1assignvariableop_116_adam_conv2d_transpose_bias_v:	ђF
7assignvariableop_117_adam_batch_normalization_5_gamma_v:	ђE
6assignvariableop_118_adam_batch_normalization_5_beta_v:	ђQ
5assignvariableop_119_adam_conv2d_transpose_1_kernel_v:ђђB
3assignvariableop_120_adam_conv2d_transpose_1_bias_v:	ђF
7assignvariableop_121_adam_batch_normalization_6_gamma_v:	ђE
6assignvariableop_122_adam_batch_normalization_6_beta_v:	ђQ
5assignvariableop_123_adam_conv2d_transpose_2_kernel_v:ђђB
3assignvariableop_124_adam_conv2d_transpose_2_bias_v:	ђF
7assignvariableop_125_adam_batch_normalization_7_gamma_v:	ђE
6assignvariableop_126_adam_batch_normalization_7_beta_v:	ђP
5assignvariableop_127_adam_conv2d_transpose_3_kernel_v:@└A
3assignvariableop_128_adam_conv2d_transpose_3_bias_v:@E
7assignvariableop_129_adam_batch_normalization_8_gamma_v:@D
6assignvariableop_130_adam_batch_normalization_8_beta_v:@O
5assignvariableop_131_adam_conv2d_transpose_4_kernel_v:DA
3assignvariableop_132_adam_conv2d_transpose_4_bias_v:
identity_134ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_100бAssignVariableOp_101бAssignVariableOp_102бAssignVariableOp_103бAssignVariableOp_104бAssignVariableOp_105бAssignVariableOp_106бAssignVariableOp_107бAssignVariableOp_108бAssignVariableOp_109бAssignVariableOp_11бAssignVariableOp_110бAssignVariableOp_111бAssignVariableOp_112бAssignVariableOp_113бAssignVariableOp_114бAssignVariableOp_115бAssignVariableOp_116бAssignVariableOp_117бAssignVariableOp_118бAssignVariableOp_119бAssignVariableOp_12бAssignVariableOp_120бAssignVariableOp_121бAssignVariableOp_122бAssignVariableOp_123бAssignVariableOp_124бAssignVariableOp_125бAssignVariableOp_126бAssignVariableOp_127бAssignVariableOp_128бAssignVariableOp_129бAssignVariableOp_13бAssignVariableOp_130бAssignVariableOp_131бAssignVariableOp_132бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_40бAssignVariableOp_41бAssignVariableOp_42бAssignVariableOp_43бAssignVariableOp_44бAssignVariableOp_45бAssignVariableOp_46бAssignVariableOp_47бAssignVariableOp_48бAssignVariableOp_49бAssignVariableOp_5бAssignVariableOp_50бAssignVariableOp_51бAssignVariableOp_52бAssignVariableOp_53бAssignVariableOp_54бAssignVariableOp_55бAssignVariableOp_56бAssignVariableOp_57бAssignVariableOp_58бAssignVariableOp_59бAssignVariableOp_6бAssignVariableOp_60бAssignVariableOp_61бAssignVariableOp_62бAssignVariableOp_63бAssignVariableOp_64бAssignVariableOp_65бAssignVariableOp_66бAssignVariableOp_67бAssignVariableOp_68бAssignVariableOp_69бAssignVariableOp_7бAssignVariableOp_70бAssignVariableOp_71бAssignVariableOp_72бAssignVariableOp_73бAssignVariableOp_74бAssignVariableOp_75бAssignVariableOp_76бAssignVariableOp_77бAssignVariableOp_78бAssignVariableOp_79бAssignVariableOp_8бAssignVariableOp_80бAssignVariableOp_81бAssignVariableOp_82бAssignVariableOp_83бAssignVariableOp_84бAssignVariableOp_85бAssignVariableOp_86бAssignVariableOp_87бAssignVariableOp_88бAssignVariableOp_89бAssignVariableOp_9бAssignVariableOp_90бAssignVariableOp_91бAssignVariableOp_92бAssignVariableOp_93бAssignVariableOp_94бAssignVariableOp_95бAssignVariableOp_96бAssignVariableOp_97бAssignVariableOp_98бAssignVariableOp_99 J
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:є*
dtype0*цJ
valueџJBЌJєB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЂ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:є*
dtype0*б
valueўBЋєB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ├
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*«
_output_shapesЏ
ў::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Ќ
dtypesї
Ѕ2є	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOpAssignVariableOp*assignvariableop_batch_normalization_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_1AssignVariableOp+assignvariableop_1_batch_normalization_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:А
AssignVariableOp_2AssignVariableOp2assignvariableop_2_batch_normalization_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_3AssignVariableOp6assignvariableop_3_batch_normalization_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_4AssignVariableOp assignvariableop_4_conv2d_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_5AssignVariableOpassignvariableop_5_conv2d_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_6AssignVariableOp.assignvariableop_6_batch_normalization_1_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_7AssignVariableOp-assignvariableop_7_batch_normalization_1_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_8AssignVariableOp4assignvariableop_8_batch_normalization_1_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_9AssignVariableOp8assignvariableop_9_batch_normalization_1_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:ћ
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv2d_1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:њ
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv2d_1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_12AssignVariableOp/assignvariableop_12_batch_normalization_2_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_13AssignVariableOp.assignvariableop_13_batch_normalization_2_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_14AssignVariableOp5assignvariableop_14_batch_normalization_2_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:ф
AssignVariableOp_15AssignVariableOp9assignvariableop_15_batch_normalization_2_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:ћ
AssignVariableOp_16AssignVariableOp#assignvariableop_16_conv2d_2_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:њ
AssignVariableOp_17AssignVariableOp!assignvariableop_17_conv2d_2_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_18AssignVariableOp/assignvariableop_18_batch_normalization_3_gammaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_19AssignVariableOp.assignvariableop_19_batch_normalization_3_betaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_20AssignVariableOp5assignvariableop_20_batch_normalization_3_moving_meanIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:ф
AssignVariableOp_21AssignVariableOp9assignvariableop_21_batch_normalization_3_moving_varianceIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:ћ
AssignVariableOp_22AssignVariableOp#assignvariableop_22_conv2d_3_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:њ
AssignVariableOp_23AssignVariableOp!assignvariableop_23_conv2d_3_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_24AssignVariableOp/assignvariableop_24_batch_normalization_4_gammaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_25AssignVariableOp.assignvariableop_25_batch_normalization_4_betaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_26AssignVariableOp5assignvariableop_26_batch_normalization_4_moving_meanIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:ф
AssignVariableOp_27AssignVariableOp9assignvariableop_27_batch_normalization_4_moving_varianceIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_28AssignVariableOp+assignvariableop_28_conv2d_transpose_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_29AssignVariableOp)assignvariableop_29_conv2d_transpose_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_30AssignVariableOp/assignvariableop_30_batch_normalization_5_gammaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_31AssignVariableOp.assignvariableop_31_batch_normalization_5_betaIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_32AssignVariableOp5assignvariableop_32_batch_normalization_5_moving_meanIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:ф
AssignVariableOp_33AssignVariableOp9assignvariableop_33_batch_normalization_5_moving_varianceIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:ъ
AssignVariableOp_34AssignVariableOp-assignvariableop_34_conv2d_transpose_1_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_35AssignVariableOp+assignvariableop_35_conv2d_transpose_1_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_36AssignVariableOp/assignvariableop_36_batch_normalization_6_gammaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_37AssignVariableOp.assignvariableop_37_batch_normalization_6_betaIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_38AssignVariableOp5assignvariableop_38_batch_normalization_6_moving_meanIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:ф
AssignVariableOp_39AssignVariableOp9assignvariableop_39_batch_normalization_6_moving_varianceIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:ъ
AssignVariableOp_40AssignVariableOp-assignvariableop_40_conv2d_transpose_2_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_41AssignVariableOp+assignvariableop_41_conv2d_transpose_2_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_42AssignVariableOp/assignvariableop_42_batch_normalization_7_gammaIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_43AssignVariableOp.assignvariableop_43_batch_normalization_7_betaIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_44AssignVariableOp5assignvariableop_44_batch_normalization_7_moving_meanIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:ф
AssignVariableOp_45AssignVariableOp9assignvariableop_45_batch_normalization_7_moving_varianceIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:ъ
AssignVariableOp_46AssignVariableOp-assignvariableop_46_conv2d_transpose_3_kernelIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_47AssignVariableOp+assignvariableop_47_conv2d_transpose_3_biasIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_48AssignVariableOp/assignvariableop_48_batch_normalization_8_gammaIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_49AssignVariableOp.assignvariableop_49_batch_normalization_8_betaIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_50AssignVariableOp5assignvariableop_50_batch_normalization_8_moving_meanIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:ф
AssignVariableOp_51AssignVariableOp9assignvariableop_51_batch_normalization_8_moving_varianceIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:ъ
AssignVariableOp_52AssignVariableOp-assignvariableop_52_conv2d_transpose_4_kernelIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_53AssignVariableOp+assignvariableop_53_conv2d_transpose_4_biasIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0	*
_output_shapes
:ј
AssignVariableOp_54AssignVariableOpassignvariableop_54_adam_iterIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_55AssignVariableOpassignvariableop_55_adam_beta_1Identity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_56AssignVariableOpassignvariableop_56_adam_beta_2Identity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_57AssignVariableOpassignvariableop_57_adam_decayIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_58AssignVariableOp&assignvariableop_58_adam_learning_rateIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_59AssignVariableOpassignvariableop_59_totalIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_60AssignVariableOpassignvariableop_60_countIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_61AssignVariableOp4assignvariableop_61_adam_batch_normalization_gamma_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:ц
AssignVariableOp_62AssignVariableOp3assignvariableop_62_adam_batch_normalization_beta_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_63AssignVariableOp(assignvariableop_63_adam_conv2d_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_64AssignVariableOp&assignvariableop_64_adam_conv2d_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_65AssignVariableOp6assignvariableop_65_adam_batch_normalization_1_gamma_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_66AssignVariableOp5assignvariableop_66_adam_batch_normalization_1_beta_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_67AssignVariableOp*assignvariableop_67_adam_conv2d_1_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_68AssignVariableOp(assignvariableop_68_adam_conv2d_1_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_69AssignVariableOp6assignvariableop_69_adam_batch_normalization_2_gamma_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_70AssignVariableOp5assignvariableop_70_adam_batch_normalization_2_beta_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_71AssignVariableOp*assignvariableop_71_adam_conv2d_2_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_72AssignVariableOp(assignvariableop_72_adam_conv2d_2_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_73AssignVariableOp6assignvariableop_73_adam_batch_normalization_3_gamma_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_74AssignVariableOp5assignvariableop_74_adam_batch_normalization_3_beta_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_75AssignVariableOp*assignvariableop_75_adam_conv2d_3_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_76AssignVariableOp(assignvariableop_76_adam_conv2d_3_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_77AssignVariableOp6assignvariableop_77_adam_batch_normalization_4_gamma_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_78AssignVariableOp5assignvariableop_78_adam_batch_normalization_4_beta_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_79AssignVariableOp2assignvariableop_79_adam_conv2d_transpose_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:А
AssignVariableOp_80AssignVariableOp0assignvariableop_80_adam_conv2d_transpose_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_81AssignVariableOp6assignvariableop_81_adam_batch_normalization_5_gamma_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_82AssignVariableOp5assignvariableop_82_adam_batch_normalization_5_beta_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_83AssignVariableOp4assignvariableop_83_adam_conv2d_transpose_1_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_84AssignVariableOp2assignvariableop_84_adam_conv2d_transpose_1_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_85AssignVariableOp6assignvariableop_85_adam_batch_normalization_6_gamma_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_86AssignVariableOp5assignvariableop_86_adam_batch_normalization_6_beta_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_87AssignVariableOp4assignvariableop_87_adam_conv2d_transpose_2_kernel_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_88AssignVariableOp2assignvariableop_88_adam_conv2d_transpose_2_bias_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_89AssignVariableOp6assignvariableop_89_adam_batch_normalization_7_gamma_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_90AssignVariableOp5assignvariableop_90_adam_batch_normalization_7_beta_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_91AssignVariableOp4assignvariableop_91_adam_conv2d_transpose_3_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_92AssignVariableOp2assignvariableop_92_adam_conv2d_transpose_3_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_93AssignVariableOp6assignvariableop_93_adam_batch_normalization_8_gamma_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_94AssignVariableOp5assignvariableop_94_adam_batch_normalization_8_beta_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_95AssignVariableOp4assignvariableop_95_adam_conv2d_transpose_4_kernel_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_96AssignVariableOp2assignvariableop_96_adam_conv2d_transpose_4_bias_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_97AssignVariableOp4assignvariableop_97_adam_batch_normalization_gamma_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:ц
AssignVariableOp_98AssignVariableOp3assignvariableop_98_adam_batch_normalization_beta_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_99AssignVariableOp(assignvariableop_99_adam_conv2d_kernel_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_100AssignVariableOp'assignvariableop_100_adam_conv2d_bias_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:ф
AssignVariableOp_101AssignVariableOp7assignvariableop_101_adam_batch_normalization_1_gamma_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_102AssignVariableOp6assignvariableop_102_adam_batch_normalization_1_beta_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:ъ
AssignVariableOp_103AssignVariableOp+assignvariableop_103_adam_conv2d_1_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_104AssignVariableOp)assignvariableop_104_adam_conv2d_1_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:ф
AssignVariableOp_105AssignVariableOp7assignvariableop_105_adam_batch_normalization_2_gamma_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_106AssignVariableOp6assignvariableop_106_adam_batch_normalization_2_beta_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:ъ
AssignVariableOp_107AssignVariableOp+assignvariableop_107_adam_conv2d_2_kernel_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_108AssignVariableOp)assignvariableop_108_adam_conv2d_2_bias_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:ф
AssignVariableOp_109AssignVariableOp7assignvariableop_109_adam_batch_normalization_3_gamma_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_110AssignVariableOp6assignvariableop_110_adam_batch_normalization_3_beta_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:ъ
AssignVariableOp_111AssignVariableOp+assignvariableop_111_adam_conv2d_3_kernel_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_112AssignVariableOp)assignvariableop_112_adam_conv2d_3_bias_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:ф
AssignVariableOp_113AssignVariableOp7assignvariableop_113_adam_batch_normalization_4_gamma_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_114AssignVariableOp6assignvariableop_114_adam_batch_normalization_4_beta_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_115AssignVariableOp3assignvariableop_115_adam_conv2d_transpose_kernel_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:ц
AssignVariableOp_116AssignVariableOp1assignvariableop_116_adam_conv2d_transpose_bias_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:ф
AssignVariableOp_117AssignVariableOp7assignvariableop_117_adam_batch_normalization_5_gamma_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_118AssignVariableOp6assignvariableop_118_adam_batch_normalization_5_beta_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_119AssignVariableOp5assignvariableop_119_adam_conv2d_transpose_1_kernel_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_120AssignVariableOp3assignvariableop_120_adam_conv2d_transpose_1_bias_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:ф
AssignVariableOp_121AssignVariableOp7assignvariableop_121_adam_batch_normalization_6_gamma_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_122AssignVariableOp6assignvariableop_122_adam_batch_normalization_6_beta_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_123AssignVariableOp5assignvariableop_123_adam_conv2d_transpose_2_kernel_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_124AssignVariableOp3assignvariableop_124_adam_conv2d_transpose_2_bias_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:ф
AssignVariableOp_125AssignVariableOp7assignvariableop_125_adam_batch_normalization_7_gamma_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_126AssignVariableOp6assignvariableop_126_adam_batch_normalization_7_beta_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_127AssignVariableOp5assignvariableop_127_adam_conv2d_transpose_3_kernel_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_128AssignVariableOp3assignvariableop_128_adam_conv2d_transpose_3_bias_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:ф
AssignVariableOp_129AssignVariableOp7assignvariableop_129_adam_batch_normalization_8_gamma_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_130AssignVariableOp6assignvariableop_130_adam_batch_normalization_8_beta_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_131AssignVariableOp5assignvariableop_131_adam_conv2d_transpose_4_kernel_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_132AssignVariableOp3assignvariableop_132_adam_conv2d_transpose_4_bias_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ▀
Identity_133Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_134IdentityIdentity_133:output:0^NoOp_1*
T0*
_output_shapes
: ╦
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
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
ц
љ
H__inference_concatenate_layer_call_and_return_conditional_losses_1033709
inputs_0
inputs_1
inputs_2
inputs_3
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ћ
concatConcatV2inputs_0inputs_1inputs_2inputs_3concat/axis:output:0*
N*
T0*1
_output_shapes
:         ђђa
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:         ђђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Є
_input_shapesv
t:         ђђ:         ђђ:         ђђ:         ђђ:[ W
1
_output_shapes
:         ђђ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:         ђђ
"
_user_specified_name
inputs/1:[W
1
_output_shapes
:         ђђ
"
_user_specified_name
inputs/2:[W
1
_output_shapes
:         ђђ
"
_user_specified_name
inputs/3
Њ
Љ
'__inference_model_layer_call_fn_1032876
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

unknown_27:ђђ

unknown_28:	ђ

unknown_29:	ђ

unknown_30:	ђ

unknown_31:	ђ

unknown_32:	ђ&

unknown_33:ђђ

unknown_34:	ђ

unknown_35:	ђ

unknown_36:	ђ

unknown_37:	ђ

unknown_38:	ђ&

unknown_39:ђђ

unknown_40:	ђ

unknown_41:	ђ

unknown_42:	ђ

unknown_43:	ђ

unknown_44:	ђ%

unknown_45:@└

unknown_46:@

unknown_47:@

unknown_48:@

unknown_49:@

unknown_50:@$

unknown_51:D

unknown_52:
identityѕбStatefulPartitionedCallо
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
:         ђђ*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456789*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1031199y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesМ
л:         ђђ:         ђђ:         ђђ:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
-
_output_shapes
:         ђђ
"
_user_specified_name
inputs/0:WS
-
_output_shapes
:         ђђ
"
_user_specified_name
inputs/1:WS
-
_output_shapes
:         ђђ
"
_user_specified_name
inputs/2:WS
-
_output_shapes
:         ђђ
"
_user_specified_name
inputs/3
П
А
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1030004

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
Џ
Ђ
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1030824

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0Ю
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЭЭђ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ђ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЭЭђY
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:         ЭЭђk
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
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
Ф
┼
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1031479

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0╦
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЩЩђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
ќ 
Г2
B__inference_model_layer_call_and_return_conditional_losses_1033304
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
9conv2d_transpose_conv2d_transpose_readvariableop_resource:ђђ?
0conv2d_transpose_biasadd_readvariableop_resource:	ђ<
-batch_normalization_5_readvariableop_resource:	ђ>
/batch_normalization_5_readvariableop_1_resource:	ђM
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:	ђO
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:	ђW
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:ђђA
2conv2d_transpose_1_biasadd_readvariableop_resource:	ђ<
-batch_normalization_6_readvariableop_resource:	ђ>
/batch_normalization_6_readvariableop_1_resource:	ђM
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource:	ђO
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:	ђW
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource:ђђA
2conv2d_transpose_2_biasadd_readvariableop_resource:	ђ<
-batch_normalization_7_readvariableop_resource:	ђ>
/batch_normalization_7_readvariableop_1_resource:	ђM
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:	ђO
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:	ђV
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource:@└@
2conv2d_transpose_3_biasadd_readvariableop_resource:@;
-batch_normalization_8_readvariableop_resource:@=
/batch_normalization_8_readvariableop_1_resource:@L
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:@U
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource:D@
2conv2d_transpose_4_biasadd_readvariableop_resource:
identityѕб3batch_normalization/FusedBatchNormV3/ReadVariableOpб5batch_normalization/FusedBatchNormV3/ReadVariableOp_1б"batch_normalization/ReadVariableOpб$batch_normalization/ReadVariableOp_1б5batch_normalization_1/FusedBatchNormV3/ReadVariableOpб7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_1/ReadVariableOpб&batch_normalization_1/ReadVariableOp_1б5batch_normalization_2/FusedBatchNormV3/ReadVariableOpб7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_2/ReadVariableOpб&batch_normalization_2/ReadVariableOp_1б5batch_normalization_3/FusedBatchNormV3/ReadVariableOpб7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_3/ReadVariableOpб&batch_normalization_3/ReadVariableOp_1б5batch_normalization_4/FusedBatchNormV3/ReadVariableOpб7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_4/ReadVariableOpб&batch_normalization_4/ReadVariableOp_1б5batch_normalization_5/FusedBatchNormV3/ReadVariableOpб7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_5/ReadVariableOpб&batch_normalization_5/ReadVariableOp_1б5batch_normalization_6/FusedBatchNormV3/ReadVariableOpб7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_6/ReadVariableOpб&batch_normalization_6/ReadVariableOp_1б5batch_normalization_7/FusedBatchNormV3/ReadVariableOpб7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_7/ReadVariableOpб&batch_normalization_7/ReadVariableOp_1б5batch_normalization_8/FusedBatchNormV3/ReadVariableOpб7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_8/ReadVariableOpб&batch_normalization_8/ReadVariableOp_1бconv2d/BiasAdd/ReadVariableOpбconv2d/Conv2D/ReadVariableOpбconv2d_1/BiasAdd/ReadVariableOpбconv2d_1/Conv2D/ReadVariableOpбconv2d_2/BiasAdd/ReadVariableOpбconv2d_2/Conv2D/ReadVariableOpбconv2d_3/BiasAdd/ReadVariableOpбconv2d_3/Conv2D/ReadVariableOpб'conv2d_transpose/BiasAdd/ReadVariableOpб0conv2d_transpose/conv2d_transpose/ReadVariableOpб)conv2d_transpose_1/BiasAdd/ReadVariableOpб2conv2d_transpose_1/conv2d_transpose/ReadVariableOpб)conv2d_transpose_2/BiasAdd/ReadVariableOpб2conv2d_transpose_2/conv2d_transpose/ReadVariableOpб)conv2d_transpose_3/BiasAdd/ReadVariableOpб2conv2d_transpose_3/conv2d_transpose/ReadVariableOpб)conv2d_transpose_4/BiasAdd/ReadVariableOpб2conv2d_transpose_4/conv2d_transpose/ReadVariableOpE
reshape/ShapeShapeinputs_0*
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
valueB:щ
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
B :ђZ
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ђY
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Л
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:ђ
reshape/ReshapeReshapeinputs_0reshape/Reshape/shape:output:0*
T0*1
_output_shapes
:         ђђG
reshape_1/ShapeShapeinputs_1*
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
valueB:Ѓ
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
B :ђ\
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ђ[
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :█
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:ё
reshape_1/ReshapeReshapeinputs_1 reshape_1/Reshape/shape:output:0*
T0*1
_output_shapes
:         ђђG
reshape_2/ShapeShapeinputs_2*
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
valueB:Ѓ
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
B :ђ\
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ђ[
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :█
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:ё
reshape_2/ReshapeReshapeinputs_2 reshape_2/Reshape/shape:output:0*
T0*1
_output_shapes
:         ђђG
reshape_3/ShapeShapeinputs_3*
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
valueB:Ѓ
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
B :ђ\
reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ђ[
reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :█
reshape_3/Reshape/shapePack reshape_3/strided_slice:output:0"reshape_3/Reshape/shape/1:output:0"reshape_3/Reshape/shape/2:output:0"reshape_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:ё
reshape_3/ReshapeReshapeinputs_3 reshape_3/Reshape/shape:output:0*
T0*1
_output_shapes
:         ђђY
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :з
concatenate/concatConcatV2reshape/Reshape:output:0reshape_1/Reshape:output:0reshape_2/Reshape:output:0reshape_3/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:         ђђі
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype0ј
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype0г
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0░
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0▒
$batch_normalization/FusedBatchNormV3FusedBatchNormV3concatenate/concat:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ:::::*
epsilon%oЃ:*
is_training( і
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0╠
conv2d/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЧЧ@*
paddingVALID*
strides
ђ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ћ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЧЧ@f

conv2d/EluEluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:         ЧЧ@ј
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0њ
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0░
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0┤
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0И
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d/Elu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЧЧ@:@:@:@:@:*
epsilon%oЃ:*
is_training( Ј
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype0М
conv2d_1/Conv2DConv2D*batch_normalization_1/FusedBatchNormV3:y:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЩЩђ*
paddingVALID*
strides
Ё
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Џ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЩЩђk
conv2d_1/EluEluconv2d_1/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЩЩђЈ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Њ
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0▒
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0х
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0┐
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_1/Elu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЩЩђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( љ
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0М
conv2d_2/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЭЭђ*
paddingVALID*
strides
Ё
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Џ
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЭЭђk
conv2d_2/EluEluconv2d_2/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЭЭђЈ
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Њ
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0▒
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0х
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0┐
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_2/Elu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЭЭђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( љ
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0М
conv2d_3/Conv2DConv2D*batch_normalization_3/FusedBatchNormV3:y:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ШШђ*
paddingVALID*
strides
Ё
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Џ
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ШШђk
conv2d_3/EluEluconv2d_3/BiasAdd:output:0*
T0*2
_output_shapes 
:         ШШђЈ
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Њ
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0▒
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0х
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0┐
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_3/Elu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ШШђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( [
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :м
concatenate_1/concatConcatV2conv2d_3/Elu:activations:0*batch_normalization_4/FusedBatchNormV3:y:0"concatenate_1/concat/axis:output:0*
N*
T0*2
_output_shapes 
:         ШШђc
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
valueB:д
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
B :Э[
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :Э[
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђя
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
valueB:«
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask┤
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0ў
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0concatenate_1/concat:output:0*
T0*2
_output_shapes 
:         ЭЭђ*
paddingVALID*
strides
Ћ
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0й
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЭЭђ{
conv2d_transpose/EluElu!conv2d_transpose/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЭЭђЈ
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Њ
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0▒
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0х
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0К
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3"conv2d_transpose/Elu:activations:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЭЭђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( [
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :м
concatenate_2/concatConcatV2conv2d_2/Elu:activations:0*batch_normalization_5/FusedBatchNormV3:y:0"concatenate_2/concat/axis:output:0*
N*
T0*2
_output_shapes 
:         ЭЭђe
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
valueB:░
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
B :Щ]
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :Щ]
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђУ
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
valueB:И
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskИ
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0ъ
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0concatenate_2/concat:output:0*
T0*2
_output_shapes 
:         ЩЩђ*
paddingVALID*
strides
Ў
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0├
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЩЩђ
conv2d_transpose_1/EluElu#conv2d_transpose_1/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЩЩђЈ
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Њ
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0▒
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0х
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0╔
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_1/Elu:activations:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЩЩђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( [
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :м
concatenate_3/concatConcatV2conv2d_1/Elu:activations:0*batch_normalization_6/FusedBatchNormV3:y:0"concatenate_3/concat/axis:output:0*
N*
T0*2
_output_shapes 
:         ЩЩђe
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
valueB:░
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
B :Ч]
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value
B :Ч]
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђУ
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
valueB:И
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskИ
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0ъ
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0concatenate_3/concat:output:0*
T0*2
_output_shapes 
:         ЧЧђ*
paddingVALID*
strides
Ў
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0├
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЧЧђ
conv2d_transpose_2/EluElu#conv2d_transpose_2/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЧЧђЈ
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Њ
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0▒
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0х
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0╔
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_2/Elu:activations:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЧЧђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( [
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :л
concatenate_4/concatConcatV2conv2d/Elu:activations:0*batch_normalization_7/FusedBatchNormV3:y:0"concatenate_4/concat/axis:output:0*
N*
T0*2
_output_shapes 
:         ЧЧ└e
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
valueB:░
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
B :ђ]
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value
B :ђ\
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@У
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
valueB:И
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskи
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@└*
dtype0Ю
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0concatenate_4/concat:output:0*
T0*1
_output_shapes
:         ђђ@*
paddingVALID*
strides
ў
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0┬
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ@~
conv2d_transpose_3/EluElu#conv2d_transpose_3/BiasAdd:output:0*
T0*1
_output_shapes
:         ђђ@ј
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:@*
dtype0њ
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:@*
dtype0░
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0┤
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0─
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_3/Elu:activations:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ@:@:@:@:@:*
epsilon%oЃ:*
is_training( [
concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :м
concatenate_5/concatConcatV2concatenate/concat:output:0*batch_normalization_8/FusedBatchNormV3:y:0"concatenate_5/concat/axis:output:0*
N*
T0*1
_output_shapes
:         ђђDe
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
valueB:░
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
B :ђ]
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value
B :ђ\
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :У
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
valueB:И
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskХ
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:D*
dtype0Ю
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0concatenate_5/concat:output:0*
T0*1
_output_shapes
:         ђђ*
paddingVALID*
strides
ў
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┬
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ|
IdentityIdentity#conv2d_transpose_4/BiasAdd:output:0^NoOp*
T0*1
_output_shapes
:         ђђщ
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesМ
л:         ђђ:         ђђ:         ђђ:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2j
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
:         ђђ
"
_user_specified_name
inputs/0:WS
-
_output_shapes
:         ђђ
"
_user_specified_name
inputs/1:WS
-
_output_shapes
:         ђђ
"
_user_specified_name
inputs/2:WS
-
_output_shapes
:         ђђ
"
_user_specified_name
inputs/3
ћ	
м
7__inference_batch_normalization_8_layer_call_fn_1035179

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallю
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1030551Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
■
б
*__inference_conv2d_2_layer_call_fn_1034130

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallУ
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
GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1030824z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ЩЩђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs
з
[
/__inference_concatenate_1_layer_call_fn_1034415
inputs_0
inputs_1
identityл
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ШШђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1030908k
IdentityIdentityPartitionedCall:output:0*
T0*2
_output_shapes 
:         ШШђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:         ШШђ:         ШШђ:\ X
2
_output_shapes 
:         ШШђ
"
_user_specified_name
inputs/0:\X
2
_output_shapes 
:         ШШђ
"
_user_specified_name
inputs/1
П
А
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1034572

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
в
┼
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1030099

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
в
┼
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1034373

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
Ю
А
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1034608

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0й
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЭЭђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЭЭђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
ђ
v
J__inference_concatenate_5_layer_call_and_return_conditional_losses_1035290
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ђ
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:         ђђDa
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:         ђђD"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::         ђђ:         ђђ@:[ W
1
_output_shapes
:         ђђ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:         ђђ@
"
_user_specified_name
inputs/1
ъ	
о
7__inference_batch_normalization_5_layer_call_fn_1034515

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЪ
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1030181і
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
№Ќ
ќ
B__inference_model_layer_call_and_return_conditional_losses_1032493	
u_vel	
v_vel	
w_vel	
pr0_2)
batch_normalization_1032361:)
batch_normalization_1032363:)
batch_normalization_1032365:)
batch_normalization_1032367:(
conv2d_1032370:@
conv2d_1032372:@+
batch_normalization_1_1032375:@+
batch_normalization_1_1032377:@+
batch_normalization_1_1032379:@+
batch_normalization_1_1032381:@+
conv2d_1_1032384:@ђ
conv2d_1_1032386:	ђ,
batch_normalization_2_1032389:	ђ,
batch_normalization_2_1032391:	ђ,
batch_normalization_2_1032393:	ђ,
batch_normalization_2_1032395:	ђ,
conv2d_2_1032398:ђђ
conv2d_2_1032400:	ђ,
batch_normalization_3_1032403:	ђ,
batch_normalization_3_1032405:	ђ,
batch_normalization_3_1032407:	ђ,
batch_normalization_3_1032409:	ђ,
conv2d_3_1032412:ђђ
conv2d_3_1032414:	ђ,
batch_normalization_4_1032417:	ђ,
batch_normalization_4_1032419:	ђ,
batch_normalization_4_1032421:	ђ,
batch_normalization_4_1032423:	ђ4
conv2d_transpose_1032427:ђђ'
conv2d_transpose_1032429:	ђ,
batch_normalization_5_1032432:	ђ,
batch_normalization_5_1032434:	ђ,
batch_normalization_5_1032436:	ђ,
batch_normalization_5_1032438:	ђ6
conv2d_transpose_1_1032442:ђђ)
conv2d_transpose_1_1032444:	ђ,
batch_normalization_6_1032447:	ђ,
batch_normalization_6_1032449:	ђ,
batch_normalization_6_1032451:	ђ,
batch_normalization_6_1032453:	ђ6
conv2d_transpose_2_1032457:ђђ)
conv2d_transpose_2_1032459:	ђ,
batch_normalization_7_1032462:	ђ,
batch_normalization_7_1032464:	ђ,
batch_normalization_7_1032466:	ђ,
batch_normalization_7_1032468:	ђ5
conv2d_transpose_3_1032472:@└(
conv2d_transpose_3_1032474:@+
batch_normalization_8_1032477:@+
batch_normalization_8_1032479:@+
batch_normalization_8_1032481:@+
batch_normalization_8_1032483:@4
conv2d_transpose_4_1032487:D(
conv2d_transpose_4_1032489:
identityѕб+batch_normalization/StatefulPartitionedCallб-batch_normalization_1/StatefulPartitionedCallб-batch_normalization_2/StatefulPartitionedCallб-batch_normalization_3/StatefulPartitionedCallб-batch_normalization_4/StatefulPartitionedCallб-batch_normalization_5/StatefulPartitionedCallб-batch_normalization_6/StatefulPartitionedCallб-batch_normalization_7/StatefulPartitionedCallб-batch_normalization_8/StatefulPartitionedCallбconv2d/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallб conv2d_2/StatefulPartitionedCallб conv2d_3/StatefulPartitionedCallб(conv2d_transpose/StatefulPartitionedCallб*conv2d_transpose_1/StatefulPartitionedCallб*conv2d_transpose_2/StatefulPartitionedCallб*conv2d_transpose_3/StatefulPartitionedCallб*conv2d_transpose_4/StatefulPartitionedCall├
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
GPU2*0J 8ѓ *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_1030637К
reshape_1/PartitionedCallPartitionedCallv_vel*
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
GPU2*0J 8ѓ *O
fJRH
F__inference_reshape_1_layer_call_and_return_conditional_losses_1030653К
reshape_2/PartitionedCallPartitionedCallw_vel*
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
GPU2*0J 8ѓ *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_1030669К
reshape_3/PartitionedCallPartitionedCallpr0_2*
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
GPU2*0J 8ѓ *O
fJRH
F__inference_reshape_3_layer_call_and_return_conditional_losses_1030685Н
concatenate/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0"reshape_1/PartitionedCall:output:0"reshape_2/PartitionedCall:output:0"reshape_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_1030696ѕ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0batch_normalization_1032361batch_normalization_1032363batch_normalization_1032365batch_normalization_1032367*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1030715д
conv2d/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_1032370conv2d_1032372*
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
GPU2*0J 8ѓ *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_1030736Ќ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_1_1032375batch_normalization_1_1032377batch_normalization_1_1032379batch_normalization_1_1032381*
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1030759▒
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_1_1032384conv2d_1_1032386*
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
GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1030780џ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_2_1032389batch_normalization_2_1032391batch_normalization_2_1032393batch_normalization_2_1032395*
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1030803▒
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_2_1032398conv2d_2_1032400*
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
GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1030824џ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_3_1032403batch_normalization_3_1032405batch_normalization_3_1032407batch_normalization_3_1032409*
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1030847▒
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_3_1032412conv2d_3_1032414*
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
GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1030868џ
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_4_1032417batch_normalization_4_1032419batch_normalization_4_1032421batch_normalization_4_1032423*
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1030891Г
concatenate_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:06batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ШШђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1030908┴
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0conv2d_transpose_1032427conv2d_transpose_1032429*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1030933б
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0batch_normalization_5_1032432batch_normalization_5_1032434batch_normalization_5_1032436batch_normalization_5_1032438*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1030956Г
concatenate_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:06batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_1030973╔
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0conv2d_transpose_1_1032442conv2d_transpose_1_1032444*
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
GPU2*0J 8ѓ *X
fSRQ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1030998ц
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0batch_normalization_6_1032447batch_normalization_6_1032449batch_normalization_6_1032451batch_normalization_6_1032453*
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1031021Г
concatenate_3/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:06batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЩЩђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_concatenate_3_layer_call_and_return_conditional_losses_1031038╔
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0conv2d_transpose_2_1032457conv2d_transpose_2_1032459*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЧЧђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1031063ц
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0batch_normalization_7_1032462batch_normalization_7_1032464batch_normalization_7_1032466batch_normalization_7_1032468*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЧЧђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1031086Ф
concatenate_4/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:06batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЧЧ└* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_concatenate_4_layer_call_and_return_conditional_losses_1031103╚
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv2d_transpose_3_1032472conv2d_transpose_3_1032474*
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
GPU2*0J 8ѓ *X
fSRQ
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1031128Б
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0batch_normalization_8_1032477batch_normalization_8_1032479batch_normalization_8_1032481batch_normalization_8_1032483*
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1031151Д
concatenate_5/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:06batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђD* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_concatenate_5_layer_call_and_return_conditional_losses_1031168╚
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0conv2d_transpose_4_1032487conv2d_transpose_4_1032489*
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
GPU2*0J 8ѓ *X
fSRQ
O__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1031192ї
IdentityIdentity3conv2d_transpose_4/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђП
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesМ
л:         ђђ:         ђђ:         ђђ:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
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

_user_specified_nameu_vel:TP
-
_output_shapes
:         ђђ

_user_specified_namev_vel:TP
-
_output_shapes
:         ђђ

_user_specified_namew_vel:TP
-
_output_shapes
:         ђђ

_user_specified_namepr0.2
Ў
┐
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1031815

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ:::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0m
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:         ђђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         ђђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
▀
b
F__inference_reshape_3_layer_call_and_return_conditional_losses_1030685

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
valueB:Л
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
B :ђR
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ђQ
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Е
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:         ђђb
IdentityIdentityReshape:output:0*
T0*1
_output_shapes
:         ђђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ђђ:U Q
-
_output_shapes
:         ђђ
 
_user_specified_nameinputs
▀
b
F__inference_reshape_3_layer_call_and_return_conditional_losses_1033692

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
valueB:Л
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
B :ђR
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ђQ
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Е
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:         ђђb
IdentityIdentityReshape:output:0*
T0*1
_output_shapes
:         ђђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ђђ:U Q
-
_output_shapes
:         ђђ
 
_user_specified_nameinputs
П
`
D__inference_reshape_layer_call_and_return_conditional_losses_1030637

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
valueB:Л
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
B :ђR
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ђQ
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Е
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:         ђђb
IdentityIdentityReshape:output:0*
T0*1
_output_shapes
:         ђђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ђђ:U Q
-
_output_shapes
:         ђђ
 
_user_specified_nameinputs
─
G
+__inference_reshape_3_layer_call_fn_1033678

inputs
identityЙ
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
GPU2*0J 8ѓ *O
fJRH
F__inference_reshape_3_layer_call_and_return_conditional_losses_1030685j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ђђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ђђ:U Q
-
_output_shapes
:         ђђ
 
_user_specified_nameinputs
═
Ъ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1034719

inputsD
(conv2d_transpose_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
B :ЩJ
stack/2Const*
_output_shapes
: *
dtype0*
value
B :ЩJ
stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђЅ
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
valueB:┘
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskњ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0╬
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*2
_output_shapes 
:         ЩЩђ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0і
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЩЩђY
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:         ЩЩђk
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђЂ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ЭЭђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
П
`
D__inference_reshape_layer_call_and_return_conditional_losses_1033635

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
valueB:Л
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
B :ђR
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ђQ
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Е
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:         ђђb
IdentityIdentityReshape:output:0*
T0*1
_output_shapes
:         ђђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ђђ:U Q
-
_output_shapes
:         ђђ
 
_user_specified_nameinputs
ъ	
о
7__inference_batch_normalization_4_layer_call_fn_1034298

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЪ
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1030068і
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
█
о
7__inference_batch_normalization_4_layer_call_fn_1034337

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЇ
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1031599z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ШШђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ШШђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ШШђ
 
_user_specified_nameinputs
ю	
о
7__inference_batch_normalization_6_layer_call_fn_1034745

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЮ
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1030325і
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
■
t
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1030908

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ђ
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*2
_output_shapes 
:         ШШђb
IdentityIdentityconcat:output:0*
T0*2
_output_shapes 
:         ШШђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:         ШШђ:         ШШђ:Z V
2
_output_shapes 
:         ШШђ
 
_user_specified_nameinputs:ZV
2
_output_shapes 
:         ШШђ
 
_user_specified_nameinputs
ј
ф
2__inference_conv2d_transpose_layer_call_fn_1034440

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1030933z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ШШђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ШШђ
 
_user_specified_nameinputs
ј
ф
4__inference_conv2d_transpose_3_layer_call_fn_1035091

inputs"
unknown:@└
	unknown_0:@
identityѕбStatefulPartitionedCallы
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
GPU2*0J 8ѓ *X
fSRQ
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1031128y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ЧЧ└: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ЧЧ└
 
_user_specified_nameinputs
є
v
J__inference_concatenate_3_layer_call_and_return_conditional_losses_1034856
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ѓ
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*2
_output_shapes 
:         ЩЩђb
IdentityIdentityconcat:output:0*
T0*2
_output_shapes 
:         ЩЩђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:         ЩЩђ:         ЩЩђ:\ X
2
_output_shapes 
:         ЩЩђ
"
_user_specified_name
inputs/0:\X
2
_output_shapes 
:         ЩЩђ
"
_user_specified_name
inputs/1
њ
г
4__inference_conv2d_transpose_2_layer_call_fn_1034874

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЧЧђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1031063z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ЧЧђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ЩЩђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs
П
А
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1034789

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
ћ	
м
7__inference_batch_normalization_1_layer_call_fn_1033879

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallю
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1029907Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
з
[
/__inference_concatenate_2_layer_call_fn_1034632
inputs_0
inputs_1
identityл
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_1030973k
IdentityIdentityPartitionedCall:output:0*
T0*2
_output_shapes 
:         ЭЭђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:         ЭЭђ:         ЭЭђ:\ X
2
_output_shapes 
:         ЭЭђ
"
_user_specified_name
inputs/0:\X
2
_output_shapes 
:         ЭЭђ
"
_user_specified_name
inputs/1
в
┼
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1030438

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
█
┴
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1035241

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
Н
м
7__inference_batch_normalization_1_layer_call_fn_1033892

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallј
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1030759y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЧЧ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         ЧЧ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ЧЧ@
 
_user_specified_nameinputs
в
┼
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1029971

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
Ф
┼
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1031707

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0╦
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЩЩђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
ќ	
м
7__inference_batch_normalization_1_layer_call_fn_1033866

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallъ
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1029876Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
П
А
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1030068

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
П
А
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1034355

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
■
t
J__inference_concatenate_3_layer_call_and_return_conditional_losses_1031038

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ђ
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*2
_output_shapes 
:         ЩЩђb
IdentityIdentityconcat:output:0*
T0*2
_output_shapes 
:         ЩЩђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:         ЩЩђ:         ЩЩђ:Z V
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs:ZV
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs
Ю
А
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1030803

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0й
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЩЩђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
н╝
щ7
B__inference_model_layer_call_and_return_conditional_losses_1033616
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
9conv2d_transpose_conv2d_transpose_readvariableop_resource:ђђ?
0conv2d_transpose_biasadd_readvariableop_resource:	ђ<
-batch_normalization_5_readvariableop_resource:	ђ>
/batch_normalization_5_readvariableop_1_resource:	ђM
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:	ђO
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:	ђW
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:ђђA
2conv2d_transpose_1_biasadd_readvariableop_resource:	ђ<
-batch_normalization_6_readvariableop_resource:	ђ>
/batch_normalization_6_readvariableop_1_resource:	ђM
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource:	ђO
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:	ђW
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource:ђђA
2conv2d_transpose_2_biasadd_readvariableop_resource:	ђ<
-batch_normalization_7_readvariableop_resource:	ђ>
/batch_normalization_7_readvariableop_1_resource:	ђM
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:	ђO
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:	ђV
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource:@└@
2conv2d_transpose_3_biasadd_readvariableop_resource:@;
-batch_normalization_8_readvariableop_resource:@=
/batch_normalization_8_readvariableop_1_resource:@L
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:@U
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource:D@
2conv2d_transpose_4_biasadd_readvariableop_resource:
identityѕб"batch_normalization/AssignNewValueб$batch_normalization/AssignNewValue_1б3batch_normalization/FusedBatchNormV3/ReadVariableOpб5batch_normalization/FusedBatchNormV3/ReadVariableOp_1б"batch_normalization/ReadVariableOpб$batch_normalization/ReadVariableOp_1б$batch_normalization_1/AssignNewValueб&batch_normalization_1/AssignNewValue_1б5batch_normalization_1/FusedBatchNormV3/ReadVariableOpб7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_1/ReadVariableOpб&batch_normalization_1/ReadVariableOp_1б$batch_normalization_2/AssignNewValueб&batch_normalization_2/AssignNewValue_1б5batch_normalization_2/FusedBatchNormV3/ReadVariableOpб7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_2/ReadVariableOpб&batch_normalization_2/ReadVariableOp_1б$batch_normalization_3/AssignNewValueб&batch_normalization_3/AssignNewValue_1б5batch_normalization_3/FusedBatchNormV3/ReadVariableOpб7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_3/ReadVariableOpб&batch_normalization_3/ReadVariableOp_1б$batch_normalization_4/AssignNewValueб&batch_normalization_4/AssignNewValue_1б5batch_normalization_4/FusedBatchNormV3/ReadVariableOpб7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_4/ReadVariableOpб&batch_normalization_4/ReadVariableOp_1б$batch_normalization_5/AssignNewValueб&batch_normalization_5/AssignNewValue_1б5batch_normalization_5/FusedBatchNormV3/ReadVariableOpб7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_5/ReadVariableOpб&batch_normalization_5/ReadVariableOp_1б$batch_normalization_6/AssignNewValueб&batch_normalization_6/AssignNewValue_1б5batch_normalization_6/FusedBatchNormV3/ReadVariableOpб7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_6/ReadVariableOpб&batch_normalization_6/ReadVariableOp_1б$batch_normalization_7/AssignNewValueб&batch_normalization_7/AssignNewValue_1б5batch_normalization_7/FusedBatchNormV3/ReadVariableOpб7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_7/ReadVariableOpб&batch_normalization_7/ReadVariableOp_1б$batch_normalization_8/AssignNewValueб&batch_normalization_8/AssignNewValue_1б5batch_normalization_8/FusedBatchNormV3/ReadVariableOpб7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_8/ReadVariableOpб&batch_normalization_8/ReadVariableOp_1бconv2d/BiasAdd/ReadVariableOpбconv2d/Conv2D/ReadVariableOpбconv2d_1/BiasAdd/ReadVariableOpбconv2d_1/Conv2D/ReadVariableOpбconv2d_2/BiasAdd/ReadVariableOpбconv2d_2/Conv2D/ReadVariableOpбconv2d_3/BiasAdd/ReadVariableOpбconv2d_3/Conv2D/ReadVariableOpб'conv2d_transpose/BiasAdd/ReadVariableOpб0conv2d_transpose/conv2d_transpose/ReadVariableOpб)conv2d_transpose_1/BiasAdd/ReadVariableOpб2conv2d_transpose_1/conv2d_transpose/ReadVariableOpб)conv2d_transpose_2/BiasAdd/ReadVariableOpб2conv2d_transpose_2/conv2d_transpose/ReadVariableOpб)conv2d_transpose_3/BiasAdd/ReadVariableOpб2conv2d_transpose_3/conv2d_transpose/ReadVariableOpб)conv2d_transpose_4/BiasAdd/ReadVariableOpб2conv2d_transpose_4/conv2d_transpose/ReadVariableOpE
reshape/ShapeShapeinputs_0*
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
valueB:щ
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
B :ђZ
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ђY
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Л
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:ђ
reshape/ReshapeReshapeinputs_0reshape/Reshape/shape:output:0*
T0*1
_output_shapes
:         ђђG
reshape_1/ShapeShapeinputs_1*
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
valueB:Ѓ
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
B :ђ\
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ђ[
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :█
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:ё
reshape_1/ReshapeReshapeinputs_1 reshape_1/Reshape/shape:output:0*
T0*1
_output_shapes
:         ђђG
reshape_2/ShapeShapeinputs_2*
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
valueB:Ѓ
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
B :ђ\
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ђ[
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :█
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:ё
reshape_2/ReshapeReshapeinputs_2 reshape_2/Reshape/shape:output:0*
T0*1
_output_shapes
:         ђђG
reshape_3/ShapeShapeinputs_3*
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
valueB:Ѓ
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
B :ђ\
reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ђ[
reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :█
reshape_3/Reshape/shapePack reshape_3/strided_slice:output:0"reshape_3/Reshape/shape/1:output:0"reshape_3/Reshape/shape/2:output:0"reshape_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:ё
reshape_3/ReshapeReshapeinputs_3 reshape_3/Reshape/shape:output:0*
T0*1
_output_shapes
:         ђђY
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :з
concatenate/concatConcatV2reshape/Reshape:output:0reshape_1/Reshape:output:0reshape_2/Reshape:output:0reshape_3/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:         ђђі
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype0ј
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype0г
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0░
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0┐
$batch_normalization/FusedBatchNormV3FusedBatchNormV3concatenate/concat:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ:::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<ђ
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0і
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0і
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0╠
conv2d/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЧЧ@*
paddingVALID*
strides
ђ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ћ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЧЧ@f

conv2d/EluEluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:         ЧЧ@ј
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0њ
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0░
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0┤
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0к
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d/Elu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЧЧ@:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<ѕ
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0њ
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0Ј
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype0М
conv2d_1/Conv2DConv2D*batch_normalization_1/FusedBatchNormV3:y:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЩЩђ*
paddingVALID*
strides
Ё
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Џ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЩЩђk
conv2d_1/EluEluconv2d_1/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЩЩђЈ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Њ
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0▒
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0х
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0═
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_1/Elu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЩЩђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<ѕ
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0њ
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0љ
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0М
conv2d_2/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЭЭђ*
paddingVALID*
strides
Ё
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Џ
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЭЭђk
conv2d_2/EluEluconv2d_2/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЭЭђЈ
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Њ
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0▒
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0х
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0═
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_2/Elu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЭЭђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<ѕ
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0њ
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0љ
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0М
conv2d_3/Conv2DConv2D*batch_normalization_3/FusedBatchNormV3:y:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ШШђ*
paddingVALID*
strides
Ё
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Џ
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ШШђk
conv2d_3/EluEluconv2d_3/BiasAdd:output:0*
T0*2
_output_shapes 
:         ШШђЈ
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Њ
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0▒
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0х
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0═
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_3/Elu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ШШђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<ѕ
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0њ
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :м
concatenate_1/concatConcatV2conv2d_3/Elu:activations:0*batch_normalization_4/FusedBatchNormV3:y:0"concatenate_1/concat/axis:output:0*
N*
T0*2
_output_shapes 
:         ШШђc
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
valueB:д
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
B :Э[
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :Э[
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђя
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
valueB:«
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask┤
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0ў
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0concatenate_1/concat:output:0*
T0*2
_output_shapes 
:         ЭЭђ*
paddingVALID*
strides
Ћ
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0й
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЭЭђ{
conv2d_transpose/EluElu!conv2d_transpose/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЭЭђЈ
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Њ
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0▒
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0х
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Н
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3"conv2d_transpose/Elu:activations:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЭЭђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<ѕ
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0њ
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0[
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :м
concatenate_2/concatConcatV2conv2d_2/Elu:activations:0*batch_normalization_5/FusedBatchNormV3:y:0"concatenate_2/concat/axis:output:0*
N*
T0*2
_output_shapes 
:         ЭЭђe
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
valueB:░
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
B :Щ]
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :Щ]
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђУ
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
valueB:И
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskИ
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0ъ
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0concatenate_2/concat:output:0*
T0*2
_output_shapes 
:         ЩЩђ*
paddingVALID*
strides
Ў
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0├
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЩЩђ
conv2d_transpose_1/EluElu#conv2d_transpose_1/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЩЩђЈ
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Њ
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0▒
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0х
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0О
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_1/Elu:activations:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЩЩђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<ѕ
$batch_normalization_6/AssignNewValueAssignVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0њ
&batch_normalization_6/AssignNewValue_1AssignVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0[
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :м
concatenate_3/concatConcatV2conv2d_1/Elu:activations:0*batch_normalization_6/FusedBatchNormV3:y:0"concatenate_3/concat/axis:output:0*
N*
T0*2
_output_shapes 
:         ЩЩђe
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
valueB:░
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
B :Ч]
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value
B :Ч]
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђУ
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
valueB:И
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskИ
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0ъ
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0concatenate_3/concat:output:0*
T0*2
_output_shapes 
:         ЧЧђ*
paddingVALID*
strides
Ў
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0├
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЧЧђ
conv2d_transpose_2/EluElu#conv2d_transpose_2/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЧЧђЈ
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Њ
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0▒
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0х
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0О
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_2/Elu:activations:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЧЧђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<ѕ
$batch_normalization_7/AssignNewValueAssignVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0њ
&batch_normalization_7/AssignNewValue_1AssignVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0[
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :л
concatenate_4/concatConcatV2conv2d/Elu:activations:0*batch_normalization_7/FusedBatchNormV3:y:0"concatenate_4/concat/axis:output:0*
N*
T0*2
_output_shapes 
:         ЧЧ└e
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
valueB:░
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
B :ђ]
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value
B :ђ\
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@У
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
valueB:И
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskи
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@└*
dtype0Ю
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0concatenate_4/concat:output:0*
T0*1
_output_shapes
:         ђђ@*
paddingVALID*
strides
ў
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0┬
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ@~
conv2d_transpose_3/EluElu#conv2d_transpose_3/BiasAdd:output:0*
T0*1
_output_shapes
:         ђђ@ј
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:@*
dtype0њ
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:@*
dtype0░
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0┤
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0м
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_3/Elu:activations:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ@:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<ѕ
$batch_normalization_8/AssignNewValueAssignVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0њ
&batch_normalization_8/AssignNewValue_1AssignVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0[
concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :м
concatenate_5/concatConcatV2concatenate/concat:output:0*batch_normalization_8/FusedBatchNormV3:y:0"concatenate_5/concat/axis:output:0*
N*
T0*1
_output_shapes
:         ђђDe
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
valueB:░
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
B :ђ]
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value
B :ђ\
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :У
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
valueB:И
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskХ
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:D*
dtype0Ю
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0concatenate_5/concat:output:0*
T0*1
_output_shapes
:         ђђ*
paddingVALID*
strides
ў
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┬
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ|
IdentityIdentity#conv2d_transpose_4/BiasAdd:output:0^NoOp*
T0*1
_output_shapes
:         ђђ┼
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesМ
л:         ђђ:         ђђ:         ђђ:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
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
:         ђђ
"
_user_specified_name
inputs/0:WS
-
_output_shapes
:         ђђ
"
_user_specified_name
inputs/1:WS
-
_output_shapes
:         ђђ
"
_user_specified_name
inputs/2:WS
-
_output_shapes
:         ђђ
"
_user_specified_name
inputs/3
Ў
┐
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1033833

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ:::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0m
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:         ђђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         ђђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
Ї
Ю
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1031151

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0И
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ@:@:@:@:@:*
epsilon%oЃ:*
is_training( m
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:         ђђ@░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
Ї
Ю
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1033959

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0И
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЧЧ@:@:@:@:@:*
epsilon%oЃ:*
is_training( m
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:         ЧЧ@░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
П
о
7__inference_batch_normalization_7_layer_call_fn_1034975

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЈ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЧЧђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1031086z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ЧЧђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЧЧђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ЧЧђ
 
_user_specified_nameinputs
П
А
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1030181

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
П
А
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1030407

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
Э#
Ю
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1034478

inputsD
(conv2d_transpose_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
valueB:┘
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
valueB:┘
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
B :ђy
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
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskњ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0я
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,                           ђ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0џ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђi
EluEluBiasAdd:output:0*
T0*B
_output_shapes0
.:,                           ђ{
IdentityIdentityElu:activations:0^NoOp*
T0*B
_output_shapes0
.:,                           ђЂ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
ѓ#
ю
O__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1030603

inputsB
(conv2d_transpose_readvariableop_resource:D-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
valueB:┘
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
valueB:┘
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
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskљ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:D*
dtype0П
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ў
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+                           Ђ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           D: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+                           D
 
_user_specified_nameinputs
┘
┐
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1029843

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
█
┴
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1030551

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
Џ
Ђ
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1034141

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0Ю
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЭЭђ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ђ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЭЭђY
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:         ЭЭђk
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
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
Ф
┼
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1031599

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0╦
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ШШђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ШШђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
ы
[
/__inference_concatenate_4_layer_call_fn_1035066
inputs_0
inputs_1
identityл
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЧЧ└* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_concatenate_4_layer_call_and_return_conditional_losses_1031103k
IdentityIdentityPartitionedCall:output:0*
T0*2
_output_shapes 
:         ЧЧ└"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         ЧЧ@:         ЧЧђ:[ W
1
_output_shapes
:         ЧЧ@
"
_user_specified_name
inputs/0:\X
2
_output_shapes 
:         ЧЧђ
"
_user_specified_name
inputs/1
кЄ
н?
 __inference__traced_save_1035793
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

identity_1ѕбMergeV2Checkpointsw
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
_temp/partЂ
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
value	B : Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ЧJ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:є*
dtype0*цJ
valueџJBЌJєB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH■
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:є*
dtype0*б
valueўBЋєB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ї=
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop2savev2_conv2d_transpose_kernel_read_readvariableop0savev2_conv2d_transpose_bias_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop4savev2_conv2d_transpose_1_kernel_read_readvariableop2savev2_conv2d_transpose_1_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop4savev2_conv2d_transpose_2_kernel_read_readvariableop2savev2_conv2d_transpose_2_bias_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop4savev2_conv2d_transpose_3_kernel_read_readvariableop2savev2_conv2d_transpose_3_bias_read_readvariableop6savev2_batch_normalization_8_gamma_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableop4savev2_conv2d_transpose_4_kernel_read_readvariableop2savev2_conv2d_transpose_4_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop<savev2_adam_batch_normalization_2_beta_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop<savev2_adam_batch_normalization_3_beta_m_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_3_bias_m_read_readvariableop=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop<savev2_adam_batch_normalization_4_beta_m_read_readvariableop9savev2_adam_conv2d_transpose_kernel_m_read_readvariableop7savev2_adam_conv2d_transpose_bias_m_read_readvariableop=savev2_adam_batch_normalization_5_gamma_m_read_readvariableop<savev2_adam_batch_normalization_5_beta_m_read_readvariableop;savev2_adam_conv2d_transpose_1_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_1_bias_m_read_readvariableop=savev2_adam_batch_normalization_6_gamma_m_read_readvariableop<savev2_adam_batch_normalization_6_beta_m_read_readvariableop;savev2_adam_conv2d_transpose_2_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_2_bias_m_read_readvariableop=savev2_adam_batch_normalization_7_gamma_m_read_readvariableop<savev2_adam_batch_normalization_7_beta_m_read_readvariableop;savev2_adam_conv2d_transpose_3_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_3_bias_m_read_readvariableop=savev2_adam_batch_normalization_8_gamma_m_read_readvariableop<savev2_adam_batch_normalization_8_beta_m_read_readvariableop;savev2_adam_conv2d_transpose_4_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_4_bias_m_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop<savev2_adam_batch_normalization_2_beta_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop<savev2_adam_batch_normalization_3_beta_v_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop/savev2_adam_conv2d_3_bias_v_read_readvariableop=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop<savev2_adam_batch_normalization_4_beta_v_read_readvariableop9savev2_adam_conv2d_transpose_kernel_v_read_readvariableop7savev2_adam_conv2d_transpose_bias_v_read_readvariableop=savev2_adam_batch_normalization_5_gamma_v_read_readvariableop<savev2_adam_batch_normalization_5_beta_v_read_readvariableop;savev2_adam_conv2d_transpose_1_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_1_bias_v_read_readvariableop=savev2_adam_batch_normalization_6_gamma_v_read_readvariableop<savev2_adam_batch_normalization_6_beta_v_read_readvariableop;savev2_adam_conv2d_transpose_2_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_2_bias_v_read_readvariableop=savev2_adam_batch_normalization_7_gamma_v_read_readvariableop<savev2_adam_batch_normalization_7_beta_v_read_readvariableop;savev2_adam_conv2d_transpose_3_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_3_bias_v_read_readvariableop=savev2_adam_batch_normalization_8_gamma_v_read_readvariableop<savev2_adam_batch_normalization_8_beta_v_read_readvariableop;savev2_adam_conv2d_transpose_4_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_4_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Ќ
dtypesї
Ѕ2є	љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:І
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

identity_1Identity_1:output:0*┼	
_input_shapes│	
░	: :::::@:@:@:@:@:@:@ђ:ђ:ђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:ђ:ђ:@└:@:@:@:@:@:D:: : : : : : : :::@:@:@:@:@ђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:@└:@:@:@:D::::@:@:@:@:@ђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:@└:@:@:@:D:: 2(
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
:ђђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:! 

_output_shapes	
:ђ:!!

_output_shapes	
:ђ:!"

_output_shapes	
:ђ:.#*
(
_output_shapes
:ђђ:!$
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
:ђђ:!*

_output_shapes	
:ђ:!+

_output_shapes	
:ђ:!,

_output_shapes	
:ђ:!-

_output_shapes	
:ђ:!.

_output_shapes	
:ђ:-/)
'
_output_shapes
:@└: 0
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
:ђђ:!Q

_output_shapes	
:ђ:!R

_output_shapes	
:ђ:!S

_output_shapes	
:ђ:.T*
(
_output_shapes
:ђђ:!U
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
:ђђ:!Y

_output_shapes	
:ђ:!Z

_output_shapes	
:ђ:![

_output_shapes	
:ђ:-\)
'
_output_shapes
:@└: ]
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
:ђђ:!u

_output_shapes	
:ђ:!v

_output_shapes	
:ђ:!w

_output_shapes	
:ђ:.x*
(
_output_shapes
:ђђ:!y
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
:ђђ:!}

_output_shapes	
:ђ:!~

_output_shapes	
:ђ:!

_output_shapes	
:ђ:.ђ)
'
_output_shapes
:@└:!Ђ
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
:D:!Ё

_output_shapes
::є

_output_shapes
: 
Ї
Ю
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1035259

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0И
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ@:@:@:@:@:*
epsilon%oЃ:*
is_training( m
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:         ђђ@░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
ПЌ
ќ
B__inference_model_layer_call_and_return_conditional_losses_1032636	
u_vel	
v_vel	
w_vel	
pr0_2)
batch_normalization_1032504:)
batch_normalization_1032506:)
batch_normalization_1032508:)
batch_normalization_1032510:(
conv2d_1032513:@
conv2d_1032515:@+
batch_normalization_1_1032518:@+
batch_normalization_1_1032520:@+
batch_normalization_1_1032522:@+
batch_normalization_1_1032524:@+
conv2d_1_1032527:@ђ
conv2d_1_1032529:	ђ,
batch_normalization_2_1032532:	ђ,
batch_normalization_2_1032534:	ђ,
batch_normalization_2_1032536:	ђ,
batch_normalization_2_1032538:	ђ,
conv2d_2_1032541:ђђ
conv2d_2_1032543:	ђ,
batch_normalization_3_1032546:	ђ,
batch_normalization_3_1032548:	ђ,
batch_normalization_3_1032550:	ђ,
batch_normalization_3_1032552:	ђ,
conv2d_3_1032555:ђђ
conv2d_3_1032557:	ђ,
batch_normalization_4_1032560:	ђ,
batch_normalization_4_1032562:	ђ,
batch_normalization_4_1032564:	ђ,
batch_normalization_4_1032566:	ђ4
conv2d_transpose_1032570:ђђ'
conv2d_transpose_1032572:	ђ,
batch_normalization_5_1032575:	ђ,
batch_normalization_5_1032577:	ђ,
batch_normalization_5_1032579:	ђ,
batch_normalization_5_1032581:	ђ6
conv2d_transpose_1_1032585:ђђ)
conv2d_transpose_1_1032587:	ђ,
batch_normalization_6_1032590:	ђ,
batch_normalization_6_1032592:	ђ,
batch_normalization_6_1032594:	ђ,
batch_normalization_6_1032596:	ђ6
conv2d_transpose_2_1032600:ђђ)
conv2d_transpose_2_1032602:	ђ,
batch_normalization_7_1032605:	ђ,
batch_normalization_7_1032607:	ђ,
batch_normalization_7_1032609:	ђ,
batch_normalization_7_1032611:	ђ5
conv2d_transpose_3_1032615:@└(
conv2d_transpose_3_1032617:@+
batch_normalization_8_1032620:@+
batch_normalization_8_1032622:@+
batch_normalization_8_1032624:@+
batch_normalization_8_1032626:@4
conv2d_transpose_4_1032630:D(
conv2d_transpose_4_1032632:
identityѕб+batch_normalization/StatefulPartitionedCallб-batch_normalization_1/StatefulPartitionedCallб-batch_normalization_2/StatefulPartitionedCallб-batch_normalization_3/StatefulPartitionedCallб-batch_normalization_4/StatefulPartitionedCallб-batch_normalization_5/StatefulPartitionedCallб-batch_normalization_6/StatefulPartitionedCallб-batch_normalization_7/StatefulPartitionedCallб-batch_normalization_8/StatefulPartitionedCallбconv2d/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallб conv2d_2/StatefulPartitionedCallб conv2d_3/StatefulPartitionedCallб(conv2d_transpose/StatefulPartitionedCallб*conv2d_transpose_1/StatefulPartitionedCallб*conv2d_transpose_2/StatefulPartitionedCallб*conv2d_transpose_3/StatefulPartitionedCallб*conv2d_transpose_4/StatefulPartitionedCall├
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
GPU2*0J 8ѓ *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_1030637К
reshape_1/PartitionedCallPartitionedCallv_vel*
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
GPU2*0J 8ѓ *O
fJRH
F__inference_reshape_1_layer_call_and_return_conditional_losses_1030653К
reshape_2/PartitionedCallPartitionedCallw_vel*
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
GPU2*0J 8ѓ *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_1030669К
reshape_3/PartitionedCallPartitionedCallpr0_2*
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
GPU2*0J 8ѓ *O
fJRH
F__inference_reshape_3_layer_call_and_return_conditional_losses_1030685Н
concatenate/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0"reshape_1/PartitionedCall:output:0"reshape_2/PartitionedCall:output:0"reshape_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_1030696є
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0batch_normalization_1032504batch_normalization_1032506batch_normalization_1032508batch_normalization_1032510*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1031815д
conv2d/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_1032513conv2d_1032515*
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
GPU2*0J 8ѓ *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_1030736Ћ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_1_1032518batch_normalization_1_1032520batch_normalization_1_1032522batch_normalization_1_1032524*
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1031761▒
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_1_1032527conv2d_1_1032529*
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
GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1030780ў
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_2_1032532batch_normalization_2_1032534batch_normalization_2_1032536batch_normalization_2_1032538*
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1031707▒
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_2_1032541conv2d_2_1032543*
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
GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1030824ў
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_3_1032546batch_normalization_3_1032548batch_normalization_3_1032550batch_normalization_3_1032552*
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1031653▒
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_3_1032555conv2d_3_1032557*
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
GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1030868ў
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_4_1032560batch_normalization_4_1032562batch_normalization_4_1032564batch_normalization_4_1032566*
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1031599Г
concatenate_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:06batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ШШђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1030908┴
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0conv2d_transpose_1032570conv2d_transpose_1032572*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1030933а
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0batch_normalization_5_1032575batch_normalization_5_1032577batch_normalization_5_1032579batch_normalization_5_1032581*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1031539Г
concatenate_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:06batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_1030973╔
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0conv2d_transpose_1_1032585conv2d_transpose_1_1032587*
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
GPU2*0J 8ѓ *X
fSRQ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1030998б
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0batch_normalization_6_1032590batch_normalization_6_1032592batch_normalization_6_1032594batch_normalization_6_1032596*
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1031479Г
concatenate_3/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:06batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЩЩђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_concatenate_3_layer_call_and_return_conditional_losses_1031038╔
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0conv2d_transpose_2_1032600conv2d_transpose_2_1032602*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЧЧђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1031063б
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0batch_normalization_7_1032605batch_normalization_7_1032607batch_normalization_7_1032609batch_normalization_7_1032611*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЧЧђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1031419Ф
concatenate_4/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:06batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЧЧ└* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_concatenate_4_layer_call_and_return_conditional_losses_1031103╚
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv2d_transpose_3_1032615conv2d_transpose_3_1032617*
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
GPU2*0J 8ѓ *X
fSRQ
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1031128А
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0batch_normalization_8_1032620batch_normalization_8_1032622batch_normalization_8_1032624batch_normalization_8_1032626*
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1031359Д
concatenate_5/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:06batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђD* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_concatenate_5_layer_call_and_return_conditional_losses_1031168╚
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0conv2d_transpose_4_1032630conv2d_transpose_4_1032632*
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
GPU2*0J 8ѓ *X
fSRQ
O__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1031192ї
IdentityIdentity3conv2d_transpose_4/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђП
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesМ
л:         ђђ:         ђђ:         ђђ:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
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

_user_specified_nameu_vel:TP
-
_output_shapes
:         ђђ

_user_specified_namev_vel:TP
-
_output_shapes
:         ђђ

_user_specified_namew_vel:TP
-
_output_shapes
:         ђђ

_user_specified_namepr0.2
Ф
┼
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1031539

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0╦
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЭЭђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЭЭђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
т
ю
O__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1035368

inputsB
(conv2d_transpose_readvariableop_resource:D-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
B :ђJ
stack/2Const*
_output_shapes
: *
dtype0*
value
B :ђI
stack/3Const*
_output_shapes
: *
dtype0*
value	B :Ѕ
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
valueB:┘
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskљ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:D*
dtype0═
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:         ђђ*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ѕ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђi
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:         ђђЂ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ђђD: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:         ђђD
 
_user_specified_nameinputs
═
Ю
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1033923

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
Ю
А
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1034825

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0й
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЩЩђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
ю	
о
7__inference_batch_normalization_3_layer_call_fn_1034167

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЮ
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1030035і
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
П
А
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1030294

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
Ф
┼
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1034626

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0╦
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЭЭђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЭЭђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
Ю
А
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1031021

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0й
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЩЩђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
ќ	
м
7__inference_batch_normalization_8_layer_call_fn_1035166

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallъ
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1030520Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
в
┼
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1034590

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
П
о
7__inference_batch_normalization_2_layer_call_fn_1034036

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЈ
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1030803z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЩЩђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs
в
┼
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1034229

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
в
┼
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1034807

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
█
о
7__inference_batch_normalization_3_layer_call_fn_1034193

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЇ
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1031653z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЭЭђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
╦
Ю
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1034502

inputsD
(conv2d_transpose_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
B :ЭJ
stack/2Const*
_output_shapes
: *
dtype0*
value
B :ЭJ
stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђЅ
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
valueB:┘
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskњ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0╬
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*2
_output_shapes 
:         ЭЭђ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0і
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЭЭђY
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:         ЭЭђk
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђЂ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ШШђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Z V
2
_output_shapes 
:         ШШђ
 
_user_specified_nameinputs
ю	
о
7__inference_batch_normalization_4_layer_call_fn_1034311

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЮ
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1030099і
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
І
Џ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1030715

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0И
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ:::::*
epsilon%oЃ:*
is_training( m
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:         ђђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         ђђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
КА
Ѕ7
"__inference__wrapped_model_1029790	
u_vel	
v_vel	
w_vel	
pr0_2?
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
-model_conv2d_1_conv2d_readvariableop_resource:@ђ=
.model_conv2d_1_biasadd_readvariableop_resource:	ђB
3model_batch_normalization_2_readvariableop_resource:	ђD
5model_batch_normalization_2_readvariableop_1_resource:	ђS
Dmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	ђU
Fmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	ђI
-model_conv2d_2_conv2d_readvariableop_resource:ђђ=
.model_conv2d_2_biasadd_readvariableop_resource:	ђB
3model_batch_normalization_3_readvariableop_resource:	ђD
5model_batch_normalization_3_readvariableop_1_resource:	ђS
Dmodel_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	ђU
Fmodel_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	ђI
-model_conv2d_3_conv2d_readvariableop_resource:ђђ=
.model_conv2d_3_biasadd_readvariableop_resource:	ђB
3model_batch_normalization_4_readvariableop_resource:	ђD
5model_batch_normalization_4_readvariableop_1_resource:	ђS
Dmodel_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	ђU
Fmodel_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	ђ[
?model_conv2d_transpose_conv2d_transpose_readvariableop_resource:ђђE
6model_conv2d_transpose_biasadd_readvariableop_resource:	ђB
3model_batch_normalization_5_readvariableop_resource:	ђD
5model_batch_normalization_5_readvariableop_1_resource:	ђS
Dmodel_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:	ђU
Fmodel_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:	ђ]
Amodel_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:ђђG
8model_conv2d_transpose_1_biasadd_readvariableop_resource:	ђB
3model_batch_normalization_6_readvariableop_resource:	ђD
5model_batch_normalization_6_readvariableop_1_resource:	ђS
Dmodel_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:	ђU
Fmodel_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:	ђ]
Amodel_conv2d_transpose_2_conv2d_transpose_readvariableop_resource:ђђG
8model_conv2d_transpose_2_biasadd_readvariableop_resource:	ђB
3model_batch_normalization_7_readvariableop_resource:	ђD
5model_batch_normalization_7_readvariableop_1_resource:	ђS
Dmodel_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:	ђU
Fmodel_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:	ђ\
Amodel_conv2d_transpose_3_conv2d_transpose_readvariableop_resource:@└F
8model_conv2d_transpose_3_biasadd_readvariableop_resource:@A
3model_batch_normalization_8_readvariableop_resource:@C
5model_batch_normalization_8_readvariableop_1_resource:@R
Dmodel_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:@T
Fmodel_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:@[
Amodel_conv2d_transpose_4_conv2d_transpose_readvariableop_resource:DF
8model_conv2d_transpose_4_biasadd_readvariableop_resource:
identityѕб9model/batch_normalization/FusedBatchNormV3/ReadVariableOpб;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1б(model/batch_normalization/ReadVariableOpб*model/batch_normalization/ReadVariableOp_1б;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpб=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1б*model/batch_normalization_1/ReadVariableOpб,model/batch_normalization_1/ReadVariableOp_1б;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOpб=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1б*model/batch_normalization_2/ReadVariableOpб,model/batch_normalization_2/ReadVariableOp_1б;model/batch_normalization_3/FusedBatchNormV3/ReadVariableOpб=model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1б*model/batch_normalization_3/ReadVariableOpб,model/batch_normalization_3/ReadVariableOp_1б;model/batch_normalization_4/FusedBatchNormV3/ReadVariableOpб=model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б*model/batch_normalization_4/ReadVariableOpб,model/batch_normalization_4/ReadVariableOp_1б;model/batch_normalization_5/FusedBatchNormV3/ReadVariableOpб=model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б*model/batch_normalization_5/ReadVariableOpб,model/batch_normalization_5/ReadVariableOp_1б;model/batch_normalization_6/FusedBatchNormV3/ReadVariableOpб=model/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1б*model/batch_normalization_6/ReadVariableOpб,model/batch_normalization_6/ReadVariableOp_1б;model/batch_normalization_7/FusedBatchNormV3/ReadVariableOpб=model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1б*model/batch_normalization_7/ReadVariableOpб,model/batch_normalization_7/ReadVariableOp_1б;model/batch_normalization_8/FusedBatchNormV3/ReadVariableOpб=model/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1б*model/batch_normalization_8/ReadVariableOpб,model/batch_normalization_8/ReadVariableOp_1б#model/conv2d/BiasAdd/ReadVariableOpб"model/conv2d/Conv2D/ReadVariableOpб%model/conv2d_1/BiasAdd/ReadVariableOpб$model/conv2d_1/Conv2D/ReadVariableOpб%model/conv2d_2/BiasAdd/ReadVariableOpб$model/conv2d_2/Conv2D/ReadVariableOpб%model/conv2d_3/BiasAdd/ReadVariableOpб$model/conv2d_3/Conv2D/ReadVariableOpб-model/conv2d_transpose/BiasAdd/ReadVariableOpб6model/conv2d_transpose/conv2d_transpose/ReadVariableOpб/model/conv2d_transpose_1/BiasAdd/ReadVariableOpб8model/conv2d_transpose_1/conv2d_transpose/ReadVariableOpб/model/conv2d_transpose_2/BiasAdd/ReadVariableOpб8model/conv2d_transpose_2/conv2d_transpose/ReadVariableOpб/model/conv2d_transpose_3/BiasAdd/ReadVariableOpб8model/conv2d_transpose_3/conv2d_transpose/ReadVariableOpб/model/conv2d_transpose_4/BiasAdd/ReadVariableOpб8model/conv2d_transpose_4/conv2d_transpose/ReadVariableOpH
model/reshape/ShapeShapeu_vel*
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
valueB:Ќ
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
B :ђ`
model/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ђ_
model/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :№
model/reshape/Reshape/shapePack$model/reshape/strided_slice:output:0&model/reshape/Reshape/shape/1:output:0&model/reshape/Reshape/shape/2:output:0&model/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ѕ
model/reshape/ReshapeReshapeu_vel$model/reshape/Reshape/shape:output:0*
T0*1
_output_shapes
:         ђђJ
model/reshape_1/ShapeShapev_vel*
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
valueB:А
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
B :ђb
model/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ђa
model/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :щ
model/reshape_1/Reshape/shapePack&model/reshape_1/strided_slice:output:0(model/reshape_1/Reshape/shape/1:output:0(model/reshape_1/Reshape/shape/2:output:0(model/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ї
model/reshape_1/ReshapeReshapev_vel&model/reshape_1/Reshape/shape:output:0*
T0*1
_output_shapes
:         ђђJ
model/reshape_2/ShapeShapew_vel*
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
valueB:А
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
B :ђb
model/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ђa
model/reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :щ
model/reshape_2/Reshape/shapePack&model/reshape_2/strided_slice:output:0(model/reshape_2/Reshape/shape/1:output:0(model/reshape_2/Reshape/shape/2:output:0(model/reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ї
model/reshape_2/ReshapeReshapew_vel&model/reshape_2/Reshape/shape:output:0*
T0*1
_output_shapes
:         ђђJ
model/reshape_3/ShapeShapepr0_2*
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
valueB:А
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
B :ђb
model/reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ђa
model/reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :щ
model/reshape_3/Reshape/shapePack&model/reshape_3/strided_slice:output:0(model/reshape_3/Reshape/shape/1:output:0(model/reshape_3/Reshape/shape/2:output:0(model/reshape_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ї
model/reshape_3/ReshapeReshapepr0_2&model/reshape_3/Reshape/shape:output:0*
T0*1
_output_shapes
:         ђђ_
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ќ
model/concatenate/concatConcatV2model/reshape/Reshape:output:0 model/reshape_1/Reshape:output:0 model/reshape_2/Reshape:output:0 model/reshape_3/Reshape:output:0&model/concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:         ђђќ
(model/batch_normalization/ReadVariableOpReadVariableOp1model_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype0џ
*model/batch_normalization/ReadVariableOp_1ReadVariableOp3model_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype0И
9model/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpBmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╝
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Н
*model/batch_normalization/FusedBatchNormV3FusedBatchNormV3!model/concatenate/concat:output:00model/batch_normalization/ReadVariableOp:value:02model/batch_normalization/ReadVariableOp_1:value:0Amodel/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Cmodel/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ:::::*
epsilon%oЃ:*
is_training( ќ
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0я
model/conv2d/Conv2DConv2D.model/batch_normalization/FusedBatchNormV3:y:0*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЧЧ@*
paddingVALID*
strides
ї
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0д
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЧЧ@r
model/conv2d/EluElumodel/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:         ЧЧ@џ
*model/batch_normalization_1/ReadVariableOpReadVariableOp3model_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0ъ
,model/batch_normalization_1/ReadVariableOp_1ReadVariableOp5model_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0╝
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0└
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0▄
,model/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3model/conv2d/Elu:activations:02model/batch_normalization_1/ReadVariableOp:value:04model/batch_normalization_1/ReadVariableOp_1:value:0Cmodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЧЧ@:@:@:@:@:*
epsilon%oЃ:*
is_training( Џ
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype0т
model/conv2d_1/Conv2DConv2D0model/batch_normalization_1/FusedBatchNormV3:y:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЩЩђ*
paddingVALID*
strides
Љ
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Г
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЩЩђw
model/conv2d_1/EluElumodel/conv2d_1/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЩЩђЏ
*model/batch_normalization_2/ReadVariableOpReadVariableOp3model_batch_normalization_2_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ъ
,model/batch_normalization_2/ReadVariableOp_1ReadVariableOp5model_batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0й
;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0┴
=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0с
,model/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3 model/conv2d_1/Elu:activations:02model/batch_normalization_2/ReadVariableOp:value:04model/batch_normalization_2/ReadVariableOp_1:value:0Cmodel/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЩЩђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( ю
$model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0т
model/conv2d_2/Conv2DConv2D0model/batch_normalization_2/FusedBatchNormV3:y:0,model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЭЭђ*
paddingVALID*
strides
Љ
%model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Г
model/conv2d_2/BiasAddBiasAddmodel/conv2d_2/Conv2D:output:0-model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЭЭђw
model/conv2d_2/EluElumodel/conv2d_2/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЭЭђЏ
*model/batch_normalization_3/ReadVariableOpReadVariableOp3model_batch_normalization_3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ъ
,model/batch_normalization_3/ReadVariableOp_1ReadVariableOp5model_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0й
;model/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0┴
=model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0с
,model/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3 model/conv2d_2/Elu:activations:02model/batch_normalization_3/ReadVariableOp:value:04model/batch_normalization_3/ReadVariableOp_1:value:0Cmodel/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЭЭђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( ю
$model/conv2d_3/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0т
model/conv2d_3/Conv2DConv2D0model/batch_normalization_3/FusedBatchNormV3:y:0,model/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ШШђ*
paddingVALID*
strides
Љ
%model/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Г
model/conv2d_3/BiasAddBiasAddmodel/conv2d_3/Conv2D:output:0-model/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ШШђw
model/conv2d_3/EluElumodel/conv2d_3/BiasAdd:output:0*
T0*2
_output_shapes 
:         ШШђЏ
*model/batch_normalization_4/ReadVariableOpReadVariableOp3model_batch_normalization_4_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ъ
,model/batch_normalization_4/ReadVariableOp_1ReadVariableOp5model_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0й
;model/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0┴
=model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0с
,model/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3 model/conv2d_3/Elu:activations:02model/batch_normalization_4/ReadVariableOp:value:04model/batch_normalization_4/ReadVariableOp_1:value:0Cmodel/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ШШђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( a
model/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ж
model/concatenate_1/concatConcatV2 model/conv2d_3/Elu:activations:00model/batch_normalization_4/FusedBatchNormV3:y:0(model/concatenate_1/concat/axis:output:0*
N*
T0*2
_output_shapes 
:         ШШђo
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
valueB:─
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
B :Эa
model/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :Эa
model/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђЧ
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
valueB:╠
&model/conv2d_transpose/strided_slice_1StridedSlice%model/conv2d_transpose/stack:output:05model/conv2d_transpose/strided_slice_1/stack:output:07model/conv2d_transpose/strided_slice_1/stack_1:output:07model/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask└
6model/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp?model_conv2d_transpose_conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0░
'model/conv2d_transpose/conv2d_transposeConv2DBackpropInput%model/conv2d_transpose/stack:output:0>model/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0#model/concatenate_1/concat:output:0*
T0*2
_output_shapes 
:         ЭЭђ*
paddingVALID*
strides
А
-model/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp6model_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0¤
model/conv2d_transpose/BiasAddBiasAdd0model/conv2d_transpose/conv2d_transpose:output:05model/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЭЭђЄ
model/conv2d_transpose/EluElu'model/conv2d_transpose/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЭЭђЏ
*model/batch_normalization_5/ReadVariableOpReadVariableOp3model_batch_normalization_5_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ъ
,model/batch_normalization_5/ReadVariableOp_1ReadVariableOp5model_batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0й
;model/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0┴
=model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0в
,model/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3(model/conv2d_transpose/Elu:activations:02model/batch_normalization_5/ReadVariableOp:value:04model/batch_normalization_5/ReadVariableOp_1:value:0Cmodel/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЭЭђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( a
model/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ж
model/concatenate_2/concatConcatV2 model/conv2d_2/Elu:activations:00model/batch_normalization_5/FusedBatchNormV3:y:0(model/concatenate_2/concat/axis:output:0*
N*
T0*2
_output_shapes 
:         ЭЭђq
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
valueB:╬
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
B :Щc
 model/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :Щc
 model/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђє
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
valueB:о
(model/conv2d_transpose_1/strided_slice_1StridedSlice'model/conv2d_transpose_1/stack:output:07model/conv2d_transpose_1/strided_slice_1/stack:output:09model/conv2d_transpose_1/strided_slice_1/stack_1:output:09model/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask─
8model/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpAmodel_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0Х
)model/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput'model/conv2d_transpose_1/stack:output:0@model/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#model/concatenate_2/concat:output:0*
T0*2
_output_shapes 
:         ЩЩђ*
paddingVALID*
strides
Ц
/model/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp8model_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Н
 model/conv2d_transpose_1/BiasAddBiasAdd2model/conv2d_transpose_1/conv2d_transpose:output:07model/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЩЩђІ
model/conv2d_transpose_1/EluElu)model/conv2d_transpose_1/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЩЩђЏ
*model/batch_normalization_6/ReadVariableOpReadVariableOp3model_batch_normalization_6_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ъ
,model/batch_normalization_6/ReadVariableOp_1ReadVariableOp5model_batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0й
;model/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0┴
=model/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0ь
,model/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3*model/conv2d_transpose_1/Elu:activations:02model/batch_normalization_6/ReadVariableOp:value:04model/batch_normalization_6/ReadVariableOp_1:value:0Cmodel/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЩЩђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( a
model/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ж
model/concatenate_3/concatConcatV2 model/conv2d_1/Elu:activations:00model/batch_normalization_6/FusedBatchNormV3:y:0(model/concatenate_3/concat/axis:output:0*
N*
T0*2
_output_shapes 
:         ЩЩђq
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
valueB:╬
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
B :Чc
 model/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value
B :Чc
 model/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђє
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
valueB:о
(model/conv2d_transpose_2/strided_slice_1StridedSlice'model/conv2d_transpose_2/stack:output:07model/conv2d_transpose_2/strided_slice_1/stack:output:09model/conv2d_transpose_2/strided_slice_1/stack_1:output:09model/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask─
8model/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpAmodel_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0Х
)model/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput'model/conv2d_transpose_2/stack:output:0@model/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0#model/concatenate_3/concat:output:0*
T0*2
_output_shapes 
:         ЧЧђ*
paddingVALID*
strides
Ц
/model/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp8model_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Н
 model/conv2d_transpose_2/BiasAddBiasAdd2model/conv2d_transpose_2/conv2d_transpose:output:07model/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЧЧђІ
model/conv2d_transpose_2/EluElu)model/conv2d_transpose_2/BiasAdd:output:0*
T0*2
_output_shapes 
:         ЧЧђЏ
*model/batch_normalization_7/ReadVariableOpReadVariableOp3model_batch_normalization_7_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ъ
,model/batch_normalization_7/ReadVariableOp_1ReadVariableOp5model_batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0й
;model/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0┴
=model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0ь
,model/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3*model/conv2d_transpose_2/Elu:activations:02model/batch_normalization_7/ReadVariableOp:value:04model/batch_normalization_7/ReadVariableOp_1:value:0Cmodel/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЧЧђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( a
model/concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :У
model/concatenate_4/concatConcatV2model/conv2d/Elu:activations:00model/batch_normalization_7/FusedBatchNormV3:y:0(model/concatenate_4/concat/axis:output:0*
N*
T0*2
_output_shapes 
:         ЧЧ└q
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
valueB:╬
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
B :ђc
 model/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value
B :ђb
 model/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@є
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
valueB:о
(model/conv2d_transpose_3/strided_slice_1StridedSlice'model/conv2d_transpose_3/stack:output:07model/conv2d_transpose_3/strided_slice_1/stack:output:09model/conv2d_transpose_3/strided_slice_1/stack_1:output:09model/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask├
8model/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpAmodel_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@└*
dtype0х
)model/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput'model/conv2d_transpose_3/stack:output:0@model/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0#model/concatenate_4/concat:output:0*
T0*1
_output_shapes
:         ђђ@*
paddingVALID*
strides
ц
/model/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp8model_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0н
 model/conv2d_transpose_3/BiasAddBiasAdd2model/conv2d_transpose_3/conv2d_transpose:output:07model/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ@і
model/conv2d_transpose_3/EluElu)model/conv2d_transpose_3/BiasAdd:output:0*
T0*1
_output_shapes
:         ђђ@џ
*model/batch_normalization_8/ReadVariableOpReadVariableOp3model_batch_normalization_8_readvariableop_resource*
_output_shapes
:@*
dtype0ъ
,model/batch_normalization_8/ReadVariableOp_1ReadVariableOp5model_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:@*
dtype0╝
;model/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0└
=model/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0У
,model/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3*model/conv2d_transpose_3/Elu:activations:02model/batch_normalization_8/ReadVariableOp:value:04model/batch_normalization_8/ReadVariableOp_1:value:0Cmodel/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ@:@:@:@:@:*
epsilon%oЃ:*
is_training( a
model/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ж
model/concatenate_5/concatConcatV2!model/concatenate/concat:output:00model/batch_normalization_8/FusedBatchNormV3:y:0(model/concatenate_5/concat/axis:output:0*
N*
T0*1
_output_shapes
:         ђђDq
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
valueB:╬
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
B :ђc
 model/conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value
B :ђb
 model/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :є
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
valueB:о
(model/conv2d_transpose_4/strided_slice_1StridedSlice'model/conv2d_transpose_4/stack:output:07model/conv2d_transpose_4/strided_slice_1/stack:output:09model/conv2d_transpose_4/strided_slice_1/stack_1:output:09model/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask┬
8model/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpAmodel_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:D*
dtype0х
)model/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput'model/conv2d_transpose_4/stack:output:0@model/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0#model/concatenate_5/concat:output:0*
T0*1
_output_shapes
:         ђђ*
paddingVALID*
strides
ц
/model/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp8model_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0н
 model/conv2d_transpose_4/BiasAddBiasAdd2model/conv2d_transpose_4/conv2d_transpose:output:07model/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђѓ
IdentityIdentity)model/conv2d_transpose_4/BiasAdd:output:0^NoOp*
T0*1
_output_shapes
:         ђђй
NoOpNoOp:^model/batch_normalization/FusedBatchNormV3/ReadVariableOp<^model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1)^model/batch_normalization/ReadVariableOp+^model/batch_normalization/ReadVariableOp_1<^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_1/ReadVariableOp-^model/batch_normalization_1/ReadVariableOp_1<^model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_2/ReadVariableOp-^model/batch_normalization_2/ReadVariableOp_1<^model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_3/ReadVariableOp-^model/batch_normalization_3/ReadVariableOp_1<^model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_4/ReadVariableOp-^model/batch_normalization_4/ReadVariableOp_1<^model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_5/ReadVariableOp-^model/batch_normalization_5/ReadVariableOp_1<^model/batch_normalization_6/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_6/ReadVariableOp-^model/batch_normalization_6/ReadVariableOp_1<^model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_7/ReadVariableOp-^model/batch_normalization_7/ReadVariableOp_1<^model/batch_normalization_8/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_8/ReadVariableOp-^model/batch_normalization_8/ReadVariableOp_1$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp&^model/conv2d_3/BiasAdd/ReadVariableOp%^model/conv2d_3/Conv2D/ReadVariableOp.^model/conv2d_transpose/BiasAdd/ReadVariableOp7^model/conv2d_transpose/conv2d_transpose/ReadVariableOp0^model/conv2d_transpose_1/BiasAdd/ReadVariableOp9^model/conv2d_transpose_1/conv2d_transpose/ReadVariableOp0^model/conv2d_transpose_2/BiasAdd/ReadVariableOp9^model/conv2d_transpose_2/conv2d_transpose/ReadVariableOp0^model/conv2d_transpose_3/BiasAdd/ReadVariableOp9^model/conv2d_transpose_3/conv2d_transpose/ReadVariableOp0^model/conv2d_transpose_4/BiasAdd/ReadVariableOp9^model/conv2d_transpose_4/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesМ
л:         ђђ:         ђђ:         ђђ:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2v
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
:         ђђ

_user_specified_nameu_vel:TP
-
_output_shapes
:         ђђ

_user_specified_namev_vel:TP
-
_output_shapes
:         ђђ

_user_specified_namew_vel:TP
-
_output_shapes
:         ђђ

_user_specified_namepr0.2
╦
Ю
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1030933

inputsD
(conv2d_transpose_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
B :ЭJ
stack/2Const*
_output_shapes
: *
dtype0*
value
B :ЭJ
stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђЅ
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
valueB:┘
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskњ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0╬
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*2
_output_shapes 
:         ЭЭђ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0і
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЭЭђY
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:         ЭЭђk
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђЂ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ШШђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Z V
2
_output_shapes 
:         ШШђ
 
_user_specified_nameinputs
─
Ю
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1031128

inputsC
(conv2d_transpose_readvariableop_resource:@└-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
B :ђJ
stack/2Const*
_output_shapes
: *
dtype0*
value
B :ђI
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@Ѕ
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
valueB:┘
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЉ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@└*
dtype0═
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:         ђђ@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ѕ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ@X
EluEluBiasAdd:output:0*
T0*1
_output_shapes
:         ђђ@j
IdentityIdentityElu:activations:0^NoOp*
T0*1
_output_shapes
:         ђђ@Ђ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ЧЧ└: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Z V
2
_output_shapes 
:         ЧЧ└
 
_user_specified_nameinputs
в
┼
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1030325

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
П
о
7__inference_batch_normalization_5_layer_call_fn_1034541

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЈ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1030956z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЭЭђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
э
t
J__inference_concatenate_5_layer_call_and_return_conditional_losses_1031168

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
:         ђђDa
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:         ђђD"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::         ђђ:         ђђ@:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs:YU
1
_output_shapes
:         ђђ@
 
_user_specified_nameinputs
Џ
Ђ
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1034285

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0Ю
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ШШђ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ђ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ШШђY
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:         ШШђk
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:         ШШђw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
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
═
Ю
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1030520

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
ъ	
о
7__inference_batch_normalization_6_layer_call_fn_1034732

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЪ
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1030294і
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Щ#
Ъ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1030265

inputsD
(conv2d_transpose_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
valueB:┘
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
valueB:┘
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
B :ђy
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
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskњ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0я
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,                           ђ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0џ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђi
EluEluBiasAdd:output:0*
T0*B
_output_shapes0
.:,                           ђ{
IdentityIdentityElu:activations:0^NoOp*
T0*B
_output_shapes0
.:,                           ђЂ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Ф
┼
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1031653

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0╦
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЭЭђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
Н
м
7__inference_batch_normalization_8_layer_call_fn_1035192

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallј
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1031151y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         ђђ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ђђ@
 
_user_specified_nameinputs
Ю
А
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1031086

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0й
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЧЧђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЧЧђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЧЧђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:         ЧЧђ
 
_user_specified_nameinputs
П
А
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1034211

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
═
Ъ
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1031063

inputsD
(conv2d_transpose_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
B :ЧJ
stack/2Const*
_output_shapes
: *
dtype0*
value
B :ЧJ
stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђЅ
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
valueB:┘
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskњ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0╬
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*2
_output_shapes 
:         ЧЧђ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0і
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЧЧђY
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:         ЧЧђk
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:         ЧЧђЂ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ЩЩђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Z V
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs
Ф
┼
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1034121

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0╦
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЩЩђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
Ф
┼
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1034265

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0╦
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЭЭђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
љ	
л
5__inference_batch_normalization_layer_call_fn_1033735

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityѕбStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1029843Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
─
G
+__inference_reshape_2_layer_call_fn_1033659

inputs
identityЙ
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
GPU2*0J 8ѓ *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_1030669j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ђђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ђђ:U Q
-
_output_shapes
:         ђђ
 
_user_specified_nameinputs
шЌ
а
B__inference_model_layer_call_and_return_conditional_losses_1032123

inputs
inputs_1
inputs_2
inputs_3)
batch_normalization_1031991:)
batch_normalization_1031993:)
batch_normalization_1031995:)
batch_normalization_1031997:(
conv2d_1032000:@
conv2d_1032002:@+
batch_normalization_1_1032005:@+
batch_normalization_1_1032007:@+
batch_normalization_1_1032009:@+
batch_normalization_1_1032011:@+
conv2d_1_1032014:@ђ
conv2d_1_1032016:	ђ,
batch_normalization_2_1032019:	ђ,
batch_normalization_2_1032021:	ђ,
batch_normalization_2_1032023:	ђ,
batch_normalization_2_1032025:	ђ,
conv2d_2_1032028:ђђ
conv2d_2_1032030:	ђ,
batch_normalization_3_1032033:	ђ,
batch_normalization_3_1032035:	ђ,
batch_normalization_3_1032037:	ђ,
batch_normalization_3_1032039:	ђ,
conv2d_3_1032042:ђђ
conv2d_3_1032044:	ђ,
batch_normalization_4_1032047:	ђ,
batch_normalization_4_1032049:	ђ,
batch_normalization_4_1032051:	ђ,
batch_normalization_4_1032053:	ђ4
conv2d_transpose_1032057:ђђ'
conv2d_transpose_1032059:	ђ,
batch_normalization_5_1032062:	ђ,
batch_normalization_5_1032064:	ђ,
batch_normalization_5_1032066:	ђ,
batch_normalization_5_1032068:	ђ6
conv2d_transpose_1_1032072:ђђ)
conv2d_transpose_1_1032074:	ђ,
batch_normalization_6_1032077:	ђ,
batch_normalization_6_1032079:	ђ,
batch_normalization_6_1032081:	ђ,
batch_normalization_6_1032083:	ђ6
conv2d_transpose_2_1032087:ђђ)
conv2d_transpose_2_1032089:	ђ,
batch_normalization_7_1032092:	ђ,
batch_normalization_7_1032094:	ђ,
batch_normalization_7_1032096:	ђ,
batch_normalization_7_1032098:	ђ5
conv2d_transpose_3_1032102:@└(
conv2d_transpose_3_1032104:@+
batch_normalization_8_1032107:@+
batch_normalization_8_1032109:@+
batch_normalization_8_1032111:@+
batch_normalization_8_1032113:@4
conv2d_transpose_4_1032117:D(
conv2d_transpose_4_1032119:
identityѕб+batch_normalization/StatefulPartitionedCallб-batch_normalization_1/StatefulPartitionedCallб-batch_normalization_2/StatefulPartitionedCallб-batch_normalization_3/StatefulPartitionedCallб-batch_normalization_4/StatefulPartitionedCallб-batch_normalization_5/StatefulPartitionedCallб-batch_normalization_6/StatefulPartitionedCallб-batch_normalization_7/StatefulPartitionedCallб-batch_normalization_8/StatefulPartitionedCallбconv2d/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallб conv2d_2/StatefulPartitionedCallб conv2d_3/StatefulPartitionedCallб(conv2d_transpose/StatefulPartitionedCallб*conv2d_transpose_1/StatefulPartitionedCallб*conv2d_transpose_2/StatefulPartitionedCallб*conv2d_transpose_3/StatefulPartitionedCallб*conv2d_transpose_4/StatefulPartitionedCall─
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
GPU2*0J 8ѓ *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_1030637╩
reshape_1/PartitionedCallPartitionedCallinputs_1*
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
GPU2*0J 8ѓ *O
fJRH
F__inference_reshape_1_layer_call_and_return_conditional_losses_1030653╩
reshape_2/PartitionedCallPartitionedCallinputs_2*
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
GPU2*0J 8ѓ *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_1030669╩
reshape_3/PartitionedCallPartitionedCallinputs_3*
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
GPU2*0J 8ѓ *O
fJRH
F__inference_reshape_3_layer_call_and_return_conditional_losses_1030685Н
concatenate/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0"reshape_1/PartitionedCall:output:0"reshape_2/PartitionedCall:output:0"reshape_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_1030696є
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0batch_normalization_1031991batch_normalization_1031993batch_normalization_1031995batch_normalization_1031997*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1031815д
conv2d/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_1032000conv2d_1032002*
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
GPU2*0J 8ѓ *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_1030736Ћ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_1_1032005batch_normalization_1_1032007batch_normalization_1_1032009batch_normalization_1_1032011*
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1031761▒
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_1_1032014conv2d_1_1032016*
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
GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1030780ў
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_2_1032019batch_normalization_2_1032021batch_normalization_2_1032023batch_normalization_2_1032025*
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1031707▒
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_2_1032028conv2d_2_1032030*
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
GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1030824ў
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_3_1032033batch_normalization_3_1032035batch_normalization_3_1032037batch_normalization_3_1032039*
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1031653▒
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_3_1032042conv2d_3_1032044*
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
GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1030868ў
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_4_1032047batch_normalization_4_1032049batch_normalization_4_1032051batch_normalization_4_1032053*
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1031599Г
concatenate_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:06batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ШШђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1030908┴
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0conv2d_transpose_1032057conv2d_transpose_1032059*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1030933а
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0batch_normalization_5_1032062batch_normalization_5_1032064batch_normalization_5_1032066batch_normalization_5_1032068*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1031539Г
concatenate_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:06batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_1030973╔
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0conv2d_transpose_1_1032072conv2d_transpose_1_1032074*
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
GPU2*0J 8ѓ *X
fSRQ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1030998б
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0batch_normalization_6_1032077batch_normalization_6_1032079batch_normalization_6_1032081batch_normalization_6_1032083*
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1031479Г
concatenate_3/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:06batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЩЩђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_concatenate_3_layer_call_and_return_conditional_losses_1031038╔
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0conv2d_transpose_2_1032087conv2d_transpose_2_1032089*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЧЧђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1031063б
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0batch_normalization_7_1032092batch_normalization_7_1032094batch_normalization_7_1032096batch_normalization_7_1032098*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЧЧђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1031419Ф
concatenate_4/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:06batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЧЧ└* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_concatenate_4_layer_call_and_return_conditional_losses_1031103╚
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv2d_transpose_3_1032102conv2d_transpose_3_1032104*
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
GPU2*0J 8ѓ *X
fSRQ
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1031128А
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0batch_normalization_8_1032107batch_normalization_8_1032109batch_normalization_8_1032111batch_normalization_8_1032113*
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1031359Д
concatenate_5/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:06batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђD* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_concatenate_5_layer_call_and_return_conditional_losses_1031168╚
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0conv2d_transpose_4_1032117conv2d_transpose_4_1032119*
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
GPU2*0J 8ѓ *X
fSRQ
O__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1031192ї
IdentityIdentity3conv2d_transpose_4/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђП
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesМ
л:         ђђ:         ђђ:         ђђ:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
 
_user_specified_nameinputs:UQ
-
_output_shapes
:         ђђ
 
_user_specified_nameinputs:UQ
-
_output_shapes
:         ђђ
 
_user_specified_nameinputs:UQ
-
_output_shapes
:         ђђ
 
_user_specified_nameinputs
њ	
л
5__inference_batch_normalization_layer_call_fn_1033722

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityѕбStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1029812Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
▀
b
F__inference_reshape_1_layer_call_and_return_conditional_losses_1030653

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
valueB:Л
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
B :ђR
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ђQ
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Е
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:         ђђb
IdentityIdentityReshape:output:0*
T0*1
_output_shapes
:         ђђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ђђ:U Q
-
_output_shapes
:         ђђ
 
_user_specified_nameinputs
Џ
┴
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1031761

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ЧЧ@:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0m
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:         ЧЧ@н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
т
ю
O__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1031192

inputsB
(conv2d_transpose_readvariableop_resource:D-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
B :ђJ
stack/2Const*
_output_shapes
: *
dtype0*
value
B :ђI
stack/3Const*
_output_shapes
: *
dtype0*
value	B :Ѕ
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
valueB:┘
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskљ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:D*
dtype0═
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:         ђђ*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ѕ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђi
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:         ђђЂ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ђђD: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:         ђђD
 
_user_specified_nameinputs
П
Ё
'__inference_model_layer_call_fn_1032350	
u_vel	
v_vel	
w_vel	
pr0_2
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

unknown_27:ђђ

unknown_28:	ђ

unknown_29:	ђ

unknown_30:	ђ

unknown_31:	ђ

unknown_32:	ђ&

unknown_33:ђђ

unknown_34:	ђ

unknown_35:	ђ

unknown_36:	ђ

unknown_37:	ђ

unknown_38:	ђ&

unknown_39:ђђ

unknown_40:	ђ

unknown_41:	ђ

unknown_42:	ђ

unknown_43:	ђ

unknown_44:	ђ%

unknown_45:@└

unknown_46:@

unknown_47:@

unknown_48:@

unknown_49:@

unknown_50:@$

unknown_51:D

unknown_52:
identityѕбStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallu_velv_velw_velpr0_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:         ђђ*F
_read_only_resource_inputs(
&$	
 !"#&'(),-./234589*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1032123y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesМ
л:         ђђ:         ђђ:         ђђ:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
-
_output_shapes
:         ђђ

_user_specified_nameu_vel:TP
-
_output_shapes
:         ђђ

_user_specified_namev_vel:TP
-
_output_shapes
:         ђђ

_user_specified_namew_vel:TP
-
_output_shapes
:         ђђ

_user_specified_namepr0.2
Ф
┼
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1031419

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0╦
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЧЧђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЧЧђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЧЧђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:         ЧЧђ
 
_user_specified_nameinputs
П
А
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1035006

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
П
А
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1029940

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
І
Е
4__inference_conv2d_transpose_4_layer_call_fn_1035308

inputs!
unknown:D
	unknown_0:
identityѕбStatefulPartitionedCallы
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
GPU2*0J 8ѓ *X
fSRQ
O__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1031192y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ђђD: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ђђD
 
_user_specified_nameinputs
█
о
7__inference_batch_normalization_2_layer_call_fn_1034049

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЇ
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1031707z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЩЩђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs
■
б
*__inference_conv2d_3_layer_call_fn_1034274

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallУ
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
GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1030868z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ШШђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ЭЭђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
▀
b
F__inference_reshape_2_layer_call_and_return_conditional_losses_1033673

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
valueB:Л
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
B :ђR
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ђQ
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Е
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:         ђђb
IdentityIdentityReshape:output:0*
T0*1
_output_shapes
:         ђђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ђђ:U Q
-
_output_shapes
:         ђђ
 
_user_specified_nameinputs
љ	
u
-__inference_concatenate_layer_call_fn_1033700
inputs_0
inputs_1
inputs_2
inputs_3
identityс
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_1030696j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ђђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Є
_input_shapesv
t:         ђђ:         ђђ:         ђђ:         ђђ:[ W
1
_output_shapes
:         ђђ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:         ђђ
"
_user_specified_name
inputs/1:[W
1
_output_shapes
:         ђђ
"
_user_specified_name
inputs/2:[W
1
_output_shapes
:         ђђ
"
_user_specified_name
inputs/3
Щ#
Ъ
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1034912

inputsD
(conv2d_transpose_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
valueB:┘
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
valueB:┘
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
B :ђy
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
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskњ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0я
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,                           ђ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0џ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђi
EluEluBiasAdd:output:0*
T0*B
_output_shapes0
.:,                           ђ{
IdentityIdentityElu:activations:0^NoOp*
T0*B
_output_shapes0
.:,                           ђЂ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
═
Ю
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1029876

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
ч
А
*__inference_conv2d_1_layer_call_fn_1033986

inputs"
unknown:@ђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallУ
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
GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1030780z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЧЧ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ЧЧ@
 
_user_specified_nameinputs
█
о
7__inference_batch_normalization_7_layer_call_fn_1034988

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЧЧђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1031419z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ЧЧђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЧЧђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ЧЧђ
 
_user_specified_nameinputs
Џ
┴
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1035277

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ@:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0m
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:         ђђ@н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
в
┼
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1030035

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
Ю
А
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1034103

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0й
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЩЩђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
ь
[
/__inference_concatenate_5_layer_call_fn_1035283
inputs_0
inputs_1
identity¤
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђD* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_concatenate_5_layer_call_and_return_conditional_losses_1031168j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ђђD"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::         ђђ:         ђђ@:[ W
1
_output_shapes
:         ђђ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:         ђђ@
"
_user_specified_name
inputs/1
═
Ѓ
%__inference_signature_wrapper_1032760	
pr0_2	
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

unknown_27:ђђ

unknown_28:	ђ

unknown_29:	ђ

unknown_30:	ђ

unknown_31:	ђ

unknown_32:	ђ&

unknown_33:ђђ

unknown_34:	ђ

unknown_35:	ђ

unknown_36:	ђ

unknown_37:	ђ

unknown_38:	ђ&

unknown_39:ђђ

unknown_40:	ђ

unknown_41:	ђ

unknown_42:	ђ

unknown_43:	ђ

unknown_44:	ђ%

unknown_45:@└

unknown_46:@

unknown_47:@

unknown_48:@

unknown_49:@

unknown_50:@$

unknown_51:D

unknown_52:
identityѕбStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallu_velv_velw_velpr0_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:         ђђ*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456789*0
config_proto 

CPU

GPU2*0J 8ѓ *+
f&R$
"__inference__wrapped_model_1029790y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*т
_input_shapesМ
л:         ђђ:         ђђ:         ђђ:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
-
_output_shapes
:         ђђ

_user_specified_namepr0.2:TP
-
_output_shapes
:         ђђ

_user_specified_nameu_vel:TP
-
_output_shapes
:         ђђ

_user_specified_namev_vel:TP
-
_output_shapes
:         ђђ

_user_specified_namew_vel
П
А
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1034067

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
█
о
7__inference_batch_normalization_5_layer_call_fn_1034554

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         ЭЭђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1031539z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЭЭђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
ъ	
о
7__inference_batch_normalization_2_layer_call_fn_1034010

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЪ
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1029940і
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
ъ	
о
7__inference_batch_normalization_7_layer_call_fn_1034949

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЪ
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1030407і
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
═
Ъ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1030998

inputsD
(conv2d_transpose_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
B :ЩJ
stack/2Const*
_output_shapes
: *
dtype0*
value
B :ЩJ
stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђЅ
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
valueB:┘
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskњ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0╬
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*2
_output_shapes 
:         ЩЩђ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0і
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЩЩђY
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:         ЩЩђk
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђЂ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ЭЭђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
█
о
7__inference_batch_normalization_6_layer_call_fn_1034771

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЇ
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
GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1031479z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЩЩђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ЩЩђ
 
_user_specified_nameinputs
Џ
┴
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1031359

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ@:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0m
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:         ђђ@н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
Ю
А
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1030956

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0й
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЭЭђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЭЭђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЭЭђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
Ю
А
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1035042

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0й
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЧЧђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЧЧђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЧЧђ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:         ЧЧђ
 
_user_specified_nameinputs
њ
г
4__inference_conv2d_transpose_1_layer_call_fn_1034657

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallЫ
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
GPU2*0J 8ѓ *X
fSRQ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1030998z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ЭЭђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
Щ#
Ъ
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1034695

inputsD
(conv2d_transpose_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
valueB:┘
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
valueB:┘
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
B :ђy
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
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskњ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0я
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,                           ђ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0џ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђi
EluEluBiasAdd:output:0*
T0*B
_output_shapes0
.:,                           ђ{
IdentityIdentityElu:activations:0^NoOp*
T0*B
_output_shapes0
.:,                           ђЂ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
■
t
J__inference_concatenate_2_layer_call_and_return_conditional_losses_1030973

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ђ
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*2
_output_shapes 
:         ЭЭђb
IdentityIdentityconcat:output:0*
T0*2
_output_shapes 
:         ЭЭђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:         ЭЭђ:         ЭЭђ:Z V
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs:ZV
2
_output_shapes 
:         ЭЭђ
 
_user_specified_nameinputs
╦
Џ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1029812

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ї
Ч
C__inference_conv2d_layer_call_and_return_conditional_losses_1033853

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0ю
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЧЧ@*
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
:         ЧЧ@X
EluEluBiasAdd:output:0*
T0*1
_output_shapes
:         ЧЧ@j
IdentityIdentityElu:activations:0^NoOp*
T0*1
_output_shapes
:         ЧЧ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ђђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
Ќ
ђ
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1033997

inputs9
conv2d_readvariableop_resource:@ђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype0Ю
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЩЩђ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ђ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ЩЩђY
EluEluBiasAdd:output:0*
T0*2
_output_shapes 
:         ЩЩђk
IdentityIdentityElu:activations:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
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
Ф
┼
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1034843

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0╦
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЩЩђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЩЩђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
┘
┐
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1033797

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ѓ#
ю
O__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1035345

inputsB
(conv2d_transpose_readvariableop_resource:D-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
valueB:┘
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
valueB:┘
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
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskљ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:D*
dtype0П
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ў
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+                           Ђ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           D: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+                           D
 
_user_specified_nameinputs
Ф
┼
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1035060

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0╦
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:         ЧЧђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:         ЧЧђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         ЧЧђ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:         ЧЧђ
 
_user_specified_nameinputs
ы#
Ю
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1035129

inputsC
(conv2d_transpose_readvariableop_resource:@└-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
valueB:┘
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
valueB:┘
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
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЉ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@└*
dtype0П
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           @*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ў
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @h
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           @z
IdentityIdentityElu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                           @Ђ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           └: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,                           └
 
_user_specified_nameinputs
є
v
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1034422
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ѓ
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*2
_output_shapes 
:         ШШђb
IdentityIdentityconcat:output:0*
T0*2
_output_shapes 
:         ШШђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:         ШШђ:         ШШђ:\ X
2
_output_shapes 
:         ШШђ
"
_user_specified_name
inputs/0:\X
2
_output_shapes 
:         ШШђ
"
_user_specified_name
inputs/1
з
Ю
(__inference_conv2d_layer_call_fn_1033842

inputs!
unknown:@
	unknown_0:@
identityѕбStatefulPartitionedCallт
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
GPU2*0J 8ѓ *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_1030736y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЧЧ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ђђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
█
┴
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1033941

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
Л
л
5__inference_batch_normalization_layer_call_fn_1033748

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityѕбStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1030715y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:         ђђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
═
Ю
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1035223

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
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
 
_user_specified_nameinputs"ѓL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*■
serving_defaultЖ
=
pr0.24
serving_default_pr0.2:0         ђђ
=
u_vel4
serving_default_u_vel:0         ђђ
=
v_vel4
serving_default_v_vel:0         ђђ
=
w_vel4
serving_default_w_vel:0         ђђP
conv2d_transpose_4:
StatefulPartitionedCall:0         ђђtensorflow/serving/predict:Дэ
Э
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-0

layer-9
layer_with_weights-1
layer-10
layer_with_weights-2
layer-11
layer_with_weights-3
layer-12
layer_with_weights-4
layer-13
layer_with_weights-5
layer-14
layer_with_weights-6
layer-15
layer_with_weights-7
layer-16
layer_with_weights-8
layer-17
layer-18
layer_with_weights-9
layer-19
layer_with_weights-10
layer-20
layer-21
layer_with_weights-11
layer-22
layer_with_weights-12
layer-23
layer-24
layer_with_weights-13
layer-25
layer_with_weights-14
layer-26
layer-27
layer_with_weights-15
layer-28
layer_with_weights-16
layer-29
layer-30
 layer_with_weights-17
 layer-31
!	optimizer
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&
signatures
╣__call__
+║&call_and_return_all_conditional_losses
╗_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
Д
'	variables
(trainable_variables
)regularization_losses
*	keras_api
╝__call__
+й&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
+	variables
,trainable_variables
-regularization_losses
.	keras_api
Й__call__
+┐&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
/	variables
0trainable_variables
1regularization_losses
2	keras_api
└__call__
+┴&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
3	variables
4trainable_variables
5regularization_losses
6	keras_api
┬__call__
+├&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
7	variables
8trainable_variables
9regularization_losses
:	keras_api
─__call__
+┼&call_and_return_all_conditional_losses"
_tf_keras_layer
В
;axis
	<gamma
=beta
>moving_mean
?moving_variance
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
к__call__
+К&call_and_return_all_conditional_losses"
_tf_keras_layer
й

Dkernel
Ebias
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
╚__call__
+╔&call_and_return_all_conditional_losses"
_tf_keras_layer
В
Jaxis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
╩__call__
+╦&call_and_return_all_conditional_losses"
_tf_keras_layer
й

Skernel
Tbias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
╠__call__
+═&call_and_return_all_conditional_losses"
_tf_keras_layer
В
Yaxis
	Zgamma
[beta
\moving_mean
]moving_variance
^	variables
_trainable_variables
`regularization_losses
a	keras_api
╬__call__
+¤&call_and_return_all_conditional_losses"
_tf_keras_layer
й

bkernel
cbias
d	variables
etrainable_variables
fregularization_losses
g	keras_api
л__call__
+Л&call_and_return_all_conditional_losses"
_tf_keras_layer
В
haxis
	igamma
jbeta
kmoving_mean
lmoving_variance
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
м__call__
+М&call_and_return_all_conditional_losses"
_tf_keras_layer
й

qkernel
rbias
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
н__call__
+Н&call_and_return_all_conditional_losses"
_tf_keras_layer
В
waxis
	xgamma
ybeta
zmoving_mean
{moving_variance
|	variables
}trainable_variables
~regularization_losses
	keras_api
о__call__
+О&call_and_return_all_conditional_losses"
_tf_keras_layer
Ф
ђ	variables
Ђtrainable_variables
ѓregularization_losses
Ѓ	keras_api
п__call__
+┘&call_and_return_all_conditional_losses"
_tf_keras_layer
├
ёkernel
	Ёbias
є	variables
Єtrainable_variables
ѕregularization_losses
Ѕ	keras_api
┌__call__
+█&call_and_return_all_conditional_losses"
_tf_keras_layer
ш
	іaxis

Іgamma
	їbeta
Їmoving_mean
јmoving_variance
Ј	variables
љtrainable_variables
Љregularization_losses
њ	keras_api
▄__call__
+П&call_and_return_all_conditional_losses"
_tf_keras_layer
Ф
Њ	variables
ћtrainable_variables
Ћregularization_losses
ќ	keras_api
я__call__
+▀&call_and_return_all_conditional_losses"
_tf_keras_layer
├
Ќkernel
	ўbias
Ў	variables
џtrainable_variables
Џregularization_losses
ю	keras_api
Я__call__
+р&call_and_return_all_conditional_losses"
_tf_keras_layer
ш
	Юaxis

ъgamma
	Ъbeta
аmoving_mean
Аmoving_variance
б	variables
Бtrainable_variables
цregularization_losses
Ц	keras_api
Р__call__
+с&call_and_return_all_conditional_losses"
_tf_keras_layer
Ф
д	variables
Дtrainable_variables
еregularization_losses
Е	keras_api
С__call__
+т&call_and_return_all_conditional_losses"
_tf_keras_layer
├
фkernel
	Фbias
г	variables
Гtrainable_variables
«regularization_losses
»	keras_api
Т__call__
+у&call_and_return_all_conditional_losses"
_tf_keras_layer
ш
	░axis

▒gamma
	▓beta
│moving_mean
┤moving_variance
х	variables
Хtrainable_variables
иregularization_losses
И	keras_api
У__call__
+ж&call_and_return_all_conditional_losses"
_tf_keras_layer
Ф
╣	variables
║trainable_variables
╗regularization_losses
╝	keras_api
Ж__call__
+в&call_and_return_all_conditional_losses"
_tf_keras_layer
├
йkernel
	Йbias
┐	variables
└trainable_variables
┴regularization_losses
┬	keras_api
В__call__
+ь&call_and_return_all_conditional_losses"
_tf_keras_layer
ш
	├axis

─gamma
	┼beta
кmoving_mean
Кmoving_variance
╚	variables
╔trainable_variables
╩regularization_losses
╦	keras_api
Ь__call__
+№&call_and_return_all_conditional_losses"
_tf_keras_layer
Ф
╠	variables
═trainable_variables
╬regularization_losses
¤	keras_api
­__call__
+ы&call_and_return_all_conditional_losses"
_tf_keras_layer
├
лkernel
	Лbias
м	variables
Мtrainable_variables
нregularization_losses
Н	keras_api
Ы__call__
+з&call_and_return_all_conditional_losses"
_tf_keras_layer
╠
	оiter
Оbeta_1
пbeta_2

┘decay
┌learning_rate<mы=mЫDmзEmЗKmшLmШSmэTmЭZmщ[mЩbmчcmЧim§jm■qm rmђxmЂymѓ	ёmЃ	Ёmё	ІmЁ	їmє	ЌmЄ	ўmѕ	ъmЅ	Ъmі	фmІ	Фmї	▒mЇ	▓mј	йmЈ	Йmљ	─mЉ	┼mњ	лmЊ	Лmћ<vЋ=vќDvЌEvўKvЎLvџSvЏTvюZvЮ[vъbvЪcvаivАjvбqvБrvцxvЦyvд	ёvД	Ёvе	ІvЕ	їvф	ЌvФ	ўvг	ъvГ	Ъv«	фv»	Фv░	▒v▒	▓v▓	йv│	Йv┤	─vх	┼vХ	лvи	ЛvИ"
	optimizer
Я
<0
=1
>2
?3
D4
E5
K6
L7
M8
N9
S10
T11
Z12
[13
\14
]15
b16
c17
i18
j19
k20
l21
q22
r23
x24
y25
z26
{27
ё28
Ё29
І30
ї31
Ї32
ј33
Ќ34
ў35
ъ36
Ъ37
а38
А39
ф40
Ф41
▒42
▓43
│44
┤45
й46
Й47
─48
┼49
к50
К51
л52
Л53"
trackable_list_wrapper
╚
<0
=1
D2
E3
K4
L5
S6
T7
Z8
[9
b10
c11
i12
j13
q14
r15
x16
y17
ё18
Ё19
І20
ї21
Ќ22
ў23
ъ24
Ъ25
ф26
Ф27
▒28
▓29
й30
Й31
─32
┼33
л34
Л35"
trackable_list_wrapper
 "
trackable_list_wrapper
М
█non_trainable_variables
▄layers
Пmetrics
 яlayer_regularization_losses
▀layer_metrics
"	variables
#trainable_variables
$regularization_losses
╣__call__
╗_default_save_signature
+║&call_and_return_all_conditional_losses
'║"call_and_return_conditional_losses"
_generic_user_object
-
Зserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Яnon_trainable_variables
рlayers
Рmetrics
 сlayer_regularization_losses
Сlayer_metrics
'	variables
(trainable_variables
)regularization_losses
╝__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
тnon_trainable_variables
Тlayers
уmetrics
 Уlayer_regularization_losses
жlayer_metrics
+	variables
,trainable_variables
-regularization_losses
Й__call__
+┐&call_and_return_all_conditional_losses
'┐"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Жnon_trainable_variables
вlayers
Вmetrics
 ьlayer_regularization_losses
Ьlayer_metrics
/	variables
0trainable_variables
1regularization_losses
└__call__
+┴&call_and_return_all_conditional_losses
'┴"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
№non_trainable_variables
­layers
ыmetrics
 Ыlayer_regularization_losses
зlayer_metrics
3	variables
4trainable_variables
5regularization_losses
┬__call__
+├&call_and_return_all_conditional_losses
'├"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Зnon_trainable_variables
шlayers
Шmetrics
 эlayer_regularization_losses
Эlayer_metrics
7	variables
8trainable_variables
9regularization_losses
─__call__
+┼&call_and_return_all_conditional_losses
'┼"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%2batch_normalization/gamma
&:$2batch_normalization/beta
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
<
<0
=1
>2
?3"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
х
щnon_trainable_variables
Щlayers
чmetrics
 Чlayer_regularization_losses
§layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
к__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
':%@2conv2d/kernel
:@2conv2d/bias
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
х
■non_trainable_variables
 layers
ђmetrics
 Ђlayer_regularization_losses
ѓlayer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
╚__call__
+╔&call_and_return_all_conditional_losses
'╔"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_1/gamma
(:&@2batch_normalization_1/beta
1:/@ (2!batch_normalization_1/moving_mean
5:3@ (2%batch_normalization_1/moving_variance
<
K0
L1
M2
N3"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
х
Ѓnon_trainable_variables
ёlayers
Ёmetrics
 єlayer_regularization_losses
Єlayer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
╩__call__
+╦&call_and_return_all_conditional_losses
'╦"call_and_return_conditional_losses"
_generic_user_object
*:(@ђ2conv2d_1/kernel
:ђ2conv2d_1/bias
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
х
ѕnon_trainable_variables
Ѕlayers
іmetrics
 Іlayer_regularization_losses
їlayer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
╠__call__
+═&call_and_return_all_conditional_losses
'═"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(ђ2batch_normalization_2/gamma
):'ђ2batch_normalization_2/beta
2:0ђ (2!batch_normalization_2/moving_mean
6:4ђ (2%batch_normalization_2/moving_variance
<
Z0
[1
\2
]3"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
х
Їnon_trainable_variables
јlayers
Јmetrics
 љlayer_regularization_losses
Љlayer_metrics
^	variables
_trainable_variables
`regularization_losses
╬__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
+:)ђђ2conv2d_2/kernel
:ђ2conv2d_2/bias
.
b0
c1"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
х
њnon_trainable_variables
Њlayers
ћmetrics
 Ћlayer_regularization_losses
ќlayer_metrics
d	variables
etrainable_variables
fregularization_losses
л__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(ђ2batch_normalization_3/gamma
):'ђ2batch_normalization_3/beta
2:0ђ (2!batch_normalization_3/moving_mean
6:4ђ (2%batch_normalization_3/moving_variance
<
i0
j1
k2
l3"
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
х
Ќnon_trainable_variables
ўlayers
Ўmetrics
 џlayer_regularization_losses
Џlayer_metrics
m	variables
ntrainable_variables
oregularization_losses
м__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
+:)ђђ2conv2d_3/kernel
:ђ2conv2d_3/bias
.
q0
r1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
х
юnon_trainable_variables
Юlayers
ъmetrics
 Ъlayer_regularization_losses
аlayer_metrics
s	variables
ttrainable_variables
uregularization_losses
н__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(ђ2batch_normalization_4/gamma
):'ђ2batch_normalization_4/beta
2:0ђ (2!batch_normalization_4/moving_mean
6:4ђ (2%batch_normalization_4/moving_variance
<
x0
y1
z2
{3"
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
х
Аnon_trainable_variables
бlayers
Бmetrics
 цlayer_regularization_losses
Цlayer_metrics
|	variables
}trainable_variables
~regularization_losses
о__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
дnon_trainable_variables
Дlayers
еmetrics
 Еlayer_regularization_losses
фlayer_metrics
ђ	variables
Ђtrainable_variables
ѓregularization_losses
п__call__
+┘&call_and_return_all_conditional_losses
'┘"call_and_return_conditional_losses"
_generic_user_object
3:1ђђ2conv2d_transpose/kernel
$:"ђ2conv2d_transpose/bias
0
ё0
Ё1"
trackable_list_wrapper
0
ё0
Ё1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Фnon_trainable_variables
гlayers
Гmetrics
 «layer_regularization_losses
»layer_metrics
є	variables
Єtrainable_variables
ѕregularization_losses
┌__call__
+█&call_and_return_all_conditional_losses
'█"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(ђ2batch_normalization_5/gamma
):'ђ2batch_normalization_5/beta
2:0ђ (2!batch_normalization_5/moving_mean
6:4ђ (2%batch_normalization_5/moving_variance
@
І0
ї1
Ї2
ј3"
trackable_list_wrapper
0
І0
ї1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
░non_trainable_variables
▒layers
▓metrics
 │layer_regularization_losses
┤layer_metrics
Ј	variables
љtrainable_variables
Љregularization_losses
▄__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
хnon_trainable_variables
Хlayers
иmetrics
 Иlayer_regularization_losses
╣layer_metrics
Њ	variables
ћtrainable_variables
Ћregularization_losses
я__call__
+▀&call_and_return_all_conditional_losses
'▀"call_and_return_conditional_losses"
_generic_user_object
5:3ђђ2conv2d_transpose_1/kernel
&:$ђ2conv2d_transpose_1/bias
0
Ќ0
ў1"
trackable_list_wrapper
0
Ќ0
ў1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
║non_trainable_variables
╗layers
╝metrics
 йlayer_regularization_losses
Йlayer_metrics
Ў	variables
џtrainable_variables
Џregularization_losses
Я__call__
+р&call_and_return_all_conditional_losses
'р"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(ђ2batch_normalization_6/gamma
):'ђ2batch_normalization_6/beta
2:0ђ (2!batch_normalization_6/moving_mean
6:4ђ (2%batch_normalization_6/moving_variance
@
ъ0
Ъ1
а2
А3"
trackable_list_wrapper
0
ъ0
Ъ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
┐non_trainable_variables
└layers
┴metrics
 ┬layer_regularization_losses
├layer_metrics
б	variables
Бtrainable_variables
цregularization_losses
Р__call__
+с&call_and_return_all_conditional_losses
'с"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
─non_trainable_variables
┼layers
кmetrics
 Кlayer_regularization_losses
╚layer_metrics
д	variables
Дtrainable_variables
еregularization_losses
С__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
5:3ђђ2conv2d_transpose_2/kernel
&:$ђ2conv2d_transpose_2/bias
0
ф0
Ф1"
trackable_list_wrapper
0
ф0
Ф1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
╔non_trainable_variables
╩layers
╦metrics
 ╠layer_regularization_losses
═layer_metrics
г	variables
Гtrainable_variables
«regularization_losses
Т__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(ђ2batch_normalization_7/gamma
):'ђ2batch_normalization_7/beta
2:0ђ (2!batch_normalization_7/moving_mean
6:4ђ (2%batch_normalization_7/moving_variance
@
▒0
▓1
│2
┤3"
trackable_list_wrapper
0
▒0
▓1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
╬non_trainable_variables
¤layers
лmetrics
 Лlayer_regularization_losses
мlayer_metrics
х	variables
Хtrainable_variables
иregularization_losses
У__call__
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Мnon_trainable_variables
нlayers
Нmetrics
 оlayer_regularization_losses
Оlayer_metrics
╣	variables
║trainable_variables
╗regularization_losses
Ж__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses"
_generic_user_object
4:2@└2conv2d_transpose_3/kernel
%:#@2conv2d_transpose_3/bias
0
й0
Й1"
trackable_list_wrapper
0
й0
Й1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
пnon_trainable_variables
┘layers
┌metrics
 █layer_regularization_losses
▄layer_metrics
┐	variables
└trainable_variables
┴regularization_losses
В__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_8/gamma
(:&@2batch_normalization_8/beta
1:/@ (2!batch_normalization_8/moving_mean
5:3@ (2%batch_normalization_8/moving_variance
@
─0
┼1
к2
К3"
trackable_list_wrapper
0
─0
┼1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Пnon_trainable_variables
яlayers
▀metrics
 Яlayer_regularization_losses
рlayer_metrics
╚	variables
╔trainable_variables
╩regularization_losses
Ь__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Рnon_trainable_variables
сlayers
Сmetrics
 тlayer_regularization_losses
Тlayer_metrics
╠	variables
═trainable_variables
╬regularization_losses
­__call__
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses"
_generic_user_object
3:1D2conv2d_transpose_4/kernel
%:#2conv2d_transpose_4/bias
0
л0
Л1"
trackable_list_wrapper
0
л0
Л1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
уnon_trainable_variables
Уlayers
жmetrics
 Жlayer_regularization_losses
вlayer_metrics
м	variables
Мtrainable_variables
нregularization_losses
Ы__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
«
>0
?1
M2
N3
\4
]5
k6
l7
z8
{9
Ї10
ј11
а12
А13
│14
┤15
к16
К17"
trackable_list_wrapper
ќ
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
 31"
trackable_list_wrapper
(
В0"
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
>0
?1"
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
M0
N1"
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
\0
]1"
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
k0
l1"
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
z0
{1"
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
Ї0
ј1"
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
а0
А1"
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
│0
┤1"
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
к0
К1"
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

ьtotal

Ьcount
№	variables
­	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
ь0
Ь1"
trackable_list_wrapper
.
№	variables"
_generic_user_object
,:*2 Adam/batch_normalization/gamma/m
+:)2Adam/batch_normalization/beta/m
,:*@2Adam/conv2d/kernel/m
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
8:6ђђ2Adam/conv2d_transpose/kernel/m
):'ђ2Adam/conv2d_transpose/bias/m
/:-ђ2"Adam/batch_normalization_5/gamma/m
.:,ђ2!Adam/batch_normalization_5/beta/m
::8ђђ2 Adam/conv2d_transpose_1/kernel/m
+:)ђ2Adam/conv2d_transpose_1/bias/m
/:-ђ2"Adam/batch_normalization_6/gamma/m
.:,ђ2!Adam/batch_normalization_6/beta/m
::8ђђ2 Adam/conv2d_transpose_2/kernel/m
+:)ђ2Adam/conv2d_transpose_2/bias/m
/:-ђ2"Adam/batch_normalization_7/gamma/m
.:,ђ2!Adam/batch_normalization_7/beta/m
9:7@└2 Adam/conv2d_transpose_3/kernel/m
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
8:6ђђ2Adam/conv2d_transpose/kernel/v
):'ђ2Adam/conv2d_transpose/bias/v
/:-ђ2"Adam/batch_normalization_5/gamma/v
.:,ђ2!Adam/batch_normalization_5/beta/v
::8ђђ2 Adam/conv2d_transpose_1/kernel/v
+:)ђ2Adam/conv2d_transpose_1/bias/v
/:-ђ2"Adam/batch_normalization_6/gamma/v
.:,ђ2!Adam/batch_normalization_6/beta/v
::8ђђ2 Adam/conv2d_transpose_2/kernel/v
+:)ђ2Adam/conv2d_transpose_2/bias/v
/:-ђ2"Adam/batch_normalization_7/gamma/v
.:,ђ2!Adam/batch_normalization_7/beta/v
9:7@└2 Adam/conv2d_transpose_3/kernel/v
*:(@2Adam/conv2d_transpose_3/bias/v
.:,@2"Adam/batch_normalization_8/gamma/v
-:+@2!Adam/batch_normalization_8/beta/v
8:6D2 Adam/conv2d_transpose_4/kernel/v
*:(2Adam/conv2d_transpose_4/bias/v
Ж2у
'__inference_model_layer_call_fn_1031310
'__inference_model_layer_call_fn_1032876
'__inference_model_layer_call_fn_1032992
'__inference_model_layer_call_fn_1032350└
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
о2М
B__inference_model_layer_call_and_return_conditional_losses_1033304
B__inference_model_layer_call_and_return_conditional_losses_1033616
B__inference_model_layer_call_and_return_conditional_losses_1032493
B__inference_model_layer_call_and_return_conditional_losses_1032636└
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
ЯBП
"__inference__wrapped_model_1029790u_velv_velw_velpr0.2"ў
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
М2л
)__inference_reshape_layer_call_fn_1033621б
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
D__inference_reshape_layer_call_and_return_conditional_losses_1033635б
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
Н2м
+__inference_reshape_1_layer_call_fn_1033640б
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
­2ь
F__inference_reshape_1_layer_call_and_return_conditional_losses_1033654б
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
Н2м
+__inference_reshape_2_layer_call_fn_1033659б
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
­2ь
F__inference_reshape_2_layer_call_and_return_conditional_losses_1033673б
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
Н2м
+__inference_reshape_3_layer_call_fn_1033678б
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
­2ь
F__inference_reshape_3_layer_call_and_return_conditional_losses_1033692б
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
О2н
-__inference_concatenate_layer_call_fn_1033700б
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
Ы2№
H__inference_concatenate_layer_call_and_return_conditional_losses_1033709б
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
ќ2Њ
5__inference_batch_normalization_layer_call_fn_1033722
5__inference_batch_normalization_layer_call_fn_1033735
5__inference_batch_normalization_layer_call_fn_1033748
5__inference_batch_normalization_layer_call_fn_1033761┤
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
ѓ2 
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1033779
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1033797
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1033815
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1033833┤
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
м2¤
(__inference_conv2d_layer_call_fn_1033842б
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
C__inference_conv2d_layer_call_and_return_conditional_losses_1033853б
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
ъ2Џ
7__inference_batch_normalization_1_layer_call_fn_1033866
7__inference_batch_normalization_1_layer_call_fn_1033879
7__inference_batch_normalization_1_layer_call_fn_1033892
7__inference_batch_normalization_1_layer_call_fn_1033905┤
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
і2Є
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1033923
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1033941
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1033959
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1033977┤
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
н2Л
*__inference_conv2d_1_layer_call_fn_1033986б
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
№2В
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1033997б
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
ъ2Џ
7__inference_batch_normalization_2_layer_call_fn_1034010
7__inference_batch_normalization_2_layer_call_fn_1034023
7__inference_batch_normalization_2_layer_call_fn_1034036
7__inference_batch_normalization_2_layer_call_fn_1034049┤
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
і2Є
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1034067
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1034085
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1034103
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1034121┤
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
н2Л
*__inference_conv2d_2_layer_call_fn_1034130б
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
№2В
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1034141б
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
ъ2Џ
7__inference_batch_normalization_3_layer_call_fn_1034154
7__inference_batch_normalization_3_layer_call_fn_1034167
7__inference_batch_normalization_3_layer_call_fn_1034180
7__inference_batch_normalization_3_layer_call_fn_1034193┤
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
і2Є
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1034211
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1034229
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1034247
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1034265┤
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
н2Л
*__inference_conv2d_3_layer_call_fn_1034274б
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
№2В
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1034285б
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
ъ2Џ
7__inference_batch_normalization_4_layer_call_fn_1034298
7__inference_batch_normalization_4_layer_call_fn_1034311
7__inference_batch_normalization_4_layer_call_fn_1034324
7__inference_batch_normalization_4_layer_call_fn_1034337┤
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
і2Є
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1034355
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1034373
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1034391
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1034409┤
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
┘2о
/__inference_concatenate_1_layer_call_fn_1034415б
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
З2ы
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1034422б
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
љ2Ї
2__inference_conv2d_transpose_layer_call_fn_1034431
2__inference_conv2d_transpose_layer_call_fn_1034440б
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
к2├
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1034478
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1034502б
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
ъ2Џ
7__inference_batch_normalization_5_layer_call_fn_1034515
7__inference_batch_normalization_5_layer_call_fn_1034528
7__inference_batch_normalization_5_layer_call_fn_1034541
7__inference_batch_normalization_5_layer_call_fn_1034554┤
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
і2Є
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1034572
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1034590
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1034608
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1034626┤
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
┘2о
/__inference_concatenate_2_layer_call_fn_1034632б
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
З2ы
J__inference_concatenate_2_layer_call_and_return_conditional_losses_1034639б
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
ћ2Љ
4__inference_conv2d_transpose_1_layer_call_fn_1034648
4__inference_conv2d_transpose_1_layer_call_fn_1034657б
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
╩2К
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1034695
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1034719б
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
ъ2Џ
7__inference_batch_normalization_6_layer_call_fn_1034732
7__inference_batch_normalization_6_layer_call_fn_1034745
7__inference_batch_normalization_6_layer_call_fn_1034758
7__inference_batch_normalization_6_layer_call_fn_1034771┤
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
і2Є
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1034789
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1034807
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1034825
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1034843┤
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
┘2о
/__inference_concatenate_3_layer_call_fn_1034849б
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
З2ы
J__inference_concatenate_3_layer_call_and_return_conditional_losses_1034856б
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
ћ2Љ
4__inference_conv2d_transpose_2_layer_call_fn_1034865
4__inference_conv2d_transpose_2_layer_call_fn_1034874б
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
╩2К
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1034912
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1034936б
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
ъ2Џ
7__inference_batch_normalization_7_layer_call_fn_1034949
7__inference_batch_normalization_7_layer_call_fn_1034962
7__inference_batch_normalization_7_layer_call_fn_1034975
7__inference_batch_normalization_7_layer_call_fn_1034988┤
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
і2Є
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1035006
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1035024
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1035042
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1035060┤
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
┘2о
/__inference_concatenate_4_layer_call_fn_1035066б
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
З2ы
J__inference_concatenate_4_layer_call_and_return_conditional_losses_1035073б
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
ћ2Љ
4__inference_conv2d_transpose_3_layer_call_fn_1035082
4__inference_conv2d_transpose_3_layer_call_fn_1035091б
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
╩2К
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1035129
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1035153б
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
ъ2Џ
7__inference_batch_normalization_8_layer_call_fn_1035166
7__inference_batch_normalization_8_layer_call_fn_1035179
7__inference_batch_normalization_8_layer_call_fn_1035192
7__inference_batch_normalization_8_layer_call_fn_1035205┤
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
і2Є
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1035223
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1035241
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1035259
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1035277┤
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
┘2о
/__inference_concatenate_5_layer_call_fn_1035283б
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
З2ы
J__inference_concatenate_5_layer_call_and_return_conditional_losses_1035290б
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
ћ2Љ
4__inference_conv2d_transpose_4_layer_call_fn_1035299
4__inference_conv2d_transpose_4_layer_call_fn_1035308б
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
╩2К
O__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1035345
O__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1035368б
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
ПB┌
%__inference_signature_wrapper_1032760pr0.2u_velv_velw_vel"ћ
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
 ѓ
"__inference__wrapped_model_1029790█P<=>?DEKLMNSTZ[\]bcijklqrxyz{ёЁІїЇјЌўъЪаАфФ▒▓│┤йЙ─┼кКлЛ│б»
ДбБ
аџю
%і"
u_vel         ђђ
%і"
v_vel         ђђ
%і"
w_vel         ђђ
%і"
pr0.2         ђђ
ф "QфN
L
conv2d_transpose_46і3
conv2d_transpose_4         ђђь
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1033923ќKLMNMбJ
Cб@
:і7
inputs+                           @
p 
ф "?б<
5і2
0+                           @
џ ь
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1033941ќKLMNMбJ
Cб@
:і7
inputs+                           @
p
ф "?б<
5і2
0+                           @
џ ╠
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1033959vKLMN=б:
3б0
*і'
inputs         ЧЧ@
p 
ф "/б,
%і"
0         ЧЧ@
џ ╠
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1033977vKLMN=б:
3б0
*і'
inputs         ЧЧ@
p
ф "/б,
%і"
0         ЧЧ@
џ ┼
7__inference_batch_normalization_1_layer_call_fn_1033866ЅKLMNMбJ
Cб@
:і7
inputs+                           @
p 
ф "2і/+                           @┼
7__inference_batch_normalization_1_layer_call_fn_1033879ЅKLMNMбJ
Cб@
:і7
inputs+                           @
p
ф "2і/+                           @ц
7__inference_batch_normalization_1_layer_call_fn_1033892iKLMN=б:
3б0
*і'
inputs         ЧЧ@
p 
ф ""і         ЧЧ@ц
7__inference_batch_normalization_1_layer_call_fn_1033905iKLMN=б:
3б0
*і'
inputs         ЧЧ@
p
ф ""і         ЧЧ@№
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1034067ўZ[\]NбK
DбA
;і8
inputs,                           ђ
p 
ф "@б=
6і3
0,                           ђ
џ №
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1034085ўZ[\]NбK
DбA
;і8
inputs,                           ђ
p
ф "@б=
6і3
0,                           ђ
џ ╬
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1034103xZ[\]>б;
4б1
+і(
inputs         ЩЩђ
p 
ф "0б-
&і#
0         ЩЩђ
џ ╬
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1034121xZ[\]>б;
4б1
+і(
inputs         ЩЩђ
p
ф "0б-
&і#
0         ЩЩђ
џ К
7__inference_batch_normalization_2_layer_call_fn_1034010ІZ[\]NбK
DбA
;і8
inputs,                           ђ
p 
ф "3і0,                           ђК
7__inference_batch_normalization_2_layer_call_fn_1034023ІZ[\]NбK
DбA
;і8
inputs,                           ђ
p
ф "3і0,                           ђд
7__inference_batch_normalization_2_layer_call_fn_1034036kZ[\]>б;
4б1
+і(
inputs         ЩЩђ
p 
ф "#і          ЩЩђд
7__inference_batch_normalization_2_layer_call_fn_1034049kZ[\]>б;
4б1
+і(
inputs         ЩЩђ
p
ф "#і          ЩЩђ№
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1034211ўijklNбK
DбA
;і8
inputs,                           ђ
p 
ф "@б=
6і3
0,                           ђ
џ №
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1034229ўijklNбK
DбA
;і8
inputs,                           ђ
p
ф "@б=
6і3
0,                           ђ
џ ╬
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1034247xijkl>б;
4б1
+і(
inputs         ЭЭђ
p 
ф "0б-
&і#
0         ЭЭђ
џ ╬
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1034265xijkl>б;
4б1
+і(
inputs         ЭЭђ
p
ф "0б-
&і#
0         ЭЭђ
џ К
7__inference_batch_normalization_3_layer_call_fn_1034154ІijklNбK
DбA
;і8
inputs,                           ђ
p 
ф "3і0,                           ђК
7__inference_batch_normalization_3_layer_call_fn_1034167ІijklNбK
DбA
;і8
inputs,                           ђ
p
ф "3і0,                           ђд
7__inference_batch_normalization_3_layer_call_fn_1034180kijkl>б;
4б1
+і(
inputs         ЭЭђ
p 
ф "#і          ЭЭђд
7__inference_batch_normalization_3_layer_call_fn_1034193kijkl>б;
4б1
+і(
inputs         ЭЭђ
p
ф "#і          ЭЭђ№
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1034355ўxyz{NбK
DбA
;і8
inputs,                           ђ
p 
ф "@б=
6і3
0,                           ђ
џ №
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1034373ўxyz{NбK
DбA
;і8
inputs,                           ђ
p
ф "@б=
6і3
0,                           ђ
џ ╬
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1034391xxyz{>б;
4б1
+і(
inputs         ШШђ
p 
ф "0б-
&і#
0         ШШђ
џ ╬
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1034409xxyz{>б;
4б1
+і(
inputs         ШШђ
p
ф "0б-
&і#
0         ШШђ
џ К
7__inference_batch_normalization_4_layer_call_fn_1034298Іxyz{NбK
DбA
;і8
inputs,                           ђ
p 
ф "3і0,                           ђК
7__inference_batch_normalization_4_layer_call_fn_1034311Іxyz{NбK
DбA
;і8
inputs,                           ђ
p
ф "3і0,                           ђд
7__inference_batch_normalization_4_layer_call_fn_1034324kxyz{>б;
4б1
+і(
inputs         ШШђ
p 
ф "#і          ШШђд
7__inference_batch_normalization_4_layer_call_fn_1034337kxyz{>б;
4б1
+і(
inputs         ШШђ
p
ф "#і          ШШђз
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1034572юІїЇјNбK
DбA
;і8
inputs,                           ђ
p 
ф "@б=
6і3
0,                           ђ
џ з
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1034590юІїЇјNбK
DбA
;і8
inputs,                           ђ
p
ф "@б=
6і3
0,                           ђ
џ м
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1034608|ІїЇј>б;
4б1
+і(
inputs         ЭЭђ
p 
ф "0б-
&і#
0         ЭЭђ
џ м
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1034626|ІїЇј>б;
4б1
+і(
inputs         ЭЭђ
p
ф "0б-
&і#
0         ЭЭђ
џ ╦
7__inference_batch_normalization_5_layer_call_fn_1034515ЈІїЇјNбK
DбA
;і8
inputs,                           ђ
p 
ф "3і0,                           ђ╦
7__inference_batch_normalization_5_layer_call_fn_1034528ЈІїЇјNбK
DбA
;і8
inputs,                           ђ
p
ф "3і0,                           ђф
7__inference_batch_normalization_5_layer_call_fn_1034541oІїЇј>б;
4б1
+і(
inputs         ЭЭђ
p 
ф "#і          ЭЭђф
7__inference_batch_normalization_5_layer_call_fn_1034554oІїЇј>б;
4б1
+і(
inputs         ЭЭђ
p
ф "#і          ЭЭђз
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1034789юъЪаАNбK
DбA
;і8
inputs,                           ђ
p 
ф "@б=
6і3
0,                           ђ
џ з
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1034807юъЪаАNбK
DбA
;і8
inputs,                           ђ
p
ф "@б=
6і3
0,                           ђ
џ м
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1034825|ъЪаА>б;
4б1
+і(
inputs         ЩЩђ
p 
ф "0б-
&і#
0         ЩЩђ
џ м
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1034843|ъЪаА>б;
4б1
+і(
inputs         ЩЩђ
p
ф "0б-
&і#
0         ЩЩђ
џ ╦
7__inference_batch_normalization_6_layer_call_fn_1034732ЈъЪаАNбK
DбA
;і8
inputs,                           ђ
p 
ф "3і0,                           ђ╦
7__inference_batch_normalization_6_layer_call_fn_1034745ЈъЪаАNбK
DбA
;і8
inputs,                           ђ
p
ф "3і0,                           ђф
7__inference_batch_normalization_6_layer_call_fn_1034758oъЪаА>б;
4б1
+і(
inputs         ЩЩђ
p 
ф "#і          ЩЩђф
7__inference_batch_normalization_6_layer_call_fn_1034771oъЪаА>б;
4б1
+і(
inputs         ЩЩђ
p
ф "#і          ЩЩђз
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1035006ю▒▓│┤NбK
DбA
;і8
inputs,                           ђ
p 
ф "@б=
6і3
0,                           ђ
џ з
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1035024ю▒▓│┤NбK
DбA
;і8
inputs,                           ђ
p
ф "@б=
6і3
0,                           ђ
џ м
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1035042|▒▓│┤>б;
4б1
+і(
inputs         ЧЧђ
p 
ф "0б-
&і#
0         ЧЧђ
џ м
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1035060|▒▓│┤>б;
4б1
+і(
inputs         ЧЧђ
p
ф "0б-
&і#
0         ЧЧђ
џ ╦
7__inference_batch_normalization_7_layer_call_fn_1034949Ј▒▓│┤NбK
DбA
;і8
inputs,                           ђ
p 
ф "3і0,                           ђ╦
7__inference_batch_normalization_7_layer_call_fn_1034962Ј▒▓│┤NбK
DбA
;і8
inputs,                           ђ
p
ф "3і0,                           ђф
7__inference_batch_normalization_7_layer_call_fn_1034975o▒▓│┤>б;
4б1
+і(
inputs         ЧЧђ
p 
ф "#і          ЧЧђф
7__inference_batch_normalization_7_layer_call_fn_1034988o▒▓│┤>б;
4б1
+і(
inputs         ЧЧђ
p
ф "#і          ЧЧђы
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1035223џ─┼кКMбJ
Cб@
:і7
inputs+                           @
p 
ф "?б<
5і2
0+                           @
џ ы
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1035241џ─┼кКMбJ
Cб@
:і7
inputs+                           @
p
ф "?б<
5і2
0+                           @
џ л
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1035259z─┼кК=б:
3б0
*і'
inputs         ђђ@
p 
ф "/б,
%і"
0         ђђ@
џ л
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1035277z─┼кК=б:
3б0
*і'
inputs         ђђ@
p
ф "/б,
%і"
0         ђђ@
џ ╔
7__inference_batch_normalization_8_layer_call_fn_1035166Ї─┼кКMбJ
Cб@
:і7
inputs+                           @
p 
ф "2і/+                           @╔
7__inference_batch_normalization_8_layer_call_fn_1035179Ї─┼кКMбJ
Cб@
:і7
inputs+                           @
p
ф "2і/+                           @е
7__inference_batch_normalization_8_layer_call_fn_1035192m─┼кК=б:
3б0
*і'
inputs         ђђ@
p 
ф ""і         ђђ@е
7__inference_batch_normalization_8_layer_call_fn_1035205m─┼кК=б:
3б0
*і'
inputs         ђђ@
p
ф ""і         ђђ@в
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1033779ќ<=>?MбJ
Cб@
:і7
inputs+                           
p 
ф "?б<
5і2
0+                           
џ в
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1033797ќ<=>?MбJ
Cб@
:і7
inputs+                           
p
ф "?б<
5і2
0+                           
џ ╩
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1033815v<=>?=б:
3б0
*і'
inputs         ђђ
p 
ф "/б,
%і"
0         ђђ
џ ╩
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1033833v<=>?=б:
3б0
*і'
inputs         ђђ
p
ф "/б,
%і"
0         ђђ
џ ├
5__inference_batch_normalization_layer_call_fn_1033722Ѕ<=>?MбJ
Cб@
:і7
inputs+                           
p 
ф "2і/+                           ├
5__inference_batch_normalization_layer_call_fn_1033735Ѕ<=>?MбJ
Cб@
:і7
inputs+                           
p
ф "2і/+                           б
5__inference_batch_normalization_layer_call_fn_1033748i<=>?=б:
3б0
*і'
inputs         ђђ
p 
ф ""і         ђђб
5__inference_batch_normalization_layer_call_fn_1033761i<=>?=б:
3б0
*і'
inputs         ђђ
p
ф ""і         ђђз
J__inference_concatenate_1_layer_call_and_return_conditional_losses_1034422цpбm
fбc
aџ^
-і*
inputs/0         ШШђ
-і*
inputs/1         ШШђ
ф "0б-
&і#
0         ШШђ
џ ╦
/__inference_concatenate_1_layer_call_fn_1034415Ќpбm
fбc
aџ^
-і*
inputs/0         ШШђ
-і*
inputs/1         ШШђ
ф "#і          ШШђз
J__inference_concatenate_2_layer_call_and_return_conditional_losses_1034639цpбm
fбc
aџ^
-і*
inputs/0         ЭЭђ
-і*
inputs/1         ЭЭђ
ф "0б-
&і#
0         ЭЭђ
џ ╦
/__inference_concatenate_2_layer_call_fn_1034632Ќpбm
fбc
aџ^
-і*
inputs/0         ЭЭђ
-і*
inputs/1         ЭЭђ
ф "#і          ЭЭђз
J__inference_concatenate_3_layer_call_and_return_conditional_losses_1034856цpбm
fбc
aџ^
-і*
inputs/0         ЩЩђ
-і*
inputs/1         ЩЩђ
ф "0б-
&і#
0         ЩЩђ
џ ╦
/__inference_concatenate_3_layer_call_fn_1034849Ќpбm
fбc
aџ^
-і*
inputs/0         ЩЩђ
-і*
inputs/1         ЩЩђ
ф "#і          ЩЩђЫ
J__inference_concatenate_4_layer_call_and_return_conditional_losses_1035073Бoбl
eбb
`џ]
,і)
inputs/0         ЧЧ@
-і*
inputs/1         ЧЧђ
ф "0б-
&і#
0         ЧЧ└
џ ╩
/__inference_concatenate_4_layer_call_fn_1035066ќoбl
eбb
`џ]
,і)
inputs/0         ЧЧ@
-і*
inputs/1         ЧЧђ
ф "#і          ЧЧ└­
J__inference_concatenate_5_layer_call_and_return_conditional_losses_1035290Аnбk
dбa
_џ\
,і)
inputs/0         ђђ
,і)
inputs/1         ђђ@
ф "/б,
%і"
0         ђђD
џ ╚
/__inference_concatenate_5_layer_call_fn_1035283ћnбk
dбa
_џ\
,і)
inputs/0         ђђ
,і)
inputs/1         ђђ@
ф ""і         ђђDл
H__inference_concatenate_layer_call_and_return_conditional_losses_1033709Ѓ¤б╦
├б┐
╝џИ
,і)
inputs/0         ђђ
,і)
inputs/1         ђђ
,і)
inputs/2         ђђ
,і)
inputs/3         ђђ
ф "/б,
%і"
0         ђђ
џ е
-__inference_concatenate_layer_call_fn_1033700Ш¤б╦
├б┐
╝џИ
,і)
inputs/0         ђђ
,і)
inputs/1         ђђ
,і)
inputs/2         ђђ
,і)
inputs/3         ђђ
ф ""і         ђђ║
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1033997qST9б6
/б,
*і'
inputs         ЧЧ@
ф "0б-
&і#
0         ЩЩђ
џ њ
*__inference_conv2d_1_layer_call_fn_1033986dST9б6
/б,
*і'
inputs         ЧЧ@
ф "#і          ЩЩђ╗
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1034141rbc:б7
0б-
+і(
inputs         ЩЩђ
ф "0б-
&і#
0         ЭЭђ
џ Њ
*__inference_conv2d_2_layer_call_fn_1034130ebc:б7
0б-
+і(
inputs         ЩЩђ
ф "#і          ЭЭђ╗
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1034285rqr:б7
0б-
+і(
inputs         ЭЭђ
ф "0б-
&і#
0         ШШђ
џ Њ
*__inference_conv2d_3_layer_call_fn_1034274eqr:б7
0б-
+і(
inputs         ЭЭђ
ф "#і          ШШђи
C__inference_conv2d_layer_call_and_return_conditional_losses_1033853pDE9б6
/б,
*і'
inputs         ђђ
ф "/б,
%і"
0         ЧЧ@
џ Ј
(__inference_conv2d_layer_call_fn_1033842cDE9б6
/б,
*і'
inputs         ђђ
ф ""і         ЧЧ@У
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1034695ћЌўJбG
@б=
;і8
inputs,                           ђ
ф "@б=
6і3
0,                           ђ
џ К
O__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1034719tЌў:б7
0б-
+і(
inputs         ЭЭђ
ф "0б-
&і#
0         ЩЩђ
џ └
4__inference_conv2d_transpose_1_layer_call_fn_1034648ЄЌўJбG
@б=
;і8
inputs,                           ђ
ф "3і0,                           ђЪ
4__inference_conv2d_transpose_1_layer_call_fn_1034657gЌў:б7
0б-
+і(
inputs         ЭЭђ
ф "#і          ЩЩђУ
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1034912ћфФJбG
@б=
;і8
inputs,                           ђ
ф "@б=
6і3
0,                           ђ
џ К
O__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1034936tфФ:б7
0б-
+і(
inputs         ЩЩђ
ф "0б-
&і#
0         ЧЧђ
џ └
4__inference_conv2d_transpose_2_layer_call_fn_1034865ЄфФJбG
@б=
;і8
inputs,                           ђ
ф "3і0,                           ђЪ
4__inference_conv2d_transpose_2_layer_call_fn_1034874gфФ:б7
0б-
+і(
inputs         ЩЩђ
ф "#і          ЧЧђу
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1035129ЊйЙJбG
@б=
;і8
inputs,                           └
ф "?б<
5і2
0+                           @
џ к
O__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1035153sйЙ:б7
0б-
+і(
inputs         ЧЧ└
ф "/б,
%і"
0         ђђ@
џ ┐
4__inference_conv2d_transpose_3_layer_call_fn_1035082єйЙJбG
@б=
;і8
inputs,                           └
ф "2і/+                           @ъ
4__inference_conv2d_transpose_3_layer_call_fn_1035091fйЙ:б7
0б-
+і(
inputs         ЧЧ└
ф ""і         ђђ@Т
O__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1035345њлЛIбF
?б<
:і7
inputs+                           D
ф "?б<
5і2
0+                           
џ ┼
O__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_1035368rлЛ9б6
/б,
*і'
inputs         ђђD
ф "/б,
%і"
0         ђђ
џ Й
4__inference_conv2d_transpose_4_layer_call_fn_1035299ЁлЛIбF
?б<
:і7
inputs+                           D
ф "2і/+                           Ю
4__inference_conv2d_transpose_4_layer_call_fn_1035308eлЛ9б6
/б,
*і'
inputs         ђђD
ф ""і         ђђТ
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1034478ћёЁJбG
@б=
;і8
inputs,                           ђ
ф "@б=
6і3
0,                           ђ
џ ┼
M__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1034502tёЁ:б7
0б-
+і(
inputs         ШШђ
ф "0б-
&і#
0         ЭЭђ
џ Й
2__inference_conv2d_transpose_layer_call_fn_1034431ЄёЁJбG
@б=
;і8
inputs,                           ђ
ф "3і0,                           ђЮ
2__inference_conv2d_transpose_layer_call_fn_1034440gёЁ:б7
0б-
+і(
inputs         ШШђ
ф "#і          ЭЭђѕ
B__inference_model_layer_call_and_return_conditional_losses_1032493┴P<=>?DEKLMNSTZ[\]bcijklqrxyz{ёЁІїЇјЌўъЪаАфФ▒▓│┤йЙ─┼кКлЛ╗би
»бФ
аџю
%і"
u_vel         ђђ
%і"
v_vel         ђђ
%і"
w_vel         ђђ
%і"
pr0.2         ђђ
p 

 
ф "/б,
%і"
0         ђђ
џ ѕ
B__inference_model_layer_call_and_return_conditional_losses_1032636┴P<=>?DEKLMNSTZ[\]bcijklqrxyz{ёЁІїЇјЌўъЪаАфФ▒▓│┤йЙ─┼кКлЛ╗би
»бФ
аџю
%і"
u_vel         ђђ
%і"
v_vel         ђђ
%і"
w_vel         ђђ
%і"
pr0.2         ђђ
p

 
ф "/б,
%і"
0         ђђ
џ ћ
B__inference_model_layer_call_and_return_conditional_losses_1033304═P<=>?DEKLMNSTZ[\]bcijklqrxyz{ёЁІїЇјЌўъЪаАфФ▒▓│┤йЙ─┼кКлЛКб├
╗би
гџе
(і%
inputs/0         ђђ
(і%
inputs/1         ђђ
(і%
inputs/2         ђђ
(і%
inputs/3         ђђ
p 

 
ф "/б,
%і"
0         ђђ
џ ћ
B__inference_model_layer_call_and_return_conditional_losses_1033616═P<=>?DEKLMNSTZ[\]bcijklqrxyz{ёЁІїЇјЌўъЪаАфФ▒▓│┤йЙ─┼кКлЛКб├
╗би
гџе
(і%
inputs/0         ђђ
(і%
inputs/1         ђђ
(і%
inputs/2         ђђ
(і%
inputs/3         ђђ
p

 
ф "/б,
%і"
0         ђђ
џ Я
'__inference_model_layer_call_fn_1031310┤P<=>?DEKLMNSTZ[\]bcijklqrxyz{ёЁІїЇјЌўъЪаАфФ▒▓│┤йЙ─┼кКлЛ╗би
»бФ
аџю
%і"
u_vel         ђђ
%і"
v_vel         ђђ
%і"
w_vel         ђђ
%і"
pr0.2         ђђ
p 

 
ф ""і         ђђЯ
'__inference_model_layer_call_fn_1032350┤P<=>?DEKLMNSTZ[\]bcijklqrxyz{ёЁІїЇјЌўъЪаАфФ▒▓│┤йЙ─┼кКлЛ╗би
»бФ
аџю
%і"
u_vel         ђђ
%і"
v_vel         ђђ
%і"
w_vel         ђђ
%і"
pr0.2         ђђ
p

 
ф ""і         ђђВ
'__inference_model_layer_call_fn_1032876└P<=>?DEKLMNSTZ[\]bcijklqrxyz{ёЁІїЇјЌўъЪаАфФ▒▓│┤йЙ─┼кКлЛКб├
╗би
гџе
(і%
inputs/0         ђђ
(і%
inputs/1         ђђ
(і%
inputs/2         ђђ
(і%
inputs/3         ђђ
p 

 
ф ""і         ђђВ
'__inference_model_layer_call_fn_1032992└P<=>?DEKLMNSTZ[\]bcijklqrxyz{ёЁІїЇјЌўъЪаАфФ▒▓│┤йЙ─┼кКлЛКб├
╗би
гџе
(і%
inputs/0         ђђ
(і%
inputs/1         ђђ
(і%
inputs/2         ђђ
(і%
inputs/3         ђђ
p

 
ф ""і         ђђ▓
F__inference_reshape_1_layer_call_and_return_conditional_losses_1033654h5б2
+б(
&і#
inputs         ђђ
ф "/б,
%і"
0         ђђ
џ і
+__inference_reshape_1_layer_call_fn_1033640[5б2
+б(
&і#
inputs         ђђ
ф ""і         ђђ▓
F__inference_reshape_2_layer_call_and_return_conditional_losses_1033673h5б2
+б(
&і#
inputs         ђђ
ф "/б,
%і"
0         ђђ
џ і
+__inference_reshape_2_layer_call_fn_1033659[5б2
+б(
&і#
inputs         ђђ
ф ""і         ђђ▓
F__inference_reshape_3_layer_call_and_return_conditional_losses_1033692h5б2
+б(
&і#
inputs         ђђ
ф "/б,
%і"
0         ђђ
џ і
+__inference_reshape_3_layer_call_fn_1033678[5б2
+б(
&і#
inputs         ђђ
ф ""і         ђђ░
D__inference_reshape_layer_call_and_return_conditional_losses_1033635h5б2
+б(
&і#
inputs         ђђ
ф "/б,
%і"
0         ђђ
џ ѕ
)__inference_reshape_layer_call_fn_1033621[5б2
+б(
&і#
inputs         ђђ
ф ""і         ђђб
%__inference_signature_wrapper_1032760ЭP<=>?DEKLMNSTZ[\]bcijklqrxyz{ёЁІїЇјЌўъЪаАфФ▒▓│┤йЙ─┼кКлЛлб╠
б 
─ф└
.
pr0.2%і"
pr0.2         ђђ
.
u_vel%і"
u_vel         ђђ
.
v_vel%і"
v_vel         ђђ
.
w_vel%і"
w_vel         ђђ"QфN
L
conv2d_transpose_46і3
conv2d_transpose_4         ђђ