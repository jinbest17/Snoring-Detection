?[
??
:
Add
x"T
y"T
z"T"
Ttype:
2	
A
AddV2
x"T
y"T
z"T"
Ttype:
2	??
x
Assign
ref"T?

value"T

output_ref"T?"	
Ttype"
validate_shapebool("
use_lockingbool(?
?
AudioMicrofrontend	
audio
filterbanks"out_type"
sample_rateint?}"
window_sizeint"
window_stepint
"
num_channelsint " 
upper_band_limitfloat% `?E" 
lower_band_limitfloat%  ?B"
smoothing_bitsint
"
even_smoothingfloat%???<"
odd_smoothingfloat%??u="$
min_signal_remainingfloat%??L="
enable_pcanbool( "
pcan_strengthfloat%33s?"
pcan_offsetfloat%  ?B"
	gain_bitsint"

enable_logbool("
scale_shiftint"
left_contextint "
right_contextint "
frame_strideint"
zero_paddingbool( "
	out_scaleint"
out_typetype0:
2
p
AudioSpectrogram	
input
spectrogram"
window_sizeint"
strideint"
magnitude_squaredbool( 
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
?
	DecodeWav
contents	
audio
sample_rate"$
desired_channelsint?????????"#
desired_samplesint?????????
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
s

VariableV2
ref"dtype?"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ?"serve*1.15.02v1.15.0-rc3-22-g590d6eef7e?A
I
wav_dataPlaceholder*
dtype0*
_output_shapes
: *
shape: 
}
decoded_sample_data	DecodeWavwav_data*
desired_samples?}*!
_output_shapes
:	?}: *
desired_channels
?
AudioSpectrogramAudioSpectrogramdecoded_sample_data*
window_size?*
magnitude_squared(*#
_output_shapes
:1?*
stride?
J
Mul/yConst*
dtype0*
_output_shapes
: *
valueB
 * ??F
P
MulMuldecoded_sample_dataMul/y*
T0*
_output_shapes
:	?}
Z
CastCastMul*
Truncate( *
_output_shapes
:	?}*

DstT0*

SrcT0
`
Reshape/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
[
ReshapeReshapeCastReshape/shape*
_output_shapes	
:?}*
T0*
Tshape0
?
AudioMicrofrontendAudioMicrofrontendReshape*
upper_band_limit% `?E*
frame_stride*
pcan_offset%  ?B*
min_signal_remaining%??L=*
right_context *
smoothing_bits
*
window_step*
	gain_bits*
sample_rate?}*
left_context *
_output_shapes

:1(*
enable_pcan(*
pcan_strength%33s?*
even_smoothing%???<*
	out_scale*
lower_band_limit%  ?B*
window_size*
zero_padding( *
num_channels(*

enable_log(*
odd_smoothing%??u=*
out_type0*
scale_shift
L
Mul_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *   =
R
Mul_1MulAudioMicrofrontendMul_1/y*
_output_shapes

:1(*
T0
`
Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"?????  
d
	Reshape_1ReshapeMul_1Reshape_1/shape*
T0*
Tshape0*
_output_shapes
:	?
?
*weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"?     *
_class
loc:@weights
?
)weights/Initializer/truncated_normal/meanConst*
valueB
 *    *
_class
loc:@weights*
dtype0*
_output_shapes
: 
?
+weights/Initializer/truncated_normal/stddevConst*
valueB
 *o?:*
_class
loc:@weights*
dtype0*
_output_shapes
: 
?
4weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal*weights/Initializer/truncated_normal/shape*
seed2 *
dtype0*
_output_shapes
:	?*

seed *
T0*
_class
loc:@weights
?
(weights/Initializer/truncated_normal/mulMul4weights/Initializer/truncated_normal/TruncatedNormal+weights/Initializer/truncated_normal/stddev*
T0*
_class
loc:@weights*
_output_shapes
:	?
?
$weights/Initializer/truncated_normalAdd(weights/Initializer/truncated_normal/mul)weights/Initializer/truncated_normal/mean*
_output_shapes
:	?*
T0*
_class
loc:@weights
?
weights
VariableV2*
	container *
shape:	?*
dtype0*
_output_shapes
:	?*
shared_name *
_class
loc:@weights
?
weights/AssignAssignweights$weights/Initializer/truncated_normal*
use_locking(*
T0*
_class
loc:@weights*
validate_shape(*
_output_shapes
:	?
g
weights/readIdentityweights*
T0*
_class
loc:@weights*
_output_shapes
:	?
|
bias/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *
_class
	loc:@bias
?
bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
_class
	loc:@bias*
	container *
shape:
?
bias/AssignAssignbiasbias/Initializer/zeros*
use_locking(*
T0*
_class
	loc:@bias*
validate_shape(*
_output_shapes
:
Y
	bias/readIdentitybias*
T0*
_class
	loc:@bias*
_output_shapes
:
x
MatMulMatMul	Reshape_1weights/read*
T0*
_output_shapes

:*
transpose_a( *
transpose_b( 
H
addAddV2MatMul	bias/read*
T0*
_output_shapes

:
G
labels_softmaxSoftmaxadd*
T0*
_output_shapes

:
Y
save/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 
n
save/SaveV2/tensor_namesConst*"
valueBBbiasBweights*
dtype0*
_output_shapes
:
g
save/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0*
_output_shapes
:
z
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbiasweights*
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
?
save/RestoreV2/tensor_namesConst"/device:CPU:0*"
valueBBbiasBweights*
dtype0*
_output_shapes
:
y
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0*
_output_shapes
:
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
_output_shapes

::*
dtypes
2
?
save/AssignAssignbiassave/RestoreV2*
use_locking(*
T0*
_class
	loc:@bias*
validate_shape(*
_output_shapes
:
?
save/Assign_1Assignweightssave/RestoreV2:1*
use_locking(*
T0*
_class
loc:@weights*
validate_shape(*
_output_shapes
:	?
6
save/restore_allNoOp^save/Assign^save/Assign_1
[
save_1/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
dtype0*
_output_shapes
: *
shape: 
?
save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_261f9c3e60f24ffbbee206f89ad71b91/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
m
save_1/ShardedFilename/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 
?
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards"/device:CPU:0*
_output_shapes
: 

save_1/SaveV2/tensor_namesConst"/device:CPU:0*"
valueBBbiasBweights*
dtype0*
_output_shapes
:
x
save_1/SaveV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0*
_output_shapes
:
?
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesbiasweights"/device:CPU:0*
dtypes
2
?
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2"/device:CPU:0*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
?
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
?
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const"/device:CPU:0*
delete_old_dirs(
?
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
?
save_1/RestoreV2/tensor_namesConst"/device:CPU:0*"
valueBBbiasBweights*
dtype0*
_output_shapes
:
{
!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0*
_output_shapes
:
?
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*
_output_shapes

::*
dtypes
2
?
save_1/AssignAssignbiassave_1/RestoreV2*
T0*
_class
	loc:@bias*
validate_shape(*
_output_shapes
:*
use_locking(
?
save_1/Assign_1Assignweightssave_1/RestoreV2:1*
T0*
_class
loc:@weights*
validate_shape(*
_output_shapes
:	?*
use_locking(
>
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1
1
save_1/restore_allNoOp^save_1/restore_shard"?B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"?
	variables??
U
	weights:0weights/Assignweights/read:02&weights/Initializer/truncated_normal:08
>
bias:0bias/Assignbias/read:02bias/Initializer/zeros:08"?
trainable_variables??
U
	weights:0weights/Assignweights/read:02&weights/Initializer/truncated_normal:08
>
bias:0bias/Assignbias/read:02bias/Initializer/zeros:08*~
serving_defaultk
#
input
Reshape_1:0	?(
output
labels_softmax:0tensorflow/serving/predict