Ć
??
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
?
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
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
E
Relu
features"T
activations"T"
Ttype:
2	
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
shared_namestring ?"serve*1.15.02unknown?f
I
wav_dataPlaceholder*
shape: *
dtype0*
_output_shapes
: 
}
decoded_sample_data	DecodeWavwav_data*!
_output_shapes
:	?}: *
desired_samples?}*
desired_channels
?
AudioSpectrogramAudioSpectrogramdecoded_sample_data*
magnitude_squared(*#
_output_shapes
:1?*
window_size?*
stride?
J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 * ??F
P
MulMuldecoded_sample_dataMul/y*
_output_shapes
:	?}*
T0
Z
CastCastMul*
Truncate( *
_output_shapes
:	?}*

SrcT0*

DstT0
`
Reshape/shapeConst*
valueB:
?????????*
_output_shapes
:*
dtype0
[
ReshapeReshapeCastReshape/shape*
Tshape0*
T0*
_output_shapes	
:?}
?
AudioMicrofrontendAudioMicrofrontendReshape*
sample_rate?}*
pcan_strength%33s?*

enable_log(*
window_size*
	gain_bits*
right_context *
scale_shift*
window_step*
_output_shapes

:1(*
smoothing_bits
*
num_channels(*
upper_band_limit% `?E*
out_type0*
min_signal_remaining%??L=*
	out_scale*
odd_smoothing%??u=*
left_context *
enable_pcan(*
pcan_offset%  ?B*
zero_padding( *
even_smoothing%???<*
frame_stride*
lower_band_limit%  ?B
L
Mul_1/yConst*
dtype0*
valueB
 *   =*
_output_shapes
: 
R
Mul_1MulAudioMicrofrontendMul_1/y*
_output_shapes

:1(*
T0
`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  
d
	Reshape_1ReshapeMul_1Reshape_1/shape*
Tshape0*
_output_shapes
:	?*
T0
h
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????1   (      
o
	Reshape_2Reshape	Reshape_1Reshape_2/shape*
T0*&
_output_shapes
:1(*
Tshape0
?
0first_weights/Initializer/truncated_normal/shapeConst*%
valueB"
            * 
_class
loc:@first_weights*
dtype0*
_output_shapes
:
?
/first_weights/Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: * 
_class
loc:@first_weights
?
1first_weights/Initializer/truncated_normal/stddevConst* 
_class
loc:@first_weights*
_output_shapes
: *
valueB
 *
?#<*
dtype0
?
:first_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal0first_weights/Initializer/truncated_normal/shape* 
_class
loc:@first_weights*
dtype0*
seed2 *&
_output_shapes
:
*
T0*

seed 
?
.first_weights/Initializer/truncated_normal/mulMul:first_weights/Initializer/truncated_normal/TruncatedNormal1first_weights/Initializer/truncated_normal/stddev* 
_class
loc:@first_weights*
T0*&
_output_shapes
:

?
*first_weights/Initializer/truncated_normalAdd.first_weights/Initializer/truncated_normal/mul/first_weights/Initializer/truncated_normal/mean* 
_class
loc:@first_weights*
T0*&
_output_shapes
:

?
first_weights
VariableV2*&
_output_shapes
:
*
dtype0*
shape:
*
shared_name *
	container * 
_class
loc:@first_weights
?
first_weights/AssignAssignfirst_weights*first_weights/Initializer/truncated_normal*
use_locking(* 
_class
loc:@first_weights*
T0*
validate_shape(*&
_output_shapes
:

?
first_weights/readIdentityfirst_weights*
T0*&
_output_shapes
:
* 
_class
loc:@first_weights
?
first_bias/Initializer/zerosConst*
_output_shapes
:*
_class
loc:@first_bias*
dtype0*
valueB*    
?

first_bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
	container *
shape:*
_class
loc:@first_bias
?
first_bias/AssignAssign
first_biasfirst_bias/Initializer/zeros*
validate_shape(*
_class
loc:@first_bias*
use_locking(*
T0*
_output_shapes
:
k
first_bias/readIdentity
first_bias*
_class
loc:@first_bias*
T0*
_output_shapes
:
?
Conv2DConv2D	Reshape_2first_weights/read*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingSAME*
T0*
explicit_paddings
 *&
_output_shapes
:*
	dilations
*
strides

V
addAddV2Conv2Dfirst_bias/read*
T0*&
_output_shapes
:
B
ReluReluadd*
T0*&
_output_shapes
:
`
Reshape_3/shapeConst*
valueB"?????  *
dtype0*
_output_shapes
:
c
	Reshape_3ReshapeReluReshape_3/shape*
_output_shapes
:	?*
Tshape0*
T0
?
3final_fc_weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"?     *#
_class
loc:@final_fc_weights
?
2final_fc_weights/Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: *#
_class
loc:@final_fc_weights
?
4final_fc_weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *#
_class
loc:@final_fc_weights*
valueB
 *
?#<
?
=final_fc_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal3final_fc_weights/Initializer/truncated_normal/shape*
_output_shapes
:	?*
dtype0*
seed2 *#
_class
loc:@final_fc_weights*
T0*

seed 
?
1final_fc_weights/Initializer/truncated_normal/mulMul=final_fc_weights/Initializer/truncated_normal/TruncatedNormal4final_fc_weights/Initializer/truncated_normal/stddev*#
_class
loc:@final_fc_weights*
_output_shapes
:	?*
T0
?
-final_fc_weights/Initializer/truncated_normalAdd1final_fc_weights/Initializer/truncated_normal/mul2final_fc_weights/Initializer/truncated_normal/mean*
T0*#
_class
loc:@final_fc_weights*
_output_shapes
:	?
?
final_fc_weights
VariableV2*
shared_name *
shape:	?*#
_class
loc:@final_fc_weights*
_output_shapes
:	?*
	container *
dtype0
?
final_fc_weights/AssignAssignfinal_fc_weights-final_fc_weights/Initializer/truncated_normal*#
_class
loc:@final_fc_weights*
validate_shape(*
_output_shapes
:	?*
use_locking(*
T0
?
final_fc_weights/readIdentityfinal_fc_weights*
_output_shapes
:	?*#
_class
loc:@final_fc_weights*
T0
?
final_fc_bias/Initializer/zerosConst*
dtype0* 
_class
loc:@final_fc_bias*
_output_shapes
:*
valueB*    
?
final_fc_bias
VariableV2*
dtype0*
_output_shapes
:* 
_class
loc:@final_fc_bias*
shape:*
	container *
shared_name 
?
final_fc_bias/AssignAssignfinal_fc_biasfinal_fc_bias/Initializer/zeros* 
_class
loc:@final_fc_bias*
use_locking(*
validate_shape(*
_output_shapes
:*
T0
t
final_fc_bias/readIdentityfinal_fc_bias*
_output_shapes
:* 
_class
loc:@final_fc_bias*
T0
?
MatMulMatMul	Reshape_3final_fc_weights/read*
T0*
transpose_b( *
transpose_a( *
_output_shapes

:
S
add_1AddV2MatMulfinal_fc_bias/read*
T0*
_output_shapes

:
I
labels_softmaxSoftmaxadd_1*
T0*
_output_shapes

:
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
_output_shapes
: *
dtype0
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
_output_shapes
: *
dtype0
?
save/SaveV2/tensor_namesConst*O
valueFBDBfinal_fc_biasBfinal_fc_weightsB
first_biasBfirst_weights*
_output_shapes
:*
dtype0
k
save/SaveV2/shape_and_slicesConst*
valueBB B B B *
_output_shapes
:*
dtype0
?
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesfinal_fc_biasfinal_fc_weights
first_biasfirst_weights*
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_output_shapes
: *
T0*
_class
loc:@save/Const
?
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*O
valueFBDBfinal_fc_biasBfinal_fc_weightsB
first_biasBfirst_weights
}
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B 
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*$
_output_shapes
::::*
dtypes
2
?
save/AssignAssignfinal_fc_biassave/RestoreV2*
_output_shapes
:*
T0*
validate_shape(*
use_locking(* 
_class
loc:@final_fc_bias
?
save/Assign_1Assignfinal_fc_weightssave/RestoreV2:1*#
_class
loc:@final_fc_weights*
validate_shape(*
_output_shapes
:	?*
use_locking(*
T0
?
save/Assign_2Assign
first_biassave/RestoreV2:2*
T0*
validate_shape(*
_output_shapes
:*
use_locking(*
_class
loc:@first_bias
?
save/Assign_3Assignfirst_weightssave/RestoreV2:3*&
_output_shapes
:
*
T0* 
_class
loc:@first_weights*
validate_shape(*
use_locking(
V
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3
[
save_1/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
_output_shapes
: *
shape: *
dtype0
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
dtype0*
shape: *
_output_shapes
: 
?
save_1/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_0957d923ceac477786175bf5d5aebe9a/part
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
S
save_1/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
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
?
save_1/SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*O
valueFBDBfinal_fc_biasBfinal_fc_weightsB
first_biasBfirst_weights
|
save_1/SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B 
?
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesfinal_fc_biasfinal_fc_weights
first_biasfirst_weights"/device:CPU:0*
dtypes
2
?
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2"/device:CPU:0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: *
T0
?
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency"/device:CPU:0*
N*
_output_shapes
:*

axis *
T0
?
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const"/device:CPU:0*
delete_old_dirs(
?
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency"/device:CPU:0*
_output_shapes
: *
T0
?
save_1/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*O
valueFBDBfinal_fc_biasBfinal_fc_weightsB
first_biasBfirst_weights*
_output_shapes
:

!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B B B *
dtype0*
_output_shapes
:
?
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*$
_output_shapes
::::*
dtypes
2
?
save_1/AssignAssignfinal_fc_biassave_1/RestoreV2*
use_locking(*
_output_shapes
:*
validate_shape(*
T0* 
_class
loc:@final_fc_bias
?
save_1/Assign_1Assignfinal_fc_weightssave_1/RestoreV2:1*
use_locking(*
validate_shape(*
_output_shapes
:	?*
T0*#
_class
loc:@final_fc_weights
?
save_1/Assign_2Assign
first_biassave_1/RestoreV2:2*
_output_shapes
:*
_class
loc:@first_bias*
validate_shape(*
T0*
use_locking(
?
save_1/Assign_3Assignfirst_weightssave_1/RestoreV2:3* 
_class
loc:@first_weights*
T0*
use_locking(*
validate_shape(*&
_output_shapes
:

b
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_2^save_1/Assign_3
1
save_1/restore_allNoOp^save_1/restore_shard"?B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"?
trainable_variables??
m
first_weights:0first_weights/Assignfirst_weights/read:02,first_weights/Initializer/truncated_normal:08
V
first_bias:0first_bias/Assignfirst_bias/read:02first_bias/Initializer/zeros:08
y
final_fc_weights:0final_fc_weights/Assignfinal_fc_weights/read:02/final_fc_weights/Initializer/truncated_normal:08
b
final_fc_bias:0final_fc_bias/Assignfinal_fc_bias/read:02!final_fc_bias/Initializer/zeros:08"?
	variables??
m
first_weights:0first_weights/Assignfirst_weights/read:02,first_weights/Initializer/truncated_normal:08
V
first_bias:0first_bias/Assignfirst_bias/read:02first_bias/Initializer/zeros:08
y
final_fc_weights:0final_fc_weights/Assignfinal_fc_weights/read:02/final_fc_weights/Initializer/truncated_normal:08
b
final_fc_bias:0final_fc_bias/Assignfinal_fc_bias/read:02!final_fc_bias/Initializer/zeros:08*~
serving_defaultk
#
input
Reshape_1:0	?(
output
labels_softmax:0tensorflow/serving/predict