#!/usr/bin/env bash

: ${WAVEGLOW:="pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt"}
: ${FASTPITCH:="pretrained_models/fastpitch/nvidia_fastpitch_210824.pt"}
: ${BATCH_SIZE:=32}
: ${PHRASES:="phrases/devset10.tsv"}
: ${OUTPUT_DIR:="./output/audio_$(basename ${PHRASES} .tsv)"}
: ${LOG_FILE:="$OUTPUT_DIR/nvlog_infer.json"}
: ${AMP:=false}
: ${TORCHSCRIPT:=false}
: ${PHONE:=true}
: ${ENERGY:=true}
: ${DENOISING:=0.01}
: ${WARMUP:=0}
: ${REPEATS:=1}
: ${CPU:=false}

: ${SPEAKER:=0}
: ${NUM_SPEAKERS:=1}

#---------added by me----------
: ${INTERPOLATE:=false}
: ${MEAN_DELTA:=false}
: ${NORMAL:=false}
# : ${NORMALISE:=false}
: ${SLOPE:=false}
: ${MEAN_F0_TGT:=None}
#------------------------------

echo -e "\nAMP=$AMP, batch_size=$BATCH_SIZE\n"

ARGS=""
ARGS+=" -i $PHRASES"
ARGS+=" -o $OUTPUT_DIR"
ARGS+=" --log-file $LOG_FILE"
ARGS+=" --fastpitch $FASTPITCH"
ARGS+=" --waveglow $WAVEGLOW"
ARGS+=" --wn-channels 256"
ARGS+=" --batch-size $BATCH_SIZE"
ARGS+=" --denoising-strength $DENOISING"
ARGS+=" --repeats $REPEATS"
ARGS+=" --warmup-steps $WARMUP"
ARGS+=" --speaker $SPEAKER"
ARGS+=" --n-speakers $NUM_SPEAKERS"
#--------------added---------------
ARGS+=" --mean_f0_tgt $MEAN_F0_TGT"
#----------------------------------
[ "$CPU" = false ]          && ARGS+=" --cuda"
[ "$CPU" = false ]          && ARGS+=" --cudnn-benchmark"
[ "$AMP" = true ]           && ARGS+=" --amp"
[ "$PHONE" = "true" ]       && ARGS+=" --p-arpabet 1.0"
[ "$ENERGY" = "true" ]      && ARGS+=" --energy-conditioning"
[ "$TORCHSCRIPT" = "true" ] && ARGS+=" --torchscript"
#-----------------------------added by me-------------------------
[ "$INTERPOLATE" = true ]   && ARGS+=" --interpolate-f0"
[ "$MEAN_DELTA" = true ]    && ARGS+=" --mean-and-delta-f0"
[ "$NORMAL" = true ]        && ARGS+=" --raw-f0"
[ "$SLOPE" = true ]         && ARGS+=" --slope-f0"
# [ "$NORMALISE" = true ]     && ARGS+=" --pitch-mean 214.72203 --pitch-std 65.72038"  #-----------------------Q:why it doesn't work?
#-----------------------------------------------------------------

mkdir -p "$OUTPUT_DIR"

python inference.py $ARGS "$@"
