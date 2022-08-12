#!/bin/sh
#
# grid engine options
#$ -P lel_hcrc_cstr_students 
#$ -N fastpitch_prep_lj
#$ -l h_rt=24:00:00
#$ -l h_vmem=1G
#$ -pe sharedmem 8
#$ -R y
#$ -o /exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2258422_Xi_Wang/job_logs/$JOB_NAME_$JOB_ID.stdout
#$ -e /exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2258422_Xi_Wang/job_logs/$JOB_NAME_$JOB_ID.stderr
#$ -M s2258422@ed.ac.uk
#$ -m beas

# initialise environment modules
. /etc/profile.d/modules.sh

module load anaconda
source activate fastpitch

set -euo pipefail

UUN=s2258422
YOUR_NAME=Xi_Wang

DS_HOME=/exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/${UUN}_${YOUR_NAME}
FP=$DS_HOME/FastPitches/PyTorch/SpeechSynthesis/FastPitch

SCRATCH=/exports/eddie/scratch/s2258422
DATA_DIR="$SCRATCH/LJSpeech-1.1"
# DATA_DIR="$FP/test_folder" #-------------------------------------------------------------------------C

cd $FP
# for FILELIST in test_file.txt \ #-------------------------------------------------------C
# for FILELIST in ljs_audio_text_train_v3.txt \
#                 ljs_audio_text_val.txt \
#                 ljs_audio_text_test.txt \
# ; do
#     # have to set smaller --n-workers than $FP/scripts/prepare_dataset.sh
#     # to work around weird qsub memory consumption
#     python prepare_dataset.py \
#         --dataset-path $DATA_DIR \
#         --wav-text-filelist filelists/$FILELIST \
#         --n-workers 1 \
#         --batch-size 1 \
#         --extract-mels #---------------------------------------------------C
        # --extract-pitch \
        # --load-pitch-from-disk \ #-------------------------------------------C
        # --interpolate-f0 \ #----------------------------------------------------C
        # --mean-and-delta-f0 #----------------------------------------------------C
    # NB: this has to use `--batch-size 1` otherwise archives get saved with
    # padding and everything ends up the wrong shape!

python prepare_dataset.py \
    --dataset-path $DATA_DIR \
    --wav-text-filelist filelists/for_duration.txt \
    --save-alignment-priors \
    --n-workers 1 

done
