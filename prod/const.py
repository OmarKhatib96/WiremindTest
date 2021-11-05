#----------------------------------------------------------------------------
# Created By  : Omar KHATIB for Wiremind   Line 3
# Created Date: 05/11/2021 
# version ='1.0'
#This module contains all the useful constants for the model
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


training_path='./Data/train.lz4'
test_path='./Data/test.lz4'
test_set=[-89,-60,-30,-20,-15,-10,-7,-6,-5,-3,-2,-1]
learning_rate=0.0001
batchsize=64
nbr_epochs=560
log_file='hist.log'
#Dropout
DO=0.3#
output_eval='./eval.csv'