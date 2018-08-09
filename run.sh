#!/bin/bash
#python3 ./pytorch-scripts/Example.py --cuda --datapath ./data/ --model Demo --aug H_FLIP
#python3 ./pytorch-scripts/Example.py --cuda --datapath ./data/ --model AlexNet --aug SCALE_H_FLIP
#python3 ./pytorch-scripts/Example.py --cuda --pretrained --datapath ./data/ --model AlexNet --aug SCALE_H_FLIP 
#python3 ./pytorch-scripts/Example.py --cuda --datapath ./data/ --model VGG16 --aug SCALE_H_FLIP 
#python3 ./pytorch-scripts/Example.py --cuda --pretrained --datapath ./data/ --model VGG16 --aug SCALE_H_FLIP 
#python3 ./pytorch-scripts/Example.py --cuda --datapath ./data/ --model ResNet18 --aug SCALE_H_FLIP 
#python3 ./pytorch-scripts/Example.py --cuda --pretrained --datapath ./data/ --model ResNet18 --aug SCALE_H_FLIP 
#python3 ./pytorch-scripts/Example.py --cuda --pretrained --datapath ./data/ --model DenseNet161 --aug SCALE_H_FLIP --batch_size 8 --#epochs 10
#python3 ./pytorch-scripts/Example.py --cuda --datapath ./data/ --model DenseNet161 --aug SCALE_H_FLIP --batch_size 8 --epochs 5



python3 ./pytorch-scripts/Example.py --cuda --datapath ./data/ --model ResNeXt --aug SCALE_H_FLIP --batch_size 32 --base_width 4 --cardinality 32 --optimizer Adam --no_preactivation --no_stochastic --no_personalized
python3 ./pytorch-scripts/Example.py --cuda --datapath ./data/ --model ResNeXt --aug SCALE_H_FLIP --batch_size 32 --base_width 4 --cardinality 32 --optimizer Adam --no_preactivation --no_stochastic
python3 ./pytorch-scripts/Example.py --cuda --datapath ./data/ --model ResNeXt --aug SCALE_H_FLIP --batch_size 32 --base_width 4 --cardinality 32 --optimizer Adam --no_preactivation --no_personalized
python3 ./pytorch-scripts/Example.py --cuda --datapath ./data/ --model ResNeXt --aug SCALE_H_FLIP --batch_size 32 --base_width 4 --cardinality 32 --optimizer Adam --no_stochastic --no_personalized


#	LR TESTS
-model ResNeXt -aug SCALE_CROP -batch_size 32 -base_width 2 -cardinality 32 -optimizer Adamax -lr 0.05 -epochs 10 --no_stochastic --no_personalized
-model ResNeXt -aug SCALE_CROP -batch_size 32 -base_width 2 -cardinality 32 -optimizer Adamax -lr 0.01 -epochs 10 --no_stochastic --no_personalized
-model ResNeXt -aug SCALE_CROP -batch_size 32 -base_width 2 -cardinality 32 -optimizer Adamax -lr 0.005 -epochs 10 --no_stochastic --no_personalized
-model ResNeXt -aug SCALE_CROP -batch_size 32 -base_width 2 -cardinality 32 -optimizer Adamax -lr 0.001 -epochs 10 --no_stochastic --no_personalized

#	LR_DECAY TESTS
-model ResNeXt -aug SCALE_CROP -batch_size 32 -base_width 2 -cardinality 32 -optimizer Adamax -lr_decay 0.5 -epochs 10 --no_stochastic --no_personalized
-model ResNeXt -aug SCALE_CROP -batch_size 32 -base_width 2 -cardinality 32 -optimizer Adamax -lr_decay 0.1 -epochs 10 --no_stochastic --no_personalized
-model ResNeXt -aug SCALE_CROP -batch_size 32 -base_width 2 -cardinality 32 -optimizer Adamax -lr_decay 0.2 -epochs 10 --no_stochastic --no_personalized

#	AUGMENTATION TESTS
-model ResNeXt -aug SCALE_CROP -batch_size 32 -base_width 2 -cardinality 32 -optimizer Adamax -epochs 10 --no_stochastic --no_personalized
-model ResNeXt -aug SCALE_JIT -batch_size 32 -base_width 2 -cardinality 32 -optimizer Adamax -epochs 10 --no_stochastic --no_personalized
-model ResNeXt -aug SCALE_DOUBLE_FLIP -batch_size 32 -base_width 2 -cardinality 32 -optimizer Adamax -epochs 10 --no_stochastic --no_personalized

#	BASE_WIDTH TESTS
-model ResNeXt -aug SCALE_CROP -batch_size 32 -base_width 1 -cardinality 32 -optimizer Adamax -epochs 10 --no_stochastic --no_personalized
-model ResNeXt -aug SCALE_CROP -batch_size 32 -base_width 2 -cardinality 32 -optimizer Adamax -epochs 10 --no_stochastic --no_personalized
-model ResNeXt -aug SCALE_CROP -batch_size 32 -base_width 3 -cardinality 32 -optimizer Adamax -epochs 10 --no_stochastic --no_personalized
-model ResNeXt -aug SCALE_CROP -batch_size 32 -base_width 4 -cardinality 32 -optimizer Adamax -epochs 10 --no_stochastic --no_personalized

#	CARDINALITY TESTS
-model ResNeXt -aug SCALE_CROP -batch_size 32 -base_width 2 -cardinality 8 -optimizer Adamax -epochs 10 --no_stochastic --no_personalized
-model ResNeXt -aug SCALE_CROP -batch_size 32 -base_width 2 -cardinality 16 -optimizer Adamax -epochs 10 --no_stochastic --no_personalized
-model ResNeXt -aug SCALE_CROP -batch_size 32 -base_width 2 -cardinality 32 -optimizer Adamax -epochs 10 --no_stochastic --no_personalized
-model ResNeXt -aug SCALE_CROP -batch_size 32 -base_width 2 -cardinality 64 -optimizer Adamax -epochs 10 --no_stochastic --no_personalized

#	LOSS TESTS
-model ResNeXt -aug SCALE_CROP -batch_size 32 -base_width 2 -cardinality 32 -optimizer Adamax -epochs 10 -loss_fun Cross -no_stochastic -no_personalized
-model ResNeXt -aug SCALE_CROP -batch_size 32 -base_width 2 -cardinality 32 -optimizer Adamax -epochs 10 -loss_fun Hinge -no_stochastic -no_personalized

#	ACTIVATION TESTS
-model ResNeXt -aug SCALE_CROP -batch_size 32 -base_width 2 -cardinality 32 -optimizer Adamax -epochs 10 -activ_fun relu -no_stochastic -no_personalized
-model ResNeXt -aug SCALE_CROP -batch_size 32 -base_width 2 -cardinality 32 -optimizer Adamax -epochs 10 -activ_fun relu6 -no_stochastic -no_personalized



# BEST
-model ResNeXt -aug SCALE_CROP -batch_size 32 -base_width 4 -cardinality 64 -optimizer Adam -epochs 25 -activ_fun relu6 -no_stochastic -no_personalized
