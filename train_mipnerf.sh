python train.py -s /scratch/leuven/mip360/bonsai -m /scratch/leuven/log/bonsai_final  --num_channels 21 --plane_size 2600   --port 6557    --contractor  --densify_grad_threshold 0.00013 --bbox_scale 0.5 --no_downsample 
python train.py -s /scratch/leuven/mip360/counter -m /scratch/leuven/log/counter_final  --num_channels 15 --plane_size 2500   --port 6557    --contractor  --no_downsample --densify_grad_threshold 0.00015 --bbox_scale 0.3
python train.py -s /scratch/leuven/mip360/kitchen -m /scratch/leuven/log/kitchen_final  --num_channels 18 --plane_size 2800   --port 6557    --contractor  --no_downsample --densify_grad_threshold 0.00014 --bbox_scale 0.4
python train.py -s /scratch/leuven/mip360/room -m /scratch/leuven/log/room_final  --num_channels 15 --plane_size 2500   --port 6557    --contractor  
python train.py -s /scratch/leuven/mip360/stump -m /scratch/leuven/log/stump_final  --num_channels 15 --plane_size 2500   --port 6557    --contractor  
python train.py -s /scratch/leuven/mip360/bicycle -m /scratch/leuven/log/bicycle_final  --num_channels 15 --plane_size 2500   --port 6557    --contractor  
python train.py -s /scratch/leuven/mip360/garden -m /scratch/leuven/log/garden_final  --num_channels 15 --plane_size 2500   --port 6557    --contractor  
python train.py -s /scratch/leuven/mip360/treehill -m /scratch/leuven/log/treehill_final  --num_channels 15 --plane_size 2500   --port 6557    --contractor  --bbox_scale 0.3
python train.py -s /scratch/leuven/mip360/flowers -m /scratch/leuven/log/flowers_final  --num_channels 15 --plane_size 2500   --port 6557    --contractor  --bbox_scale 0.3
