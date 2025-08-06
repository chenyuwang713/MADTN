python main.py --Max_train_steps 600000 --device_num 10 --max_step 100 --max_channel_num 10
python main.py --Max_train_steps 600000 --device_num 20 --max_step 100 --max_channel_num 10 --write 0 
python main.py --Max_train_steps 600000 --device_num 20 --max_step 100 --max_channel_num 10





python main.py --env_mode edge --Max_train_steps 200000 --device_num 10 --max_step 100 --max_channel_num 10 --write 0 
--model_save 0


python main.py --env_mode edge --Max_train_steps 200000 --device_num 20 --max_step 100 --max_channel_num 10 --write 1 --model_save 0


v1 Changes
1. Change the evaluation process, stophotistic policy to a determinstic policy
2. Change the sample space from multi-binary to discrete.

v2 Changes
1. Test the discrete action space with pure optimization goal for edge_AoI
2. Change back the discrete action space to multi-binary aciton space, and increase the number of device for more dynamics.



# Key Choice
1. Sampling Action Space: multibinary or discrete 
