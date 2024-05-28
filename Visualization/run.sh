python main_synthetic.py --steps 20000 --sep 2000

python main_realworld.py --dataset cora --model GCN --seed 123 --steps 50000 --sep 2000 --hidden_dim 16 --num_layers 2 --lr 0.01 --weight_decay 5e-4 --momentum 0.9

python main_realworld.py --dataset texas --model GCN --seed 100 --steps 20000 --sep 800 --hidden_dim 16 --num_layers 2 --lr 0.0003 --weight_decay 5e-4 --momentum 0.9