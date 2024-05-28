python main.py --dataset cora --algorithm LP --lr 10 --alpha 1.0 --k 3 --gamma 1 --steps 20000 --fixed_split 

python main.py --dataset citeseer --algorithm LP --lr 10 --alpha 1.0 --k 3 --gamma 20 --steps 1000 --fixed_split 

python main.py --dataset pubmed --algorithm LP --lr 100 --alpha 1.0 --k 4 --gamma 1 --steps 50000 --fixed_split 

python main.py --dataset computers --algorithm LP --lr 1 --alpha 1.0 --k 1 --gamma 0.25 --steps 100 --runs 1

python main.py --dataset photo --algorithm LP --lr 1 --alpha 1.0 --k 3 --gamma 0.1 --steps 500 --runs 1

python main.py --dataset cs --algorithm LP --lr 1 --alpha 1.0 --k 1 --gamma 20 --steps 5000 --runs 1

python main.py --dataset physics --algorithm LP --lr 1 --alpha 1.0 --k 1 --gamma 20 --steps 5000 --runs 1

python main.py --dataset roman-empire --algorithm LP --lr 1 --alpha 1.0 --k 0 --gamma 1 --steps 10000 --runs 10

python main.py --dataset amazon-ratings --algorithm LP --lr 2 --alpha 1.0 --k 0 --gamma 1 --steps 50000 --runs 10

python main.py --dataset minesweeper --algorithm LP --lr 1 --alpha 1 --k 1 --gamma 10 --steps 1000 --runs 10

python main.py --dataset tolokers --algorithm LP --lr 1 --alpha 0.5 --k 3 --gamma 10 --steps 1000 --runs 10

python main.py --dataset questions --algorithm LP --lr 5 --alpha 1.0 --k 2 --gamma 1 --steps 10000 --runs 10