#python -m ipdb -c continue data_processing/process_model_data.py --env jaxmaze  --df --models qlearning usfa preplay_new dfs bfs
#python -m ipdb -c continue data_processing/process_model_data.py --env jaxmaze  --df --episodes --models bfs dfs


python -m ipdb -c continue data_processing/process_model_data.py --env craftax  --df --models qlearning usfa preplay_new dfs bfs
python -m ipdb -c continue data_processing/process_model_data.py --env craftax  --df --episodes --models bfs dfs
