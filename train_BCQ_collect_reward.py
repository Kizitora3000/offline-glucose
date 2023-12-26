import subprocess
import csv
from concurrent.futures import ThreadPoolExecutor

def run_script(id):
    result = subprocess.run(['python', 'train_BCQ_of_diabetes.py', str(id)], capture_output=True, text=True)
    return id, result.stdout.strip()

with open('ids.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ID', 'RL_Reward'])

    # リソースの都合で同時に10回までしか実行できないので、10回×10回で100回分計算
    for i in range(2, 10):
        id = 100000 * i
        # ThreadPoolExecutorを使用して並列実行
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(run_script, id + j * 10000) for j in range(1, 11)]
            for future in futures:
                id_result, rl_reward = future.result()
                print(id_result, rl_reward)
                writer.writerow([id_result, rl_reward])
