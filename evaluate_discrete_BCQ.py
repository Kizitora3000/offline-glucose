import subprocess
import csv
from concurrent.futures import ThreadPoolExecutor

def run_script(id):
    result = subprocess.run(['python', 'train_discrete_BCQ.py', str(id)], capture_output=True, text=True)
    return id, result.stdout.strip()

with open('ids.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ID', 'RL_Reward'])

    # ThreadPoolExecutorを使用して並列実行
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run_script, j * 10000) for j in range(1, 11)]
        for future in futures:
            id_result, rl_reward = future.result()
            print(id_result, rl_reward)
            writer.writerow([id_result, rl_reward])
