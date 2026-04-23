from dataProcess import dataProcess, dataProcessTAIEX
import time
if __name__ == "__main__":
    start_time = time.time()
    dataTAIEX = dataProcessTAIEX()
    datas = dataProcess(dataTAIEX, 21)
    datas.to_pickle(f"cache/{time.strftime('%Y%m%d')}_TrainingDataset.pkl")
    print(f"RS Rating 計算完成，耗時 {time.time() - start_time:.2f} 秒")
