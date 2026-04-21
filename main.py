from dataProcess import dataProcess, dataProcessTAIEX
import time
if __name__ == "__main__":
    start_time = time.time()
    dataTAIEX = dataProcessTAIEX()
    datas = dataProcess(dataTAIEX)
    print(f"RS Rating 計算完成，耗時 {time.time() - start_time:.2f} 秒")
