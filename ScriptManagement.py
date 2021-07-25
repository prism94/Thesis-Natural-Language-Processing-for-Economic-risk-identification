import multiprocessing
from RSS_news_scrapper_Headlines_def import fetch_data
import pickle
import datetime
import time

with open('asset_list.pkl', 'rb') as f:
    asset_list = pickle.load(f)

process_number = 4

finished_list = []

finish_date = datetime.datetime(2018, 1, 1)

#fetch_data(asset_list[0], finish_date)

proc = {}

if __name__ == '__main__':
    open_processes = 0
    while True:
        if open_processes < process_number:
            if len(asset_list) > 0:
                asset = asset_list[0]
                print(f'Init {asset}')
                del asset_list[0]
                proc[asset] = multiprocessing.Process(target = fetch_data, args=[asset, finish_date])
                proc[asset].start()
                open_processes += 1
        else:
            del_list = []
            for ass in proc.keys():
                if proc[ass].is_alive() == False:
                    open_processes -= 1
                    proc[ass].terminate()
                    del_list.append(ass)
                    
                    print(ass, 'is done')
                    print(f'asset_list has {len(asset_list)} left')
            for _del_ in del_list:
                finished_list.append(_del_)
                with open(f'Finished_List.pkl', 'wb') as f:
                    pickle.dump(finished_list, f)
                del proc[_del_]
        
        time.sleep(30)
    

