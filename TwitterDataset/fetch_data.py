from TwitterDataset import TwitterDataset
import time
from datetime import timedelta

if __name__ == '__main__':
    # 设置代理
    proxies = {
        "http": "http://127.0.0.1:7897",
        "https": "http://127.0.0.1:7897",
    }

    interval = timedelta(minutes=60)
    # change range to amount of days of dataset to fetch
    for dataset_no in range(1,10):
        print(f"Fetching dataset {dataset_no}")
        # twitterDataset = TwitterDataset(timedelt=interval, threads=2, dataset_dir=f"./dataset{dataset_no}")
        twitterDataset = TwitterDataset(timedelt=interval, threads=2, dataset_dir=f"./dataset{dataset_no}", proxies=proxies)
        topics = twitterDataset.load_topics()
        # twitterDataset.set_threads(len(topics))
        twitterDataset.init_api_handlers()
        twitterDataset.get_tweets()
        for i in range(timedelta(days=1)//interval):
            print(f"Updating dataset {dataset_no}, update no: {i}")
            twitterDataset.update_tweets()