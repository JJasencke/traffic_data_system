import time
from collector import setup_logger, collect_once

INTERVAL_SECONDS = 60


def main():
    logger = setup_logger()
    logger.info("自动采集程序启动")
    print("自动采集程序启动")

    while True:
        start_time = time.time()

        try:
            print("开始执行一轮采集...")
            collect_once()
            print("本轮采集完成")
        except Exception as e:
            logger.exception(f"本轮采集执行失败: {e}")
            print(f"本轮采集执行失败: {e}")

        elapsed = time.time() - start_time
        sleep_time = max(0, INTERVAL_SECONDS - elapsed)

        logger.info(f"本轮总耗时 {elapsed:.2f} 秒，休眠 {sleep_time:.2f} 秒")
        print(f"本轮总耗时 {elapsed:.2f} 秒，休眠 {sleep_time:.2f} 秒")

        time.sleep(sleep_time)


if __name__ == "__main__":
    main()