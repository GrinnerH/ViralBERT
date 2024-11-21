import requests
import logging

# 设置日志级别为调试
logging.basicConfig(level=logging.DEBUG)

# 设置请求头
headers = {
    "Authorization": "Bearer YOUR_BEARER_TOKEN"
}

# 配置代理
proxies = {
    "http": "http://127.0.0.1:7897",  # 示例代理地址
    "https": "http://127.0.0.1:7897",  # 示例代理地址
}

# 发送请求并使用代理
response = requests.get("https://api.twitter.com/2/tweets/search/recent", headers=headers, proxies=proxies)

# 打印请求和响应的详细信息
print("Request Headers:")
for key, value in response.request.headers.items():
    print(f"{key}: {value}")

print("\nResponse Headers:")
for key, value in response.headers.items():
    print(f"{key}: {value}")

