import requests
from pathlib import Path

from utils import *


class Checker:

    def __init__(
        self, 
        url, 
        app_id, 
        app_secret, 
        timeout = 5
    ):
        
        self.url = url
        self.headers = {
            "app_id" : app_id,
            "app_secret" : app_secret
        }
        self.timeout = timeout
        self.saved = {}
    
    @classmethod
    def from_file(
        self, 
        filename, 
        timeout=5
    ):
        p = Path(filename)
        if not p.is_file():
            raise FileNotFoundError(f"File {str(p.absolute())} not found.")
        p = str(p.absolute())
        with open(p, "r") as f:
            contents = f.readlines()[: 3]
        if len(contents) < 3:
            raise MissingInfoException(f"File {p} is not complete.")
        url, app_id, app_secret = [l.strip() for l in contents]
        return Checker(url, app_id, app_secret, timeout)

    def query(
        self, 
        code, 
        issue
    ):
        winning = self.saved.get((code, issue), None)
        if not winning:
            payload = {
                "code" : code,
                "expect" : issue
            }
            r = requests.get(self.url, params=payload, headers=self.headers, timeout=self.timeout)
            if  r.status_code != requests.codes.ok:
                raise requests.RequestException(f"网络请求出错，code: {r.status_code}，请检查网络连接。")
            js = r.json()
            if js["code"] != 1:
                raise requests.RequestException(f"没有查询到期号为{issue}的彩票开奖信息，请确认是否已开奖。")
            winning = js["data"]["openCode"]
            if not winning:
                raise MissingInfoException("接口返回数据中未解析到开奖号码。")
            winning = winning_process(winning, code)
            self.saved[(code, issue)] = winning
        return winning
    
    def __call__(
        self, 
        code, 
        issue, 
        numbers
    ):
        winning = self.query(code, issue)
        hits = hit_check(numbers, winning)
        return hits, winning

