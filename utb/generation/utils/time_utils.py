import random
import time
import os

from datetime import datetime


def get_random_date():
    a1 = (2024, 1, 1, 0, 0, 0, 0, 0, 0)                                   
    a2 = (2024, 12, 31, 23, 59, 59, 0, 0, 0)                                   

    start = int(time.mktime(a1))           
    end = int(time.mktime(a2))           

    t = random.randint(start, end)                    
    date_touple = time.localtime(t)              
    date = time.strftime("%Y-%m-%d %H:%M:%S", date_touple)                             
    date_obj = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    weekday_num = date_obj.weekday()
    language = os.getenv("LANGUAGE")
    if language == "zh":
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    else:
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = weekdays[weekday_num]
    date = date + " " + weekday
    return date


if __name__ == "__main__":
    date = get_random_date()
    print(date)
