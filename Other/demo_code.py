# 利用requests 模拟登陆
import requests
import handlePicture.calcPicture as calcPicture
import os
import time
import datetime
import random
from bs4 import BeautifulSoup

# 全局数据
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.96 Safari/537.36',
    'Referer': 'http://kq.neusoft.com'
}  # 请求头
post_data = {}  # 登录信息


def relay_short():
    time.sleep(1)


def relay_long():
    time.sleep(3)


def open_mnist_cnn():
    print('开启数字识别模型')
    calcPicture.open_mnist_cnn()


def close_mnist_cnn():
    print('关闭数字识别模型')
    calcPicture.close_mnist_cnn()


def transform_verification_image_to_code():
    print('将验证码图片转换为数字')
    # 转化验证码
    result = calcPicture.image2num_mnist_cnn(os.path.dirname(__file__) + '/Image/imageRandeCode.png')
    print('验证码识别为' + result)
    return result


def get_login_html(session):
    print('获取登录页面，开始解析html')
    resp = session.get('http://kq.neusoft.com', headers=headers)
    return BeautifulSoup(resp.text, 'html.parser')


def get_verification_image(session):
    print('获取验证码，并将验证码保存在/Image/imageRandeCode.png')
    resp = session.get('http://kq.neusoft.com/imageRandeCode')
    image = resp.content
    with open(os.path.dirname(__file__) + '/Image/imageRandeCode.png', 'wb+') as f:
        f.write(image)  # 写文件
    relay_short()


def update_post_data(soup, code_result):
    print('更新post请求参数')
    find_input = soup.find_all('input')
    key_time = find_input[2]['name']
    neusoft_key = find_input[3]['value']
    user_name = find_input[4]['name']
    password = find_input[5]['name']
    code = find_input[6]['name']
    post_data['login'] = 'true'
    post_data['neusoft_attendance_online'] = ''
    post_data[key_time] = ''
    post_data['neusoft_key'] = neusoft_key
    post_data[user_name] = 'rongxn'
    post_data[password] = 'RONGnx0429'
    post_data[code] = code_result


def check_login_is_success(response):
    if response == '':
        return False
    print('校验模拟登录是否成功')
    param1 = str(response.status_code)
    param2 = str(response.url)
    if param1 == '200' and param2 == 'http://kq.neusoft.com/attendance.jsp':
        print('模拟登录成功')
        print(response.text)
        return True
    print('模拟登录失败')
    relay_long()
    return False


def login():
    # 通过requests获取session
    session = requests.Session()
    # 发送数据，获取登录页面，解析页面数据
    soup = get_login_html(session)
    # 登录
    response = ''  # 登录相应
    connection_count = 0
    open_mnist_cnn()  # 开启验证码识别
    while not check_login_is_success(response):
        connection_count = connection_count + 1
        print('尝试登录：' + str(connection_count))
        # 获取验证码
        get_verification_image(session)
        # 解析验证码
        code_result = transform_verification_image_to_code()
        # code_result = input('请验证，验证码解析是否正确，并输入正确验证码：')
        # 更新post_data
        update_post_data(soup, code_result)
        # 验证登录数据
        print(post_data)
        print(headers)
        response = session.post('http://kq.neusoft.com/login.jsp', post_data, headers)
    close_mnist_cnn()  # 关闭验证码识别
    # 打卡
    print('准备打卡')
    # session.post()
    # session.get('http://kq.neusoft.com/record.jsp', headers=headers)


def timer(hour, minute, second):
    while True:
        now = datetime.datetime.now()
        h = now.hour
        m = now.minute
        s = now.second
        if hour == h and minute == m and second == s:
            print('打卡开始：' + str(now))
            login()
            print('打卡结束')
            time.sleep(120)


if __name__ == '__main__':
    # timer(18, random.randint(0, 20), random.randint(0, 59))
    login()
