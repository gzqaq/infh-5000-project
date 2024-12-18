import cv2
import random


def face_detect(img, cname):
    # 脸部检测代码
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_hist = cv2.equalizeHist(img_gray)
    face_cascade = cv2.CascadeClassifier(cname)
    faces = face_cascade.detectMultiScale(img_hist)
    return faces

def christmas_hat(fname, mode):
    img = cv2.imread(fname)
    cnames = ["data/haarcascade_frontalface_alt.xml",
              "data/haarcascade_frontalcatface.xml",
              "data/lbpcascade_animeface.xml"]
    cname = cnames[mode]

    # 脸部检测
    faces = face_detect(img, cname)
    # 读取圣诞帽
    hats = [cv2.imread(f'img/hats/hat_{i+1}.png', -1) for i in range(3)]
    for face in faces:
        hat = random.choice(hats) # 随机选择一个帽子
        scale = face[3] / hat.shape[0] * 2  # 设置缩放因子
        hat = cv2.resize(hat, (0, 0), fx=scale, fy=scale) # 调整帽子大小
        x_offset = int(face[0] + face[2] / 2 - hat.shape[1] / 2) # 计算帽子的x偏移
        y_offset = int(face[1] - hat.shape[0] / 2)  # 计算帽子的y偏移
        # 计算贴图位置，注意防止超出边界的情况
        x1 = max(x_offset, 0)
        x2 = min(x_offset + hat.shape[1], img.shape[1])
        y1 = max(y_offset, 0)
        y2 = min(y_offset + hat.shape[0], img.shape[0])
        hat_x1 = max(0, -x_offset)
        hat_x2 = hat_x1 + x2 - x1
        hat_y1 = max(0, -y_offset)
        hat_y2 = hat_y1 + y2 - y1
        # 透明部分的处理
        alpha_h = hat[hat_y1:hat_y2, hat_x1:hat_x2, 3] / 255
        alpha = 1 - alpha_h
        # 按3个通道合并图片
        for c in range(3):
            img[y1:y2, x1:x2, c] = alpha_h * hat[hat_y1:hat_y2, hat_x1:hat_x2, c] + alpha * img[y1:y2, x1:x2, c]

    # 保存最终结果
    cv2.imwrite(f'img/result/{fname.split("/")[-1]}', img)
    return f'img/result/{fname.split("/")[-1]}'


# 使用例
if __name__ == "__main__":
    result = christmas_hat('img/image.png', 0) # 这里输入要添加圣诞帽的图像路径，模式0人 1猫 2动漫
    print(result)
