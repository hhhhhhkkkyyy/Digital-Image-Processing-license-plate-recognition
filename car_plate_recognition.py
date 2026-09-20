import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import time


def order_points(pts):
    """对四个点进行排序：左上、右上、右下、左下"""
    # 初始化坐标点
    rect = np.zeros((4, 2), dtype="float32")

    # 左上角点有最小的x+y和，右下角点有最大的x+y和
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 右上角点有最小的x-y差，左下角点有最大的x-y差
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    """透视变换，获取校正后的车牌图像"""
    # 对点进行排序
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算新图像的宽度
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    # 计算新图像的高度
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # 定义目标点
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 计算变换矩阵并应用
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


class ImageProcessor:
    @staticmethod
    def locate_plate(img):
        """车牌定位"""
        # 1. 颜色空间转换
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 蓝色车牌范围
        lower_blue = np.array([90, 70, 60])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # 2. 形态学操作优化（增强连接性）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))

        # 强化闭操作（连接蓝色区域）
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)

        # 强化开操作（去除小噪点）
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=3)

        # 3. 边缘增强（优化Canny参数）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 60, 150)

        # 结合颜色和边缘特征
        combined = cv2.bitwise_and(opened, edges)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 查找轮廓
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_plate = None
        best_score = 0

        img_area = img.shape[0] * img.shape[1]
        min_area = img_area * 0.005  # 最小面积为图像0.5%
        max_area = img_area * 0.25  # 最大面积为图像15%
        for contour in contours:
            # 筛选符合条件的轮廓
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            # 获取最小外接矩形
            rect = cv2.minAreaRect(contour)
            (cx, cy), (w, h), angle = rect

            # 确保宽度大于高度
            if w < h:
                w, h = h, w
                angle += 90

            aspect_ratio = w / (h + 1e-5)  # 避免除以零

            # 关键改进：优先宽高比（中国车牌标准3.14：1）
            aspect_diff = abs(aspect_ratio - 3.14)
            if 2.0 <= aspect_ratio <= 6.0:  # 更宽松的范围
                # 改进评分机制：宽高比匹配度>面积

                score = (1 / (aspect_diff + 0.1)) * 100 + area

                if score > best_score:
                    best_score = score
                    best_plate = (contour, rect)

        marked_img = img.copy()

        if best_plate is not None:
            contour, rect = best_plate
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype=np.float32)

            # 绘制轮廓和顶点
            cv2.drawContours(marked_img, [box.astype(np.int32)], -1, (0, 0, 255), 3)

            try:
                # 透视变换（改进稳定性）
                if max(box[:, 0]) - min(box[:, 0]) > 20 and max(box[:, 1]) - min(box[:, 1]) > 20:
                    warped = four_point_transform(img, box)
                    return marked_img, box, warped
                else:
                    # 如果点集太近，改用边界框
                    x, y, w, h = cv2.boundingRect(contour)
                    warped = img[y:y + h, x:x + w]
                    return marked_img, box, warped
            except:
                # 变换失败时使用边界框
                x, y, w, h = cv2.boundingRect(contour)
                warped = img[y:y + h, x:x + w]
                return marked_img, box, warped

        # 未找到车牌的处理
        print("未找到合适的车牌轮廓")
        return img, None, None

    @staticmethod
    def segment_plate_chars(binary_plate):
        """等宽切割法，保证7个字符不偏位"""
        h, w = binary_plate.shape
        char_num = 7

        # 1. 先对二值图像做竖直投影，找出左右边界
        vertical_sum = np.sum(binary_plate, axis=0)
        # 查找第一个和最后一个非零投影点
        cols = np.where(vertical_sum > 0)[0]
        if len(cols) == 0:
            print("Warning: Vertical projection resulted in all zeros.")
            return []  # 返回空列表或处理错误

        left = cols[0]
        right = cols[-1]

        plate_width = right - left + 1

        if plate_width <= 0:
            print("Warning: Plate width is zero or negative after projection.")
            return []

        # 2. 等宽分割
        char_width = plate_width // char_num
        char_images = []
        for i in range(char_num):
            x1 = left + i * char_width
            # 确保最后一个字符包含右边界
            x2 = left + (i + 1) * char_width if i < char_num - 1 else right + 1
            # 防止切片越界
            x1 = max(0, x1)
            x2 = min(w, x2)

            if x1 >= x2:  # 避免无效切片
                print(f"Warning: Invalid slice for character {i}: x1={x1}, x2={x2}")
                continue  # 跳过当前字符

            char_img = binary_plate[:, x1:x2]
            char_images.append(char_img)

        # 确保返回7个字符，如果不够则填充空白图像
        while len(char_images) < char_num:
            # 创建一个空白的字符图像 (例如，与标准化后的图像大小一致)
            target_height = 60
            target_width = int(target_height * 0.6)
            blank_char = np.zeros((target_height, target_width), dtype=np.uint8)
            char_images.append(blank_char)

        return char_images

    @staticmethod
    def normalize_char_image(char_img, target_size=None):
        """标准化字符图像"""
        if len(char_img.shape) == 3:
            char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)

        # 1. 计算需要添加的边框
        target_ratio = 0.6  # 宽高比 (可以根据实际调整)
        target_height = 60  # 目标高度 (可以根据实际调整)
        target_width = int(target_height * target_ratio)

        h, w = char_img.shape

        if h <= 0 or w <= 0:
            print("Warning: Input character image has zero or negative dimension.")
            # 返回一个空白图像
            return np.zeros((target_height, target_width), dtype=np.uint8)

        # 2. 保持原始宽高比调整大小，确保最大边适应目标大小
        scale = min(target_width / w, target_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 使用插值方法调整大小 (INTER_AREA 用于缩小，INTER_LINEAR 或 INTER_CUBIC 用于放大)
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(char_img, (new_w, new_h), interpolation=interpolation)

        # 3. 创建目标图像并居中放置字符
        result = np.zeros((target_height, target_width), dtype=np.uint8)  # 黑色背景
        x_offset = (target_width - new_w) // 2
        y_offset = (target_height - new_h) // 2

        # 确保放置区域在目标图像范围内
        result[max(0, y_offset):min(target_height, y_offset + new_h),
        max(0, x_offset):min(target_width, x_offset + new_w)] = resized

        return result

    @staticmethod
    def load_templates(template_dir):
        """安全加载模板图像并进行标准化处理"""
        templates = {}
        # 支持的字符类型
        char_types = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                      'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M',
                      'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                      '京', '津', '冀', '晋', '蒙', '辽', '吉', '黑', '沪', '苏',
                      '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤', '桂',
                      '琼', '渝', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新']

        # 确保模板目录存在
        if not os.path.exists(template_dir):
            print(f"错误: 模板目录 {template_dir} 不存在!")
            messagebox.showerror('错误', f'模板目录 {template_dir} 不存在!')
            return templates

        loaded_count = 0
        for char in char_types:
            char_dir = os.path.join(template_dir, char)
            if not os.path.isdir(char_dir):
                print(f"警告: 未找到模板目录 {char}")
                continue

            template_files = [f for f in os.listdir(char_dir)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if not template_files:
                print(f"警告: 模板目录 {char} 中无图像文件")
                continue

            template_path = os.path.join(char_dir, template_files[0])

            try:
                with open(template_path, 'rb') as file:
                    template_data = file.read()
                template = cv2.imdecode(np.frombuffer(template_data, np.uint8), cv2.IMREAD_GRAYSCALE)
                if template is None:
                    print(f"警告: 无法加载模板文件 {template_path} (可能损坏)")
                    continue

                template = cv2.resize(template, (20, 40))
                _, template = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                templates[char] = {
                    'original': template,
                    'inverted': cv2.bitwise_not(template)
                }
                loaded_count += 1
                print(f"已加载模板: {char}")

            except Exception as e:
                print(f"加载模板 {char} 时出错: {str(e)}")
                continue

        if loaded_count == 0:
            print("错误: 未能加载任何模板文件！")
            messagebox.showwarning('警告', '未能加载任何模板文件！请检查模板目录。')
        else:
            print(f"成功加载 {loaded_count}/{len(char_types)} 个模板文件")

        return templates

    @staticmethod
    def match_template(char_img, templates):
        """使用改进的模板匹配识别单个字符"""
        if not templates:
            print("错误: 没有可用的模板!")
            return None

        # 输入图像预处理
        if len(char_img.shape) == 3:
            char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)

        # 对比度调整为100%
        char_img = cv2.equalizeHist(char_img)

        # 标准化尺寸
        char_img = cv2.resize(char_img, (20, 40))

        # 多种预处理方式
        _, char_bin = cv2.threshold(char_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        char_inv = cv2.bitwise_not(char_bin)

        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        char_erode = cv2.erode(char_bin, kernel)
        char_dilate = cv2.dilate(char_bin, kernel)

        # 开运算（先腐蚀后膨胀）
        char_open = cv2.morphologyEx(char_bin, cv2.MORPH_OPEN, kernel, iterations=1)

        # 所有预处理变体
        char_variants = {
            'binary': char_bin,
            'inverted': char_inv,
            'eroded': char_erode,
            'dilated': char_dilate,
            'opened': char_open
        }

        best_match = None
        best_score = -1

        # 对每个模板进行匹配
        for char, template_data in templates.items():
            template = template_data['original']
            template_inv = template_data['inverted']

            # 尝试所有预处理组合
            for c_name, c_img in char_variants.items():
                # 匹配原始模板
                score1 = cv2.matchTemplate(c_img, template, cv2.TM_CCOEFF_NORMED)[0][0]
                # 匹配反转模板
                score2 = cv2.matchTemplate(c_img, template_inv, cv2.TM_CCOEFF_NORMED)[0][0]

                # 取最高分
                current_score = max(score1, score2)

                # 更新最佳匹配
                if current_score > best_score:
                    best_score = current_score
                    best_match = char

        print(f'字符匹配分数: {best_score:.3f}, 匹配字符: {best_match}')

        if best_score > 0.18:  # 阈值可再降
            return best_match
        return None

    @staticmethod
    def preprocess_plate_image(plate_img):
        """车牌图像预处理"""
        # 转为灰度图
        gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        # 二值化
        _, binary_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary_plate

    @staticmethod
    def extract_and_process_plate(img):
        """提取并处理车牌图像"""
        # 定位车牌
        marked_img, box, plate_img = ImageProcessor.locate_plate(img)

        if plate_img is None:
            return None, None

        # 预处理车牌
        processed_plate = plate_img.copy()
        if len(processed_plate.shape) == 2:
            processed_plate = cv2.cvtColor(processed_plate, cv2.COLOR_GRAY2BGR)

        # 添加边框和标题
        border_color = (0, 255, 0)
        border_thickness = 2
        processed_plate = cv2.copyMakeBorder(processed_plate,
                                             border_thickness, border_thickness,
                                             border_thickness, border_thickness,
                                             cv2.BORDER_CONSTANT, value=border_color)

        # 添加文字
        title = "Extracted Plate"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_color = border_color
        font_thickness = 2
        text_size, _ = cv2.getTextSize(title, font, font_scale, font_thickness)
        text_width = text_size[0]

        # 创建标题背景
        title_background = np.zeros((text_size[1] + 20, processed_plate.shape[1], 3), dtype=np.uint8)
        title_background[:, :] = (50, 50, 50)  # 深灰色背景

        # 在背景上添加标题
        text_x = (title_background.shape[1] - text_width) // 2
        text_y = (title_background.shape[0] + text_size[1]) // 2
        cv2.putText(title_background, title, (text_x, text_y),
                    font, font_scale, font_color, font_thickness)

        # 合并标题和车牌图像
        combined = np.vstack((title_background, processed_plate))

        return plate_img, combined


class Application:
    def __init__(self, root):
        self.root = root
        self.root.title("车牌识别")
        self.root.geometry("1300x850")  # 增加窗口大小以容纳三个图像区域

        # 初始化变量
        self.original_label = None
        self.processed_label = None
        self.plate_label = None
        self.original_image = None
        self.processed_image = None
        self.plate_image = None
        self.char_images = []
        self.char_labels = []
        self.templates = None
        self.recognition_result = ""
        self.status_var = tk.StringVar()  # 状态提示变量

        # 创建UI
        self.create_widgets()

        # 加载模板
        self.load_templates()
        self.status_var.set("已加载车牌识别系统")  # 设置初始状态

    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 状态栏
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=5)

        # 左侧控制面板
        control_panel = ttk.LabelFrame(main_frame, text="图像处理控制", width=250)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # 添加控制按钮前的分隔线
        ttk.Separator(control_panel, orient='horizontal').pack(fill='x', pady=5)

        # 图像加载按钮
        load_frame = ttk.Frame(control_panel)
        load_frame.pack(fill=tk.X, padx=5, pady=5)
        self.load_button = ttk.Button(load_frame, text="加载图像", command=self.load_image)
        self.load_button.pack(fill=tk.X, side=tk.LEFT, expand=True)

        # 提取按钮
        extract_frame = ttk.Frame(control_panel)
        extract_frame.pack(fill=tk.X, padx=5, pady=5)
        self.extract_button = ttk.Button(extract_frame, text="车牌提取",
                                         command=self.extract_plate,
                                         state=tk.DISABLED)
        self.extract_button.pack(fill=tk.X, side=tk.LEFT, expand=True)

        # 车牌定位
        locate_frame = ttk.Frame(control_panel)
        locate_frame.pack(fill=tk.X, padx=5, pady=5)
        self.locate_button = ttk.Button(locate_frame, text="车牌定位",
                                        command=self.locate_plate,
                                        state=tk.DISABLED)
        self.locate_button.pack(fill=tk.X, side=tk.LEFT, expand=True)

        # 字符分割
        segment_frame = ttk.Frame(control_panel)
        segment_frame.pack(fill=tk.X, padx=5, pady=5)
        self.segment_button = ttk.Button(segment_frame, text="字符分割",
                                         command=self.segment_chars,
                                         state=tk.DISABLED)
        self.segment_button.pack(fill=tk.X, side=tk.LEFT, expand=True)

        # 字符识别
        recognize_frame = ttk.Frame(control_panel)
        recognize_frame.pack(fill=tk.X, padx=5, pady=5)
        self.recognize_button = ttk.Button(recognize_frame, text="字符识别",
                                           command=self.recognize_chars,
                                           state=tk.DISABLED)
        self.recognize_button.pack(fill=tk.X, side=tk.LEFT, expand=True)

        # 添加控制按钮后的分隔线
        ttk.Separator(control_panel, orient='horizontal').pack(fill='x', pady=5)

        # 识别结果区域
        self.result_frame = ttk.LabelFrame(control_panel, text="识别结果")
        self.result_frame.pack(fill=tk.X, padx=5, pady=5)

        self.result_var = tk.StringVar()
        self.result_var.set("等待识别...")
        self.result_label = ttk.Label(self.result_frame, textvariable=self.result_var,
                                      font=('Arial', 16, 'bold'), foreground='blue',
                                      anchor=tk.CENTER)
        self.result_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # 右侧图像显示区域
        image_display = ttk.Frame(main_frame)
        image_display.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 使用三列布局
        col1_frame = ttk.Frame(image_display)
        col1_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        col2_frame = ttk.Frame(image_display)
        col2_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        col3_frame = ttk.Frame(image_display)
        col3_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 原始图像（第一列）
        self.original_group = ttk.LabelFrame(col1_frame, text="原始图像")
        self.original_group.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.original_label = ttk.Label(self.original_group, text="请加载图像", anchor="center")
        self.original_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 处理后图像（第二列）
        self.processed_group = ttk.LabelFrame(col2_frame, text="处理后图像")
        self.processed_group.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.processed_label = ttk.Label(self.processed_group, text="处理后图像将显示在这里", anchor="center")
        self.processed_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 车牌图像（第三列）
        self.plate_group = ttk.LabelFrame(col3_frame, text="车牌图像")
        self.plate_group.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.plate_label = ttk.Label(self.plate_group, text="车牌将显示在这里", anchor="center")
        self.plate_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def load_templates(self):
        """加载模板"""
        # 假设模板在当前目录的templates文件夹下
        template_dir = "templates"
        if not os.path.exists(template_dir):
            messagebox.showwarning("警告", f"未找到模板目录: {template_dir}")
            self.status_var.set(f"警告: 未找到模板目录 {template_dir}")
            return

        self.status_var.set("正在加载模板...")
        self.root.update()
        self.templates = ImageProcessor.load_templates(template_dir)
        self.status_var.set("模板加载完成")

    def load_image(self):
        """加载图像"""
        file_path = filedialog.askopenfilename(
            title="打开图像",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.tif")])

        if file_path:
            self.original_image = cv2.imread(file_path)
            if self.original_image is not None:
                self.status_var.set(f"已加载图像: {os.path.basename(file_path)}")

                # 启用所有按钮
                self.extract_button['state'] = tk.NORMAL
                self.locate_button['state'] = tk.NORMAL
                self.segment_button['state'] = tk.NORMAL
                self.recognize_button['state'] = tk.NORMAL

                # 显示原始图像
                self.display_image(self.original_image, self.original_label)

                # 重置其他显示
                self.display_placeholder(self.processed_label, "处理后图像将显示在这里")
                self.display_placeholder(self.plate_label, "车牌将显示在这里")

                for label in self.char_labels:
                    label.config(image='')

                # 重置处理结果
                self.processed_image = None
                self.plate_image = None
                self.char_images = []
                self.recognition_result = ""
                self.result_var.set("等待识别...")
            else:
                messagebox.showerror("错误", "无法加载图像文件！")
                self.status_var.set("错误: 无法加载图像文件")

    def display_placeholder(self, label, text):
        """显示占位文本"""
        # 创建一个简单的占位图像
        width = 300
        height = 200

        # 创建白色背景
        placeholder = np.ones((height, width, 3), dtype=np.uint8) * 255

        # 添加文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_color = (100, 100, 100)  # 灰色
        font_thickness = 2
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = int((width - text_size[0]) / 2)
        text_y = int((height + text_size[1]) / 2)
        cv2.putText(placeholder, text, (text_x, text_y), font,
                    font_scale, font_color, font_thickness)

        # 显示
        self.display_image(placeholder, label)

    def display_image(self, image, label, max_size=400):
        """显示OpenCV图像到Tkinter标签"""
        if image is None:
            return

        # 调整图像大小以适应标签
        height, width = image.shape[:2]
        ratio = min(max_size / width, max_size / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)

        # 转换颜色空间
        if len(image.shape) == 2:  # 灰度图
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:  # 彩色图
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 调整大小
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # 转换为PIL图像
        pil_image = Image.fromarray(image)

        # 转换为Tkinter PhotoImage
        photo = ImageTk.PhotoImage(pil_image)

        # 更新标签
        label.config(image=photo)
        label.image = photo  # 保持引用

    def locate_plate(self):
        """车牌定位"""
        if self.original_image is None:
            messagebox.showerror("警告", "请先加载图像！")
            self.status_var.set("警告: 请先加载图像")
            return

        self.status_var.set("正在进行车牌定位...")
        self.root.update()

        # 定位车牌
        marked_img, box, plate_img = ImageProcessor.locate_plate(self.original_image)

        if box is not None:
            self.processed_image = marked_img
            self.plate_image = plate_img
            self.display_image(marked_img, self.processed_label)

            # 自动显示车牌
            self.display_plate()
            self.status_var.set("车牌定位成功")
        else:
            messagebox.showinfo("提示", "未检测到车牌")
            self.status_var.set("未检测到车牌")

    def extract_plate(self):
        """车牌提取"""
        if self.original_image is None:
            messagebox.showerror("警告", "请先加载图像！")
            self.status_var.set("警告: 请先加载图像")
            return

        self.status_var.set("正在提取车牌...")
        self.root.update()

        # 提取并处理车牌图像
        plate_img, processed_plate = ImageProcessor.extract_and_process_plate(self.original_image)

        if plate_img is not None:
            self.plate_image = plate_img
            self.display_image(processed_plate, self.plate_label)
            self.status_var.set("车牌提取成功")
        else:
            messagebox.showinfo("提示", "无法提取车牌")
            self.status_var.set("无法提取车牌")

    def display_plate(self):
        """显示提取的车牌图像"""
        if self.plate_image is not None:
            plate_img, processed_plate = ImageProcessor.extract_and_process_plate(self.original_image)
            self.display_image(processed_plate, self.plate_label)

    def segment_chars(self):
        """字符分割 - 显示分割后的字符图像"""
        if self.plate_image is None:
            messagebox.showerror("警告", "请先进行车牌定位！")
            self.status_var.set("警告: 请先进行车牌定位")
            return

        self.status_var.set("正在进行字符分割...")
        self.root.update()

        # 预处理车牌图像
        binary_plate = ImageProcessor.preprocess_plate_image(self.plate_image)

        # 分割字符
        self.char_images = ImageProcessor.segment_plate_chars(binary_plate)

        if len(self.char_images) > 0:
            # 创建分割字符的合成图像
            char_display = self._create_char_display_image(self.char_images)

            # 显示分割结果
            self.display_image(char_display, self.processed_label)
            self.status_var.set(f"成功分割出 {len(self.char_images)} 个字符")
        else:
            self.status_var.set("字符分割失败")
            messagebox.showinfo("提示", "未分割出字符")

    def recognize_chars(self):
        """字符识别 - 仅输出最终识别结果"""
        # 输入验证
        if not self.char_images:
            messagebox.showerror("警告", "请先进行字符分割！")
            self.status_var.set("警告: 请先进行字符分割")
            return

        if not self.templates:
            messagebox.showerror("警告", "未加载模板文件！")
            self.status_var.set("警告: 未加载模板文件")
            return

        self.status_var.set("正在进行字符识别...")
        self.root.update()

        # 初始化识别结果
        plate_text = ""

        # 逐个字符进行识别
        for i, char_img in enumerate(self.char_images):
            # 标准化字符图像
            normalized_char = ImageProcessor.normalize_char_image(char_img)

            # 模板匹配
            matched_char = ImageProcessor.match_template(normalized_char, self.templates)

            # 可视化识别过程
            char_display = self._create_char_display_with_index(i, matched_char)
            self.display_image(char_display, self.processed_label)
            self.root.update()

            # 添加短暂延迟以便观察
            time.sleep(0.3)

            # 拼接识别结果
            plate_text += matched_char if matched_char else "?"

            self.status_var.set(f"识别中: {plate_text}")

        # 保存并显示最终结果
        self.recognition_result = plate_text
        self.result_var.set(f"识别结果: {plate_text}")
        self.status_var.set(f"识别完成: {plate_text}")

    def _create_char_display_image(self, char_images):
        """创建分割字符的合成图像"""
        char_height = 60
        char_width = 40
        margin = 5

        # 创建空白画布
        canvas_height = char_height + 2 * margin
        canvas_width = len(char_images) * (char_width + margin) + margin
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        canvas.fill(255)  # 白色背景

        # 将每个字符放置在画布上
        for i, char_img in enumerate(char_images):
            normalized_char = ImageProcessor.normalize_char_image(char_img, (char_height, char_width))

            if len(normalized_char.shape) == 2:
                normalized_char = cv2.cvtColor(normalized_char, cv2.COLOR_GRAY2BGR)

            x = margin + i * (char_width + margin)
            y = margin

            h, w = normalized_char.shape[:2]
            canvas[y:y + h, x:x + w] = normalized_char

            # 添加字符编号
            cv2.putText(canvas, str(i + 1), (x + 5, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 添加标题
        title = "分割字符"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_color = (0, 0, 255)
        font_thickness = 2
        text_size, _ = cv2.getTextSize(title, font, font_scale, font_thickness)
        text_x = (canvas_width - text_size[0]) // 2
        text_y = text_size[1] + 5

        # 扩展画布添加标题
        title_height = 30
        final_canvas = np.zeros((canvas_height + title_height, canvas_width, 3), dtype=np.uint8)
        final_canvas.fill(50)  # 深灰色背景

        # 在背景上添加标题
        cv2.putText(final_canvas, title, (text_x, text_y), font,
                    font_scale, font_color, font_thickness)

        # 合并标题和字符图像
        final_canvas[title_height:, :] = canvas

        return final_canvas

    def _create_char_display_with_index(self, index, matched_char):
        """创建带高亮和识别结果的字符图像"""
        # 创建基础分割图像
        char_display = self._create_char_display_image(self.char_images)

        # 检查是否需要扩展图像以显示结果
        title_height = 30
        result_row = 0

        # 检查图像底部是否已有足够空间显示结果
        height, width, _ = char_display.shape

        # 创建结果文字行
        result_text = f"识别进度: {self.recognition_result}{matched_char if matched_char else ''}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (0, 255, 0) if matched_char else (0, 0, 255)
        font_thickness = 2
        text_size, _ = cv2.getTextSize(result_text, font, font_scale, font_thickness)

        # 添加新行以显示结果
        new_height = height + text_size[1] + 15
        result_canvas = np.zeros((new_height, width, 3), dtype=np.uint8)

        # 添加原内容
        result_canvas[:height, :] = char_display

        # 在顶部添加分隔线
        cv2.line(result_canvas, (0, height), (width, height), (200, 200, 200), 1)

        # 添加结果文本
        text_x = (width - text_size[0]) // 2
        text_y = height + (new_height - height + text_size[1]) // 2 + 5
        cv2.putText(result_canvas, result_text, (text_x, text_y),
                    font, font_scale, font_color, font_thickness)

        # 高亮当前字符
        char_height = 60
        char_width = 40
        margin = 5
        x = margin + index * (char_width + margin)
        y = title_height + margin

        # 绘制高亮框
        highlight_color = (0, 255, 0) if matched_char else (0, 0, 255)
        cv2.rectangle(result_canvas,
                      (x - 1, y - 1),
                      (x + char_width + 1, y + char_height + 1),
                      highlight_color, 2)

        return result_canvas


if __name__ == "__main__":
    root = tk.Tk()
    app = Application(root)
    root.mainloop()