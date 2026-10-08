import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import cv2
import os


def resize_keep_aspect_ratio(image, target_width=None, target_height=None):
    
    (h, w) = image.shape[:2]
    
    if target_width is None and target_height is None: 
        return image
    
    #設定寬度1200
    if target_height is None:
        if target_width >= w: return image
        ratio = target_width / float(w)
        new_h = int(h * ratio)
        print(f"圖片尺寸調整為: {target_width}x{new_h}")
        return cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_AREA)
    
    else:
        img_ratio = w / h
        canvas_ratio = target_width / target_height
        
        if img_ratio > canvas_ratio:
            # 圖片比較寬，以寬度為準
            new_w = target_width
            new_h = int(new_w / img_ratio)
        else:
            # 圖片比較高，以高度為準
            new_h = target_height
            new_w = int(new_h * img_ratio)
            
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def smooth_region_borders(img_index_map, smoothness_factor):
    #放大 -> 中值模糊 -> 縮小
    print(f" 平滑度={smoothness_factor})")
    h, w = img_index_map.shape
    upscaled = cv2.resize(img_index_map, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)
    kernel_size = smoothness_factor * 2 + 1
    smoothed_upscaled = cv2.medianBlur(upscaled, kernel_size)
    smoothed_upscaled = cv2.medianBlur(smoothed_upscaled, 5) 
    smooth_map = cv2.resize(smoothed_upscaled, (w, h), interpolation=cv2.INTER_NEAREST)
    return smooth_map

def merge_tiny_regions(index_map, min_pixels):
    #merge 小區域
    cleaned_map = index_map.copy()
    iterations = 5 
    print(f"(門檻={min_pixels}px)")

    for i in range(iterations):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_map, connectivity=4)
        tiny_count = 0
        for label_idx in range(num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area < min_pixels:
                tiny_count += 1
                mask = (labels == label_idx).astype(np.uint8)
                dilated_mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
                border_mask = dilated_mask - mask
                neighbor_indices = cleaned_map[border_mask > 0]
                if len(neighbor_indices) > 0:
                    counts = np.bincount(neighbor_indices)
                    dominant_neighbor = np.argmax(counts)
                    cleaned_map[mask > 0] = dominant_neighbor
        if tiny_count == 0: break
    return cleaned_map

def generate_slic_pipeline(cv_img, target_width, k_colors=16):

    #縮放 -> 前處理 -> SLIC -> K-Means -> 映射 -> 合併 -> 圓滑

    #調整大小
    img = resize_keep_aspect_ratio(cv_img, target_width=target_width)
    h_img, w_img = img.shape[:2]
    

    #雙邊濾波
    img_blurred = cv2.bilateralFilter(img, d=9, sigmaColor=25, sigmaSpace=75)
    
    #中值模糊
    img_blurred = cv2.medianBlur(img_blurred, 3)

    #SLIC 分割 (限制 2000 個區塊)
    max_superpixels = 2000
    total_pixels = h_img * w_img
    ideal_area = total_pixels / max_superpixels
    region_size = int(np.sqrt(ideal_area))
    if region_size < 10: region_size = 10
    
    print(f"SLIC (目標<=2000, Size={region_size})")
    
    try:
        img_lab = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2Lab)
        slic = cv2.ximgproc.createSuperpixelSLIC(img_lab, region_size=region_size, ruler=30.0)
        slic.iterate(10)
        labels = slic.getLabels()
        num_slic = slic.getNumberOfSuperpixels()
    except AttributeError:
        print("錯誤：找不到 cv2.ximgproc")
        return None, None, None

    #K-Means
    print(f"K-Means ({k_colors} 色)")
    data = img_blurred.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(data, k_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)

    # 5. Mapping
    final_index_map = np.zeros((h_img, w_img), dtype=np.uint8)
    
    for i in range(num_slic):
        mask = (labels == i)
        if not np.any(mask): continue
        # 取平均色
        mean_color = cv2.mean(img_blurred, mask=mask.astype(np.uint8))[:3]
        mean_color_arr = np.array(mean_color, dtype=np.float32)
        distances = np.linalg.norm(centers - mean_color_arr, axis=1)
        final_index_map[mask] = np.argmin(distances)

    #合併小區域
    min_paintable_area = 30
    final_index_map = merge_tiny_regions(final_index_map, min_paintable_area)
    
    #圓滑化
    final_index_map = smooth_region_borders(final_index_map, smoothness_factor=9)

    return final_index_map, centers, img


class PaintByNumbersGame:
    def __init__(self, root):
        self.root = root
        self.root.title("AI 數字油畫遊戲 - 自動填補微小區域版")
        
        # 畫布顯示區域
        self.canvas_width = 1000
        self.canvas_height = 800
        
        self.current_color = None
        self.original_img = None
        self.template_img = None # PIL Image (顯示用線稿)
        
        # 狀態陣列
        self.painted_array = None # 紀錄已填色的區域
        self.color_map = {}       # {1: (R,G,B), ...}
        
        # 核心地圖
        self.color_map_array = None   # 答案圖 (數值 1~16)
        self.unique_region_map = None # 唯一區域 ID (數值 1~N)
        self.region_to_color = {}     # {RegionID: ColorID}
        
        self.is_individual_region_mode = True
        
        # 縮放和平移
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.is_dragging = False
        
        # 鉛筆圖示
        self.pencil_img = None
        self.pencil_icon = None
        self.pencil_id = None
        
        # 按鈕和完成狀態追蹤
        self.color_buttons = {} 
        self.color_completion_status = {} 
        
        self.setup_ui()

    def setup_ui(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 右側色表
        self.palette_frame = tk.Frame(main_frame, width=350, bg="#e8e8e8")
        self.palette_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        self.palette_frame.pack_propagate(False)
        
        tk.Label(self.palette_frame, text="色表", font=("Arial", 18, "bold"), bg="#e8e8e8").pack(pady=15)
        
        scroll_frame = tk.Frame(self.palette_frame, bg="#e8e8e8")
        scroll_frame.pack(fill=tk.BOTH, expand=True, padx=5)
        
        self.palette_canvas = tk.Canvas(scroll_frame, bg="#e8e8e8", highlightthickness=0)
        scrollbar = tk.Scrollbar(scroll_frame, orient="vertical", command=self.palette_canvas.yview)
        
        self.palette_container = tk.Frame(self.palette_canvas, bg="#e8e8e8")
        self.palette_canvas.create_window((0, 0), window=self.palette_container, anchor="nw", width=320)
        self.palette_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.palette_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.palette_container.bind("<Configure>", lambda e: self.palette_canvas.configure(scrollregion=self.palette_canvas.bbox("all")))

        # 左側畫布
        canvas_frame = tk.Frame(main_frame)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        top_btn_frame = tk.Frame(canvas_frame)
        top_btn_frame.pack(pady=10)
        
        self.upload_btn = tk.Button(top_btn_frame, text="上傳圖片", command=self.upload_image, font=("Arial", 14), bg="#4CAF50", fg="white", padx=15)
        self.upload_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_btn = tk.Button(top_btn_frame, text="重置畫布", command=self.reset_canvas, font=("Arial", 14), bg="#ff9800", fg="white", padx=15)
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        self.zoom_reset_btn = tk.Button(top_btn_frame, text="重置視圖", command=self.reset_zoom, font=("Arial", 14), bg="black", fg="white", padx=15)
        self.zoom_reset_btn.pack(side=tk.LEFT, padx=5)

        mode_btn_frame = tk.Frame(canvas_frame)
        mode_btn_frame.pack(pady=5)
        self.mode1_btn = tk.Button(mode_btn_frame, text="模式一：分開區域", command=lambda: self.set_mode(True), font=("Arial", 12), bg="#2196F3", fg="white", width=15)
        self.mode1_btn.pack(side=tk.LEFT, padx=5)
        self.mode2_btn = tk.Button(mode_btn_frame, text="模式二：相同顏色", command=lambda: self.set_mode(False), font=("Arial", 12), bg="#E0E0E0", fg="black", width=15)
        self.mode2_btn.pack(side=tk.LEFT, padx=5)

        self.zoom_label = tk.Label(canvas_frame, text="準備就緒", font=("Arial", 10), fg="gray")
        self.zoom_label.pack()
        
        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_width, height=self.canvas_height, bg="white", cursor="crosshair")
        self.canvas.pack(padx=10, pady=10)
        
        # 綁定
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<ButtonPress-3>", self.on_drag_start)
        self.canvas.bind("<B3-Motion>", self.on_drag_move)
        self.canvas.bind("<ButtonRelease-3>", self.on_drag_end)
        self.canvas.bind("<MouseWheel>", self.on_zoom)
        self.canvas.bind("<Button-4>", lambda e: self.on_zoom(e, direction=1))
        self.canvas.bind("<Button-5>", lambda e: self.on_zoom(e, direction=-1))
        self.canvas.bind("<Motion>", self.on_mouse_move) 
        
        # 嘗試載入鉛筆圖示
        try:
           if os.path.exists("pencil.png"):
               self.pencil_img = Image.open("pencil.png").convert("RGBA")
               self.pencil_img = self.pencil_img.resize((32,32), Image.Resampling.LANCZOS)
               self.pencil_icon = ImageTk.PhotoImage(self.pencil_img)
        except:
            pass

        self.update_mode_button_appearance()

    def set_mode(self, is_individual):
        if self.is_individual_region_mode != is_individual:
            self.is_individual_region_mode = is_individual
            self.update_mode_button_appearance()
            if self.template_img is not None:
                mode_name = "分開區域填色" if is_individual else "一起填色"
                messagebox.showinfo("模式切換", f"模式已切換為：{mode_name}。")
    
    def update_mode_button_appearance(self):
        if self.is_individual_region_mode:
            self.mode1_btn.config(bg="#2196F3", fg="white")
            self.mode2_btn.config(bg="#E0E0E0", fg="black")
        else:
            self.mode1_btn.config(bg="#E0E0E0", fg="black")
            self.mode2_btn.config(bg="#2196F3", fg="white")
        self.display_image() 

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if file_path:
            self.zoom_label.config(text="處理中... (SLIC 演算法運算中)")
            self.root.update()
            try:
                self.process_image_with_slic(file_path)
            except Exception as e:
                import traceback
                traceback.print_exc()
                messagebox.showerror("錯誤", f"處理失敗: {str(e)}")
                self.zoom_label.config(text="發生錯誤")

    def auto_fill_tiny_regions(self, min_area=10):
        h, w = self.color_map_array.shape
        
        unique_colors = np.unique(self.color_map_array)
        count_filled = 0
        
        for color_num in unique_colors:
            if color_num == 0: continue 
            
            # 找出該顏色的所有區域
            mask = (self.color_map_array == color_num).astype(np.uint8)
            # 連通分量分析
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
            
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                
                #面積小於門檻
                if area < min_area:
                    component_mask = (labels == i)
                    self.painted_array[component_mask] = color_num
                    count_filled += 1
                

    def process_image_with_slic(self, file_path):
        #執行 SLIC 載入到 GUI
        stream = open(file_path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        cv_img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        
        if cv_img is None: return

        # 設定運算寬度
        target_w = 1200 
        
        # 執行 SLIC Pipeline
        index_map, centers, resized_img = generate_slic_pipeline(cv_img, target_width=target_w, k_colors=16)
        
        if index_map is None: return

        self.original_img_cv = resized_img
        h, w = index_map.shape
        
        # 轉換 index (0留給線條)
        self.color_map_array = index_map.astype(np.int32) + 1 
        
        # 建立顏色表
        self.color_map = {}
        for i, color in enumerate(centers):
            self.color_map[i + 1] = (int(color[2]), int(color[1]), int(color[0]))

        # 建立點擊區域
        self.create_unique_regions()
        
        # 建立視覺模板 (白底黑線+數字)
        self.create_numbered_template_cv()

        # 初始化填色狀態
        self.painted_array = np.zeros((h, w), dtype=np.int32)
        
        self.auto_fill_tiny_regions(min_area=30)

        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.current_color = None
        
        self.update_palette()
        self.display_image()
        self.zoom_label.config(text="準備就緒！")

    def create_unique_regions(self):
        #用於模式一
        self.unique_region_map = np.zeros_like(self.color_map_array, dtype=np.int32)
        self.region_to_color = {}
        current_id = 1
        
        for color_num in self.color_map.keys():
            mask = (self.color_map_array == color_num).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
            
            for i in range(1, num_labels):
                self.unique_region_map[labels == i] = current_id
                self.region_to_color[current_id] = color_num
                current_id += 1

    def create_numbered_template_cv(self):
        #生成線稿與數字
        h, w = self.color_map_array.shape
        
        #產生邊緣
        kernel = np.ones((2,2), np.uint8)
        map_float = self.color_map_array.astype(float)
        edges = cv2.subtract(cv2.dilate(map_float, kernel), cv2.erode(map_float, kernel))
        edges = (edges > 0).astype(np.uint8) * 255
        edges_inv = cv2.bitwise_not(edges)
        
        canvas_pil = Image.fromarray(cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2RGB))
        draw = ImageDraw.Draw(canvas_pil)
        
        font_size = max(10, int(w / 80)) 
        try: font = ImageFont.truetype("arial.ttf", font_size)
        except: font = ImageFont.load_default()
        
        min_label_area = 30
        edge_margin = 2
        
        # 標註數字
        for color_num in self.color_map.keys():
            mask = (self.color_map_array == color_num).astype(np.uint8) * 255
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
            
            for i in range(1, num_labels):
                x, y, rw, rh, area = stats[i]
                
                if area > min_label_area:
                    roi = (labels[y:y+rh, x:x+rw] == i).astype(np.uint8) * 255
                    padded = cv2.copyMakeBorder(roi, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
                    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
                    _, max_val, _, max_loc = cv2.minMaxLoc(dist)
                    
                    final_x = x + max_loc[0] - 1
                    final_y = y + max_loc[1] - 1
                    
                    text = str(color_num)
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                    
                    draw_x = final_x - text_w // 2
                    draw_y = final_y - text_h // 2
                    
                    # 強制限制在邊界內
                    draw_x = max(edge_margin, min(draw_x, w - text_w - edge_margin))
                    draw_y = max(edge_margin, min(draw_y, h - text_h - edge_margin))
                    
                    draw.text((draw_x, draw_y), text, fill="black", font=font)
        
        self.numbered_template = canvas_pil 
        self.template_img = canvas_pil 

    def display_image(self):
        if self.template_img is None: return
        
        display_img = self.numbered_template.copy()
        
        if np.any(self.painted_array > 0):
            h, w = self.painted_array.shape
            color_layer = np.zeros((h, w, 3), dtype=np.uint8)
            mask_layer = (self.painted_array > 0)
            
            for c_num, rgb in self.color_map.items():
                color_layer[self.painted_array == c_num] = rgb
            
            color_pil = Image.fromarray(color_layer)
            mask_pil = Image.fromarray(mask_layer.astype(np.uint8) * 255)
            display_img.paste(color_pil, mask=mask_pil)

        zoomed_width = int(display_img.width * self.zoom_level)
        zoomed_height = int(display_img.height * self.zoom_level)
        self.zoomed_img = display_img.resize((zoomed_width, zoomed_height), Image.Resampling.NEAREST)
        
        self.photo = ImageTk.PhotoImage(self.zoomed_img)
        self.canvas.delete("all")
        
        x = (self.canvas_width - zoomed_width) // 2 + self.pan_x
        y = (self.canvas_height - zoomed_height) // 2 + self.pan_y
        self.img_offset = (x, y)
        
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo)
        
        if self.pencil_icon:
            if self.pencil_id: self.canvas.delete(self.pencil_id)
            self.pencil_id = self.canvas.create_image(0, 0, image=self.pencil_icon)
            
        mode_text = "分開區域" if self.is_individual_region_mode else "相同顏色"
        self.zoom_label.config(text=f"模式: {mode_text} | 縮放: {self.zoom_level:.1f}x")

    def on_mouse_move(self,event):
        if self.pencil_id:
            self.canvas.coords(self.pencil_id, event.x, event.y)
            self.canvas.tag_raise(self.pencil_id)
    
    def on_zoom(self, event, direction=None):
        if self.template_img is None: return
        if direction is not None: scale_factor = 1.2 if direction==1 else 1/1.2
        elif event.delta>0: scale_factor=1.2
        else: scale_factor=1/1.2
        self.zoom_level*=scale_factor
        self.zoom_level=max(0.1, min(self.zoom_level, 10.0))
        self.display_image()
    
    def reset_zoom(self):
        self.zoom_level=1.0
        self.pan_x=0
        self.pan_y=0
        self.display_image()
    
    def on_drag_start(self,event):
        self.is_dragging=True
        self.drag_start_x=event.x
        self.drag_start_y=event.y
        self.canvas.config(cursor="fleur")
    
    def on_drag_move(self,event):
        if self.is_dragging:
            dx = event.x-self.drag_start_x
            dy = event.y-self.drag_start_y
            self.pan_x+=dx
            self.pan_y+=dy
            self.drag_start_x=event.x
            self.drag_start_y=event.y
            self.display_image()
    
    def on_drag_end(self,event):
        self.is_dragging=False
        self.canvas.config(cursor="crosshair")
    
    # --- 調色盤邏輯 ---
    def update_palette(self):
        for widget in self.palette_container.winfo_children(): widget.destroy()
        self.color_buttons={}
        if self.color_map_array is not None: self.check_all_colors_completion()

        row, col = 0, 0
        grid_frame=tk.Frame(self.palette_container,bg="#e8e8e8")
        grid_frame.pack(fill=tk.X,expand=True)
        grid_frame.grid_columnconfigure(0,weight=1)
        grid_frame.grid_columnconfigure(1,weight=1)
        
        for num,color in sorted(self.color_map.items()):
            color_hex_bg="#{:02x}{:02x}{:02x}".format(*color) 
            btn = tk.Label(grid_frame, text=str(num), font=("Arial",20,"bold"),
                           bg=color_hex_bg, width=6, height=2, relief=tk.RAISED, bd=3, cursor="hand2")
            
            fg_color = "black" if (color[0]*0.299 + color[1]*0.587 + color[2]*0.114) > 128 else "white"
            btn.config(fg=fg_color)
            
            self.color_buttons[num]=btn
            btn.bind("<Button-1>", lambda e,n=num,c=color,l=btn:self.select_color(n,c,l))
            self.update_color_button_status(num)
            
            btn.grid(row=row,column=col,padx=5,pady=5,sticky="ew")
            col+=1
            if col>1: col=0; row+=1

    def check_all_colors_completion(self):
        if self.color_map_array is None or self.painted_array is None: return
        for c_num in self.color_map.keys():
            req = (self.color_map_array == c_num)
            done = (self.painted_array == c_num)
            is_completed = not np.any(req & (~done))
            self.color_completion_status[c_num] = is_completed
            if c_num in self.color_buttons: self.update_color_button_status(c_num)

    def update_color_button_status(self, c_num):
        if c_num not in self.color_buttons: return
        btn = self.color_buttons[c_num]
        is_done = self.color_completion_status.get(c_num, False)
        
        if is_done:
            btn.config(state=tk.DISABLED, relief=tk.FLAT, bd=1, bg="#AAAAAA", fg="#FFFFFF", cursor="arrow")
            if self.current_color and self.current_color[0] == c_num: self.current_color = None
        else:
            orig_color = self.color_map[c_num]
            hex_c = "#{:02x}{:02x}{:02x}".format(*orig_color)
            fg_c = "black" if (orig_color[0]*0.299 + orig_color[1]*0.587 + orig_color[2]*0.114) > 128 else "white"
            btn.config(state=tk.NORMAL, relief=tk.RAISED, bd=3, bg=hex_c, fg=fg_c, cursor="hand2")
            
            if self.current_color and self.current_color[0] == c_num:
                btn.config(relief=tk.SUNKEN, bd=5)
    
    def select_color(self,num,color,label):
        if label.cget('state') == tk.DISABLED: return
        self.current_color=(num,color)
        for n,btn in self.color_buttons.items():
            if self.color_completion_status.get(n, False): continue
            if n==num: btn.config(relief=tk.SUNKEN,bd=5)
            else: btn.config(relief=tk.RAISED,bd=3)
    
    def on_canvas_click(self,event):
        if self.current_color is None or self.unique_region_map is None: return
        
        img_x = int((event.x - self.img_offset[0]) / self.zoom_level)
        img_y = int((event.y - self.img_offset[1]) / self.zoom_level)
        
        h, w = self.unique_region_map.shape
        if 0 <= img_x < w and 0 <= img_y < h:
            target_uid = self.unique_region_map[img_y, img_x]
            target_c_num = self.color_map_array[img_y, img_x]
            
            if target_uid == 0:
                if target_c_num > 0: correct = target_c_num
                else: return
            else:
                correct = self.region_to_color.get(target_uid)
            
            if correct is None: return
            
            if self.color_completion_status.get(self.current_color[0], False): return

            if self.is_individual_region_mode:
                self.fill_region_individual(target_uid, correct)
            else:
                self.fill_region_same_color(correct)
                
            self.display_image()
            self.check_all_colors_completion()
            self.check_completion()
    
    def fill_region_individual(self, uid, correct):
        curr_num, _ = self.current_color
        if uid == 0: pass 
        if correct == curr_num:
            self.painted_array[self.unique_region_map == uid] = curr_num
        else:
            messagebox.showinfo("提示", f"這裡應該填 {correct} 號顏色哦！")
    
    def fill_region_same_color(self, correct):
        curr_num, _ = self.current_color
        if correct == curr_num:
            self.painted_array[self.color_map_array == curr_num] = curr_num
        else:
            messagebox.showinfo("提示", f"這裡應該填 {correct} 號顏色哦！")
    
    def check_completion(self):
        unfilled = (self.painted_array == 0) & (self.color_map_array > 0)
        if not np.any(unfilled):
            messagebox.showinfo("恭喜！", "你完成了這幅數字油畫！🎉")
            
            h, w = self.painted_array.shape
            res_arr = np.zeros((h, w, 3), dtype=np.uint8)
            for c, rgb in self.color_map.items():
                res_arr[self.painted_array == c] = rgb
            
            comp_img = Image.fromarray(res_arr)
            
            # --- 相框 ---
            outer = 30; middle = 15; inner = 5
            tot_w = comp_img.width + 2*(outer+middle+inner)
            tot_h = comp_img.height + 2*(outer+middle+inner)
            
            framed = Image.new("RGB", (tot_w, tot_h), (0,0,0))
            mid_frame = Image.new("RGB", (comp_img.width+2*(middle+inner), comp_img.height+2*(middle+inner)), (255,255,255))
            inn_frame = Image.new("RGB", (comp_img.width+2*inner, comp_img.height+2*inner), (0,0,0))
            
            inn_frame.paste(comp_img, (inner, inner))
            mid_frame.paste(inn_frame, (middle, middle))
            framed.paste(mid_frame, (outer, outer))
            
            framed.show()

    def reset_canvas(self):
        if self.template_img is not None:
            self.painted_array.fill(0)
            self.auto_fill_tiny_regions(min_area=30)
            
            self.display_image()
            self.current_color = None
            self.check_all_colors_completion()
            self.update_palette()

if __name__=="__main__":
    root=tk.Tk()
    root.minsize(1400,900)
    app=PaintByNumbersGame(root)
    root.mainloop()