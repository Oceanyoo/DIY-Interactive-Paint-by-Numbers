import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import cv2
from scipy import stats, ndimage
import os
import math

def merge_small_regions_fast(img, min_area_threshold):
    # 強制確保輸入是 uint8
    img = img.astype(np.uint8)
    h, w, c = img.shape
    
    img_int = img[:,:,0].astype(np.uint32) * 65536 + \
              img[:,:,1].astype(np.uint32) * 256 + \
              img[:,:,2].astype(np.uint32)
    
    result_img = img.copy()
    unique_colors = np.unique(img_int)
    
    kernel = np.ones((3,3), np.uint8)

    for color_val in unique_colors:
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[img_int == color_val] = 255
        
        num_labels, labels, cc_stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        for i in range(1, num_labels):
            area = cc_stats[i, cv2.CC_STAT_AREA]
            if area < min_area_threshold:
                # 簡單填補邏輯
                x, y, rw, rh = cc_stats[i, cv2.CC_STAT_LEFT], cc_stats[i, cv2.CC_STAT_TOP], cc_stats[i, cv2.CC_STAT_WIDTH], cc_stats[i, cv2.CC_STAT_HEIGHT]
                roi_labels = labels[y:y+rh, x:x+rw]
                roi_img = result_img[y:y+rh, x:x+rw]
                blob_mask_roi = (roi_labels == i).astype(np.uint8) * 255
                
                dilated = cv2.dilate(blob_mask_roi, kernel, iterations=1)
                neighbor_mask = cv2.subtract(dilated, blob_mask_roi)
                neighbor_pixels = roi_img[neighbor_mask > 0]
                
                if len(neighbor_pixels) > 0:
                    nb_ints = neighbor_pixels[:,0].astype(np.uint32) * 65536 + \
                              neighbor_pixels[:,1].astype(np.uint32) * 256 + \
                              neighbor_pixels[:,2].astype(np.uint32)
                    mode_res = stats.mode(nb_ints, keepdims=False)
                    mode_val = mode_res[0]
                    if isinstance(mode_val, np.ndarray): mode_val = mode_val.item()
                    
                    b = int(mode_val % 256)
                    g = int((mode_val // 256) % 256)
                    r = int((mode_val // 65536))
                    roi_img[blob_mask_roi > 0] = [b, g, r]
                    
    return result_img

def remove_tiny_islands_from_map(label_map, min_pixel_size):
    h, w = label_map.shape
    clean_map = label_map.copy()
    unique_ids = np.unique(clean_map)
    
    for uid in unique_ids:
        if uid == 0: continue 
        mask = (clean_map == uid).astype(np.uint8) * 255
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_pixel_size:
                blob_mask = (labels == i).astype(np.uint8) * 255
                dilated = cv2.dilate(blob_mask, np.ones((3,3), np.uint8))
                neighbor_mask = (dilated > 0) & (blob_mask == 0)
                neighbor_ids = clean_map[neighbor_mask]
                
                if len(neighbor_ids) > 0:
                    counts = np.bincount(neighbor_ids)
                    best_neighbor = np.argmax(counts)
                    clean_map[blob_mask > 0] = best_neighbor
    return clean_map

def get_optimal_font_scale(image_width):
    scale = 0.35 * (image_width / 1000)
    thickness = max(1, int(1.2 * (image_width / 1000)))
    return max(0.3, scale), thickness

# ==========================================
# 主程式：整合 GUI (DDD.py) 與 核心 (3.py)
# ==========================================
class PaintByNumbersGame:
    def __init__(self, root):
        self.root = root
        self.root.title("數字油畫遊戲 - 完美演算法整合版")
        
        # 畫布顯示區域
        self.canvas_width = 1000
        self.canvas_height = 800
        
        self.current_color = None
        self.original_img = None
        self.template_img = None # 線稿圖 (PIL)
        self.painted_array = None # 填色狀態 (numpy)
        self.color_map = {} # {id: (R, G, B)}
        
        # 核心雙重地圖
        self.color_map_array = None      # 像素對應顏色 ID (1~12)
        self.unique_region_map = None    # 像素對應唯一區域 ID
        self.region_to_color = {}        # 區域 ID -> 顏色 ID
        
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
        
        palette_label = tk.Label(self.palette_frame, text="色表", 
                                 font=("Arial", 18, "bold"), bg="#e8e8e8")
        palette_label.pack(pady=15)
        
        scroll_frame = tk.Frame(self.palette_frame, bg="#e8e8e8")
        scroll_frame.pack(fill=tk.BOTH, expand=True, padx=5)
        
        self.palette_canvas = tk.Canvas(scroll_frame, bg="#e8e8e8", highlightthickness=0)
        scrollbar = tk.Scrollbar(scroll_frame, orient="vertical", command=self.palette_canvas.yview)
        
        self.palette_container = tk.Frame(self.palette_canvas, bg="#e8e8e8")
        self.palette_canvas.create_window((0, 0), window=self.palette_container, anchor="nw", width=320)
        self.palette_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.palette_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        def update_scroll(event=None):
            self.palette_canvas.configure(scrollregion=self.palette_canvas.bbox("all"))
        self.palette_container.bind("<Configure>", update_scroll)

        # 左側畫布和按鈕
        canvas_frame = tk.Frame(main_frame)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        top_btn_frame = tk.Frame(canvas_frame)
        top_btn_frame.pack(pady=10)
        
        self.upload_btn = tk.Button(top_btn_frame, text="上傳圖片", command=self.upload_image, 
                                 font=("Arial", 14, "bold"), bg="#4CAF50", fg="white", padx=15, pady=5)
        self.upload_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_btn = tk.Button(top_btn_frame, text="重置畫布", command=self.reset_canvas, 
                                 font=("Arial", 14, "bold"), bg="#ff9800", fg="white", padx=15, pady=5)
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        self.zoom_reset_btn = tk.Button(top_btn_frame, text="重置視圖", command=self.reset_zoom, 
                                     font=("Arial", 14, "bold"), bg="black", fg="white", padx=15, pady=5)
        self.zoom_reset_btn.pack(side=tk.LEFT, padx=5)

        mode_btn_frame = tk.Frame(canvas_frame)
        mode_btn_frame.pack(pady=5)
        
        self.mode1_btn = tk.Button(mode_btn_frame, text="模式一：分開區域填色", command=lambda: self.set_mode(True),
                                 font=("Arial", 12, "bold"), bg="#2196F3", fg="white", padx=15, pady=5)
        self.mode1_btn.pack(side=tk.LEFT, padx=5)
        
        self.mode2_btn = tk.Button(mode_btn_frame, text="模式二：一起填色", command=lambda: self.set_mode(False),
                                 font=("Arial", 12, "bold"), bg="#E0E0E0", fg="black", padx=15, pady=5)
        self.mode2_btn.pack(side=tk.LEFT, padx=5)

        self.zoom_label = tk.Label(canvas_frame, text="滾輪縮放 | 右鍵拖曳移動 | 左鍵填色", 
                                 font=("Arial", 10), fg="gray")
        self.zoom_label.pack()
        
        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_width, height=self.canvas_height, 
                                 bg="white", cursor="crosshair")
        self.canvas.pack(padx=10, pady=10)
        
        # 綁定事件
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<ButtonPress-3>", self.on_drag_start)
        self.canvas.bind("<B3-Motion>", self.on_drag_move)
        self.canvas.bind("<ButtonRelease-3>", self.on_drag_end)
        self.canvas.bind("<MouseWheel>", self.on_zoom)
        self.canvas.bind("<Button-4>", lambda e: self.on_zoom(e, direction=1))
        self.canvas.bind("<Button-5>", lambda e: self.on_zoom(e, direction=-1))
        self.canvas.bind("<Motion>", self.on_mouse_move) 
        
        # 載入鉛筆圖 (如果有的話)
        try:
           self.pencil_img = Image.open("pencil.png").convert("RGBA")
           self.pencil_img = self.pencil_img.resize((32,32), Image.Resampling.LANCZOS)
           self.pencil_icon = ImageTk.PhotoImage(self.pencil_img)
        except:
            self.pencil_icon = None

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
        file_path = filedialog.askopenfilename(
            title="選擇圖片",
            filetypes=[("圖片檔案", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if file_path:
            self.zoom_label.config(text="運算中... (OpenCV 智能去噪與分割)", fg="red")
            self.root.update() # 強制刷新介面
            try:
                self.load_and_process_image(file_path)
                self.zoom_label.config(text="處理完成！請開始填色。", fg="gray")
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.zoom_label.config(text="發生錯誤", fg="red")
                messagebox.showerror("錯誤", f"處理失敗：{str(e)}")
    
    # =========================================================
    #  核心替換：使用 3.py 的 OpenCV 流程
    # =========================================================
    def load_and_process_image(self, file_path):
        # 1. 讀取 (支援中文路徑)
        try:
            cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            cv_img = None
        
        if cv_img is None:
            raise Exception("無法讀取圖片")

        # 2. Resize
        h, w = cv_img.shape[:2]
        ratio = min(self.canvas_width/w, self.canvas_height/h)
        new_w, new_h = int(w*ratio), int(h*ratio)
        img = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 3. MeanShift & KMeans
        print("   >> MeanShift & KMeans...")
        shifted = cv2.pyrMeanShiftFiltering(img, sp=30, sr=50)
        
        k_colors = 16 # 可以根據需求調整
        shifted_lab = cv2.cvtColor(shifted, cv2.COLOR_BGR2LAB)
        data = shifted_lab.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, label, center = cv2.kmeans(data, k_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        center = np.uint8(center)
        center_bgr = cv2.cvtColor(np.array([center], dtype=np.uint8), cv2.COLOR_LAB2BGR)[0]
        
        # 建立 color_map (BGR -> RGB)
        self.color_map = {}
        for i, bgr in enumerate(center_bgr):
            self.color_map[i+1] = (int(bgr[2]), int(bgr[1]), int(bgr[0]))

        res = center[label.flatten()]
        quantized_img = cv2.cvtColor(res.reshape((new_h, new_w, 3)), cv2.COLOR_LAB2BGR)

        # 4. 區域合併 (Merge)
        min_size = int(new_w * new_h * 0.001)
        print(f"   >> 合併微小區域 (閾值={min_size})...")
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        quantized_clean = cv2.morphologyEx(quantized_img, cv2.MORPH_OPEN, morph_kernel, iterations=1)
        
        merged_img = merge_small_regions_fast(quantized_clean, min_size)
        merged_img = merged_img.astype(np.uint8) # 安全鎖
        final_smooth_img = cv2.medianBlur(merged_img, 7) # 平滑化

        # 5. 強制清洗 Color Map (去除幽靈框線)
        print("   >> 強制清洗 Color Map...")
        img_float = final_smooth_img.astype(np.float32)
        min_dist = np.full((new_h, new_w), np.inf, dtype=np.float32)
        self.color_map_array = np.zeros((new_h, new_w), dtype=np.int32)
        
        for idx, c_bgr in enumerate(center_bgr):
            dist = np.sum((img_float - c_bgr)**2, axis=2)
            mask = dist < min_dist
            min_dist[mask] = dist[mask]
            self.color_map_array[mask] = idx + 1
            
        # 這裡設定絕對最小值，例如 100~200px，確保每個框都能點擊
        self.color_map_array = remove_tiny_islands_from_map(self.color_map_array, min_pixel_size=200)

        # 6. 生成線稿 (Sobel)
        print("   >> 生成 Sobel 線稿...")
        dx = cv2.Sobel(self.color_map_array.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(self.color_map_array.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
        mag = cv2.magnitude(dx, dy)
        _, edge_mask = cv2.threshold(mag, 0.1, 255, cv2.THRESH_BINARY)
        
        line_art = np.ones((new_h, new_w, 3), dtype=np.uint8) * 255
        line_art[edge_mask > 0] = (0, 0, 0) 

        # 7. 生成遊戲資料與標號
        self.unique_region_map = np.zeros_like(self.color_map_array, dtype=np.int32)
        self.region_to_color = {}
        current_region_id = 1
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale_base, thickness = get_optimal_font_scale(new_w)
        label_min_size = 50 
        
        print("   >> 標註數字...")
        candidate_labels = []

        for color_id in range(1, k_colors + 1):
            mask = (self.color_map_array == color_id).astype(np.uint8) * 255
            num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            for i in range(1, num):
                area = stats[i, cv2.CC_STAT_AREA]
                
                # 記錄區域 ID (即使沒標數字，也要能點擊)
                if area > 10: 
                    reg_mask = (labels == i)
                    self.unique_region_map[reg_mask] = current_region_id
                    self.region_to_color[current_region_id] = color_id
                    current_region_id += 1
                    
                    if area > label_min_size:
                        dist_map = cv2.distanceTransform(reg_mask.astype(np.uint8), cv2.DIST_L2, 3)
                        _, max_val, _, max_loc = cv2.minMaxLoc(dist_map)
                        
                        candidate_labels.append({
                            'text': str(color_id),
                            'cx': max_loc[0],
                            'cy': max_loc[1],
                            'area': area,
                            'max_val': max_val
                        })

        # 排序與距離過濾
        candidate_labels.sort(key=lambda x: x['area'], reverse=True)
        drawn_labels = []
        min_label_dist = 30
        
        for cand in candidate_labels:
            cx, cy = cand['cx'], cand['cy']
            
            too_close = False
            for drawn in drawn_labels:
                dist = math.sqrt((cx - drawn['cx'])**2 + (cy - drawn['cy'])**2)
                if dist < min_label_dist:
                    too_close = True
                    break
            if too_close: continue
            
            text = cand['text']
            scale = min(font_scale_base * 1.8, max(0.4, cand['max_val'] / 25))
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            
            tx = int(cx - tw/2)
            ty = int(cy + th/2)
            tx = max(5, min(tx, new_w - tw - 5))
            ty = max(th+5, min(ty, new_h - 5))

            cv2.putText(line_art, text, (tx, ty), font, scale, (80, 80, 80), thickness, cv2.LINE_AA)
            drawn_labels.append({'cx': cx, 'cy': cy})

        # 8. 最終設定
        self.template_img = Image.fromarray(line_art) # 轉回 PIL 供 display_image 使用
        self.painted_array = np.zeros((new_h, new_w), dtype=np.int32)
        
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.current_color = None
        
        self.display_image()
        self.update_palette() # 更新按鈕

    # =========================================================
    #  GUI 顯示邏輯 (優化：使用 Numpy Masking 加速渲染)
    # =========================================================
    def display_image(self):
        if self.template_img is None: return
        
        # 1. 取得底圖 (線稿)
        display_np = np.array(self.template_img)
        
        # 2. 疊加填色 (使用 Numpy 加速，不再用 putpixel)
        mask_painted = (self.painted_array > 0)
        
        if np.any(mask_painted):
            # 建立一個全彩圖層
            color_layer = np.zeros_like(display_np)
            
            # 針對每個已填的顏色 ID，填入對應 RGB
            # 找出目前畫面上用到的所有顏色 ID
            active_ids = np.unique(self.painted_array[mask_painted])
            
            for cid in active_ids:
                if cid in self.color_map:
                    color_layer[self.painted_array == cid] = self.color_map[cid]
            
            # 覆蓋上去 (簡單覆蓋，保留線條需要更複雜的 blend，這裡先做填滿)
            display_np[mask_painted] = color_layer[mask_painted]

        # 3. 轉回 PIL 進行縮放顯示
        display_img = Image.fromarray(display_np)
        
        zoomed_width = int(display_img.width * self.zoom_level)
        zoomed_height = int(display_img.height * self.zoom_level)
        self.zoomed_img = display_img.resize((zoomed_width, zoomed_height), Image.Resampling.NEAREST)
        
        self.photo = ImageTk.PhotoImage(self.zoomed_img)
        self.canvas.delete("all")
        
        x = (self.canvas_width - zoomed_width) // 2 + self.pan_x
        y = (self.canvas_height - zoomed_height) // 2 + self.pan_y
        self.img_offset = (x, y)
        
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo)
        
        # 加鉛筆圖
        if self.pencil_icon:
            pencil_size = int(32 * self.zoom_level)
            pencil_resized = self.pencil_img.resize((pencil_size, pencil_size), Image.Resampling.LANCZOS)
            self.pencil_icon_tk = ImageTk.PhotoImage(pencil_resized) # 避免被回收
            if self.pencil_id:
                self.canvas.delete(self.pencil_id)
            self.pencil_id = self.canvas.create_image(0, 0, image=self.pencil_icon_tk)

        mode_text = "分開區域填色 (DIY 模式)" if self.is_individual_region_mode else "一起填色 (DIY_3 模式)"
        self.zoom_label.config(text=f"滾輪縮放 | 右鍵拖曳移動 | 左鍵填色 | 當前縮放: {self.zoom_level:.1f}x | 模式: {mode_text}")

    def on_mouse_move(self,event):
        if self.pencil_id:
            self.canvas.coords(self.pencil_id,event.x,event.y)
    
    def on_zoom(self, event, direction=None):
        if self.template_img is None: return
        if direction is not None: scale_factor = 1.2 if direction==1 else 1/1.2
        elif event.delta>0: scale_factor=1.2
        else: scale_factor=1/1.2
        self.zoom_level*=scale_factor
        self.zoom_level=max(0.5,min(self.zoom_level,10.0))
        self.display_image()
    
    def reset_zoom(self):
        if self.template_img is None: return
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
        """重新生成調色盤，並檢查顏色完成狀態"""
        for widget in self.palette_container.winfo_children(): 
            widget.destroy()
        
        self.color_buttons={}
        
        if self.color_map_array is not None:
             self.check_all_colors_completion()

        row=0
        col=0
        grid_frame=tk.Frame(self.palette_container,bg="#e8e8e8")
        grid_frame.pack(fill=tk.X,expand=True)
        grid_frame.grid_columnconfigure(0,weight=1)
        grid_frame.grid_columnconfigure(1,weight=1)
        
        for num,color in sorted(self.color_map.items()):
            color_hex_bg="#{:02x}{:02x}{:02x}".format(*color) 
            
            color_label = tk.Label(grid_frame,text=str(num),
                                    font=("Arial",20,"bold"),
                                    bg=color_hex_bg, 
                                    width=6,height=2,
                                    relief=tk.RAISED,borderwidth=3,
                                    cursor="hand2")
            
            color_label.config(fg="black" if sum(color)>400 else "white")
            self.color_buttons[num]=color_label
            color_label.bind("<Button-1>", lambda e,n=num,c=color,l=color_label:self.select_color(n,c,l))
            
            self.update_color_button_status(num)
            
            color_label.grid(row=row,column=col,padx=5,pady=5,sticky="ew")
            col+=1
            if col>1: col=0; row+=1

    def check_all_colors_completion(self):
        """檢查並更新所有顏色是否已完成"""
        if self.color_map_array is None or self.painted_array is None:
            return
            
        for color_num in self.color_map.keys():
            required_mask = (self.color_map_array == color_num)
            painted_mask = (self.painted_array == color_num)
            unpainted_required = required_mask & (~painted_mask)
            
            is_completed = not np.any(unpainted_required)
            self.color_completion_status[color_num] = is_completed
            
            if color_num in self.color_buttons:
                self.update_color_button_status(color_num)

    def update_color_button_status(self, color_num):
        """更新單個顏色按鈕的外觀"""
        if color_num not in self.color_buttons:
            return
            
        button = self.color_buttons[color_num]
        is_completed = self.color_completion_status.get(color_num, False)
        original_color = self.color_map[color_num]
        original_hex = "#{:02x}{:02x}{:02x}".format(*original_color)

        if is_completed:
            button.config(bg="#AAAAAA", cursor="arrow", state=tk.DISABLED, 
                          relief=tk.FLAT, borderwidth=1, fg="white")
            button.unbind("<Button-1>")
            if self.current_color and self.current_color[0] == color_num:
                self.current_color = None
        else:
            button.config(bg=original_hex, cursor="hand2", state=tk.NORMAL,
                          relief=tk.RAISED, borderwidth=3, 
                          fg="black" if sum(original_color)>400 else "white")
            button.bind("<Button-1>", lambda e,n=color_num,c=original_color,l=button:self.select_color(n,c,l))
            
        if self.current_color and self.current_color[0] == color_num and not is_completed:
             button.config(relief=tk.SUNKEN, borderwidth=5)
    
    def select_color(self,num,color,label):
        if label.cget('state') == tk.DISABLED: return
        self.current_color=(num,color)
        for n,btn in self.color_buttons.items():
            if self.color_completion_status.get(n, False): continue
            if n==num: btn.config(relief=tk.SUNKEN,borderwidth=5)
            else: btn.config(relief=tk.RAISED,borderwidth=3)
    
    def on_canvas_click(self,event):
        if self.current_color is None or self.unique_region_map is None:
            messagebox.showwarning("提示","請先選擇一個顏色！")
            return
        
        x=int((event.x-self.img_offset[0])/self.zoom_level)
        y=int((event.y-self.img_offset[1])/self.zoom_level)
        
        if 0<=x<self.template_img.width and 0<=y<self.template_img.height:
            target_unique_id=self.unique_region_map[y,x]
            target_color_num=self.color_map_array[y,x]
            
            if target_unique_id==0:
                # 點到邊框或無效區，嘗試看底層顏色
                if target_color_num>0: correct_color_num=target_color_num
                else: return
            else:
                correct_color_num=self.region_to_color.get(target_unique_id)
            
            if correct_color_num is None: return
            
            selected_color_num = self.current_color[0]
            if self.color_completion_status.get(selected_color_num, False):
                messagebox.showwarning("提示",f"你選取的 {selected_color_num} 號顏色已完成！")
                self.current_color = None
                return
            
            if self.is_individual_region_mode:
                self.fill_region_individual(target_unique_id,correct_color_num)
            else:
                self.fill_region_same_color(correct_color_num)
                
            self.display_image()
            self.check_all_colors_completion()
            self.check_completion()
    
    def fill_region_individual(self,unique_id,correct_color_num):
        color_num,_=self.current_color
        if unique_id==0:
            # 特例：如果是幽靈區塊（沒ID但有顏色），允許填色
            if correct_color_num == color_num:
                # 我們需要建立一個臨時mask，因為它沒有 unique_id
                # 這裡簡單處理：填補點擊位置周圍相同顏色的像素
                # 但這比較複雜，我們假設 3.py 的 remove_tiny_islands 已經解決大部分問題
                # 如果還是點到邊界，提示一下
                messagebox.showinfo("提示",f"這裡是邊界或微小區域，請嘗試填入 {correct_color_num} 號顏色。")
            return

        if correct_color_num==color_num:
            mask=(self.unique_region_map==unique_id)
            self.painted_array[mask]=color_num
        else:
            messagebox.showinfo("提示",f"這個獨立區域應該填 {correct_color_num} 號顏色哦！")
    
    def fill_region_same_color(self,correct_color_num):
        color_num,_=self.current_color
        if correct_color_num==color_num:
            final_mask=(self.color_map_array==correct_color_num)
            self.painted_array[final_mask]=color_num
        else:
            messagebox.showinfo("提示",f"這個區域應該填 {correct_color_num} 號顏色哦！")
    
    def check_completion(self):
        unfilled = (self.painted_array == 0) & (self.color_map_array > 0)
        if not np.any(unfilled):
            messagebox.showinfo("恭喜！", "你完成了這幅數字油畫！🎉")
            
            # 生成完成圖像
            completed_img = Image.new("RGB", self.template_img.size, "white")
            completed_img.paste(self.template_img) # 貼上線稿作為底
            
            # 填入顏色
            np_img = np.array(completed_img)
            for cid, rgb in self.color_map.items():
                np_img[self.painted_array == cid] = rgb
            
            completed_img = Image.fromarray(np_img)

            # --- 相框設置 ---
            outer_border = 30  # 最外層粗黑框
            middle_border = 15 # 白色框
            inner_border = 5   # 細黑框

            total_width = completed_img.width + 2*(outer_border + middle_border + inner_border)
            total_height = completed_img.height + 2*(outer_border + middle_border + inner_border)

            framed_img = Image.new("RGB", (total_width, total_height), (0,0,0))
            middle_frame = Image.new("RGB",
                                    (completed_img.width + 2*(middle_border + inner_border),
                                    completed_img.height + 2*(middle_border + inner_border)),
                                    (255,255,255))
            inner_frame = Image.new("RGB",
                                    (completed_img.width + 2*inner_border, completed_img.height + 2*inner_border),
                                    (0,0,0))

            inner_frame.paste(completed_img, (inner_border, inner_border))
            middle_frame.paste(inner_frame, (middle_border, middle_border))
            framed_img.paste(middle_frame, (outer_border, outer_border))
            framed_img.show()
    
    def reset_canvas(self):
        if self.template_img is not None:
            self.painted_array=np.zeros((self.template_img.height,self.template_img.width),dtype=np.int32)
            self.display_image()
            self.current_color=None
            self.check_all_colors_completion() 
            if hasattr(self,'color_buttons'):
                for btn in self.color_buttons.values():
                    if btn.cget('state') != tk.DISABLED:
                        btn.config(relief=tk.RAISED,borderwidth=3)

if __name__=="__main__":
    root=tk.Tk()
    root.minsize(1400,900)
    app=PaintByNumbersGame(root)
    root.mainloop()