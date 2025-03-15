# Machine Vision Project / Hamid Reza Heidari  / Milad Mohammadi

# PHASE 5 --> Project GUI

# Import library
from customtkinter import *
from PIL import Image
import numpy as np
import pandas as pd
import cv2
from itertools import combinations
import os

from imageio import imread
from ultralytics import YOLO

def Enhancement(image, clipLimit=2.0, tileGridSize=(8, 8)):

    trained_model = YOLO("best.onnx", task="detect")
    objects = trained_model.predict(source=image, max_det=1)

    # calculate bounding box
    for r in objects:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]

    # Define the crop region (x1, y1, x2, y2)
    x1, y1, x2, y2 = b
    x1 = int(x1.item())
    x2 = int(x2.item())
    y1 = int(y1.item())
    y2 = int(y2.item())

    #Crop the image
    cropped_image = image[y1:y2, x1:x2]

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # Step 3: Create CLAHE object
    # Clip limit: Threshold for contrast limiting
    # Tile grid size: Size of grid for histogram equalization
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    # Step 4: Apply CLAHE
    enhanced_image = clahe.apply(gray)
    return enhanced_image

output_folder="Geometric_Feature_Extraction"

# Create enrollment function
def enrollment(file_path, segment_length,
               ksize, d, up_limit, down_limit, crop_width, crop_height, save, crop=True, enhance_mode=False):
    file_name = os.path.basename(file_path)

    # Import Database

    feature_vector_database = pd.read_excel("Feature_Vector_Database.xlsx", index_col=0)

    # Set output folder for processed images
    os.makedirs(output_folder, exist_ok=True)

    def geometric_recognition(image, crop_mode=True):

        if crop_mode:
            height = image.shape[0]
            width = image.shape[1]

            # Calculate the center crop region
            center_x, center_y = width // 2, height // 2

            # Define the crop box
            x1 = max(0, center_x - crop_width // 2)
            y1 = max(0, center_y - crop_height // 2)
            x2 = min(width, center_x + crop_width // 2)
            y2 = min(height, center_y + crop_height // 2)

            # Crop the image
            cropped_image = image[y1:y2, x1:x2]

            # Resize to ensure the exact dimensions
            image = cv2.resize(cropped_image, (crop_width, crop_height))

        if enhance_mode:
            image = cv2.imread(file_path)
            image = Enhancement(image)

        ##--> Section I : Pre processing
        blurred_image = cv2.GaussianBlur(image, ksize, d)

        ##--> Section II : Edge Detection
        edges = cv2.Canny(blurred_image, down_limit, up_limit)

        ##--> Section III : Parameter Extraction

        non_zero_coords = np.column_stack(np.where(edges > 0))

        # Compute all pairwise distances and find the farthest pair
        farthest_pair = None
        max_distance = 0

        for (p1, p2) in combinations(non_zero_coords, 2):

            distance = np.linalg.norm(p1 - p2)
            if distance > max_distance:
                max_distance = distance
                farthest_pair = (p1, p2)

        # Extract the coordinates of the farthest points
        point1, point2 = farthest_pair
        FV1 = max_distance

        if point1[0] < point2[0]:
            U_max = point1
            L_max = point2
        else:
            U_max = point2
            L_max = point1

        # Draw a line between these two points
        final_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Convert to BGR for colored line
        cv2.line(final_image, tuple(point1[::-1]), tuple(point2[::-1]), (0, 255, 255), 1)

        def kernel_to_pixel(kernel, image, p):
            K = np.zeros((3, 3))

            K[0, 0] = kernel[0, 0] * image[p[0] - 1, p[1] - 1]
            K[0, 1] = kernel[0, 1] * image[p[0] - 1, p[1]]
            K[0, 2] = kernel[0, 2] * image[p[0] - 1, p[1] + 1]
            K[1, 0] = kernel[1, 0] * image[p[0], p[1] - 1]

            K[1, 2] = kernel[1, 2] * image[p[0], p[1] + 1]
            K[2, 0] = kernel[2, 0] * image[p[0] + 1, p[1] - 1]
            K[2, 1] = kernel[2, 0] * image[p[0] + 1, p[1]]
            K[2, 2] = kernel[2, 2] * image[p[0] + 1, p[1] + 1]

            return K

        def find_point(binary_image, start_point, T, dir):
            p = start_point
            kernel = np.ones((3, 3))

            if dir == "left":
                for i in range(T):
                    K = kernel_to_pixel(kernel, binary_image, p)
                    if K[2, 0] > 0:
                        p = (p[0] + 1, p[1] - 1)
                    elif K[2, 1] > 0:
                        p = (p[0] + 1, p[1])
                    elif K[1, 0] > 0:
                        p = (p[0], p[1] - 1)
                    elif K[0, 0] > 0:
                        p = (p[0] - 1, p[1] - 1)
                    elif K[0, 1] > 0:
                        p = (p[0] - 1, p[1])
                    elif K[0, 2] > 0:
                        p = (p[0] - 1, p[1] + 1)
                    elif K[1, 2] > 0:
                        p = (p[0], p[1] + 1)
                    elif K[2, 2] > 0:
                        p = (p[0] + 1, p[1] + 1)

            elif dir == "right":
                for i in range(T):
                    K = kernel_to_pixel(kernel, binary_image, p)
                    if K[1, 2] > 0:
                        p = (p[0], p[1] + 1)
                    elif K[2, 2] > 0:
                        p = (p[0] + 1, p[1] + 1)
                    elif K[2, 1] > 0:
                        p = (p[0] + 1, p[1])
                    elif K[0, 2] > 0:
                        p = (p[0] - 1, p[1] + 1)
                    elif K[0, 1] > 0:
                        p = (p[0] - 1, p[1])
                    elif K[2, 0] > 0:
                        p = (p[0] + 1, p[1] - 1)
                    elif K[1, 0] > 0:
                        p = (p[0], p[1] - 1)
                    elif K[0, 0] > 0:
                        p = (p[0] - 1, p[1] - 1)
            return p

        Ulb = find_point(edges, U_max, segment_length, "left")
        Urb = find_point(edges, U_max, segment_length, "right")
        Llb = find_point(edges, L_max, segment_length, "left")

        cv2.circle(final_image, (Ulb[1], Ulb[0]), 5, (139, 69, 19))
        cv2.circle(final_image, (Urb[1], Urb[0]), 5, (139, 0, 139))
        cv2.circle(final_image, (Llb[1], Llb[0]), 5, (127, 255, 0))

        # Find Umin, Lmin
        d1 = np.linalg.norm(np.array(Ulb) - np.array(Llb))
        d2 = np.linalg.norm(np.array(Urb) - np.array(Llb))

        if d1 < d2:
            U_min = Ulb
            FV2 = d1
        else:
            U_min = Urb
            FV2 = d2
        L_min = Llb

        cv2.line(final_image, tuple(U_min[::-1]), tuple(L_min[::-1]), (128, 0, 128), 1)

        # calculate CI
        Cmax = np.array([int((U_max[0] + L_max[0]) / 2), int((U_max[1] + L_max[1]) / 2)])
        cv2.circle(final_image, (Cmax[1], Cmax[0]), 5, (255, 69, 0))

        Ilb = None
        min_distance = image.shape[0]

        for y in range(image.shape[0]):
            x = 0
            row_meet = False
            while not row_meet:
                p = [y, x]
                if edges[p[0], p[1]] > 0:
                    row_meet = True
                    distance = np.linalg.norm(p - np.array(Cmax))

                    if distance < min_distance:
                        min_distance = distance
                        Ilb = p
                elif x < image.shape[1] - 1:
                    x = x + 1
                else:
                    row_meet = True

        CI = np.linalg.norm(Ilb - np.array(Cmax))

        cv2.circle(final_image, (Ilb[1], Ilb[0]), 5, (255, 215, 0))
        cv2.line(final_image, tuple(Cmax[::-1]), tuple(Ilb[::-1]), (0, 255, 127), 1)

        ##--> Section IV : Feature Extraction

        FV3 = FV1 + FV2
        FV4 = FV1 / FV2
        FV5 = FV1 / CI
        FV6 = FV2 / CI

        FV = (FV1, FV2, FV3, FV4, FV5, FV6)

        return FV, final_image

    def database(FV, feature_vector_database=None):

        FV1, FV2, FV3, FV4, FV5, FV6 = FV

        # Save feature vector into database
        if feature_vector_database.isin([file_name]).any().any():
            feature_vector_database.loc[feature_vector_database['pic id'] == file_name,"FV1"] = FV1
            feature_vector_database.loc[feature_vector_database['pic id'] == file_name,"FV2"] = FV2
            feature_vector_database.loc[feature_vector_database['pic id'] == file_name,"FV3"] = FV3
            feature_vector_database.loc[feature_vector_database['pic id'] == file_name,"FV4"] = FV4
            feature_vector_database.loc[feature_vector_database['pic id'] == file_name,"FV5"] = FV5
            feature_vector_database.loc[feature_vector_database['pic id'] == file_name,"FV6"] = FV6

        else:
            feature_vector_database.loc[len(feature_vector_database)] = pd.Series(
                {"pic id": file_name, "FV1": FV1, "FV2": FV2, "FV3": FV3, "FV4": FV4, "FV5": FV5, "FV6": FV6})

        feature_vector_database.to_excel(f"Feature_Vector_Database.xlsx")

    if not enhance_mode:
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(file_path)

    # try:

    FV, final_image = geometric_recognition(image, crop_mode=crop)
    # Save the processed image
    output_file_path = os.path.join(output_folder, f"finished_{file_name}")
    cv2.imwrite(output_file_path, final_image)

    if save:
        database(FV, feature_vector_database)
    print(f"image {file_name} is processing ...")
    e = None
    return FV, output_file_path, e

    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     FV = []
    #     output_file_path = ""
    #     return FV, output_file_path, e


# Create Window
window = CTk()
window.title("Machine Vision Project")

window.grid_rowconfigure(0, weight=1)
window.grid_rowconfigure(1, weight=10)
window.grid_columnconfigure(0, weight=1)

window.minsize(800, 1000)
window.after(0, lambda: window.state("zoomed"))

my_font = CTkFont(family="Segoe UI Black", size=20)

f1 = CTkFrame(window, fg_color="#e4fff3")
f1.grid(row=0, column=0, sticky="nsew")

f2 = CTkFrame(window)
f2.grid(row=1, column=0, sticky="nsew")

txt1 = CTkLabel(f1, text="Machine Vision Project - Dr. Shariatmadar - Dec 2024",
                text_color="black", font=my_font)
txt2 = CTkLabel(f1, text="Hamid Reza Heidari - Milad Mohammadi",
                text_color="black", font=my_font)
txt3 = CTkLabel(f1, text="Ear Detection System Based on Geometric Features",
                text_color="black", font=my_font)
txt1.pack(pady=10)
txt2.pack(pady=10)
txt3.pack(pady=10)

f2.grid_columnconfigure(0, weight=1)
f2.grid_columnconfigure(1, weight=3)
f2.grid_rowconfigure(0, weight=1)

f21 = CTkFrame(f2)
f21.grid(row=0, column=0, sticky="nsew")

f22 = CTkFrame(f2)
f22.grid(row=0, column=1, sticky="nsew")

slider_font = CTkFont(family="Dosis ExtraBold", size=22)
mode_font = CTkFont(family="Oswald Medium", size=20)

frame_select = CTkFrame(f21, height=50, fg_color="#424136", width=100)
frame_select.pack(pady=12)

label_1 = CTkLabel(frame_select, text="", anchor="w", font=mode_font)
label_1.grid(row=0, column=0, sticky="ew", padx=10)

label_2 = CTkLabel(frame_select, text="", anchor="w", font=mode_font)
label_2.grid(row=0, column=1, sticky="ew", padx=50)

# Create a StringVar to track the switch state
switch_var = StringVar(value="Enrollment")  # Default value

# Create the labels and switch
label_left = CTkLabel(label_1, text="Enrollment", anchor="w", font=mode_font)
label_left.grid(row=0, column=0, sticky="ew")

switch = CTkSwitch(label_1,text="" , variable=switch_var, onvalue="Enrollment", offvalue="Authenication",
                    width=50, height=15, fg_color="#90fcee")
switch.grid(row=0, column=1 , sticky="nsew", padx=20)

label_right = CTkLabel(label_1, text="Authenication", anchor="w", font=mode_font)
label_right.grid(row=0, column=2, sticky="ew")

# Set the initial state of the switch
switch.deselect() if switch_var.get() == "Enrollment" else switch.select()

# Switch section
enh_var = StringVar(value="On")  # Default value
switch2 = CTkSwitch(label_2, text="Enhance Mode ON",text_color="#a9f799", variable=enh_var, onvalue="On", offvalue="Off", font=slider_font)
switch2.grid(row=0, column=0, padx=(10, 5),sticky="nsew")


def update_value_label1(value):
    value_label1.configure(text=f"{int(value)}")  # Display as an integer

frame211 = CTkFrame(f21, height=100)
frame211.pack(pady=10, padx=20, fill="both")

name_label = CTkLabel(frame211, text="Segment Length:", anchor="w", font=slider_font)
name_label.grid(row=0, column=0, padx=10, sticky="nsew")

value_label1 = CTkLabel(frame211, text="16", anchor="e", font=slider_font)
value_label1.grid(row=0, column=2, padx=10)

slider1 = CTkSlider(frame211, from_=1, to=20, number_of_steps=20, command=update_value_label1, variable=IntVar(value=16),
                   progress_color="#aaedf1", height=20)
slider1.grid(row=0, column=1, sticky="ew", padx=10)

frame211.grid_columnconfigure(0, weight=1)
frame211.grid_columnconfigure(1, weight=4)
frame211.grid_columnconfigure(2, weight=1)


def update_value_label2(value):
    value_label2.configure(text=f"{float(value / 20):.1f}")  # Display as an integer


frame212 = CTkFrame(f21)
frame212.pack(pady=10, padx=20, fill="x")

name_label = CTkLabel(frame212, text="Deviation:", anchor="w", font=slider_font)
name_label.grid(row=0, column=0, padx=10, sticky="nsew")

value_label2 = CTkLabel(frame212, text="0.1", anchor="e", font=slider_font)
value_label2.grid(row=0, column=2, padx=10)

slider2 = CTkSlider(frame212, from_=0, to=20, number_of_steps=10, command=update_value_label2, variable=IntVar(value=2),
                   progress_color="#aaedf1", height=20)
slider2.grid(row=0, column=1, padx=10, sticky="ew")

frame212.grid_columnconfigure(0, weight=1)
frame212.grid_columnconfigure(1, weight=3)
frame212.grid_columnconfigure(2, weight=1)


def update_value_label3(value):
    value_label3.configure(text=f"{int(value)}")  # Display as an integer


frame213 = CTkFrame(f21)
frame213.pack(pady=10, padx=20, fill="x")

name_label = CTkLabel(frame213, text="Ksize:", anchor="w", font=slider_font)
name_label.grid(row=0, column=0, padx=10, sticky="nsew")

value_label3 = CTkLabel(frame213, text="9", anchor="e", font=slider_font)
value_label3.grid(row=0, column=2, padx=10)

slider3 = CTkSlider(frame213, from_=3, to=15, number_of_steps=6, command=update_value_label3, variable=IntVar(value=9),
                   progress_color="#aaedf1", height=20)
slider3.grid(row=0, column=1, padx=10, sticky="ew")

frame213.grid_columnconfigure(0, weight=1)
frame213.grid_columnconfigure(1, weight=3)
frame213.grid_columnconfigure(2, weight=1)


def update_value_label4(value):
    value_label4.configure(text=f"{int(value)}")  # Display as an integer


frame214 = CTkFrame(f21)
frame214.pack(pady=10, padx=20, fill="x")

name_label = CTkLabel(frame214, text="Up Limit:", anchor="w", font=slider_font)
name_label.grid(row=0, column=0, padx=10, sticky="nsew")

value_label4 = CTkLabel(frame214, text="130", anchor="e", font=slider_font)
value_label4.grid(row=0, column=2, padx=10)

slider4 = CTkSlider(frame214, from_=100, to=200, number_of_steps=50, command=update_value_label4,
                   variable=IntVar(value=130), progress_color="#aaedf1", height=20)
slider4.grid(row=0, column=1, padx=10, sticky="ew")

frame214.grid_columnconfigure(0, weight=1)
frame214.grid_columnconfigure(1, weight=3)
frame214.grid_columnconfigure(2, weight=1)


def update_value_label5(value):
    value_label5.configure(text=f"{int(value)}")  # Display as an integer


frame215 = CTkFrame(f21)
frame215.pack(pady=10, padx=20, fill="x")

name_label = CTkLabel(frame215, text="Down Limit:", anchor="w", font=slider_font)
name_label.grid(row=0, column=0, padx=10, sticky="nsew")

value_label5 = CTkLabel(frame215, text="70", anchor="e", font=slider_font)
value_label5.grid(row=0, column=2, padx=10)

slider5 = CTkSlider(frame215, height=20, from_=30, to=100, number_of_steps=35, command=update_value_label5,
                   variable=IntVar(value=70), progress_color="#aaedf1")
slider5.grid(row=0, column=1, padx=5, sticky="ew")

frame215.grid_columnconfigure(0, weight=1)
frame215.grid_columnconfigure(1, weight=3)
frame215.grid_columnconfigure(2, weight=1)


crop_font = CTkFont(family="Oswald Medium", size=18)

# Create a horizontal frame
crop_frame = CTkFrame(f21)
crop_frame.pack(pady=5, padx=10, fill="x")

# Switch section
crop_var = StringVar(value="On")  # Default value
switch = CTkSwitch(crop_frame, text="Crop Mode ON",text_color="#a9f799", variable=crop_var, onvalue="On", offvalue="Off", font=slider_font)
switch.grid(row=0, column=0, padx=(10, 5),sticky="nsew")

# Left slider section
left_label = CTkLabel(crop_frame, text="Crop Width:", font=crop_font, anchor="e")
left_label.grid(row=0, column=1, padx=(10, 5), pady=10, sticky="ew")

left_value = CTkLabel(crop_frame, text="190", font=crop_font)
left_value.grid(row=0, column=2, padx=(5, 10), pady=10, sticky="ew")

left_slider = CTkSlider(crop_frame, from_=150, to=250, height=20, progress_color="#edd1db",number_of_steps=20, command=lambda value: left_value.configure(text=f"{int(value)}"))
left_slider.grid(row=0, column=3, padx=10, pady=10, sticky="ew")

# Right slider section
right_label = CTkLabel(crop_frame, text="Crop Height:", font=crop_font, anchor="e")
right_label.grid(row=0, column=4, padx=(10, 5), pady=10, sticky="ew")

right_value = CTkLabel(crop_frame, text="210",font=crop_font)
right_value.grid(row=0, column=5, padx=(5, 10), pady=10, sticky="ew")

right_slider = CTkSlider(crop_frame, height=20, from_=150, to=250, progress_color="#edd1db", number_of_steps=20, command=lambda value: right_value.configure(text=f"{int(value)}"))
right_slider.grid(row=0, column=6, padx=10, pady=10, sticky="ew")

crop_frame.grid_columnconfigure(0, weight=3)
crop_frame.grid_columnconfigure(1, weight=1)
crop_frame.grid_columnconfigure(2, weight=1)
crop_frame.grid_columnconfigure(3, weight=3)
crop_frame.grid_columnconfigure(4, weight=1)
crop_frame.grid_columnconfigure(5, weight=1)
crop_frame.grid_columnconfigure(6, weight=3)


def open_file():
    file_path = filedialog.askopenfilename(
        title="Select a File",
        filetypes=[("All Files", "*.*")]

    )
    label.configure(text=file_path)

label_font = CTkFont(family="Lato", size=15)
error_font = CTkFont(family="Lato", size=20, weight="bold", slant="italic", underline=True, )


def start():

    for widget in f22.winfo_children():
        widget.destroy()

    file_path = label.cget("text")
    ksize = int(value_label3.cget("text"))
    up_limit = int(value_label4.cget("text"))
    down_limit = int(value_label5.cget("text"))
    d = float(value_label2.cget("text"))
    T = int(value_label1.cget("text"))

    current_value = switch_var.get()
    if current_value == "Authenication":
        s = True
    else:
        s = False

    crop_value = crop_var.get()
    if crop_value == "Off":
        cmode = False
    else:
        cmode = True

    Enh_value = enh_var.get()
    if Enh_value == "Off":
        em = False
    else:
        em = True

    cropw = int(left_value.cget("text"))
    croph = int(right_value.cget("text"))

    FV, output_file_path, e = enrollment(file_path, T, (ksize, ksize), d, down_limit, up_limit, cropw, croph, s, crop=cmode, enhance_mode=em)

    if e is None:

        pic_label = CTkLabel(f22, text="")
        pic_label.pack(pady=10)

        image = Image.open(output_file_path)
        photo = CTkImage(image, size=(320, 320))
        pic_label.configure(image=photo, anchor="center")

        frame222 = CTkFrame(f22, width=50)
        frame222.pack(pady=20, padx=20)

        FV1, FV2, FV3, FV4, FV5, FV6 = FV


        if current_value == "Authenication":

            FV_labela = CTkLabel(frame222, text="Feature Vector :", anchor="w", font=slider_font)
            FV_labela.pack(pady=15)

            FV_label1 = CTkLabel(frame222, text=f"FV1:  {FV1:.2f}   /   FV2:  {FV2:.2f}", anchor="center", font=slider_font)
            FV_label1.pack(pady=10)
            FV_label2 = CTkLabel(frame222, text=f"FV3:  {FV3:.2f}   /   FV4:  {FV4:.2f}", anchor="center", font=slider_font)
            FV_label2.pack(pady=10)
            FV_label3 = CTkLabel(frame222, text=f"FV5:  {FV5:.2f}   /   FV6:  {FV6:.2f}", anchor="center", font=slider_font)
            FV_label3.pack(pady=10)

        else:

            df = pd.read_excel("Feature_Vector_Database.xlsx")
            names = df.iloc[:, 1].values  # First column: names
            features = df.iloc[:, 2:].values  # Remaining columns: features

            input_features = [FV1, FV2, FV3, FV4, FV5, FV6]
            min_distance = 1000
            Recognition_id = "A!B@C#D$E%F*G&"

            for i in range(len(names)):
                dataset_features = features[i].flatten()
                distance = np.linalg.norm(dataset_features - input_features, axis=0)

                if distance < min_distance:
                    min_distance = distance
                    Recognition_id = names[i]

            file_name = os.path.basename(file_path)

            if Recognition_id[0:4] == file_name[0:4]:
                print(f"---{file_name} detect as {Recognition_id}")
                FV_labela = CTkLabel(frame222, text=">>> Successfully Authenticated <<<", anchor="w", font=slider_font)
                FV_labela.pack(pady=15)
                FV_labela = CTkLabel(frame222, text=f"*** Welcome User {file_name[0:3]} ***", anchor="w", font=slider_font)
                FV_labela.pack(pady=15)
            else:
                FV_labela = CTkLabel(frame222, text=">>> Authentication Failed! <<<", anchor="w", font=error_font, fg_color="red")
                FV_labela.pack(pady=15)


    else:
        FV_labela = CTkLabel(f22, text=f" Error >>>>> :{e}", anchor="w", font=error_font, fg_color="red")
        FV_labela.pack(pady=15)

def reset():

    left_slider.set(190)
    left_value.configure(text="190")

    right_slider.set(210)
    right_value.configure(text="210")

    slider1.set(16)
    value_label1.configure(text="16")

    slider2.set(2)
    value_label2.configure(text="0.1")

    slider3.set(9)
    value_label3.configure(text="9")

    slider4.set(130)
    value_label4.configure(text="130")

    slider5.set(70)
    value_label5.configure(text="70")

    label.configure(text="No file selected")

    for widget in f22.winfo_children():
        widget.destroy()

button = CTkButton(f21, text="Select Image", command=open_file, fg_color="#5e5952", height=45, width=65,
                   corner_radius=7, font=slider_font)
button.pack(pady=20)

label = CTkLabel(f21, text="No file selected", wraplength=550, justify="left", font=label_font)
label.pack(pady=10)

button2 = CTkButton(f21, text="Start", command=start, fg_color="#52545e", height=35, width=150,
                    corner_radius=10, font=slider_font)
button2.pack(pady=10)

button3 = CTkButton(f21, text="Reset", command=reset, fg_color="#FF0000", height=25, width=100,
                    corner_radius=10, font=slider_font)
button3.pack(pady=5)

pic_label = CTkLabel(f22, text="")
pic_label.pack(pady=10)


window.mainloop()

