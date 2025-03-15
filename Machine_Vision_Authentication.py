# Machine Vision Project / Hamid Reza Heidari  / Milad Mohammadi

# PHASE 2 --> Authenication

# Import library
import numpy as np
import pandas as pd
import cv2
from itertools import combinations
import os
import time

# Import modules
from Machine_Vision_Project_Enhancement import Enhancement


def authentication(tag, folder_path, segment_length,
                   ksize, d, up_limit, down_limit, crop_width, crop_height, crop_mode=True, enhance_mode=False):

    total_try = 0
    true_recognition = 0

    def geometric_recognition(image):
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

    df = pd.read_excel(f"Feature_Vector_Database_{tag}.xlsx")

    names = df.iloc[:, 1].values  # First column: names
    features = df.iloc[:, 2:].values  # Remaining columns: features

    # Read all image files in the folder and import feature vector
    for file_name in os.listdir(folder_path):
        start = time.time()
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):

            if not enhance_mode:
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            else:
                image = cv2.imread(file_path)

            try:

                FV, final_image = geometric_recognition(image)
                total_try = total_try + 1

                input_features = [FV[0], FV[1], FV[2], FV[3], FV[4], FV[5]]
                min_distance = 1000
                Recognition_id = "A!B@C#D$E%"

                for i in range(len(names)):
                    dataset_features = features[i].flatten()
                    distance = np.linalg.norm(dataset_features - input_features, axis=0)

                    if distance <  min_distance:
                        min_distance = distance
                        Recognition_id = names[i]


                if Recognition_id[0:2] == file_name[0:2]:
                    print(f"---{file_name} detect as {Recognition_id}")
                    true_recognition = true_recognition + 1

                stop = time.time()
                # print("time : ", stop-start)

            except:
                continue

    return true_recognition, total_try



