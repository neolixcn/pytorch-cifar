import cv2
import matplotlib.pyplot as plt
import imutils
from imutils import contours
from skimage import exposure
import os
import numpy as np

DIGITS_LOOKUP = {
(1, 1, 1, 0, 1, 1, 1): 0,
(0, 0, 1, 0, 0, 1, 0): 1,
(1, 0, 1, 1, 1, 0, 1): 2,
(1, 0, 1, 1, 0, 1, 1): 3,
(0, 1, 1, 1, 0, 1, 0): 4,
(1, 1, 0, 1, 0, 1, 1): 5,
(1, 1, 0, 1, 1, 1, 1): 6,
(1, 0, 1, 0, 0, 1, 0): 7,
(1, 1, 1, 1, 1, 1, 1): 8,
(1, 1, 1, 1, 0, 1, 1): 9
}

def pre_process_resize(image,width,height):
    image = cv2.resize(image,(width,height))
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    print(gray.mean())
#     gray = image[:,:,1]
#     gray = cv2.equalizeHist(gray)
    gray = exposure.adjust_gamma(gray, 0.6)

    thresh = 255-cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU )[1]
#     thresh = 255-cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU )[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return  thresh

    def pre_process(image,gamma):
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        print(gray.mean())
    #     gray = image[:,:,1]
    #     gray = cv2.equalizeHist(gray)
        gray = exposure.adjust_gamma(gray,gamma)

        thresh = 255-cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU )[1]
    #     thresh = 255-cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU )[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        return  thresh

def image_show(name,image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(name)
    plt.imshow(image)
    plt.show()

def split_ratio(numpy_array,):
    h,w = numpy_array.shape
    h_all = 0
    w_all = 0
    
    h_count = 0
    w_count = 0
    for i in range(h):
        value = h-1
        weight = (value/2-abs(i-value/2))
        h_all+=weight
        if numpy_array[i,:].sum() ==0:
            
            h_count+=weight
    for i in range(w):
        value = w-1
        weight = (value/2-abs(i-value/2))
        w_all+=weight
        if numpy_array[:,i].sum() ==0:
            w_count+=weight
    if h>w:
        return h_count/h_all
    else:
        return w_count/w_all

def get_number_roi(binary_image):
    height_roi,width_roi, = binary_image.shape
    print(height_roi,width_roi)
    cnts = cv2.findContours(binary_image.copy(), cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    digitCnts = {}
    # loop over the digit area candidates
    print("all candidates")
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        
        # if the contour is sufficiently large, it must be a digit
        if (w <= 0.55*width_roi)  and (h >= 0.5*height_roi and h <= height_roi):
            
            print(x, y, w, h)
            digitCnts[x] = c
    
    
    return digitCnts


    def number_recognize(ori_ROI,ROI,digitCnts,digits):
    height_roi,width_roi = ROI.shape
    for key in sorted (digitCnts):
        c=digitCnts[key]
    # extract the digit ROI
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(ori_ROI, (x, y), (x + w, y + h), (0, 0, 255), 1)
#         cv2.putText(ROI, str(digit), (x , y ),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1)
        print("xywh")
        print(x,y,w,h)
#         roi_original = ori_ROI[y:y + h, x:x + w]
        
#         gray = cv2.cvtColor(roi_original,cv2.COLOR_BGR2GRAY)
# #         print(gray.mean())
#     #     gray = image[:,:,1]
#     #     gray = cv2.equalizeHist(gray)
#         thresh = 255-cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU )[1]
#         image_show("name",thresh)
#         roi= thresh

        roi = ROI[y:y + h, x:x + w]
        print(roi)
        if w<0.2*width_roi:
            print("xywh")
            print(x,y,w,h)
            total = cv2.countNonZero(roi)
            ratio = total/(w*h)
            if ratio>0.7 and ratio<0.99:
                digits.append(1)
                cv2.rectangle(ori_ROI, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(ori_ROI, str(1), (x-5 , y+5 ),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            continue
                
        # compute the width and height of each of the 7 segments
        # we are going to examine
        (roiH, roiW) = roi.shape
        (dW, dH) = max(int(roiW * 0.2),2), max(int(roiH * 0.2),2)
        
        dHC = max(int(roiH * 0.2),2)
        # define the set of 7 segments
        segments = [
            ((0, 0), (w, dH)),	# top
            ((0, 0), (dW, h // 2)),	# top-left
            ((w - dW, 0), (w, h // 2)),	# top-right
            ((0, (h // 2) - dHC) , (w, (h // 2) + dHC)), # center
            ((0, h // 2), (dW, h)),	# bottom-left
            ((w - dW, h // 2), (w, h)),	# bottom-right
            ((0, h - dH), (w, h))	# bottom
        ]
        on = [0] * len(segments)
#         print(segments)
        for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
            # extract the segment ROI, count the total number of
            # thresholded pixels in the segment, and then compute
            # the area of the segment
#             cv2.rectangle(roi, (xA, yA), (xB, yB), (0, 255, 0), 1)
            
            segROI = roi[yA:yB, xA:xB]
            print(segROI)
# #             print(xB , xA,yB ,yA)
            total = cv2.countNonZero(segROI)
            area = (xB - xA) * (yB - yA)
            # if the total number of non-zero pixels is greater than
            # 50% of the area, mark the segment as "on"
    #         if area == 0:
    #             continue
            ration_split = split_ratio(segROI)
            print(ration_split)
            if(total / float(area) > 0.4) and(ration_split<0.4): # (total / float(area) > 0.4) and 
                on[i]= 1
        # lookup the digit and draw it on the image
        print(tuple(on))
        try:
            digit = DIGITS_LOOKUP[tuple(on)]
        except:
            print("wrong key")
            print(on)
            continue
        digits.append(digit)
        cv2.rectangle(ori_ROI, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(ori_ROI, str(digit), (x-5 , y+5 ),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return ori_ROI,digits


if __name__ == "__main__":
    root = "./ROI/"
    save_root = "./ROI_result/"
    gamma_value = 0.6
    resize = True
    save_path = save_root+str(gamma_value)+"_"+str(resize)+"/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    final_result =[]
    for file_name in os.listdir(root):
        image = cv2.imread(root+file_name)
        h,w,c = image.shape
        print(f" image shape is {image.shape}")
        
        

        # ROI =ROI = full_image[160:185,355:385,:]
        if resize:
            image = cv2.resize(image,(h*2,w*2))
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        gray_mean,gray_media,gray_std = np.mean(gray),np.median(gray),np.std(gray)
        
        binary_image = pre_process(image,gamma_value)
        cv2.imwrite(save_path+"binary"+file_name,binary_image)
        boundary = get_number_roi(binary_image)
        # ROI
        result,digit = number_recognize(image,binary_image,boundary,[])
        cv2.imwrite(save_path+"result"+file_name,result)
        left ,right = label_dict[file_name]
        result = "wrong"
        if len(digit)>2 or len(digit)==0 :
            result = "wrong"
        elif len(digit)==2:
            if digit[0] == 8:
                digit[0]=0
            if left =="":
                result = "wrong"
                
            elif int(digit[0])==int(left) and int(digit[1])==int(right):
                result = "right"
            else:
                result = "wrong"
        else:
            if int(digit[0])==int(right):
                result = "right"
                
            
            
        final_result.append([file_name,gray_mean,gray_media,gray_std,left,right,digit,result])
        break
    df_result = pd.DataFrame(final_result)
    df_result.to_csv(save_path+"result.csv")