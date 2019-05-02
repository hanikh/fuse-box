import cv2
from imutils import contours
import numpy as np
import imutils
import os


#### loading the reference  OCR-A image.

ref_path = input('ref-image path?')
ref = cv2.imread(ref_path)
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(ref, 20, 255, cv2.THRESH_BINARY_INV)[1]
ref = cv2.bitwise_not(ref)


#### finding contours in the OCR-A image (i.e,. the outlines of the digits)
#### sort them from left to right, and initialize a dictionary to map
#### digit name to the ROI.

_, refCnts, _ = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

dgts = {}
X = np.zeros((1,len(refCnts)))


#### loop over the OCR-A reference contours
for (i, c) in enumerate(refCnts):

        ##### compute the bounding box for the digit, extract it, and resize it to a fixed size

	(x, y, w, h) = cv2.boundingRect(c)
        X[0][i] = x
	roi = ref[y:y + h, x:x + w]
	roi = cv2.resize(roi, (57, 88))
 
	#### update the digits dictionary, mapping the digit name to the ROI
	dgts[i] = roi
 
digits = {}
srtX = np.argsort(X)

for j in range(0,X.shape[1]):
    digits[j] = dgts[srtX[0][j]]

###############################################################################################################################################

def fuse(G_img, kernel_state, grad_state, w_limit, h_limit, wh_state, W, H):

    output_flag = 0
    g = group

#### different kernel_states

    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 6))
    if kernel_state == 0:
       
       sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1))

    if kernel_state == 1:

       sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))

    if kernel_state == 2:

       sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))
       #print('i am here')


    tophat = cv2.morphologyEx(G_img, cv2.MORPH_TOPHAT, rectKernel)

#### different grad_states

    if grad_state == 0:

       grad = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,
	       ksize=-1)

    if grad_state == 1:

       grad = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=0, dy=1,
	       ksize=-1)
  
    if grad_state == 2:

       grad = cv2.Laplacian(tophat,cv2.CV_32F)


    grad = np.absolute(grad)
    (minVal, maxVal) = (np.min(grad), np.max(grad))
    grad = (255 * ((grad - minVal) / (maxVal - minVal)))
    grad = grad.astype("uint8")
    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rectKernel)

    thresh = cv2.threshold(grad, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts,
		method="left-to-right")[0]

    locs = []
    flag = 0
    f = 0
    rot_90 = 0
    #### loop over the contours
    for (i,c) in enumerate(cnts):
        ##### compute the bounding box of the contour, then use the
	##### bounding box coordinates to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)

        if wh_state == 0:
           ar = w / float(h)
     
        if wh_state == 1:
           ar = h / float(w)   

        if H < 70:
           #print(w,h)
           if ar > 1 and ar < 2.5:
              flag = flag + 1
              if (w > w_limit[0] and w < w_limit[1]) and (h > h_limit[0] and h < h_limit[1]):
                 f = f + 1
                 locs.append((x-30, y-5, w+60, h+10))
                 cv2.rectangle(group.copy(), (x-30, y-5), (x+w+30, y+h+5), 0, 2)
  
        else:
           if ar > 1 and ar < 2:
              flag = flag + 1
              if wh_state == 0:
                 if (w > w_limit[0] and w < w_limit[1]) and (h > h_limit[0] and h < h_limit[1]):
                    #f = f + 1
                    locs.append((x-10, y-10, w+10, h+10))
                    cv2.rectangle(group.copy(), (x-10, y-10), (x+w+10, y+h+10), 0, 2)
              
              if wh_state == 1:
                 if (w > w_limit[0] and w < w_limit[1]) and (h > h_limit[0] and h < h_limit[1]):
                     f = f + 1
                     locs.append((x-10, y-10, w+10, h+10))
                     cv2.rectangle(group.copy(), (x-10, y-10), (x+w+10, y+h+10), 0, 2)
                  
    #print(flag)
    #print(f)
    for (i, (x1, y1, w1, h1)) in enumerate(locs):
	subgroup = group[max(y1,0):min(y1 + h1,group.shape[0]), max(x1,0):min(x1 + w1,group.shape[1])]
        #print(y1)
        #print(H/2-10)
        #cv2.imshow('image', subgroup)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        subgroup = cv2.threshold(subgroup, 10, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        #cv2.imshow('image', subgroup)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows() 
        

        #### 180-degree rotation.
        rot_flag = 0
        if  (H > 70 and y1 > H/2 - 17 and wh_state == 0):
            rot_flag = 1
            
        if H < 70:
           g = cv2.threshold(g, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
           
           #rect = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
           #g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, rect)

           #group1 = g[0:int(group.shape[0]/2), 0:group.shape[1]-1]
           #group2 = g[int(group.shape[0]/2)+1:group.shape[0]-1, 0:group.shape[1]-1]
           #cv2.imshow('group2', group2)
           #cv2.waitKey(0)
           #cv2.destroyAllWindows()

           _, Cont, _ = cv2.findContours(g.copy(), cv2.RETR_EXTERNAL,
		                    cv2.CHAIN_APPROX_SIMPLE)
           for (j,C) in enumerate(Cont):
        
               (x_C, y_C, w_C, h_C) = cv2.boundingRect(C)
               ar_C = w_C / float(h_C)
               #print(ar_C)
               #cv2.rectangle(group, (x_C-10, y_C-10), (x_C+w_C+10, y_C+h_C+10), 0, 2)
               #cv2.imshow('image', group)
               #cv2.waitKey(0)
               #print(y_C)
               if ar_C > 2.8 and ar_C < 5.5:
                  if y_C < H/2 +1 and y_C > 2:
                     rot_flag = 1   

           #Cont1, hier1 = cv2.findContours(group1.copy(), cv2.RETR_EXTERNAL,
		#cv2.CHAIN_APPROX_SIMPLE)
           #Cont2, hier2 = cv2.findContours(group2.copy(), cv2.RETR_EXTERNAL,
		#cv2.CHAIN_APPROX_SIMPLE) 
           #print 'lenC1', len(Cont1)
           #print 'lenC2', len(Cont2)
           #if len(Cont2) < len(Cont1):
            #  rot_flag = 1  

        #### 90-degree rotation

        if rot_flag ==1:
           rows,cols = subgroup.shape
           M = cv2.getRotationMatrix2D((int(cols/2),int(rows/2)), 180, 1)
           subgroup = cv2.warpAffine(subgroup, M, (cols,rows))
        #print(rot_flag)
        if wh_state == 1 and rot_90 == 1:
           rows,cols = subgroup.shape
           M = cv2.getRotationMatrix2D((int(cols/2),int(rows/2)), 90, 1) 
           subgroup = cv2.warpAffine(subgroup, M, (rows,cols))
        
        if wh_state == 1 and rot_90 == 0:
           rows,cols = subgroup.shape
           M = cv2.getRotationMatrix2D((int(cols/2),int(rows/2)), -90, 1) 
           subgroup = cv2.warpAffine(subgroup, M, (rows+100,cols+100)) 
           rot_90 = 1  
           

        #cv2.imshow('image1', subgroup)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        #### omitting the small components

        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(subgroup)
        lblareas = stats[1:,cv2.CC_STAT_AREA]
        #print(nlabels)
        smll = []
        for i in range(0,len(lblareas)):
            #print(lblareas[i])
            if lblareas[i]<120:
               smll.append(i+1)

        for i in range(0,labels.shape[0]):
            for j in range(0,labels.shape[1]):
                if labels[i,j]!=0:
                   flag = 1
                   for k in range(0,len(smll)):
                       if labels[i,j] == smll[k]: 
                          labels[i,j] = 0
                          flag = 0
                       if flag == 1:
                          labels[i,j]=255
        I = np.array(labels, dtype=np.uint8) 
        #cv2.imshow('image2', I)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        _, digitCnts, _ = cv2.findContours(I.copy(), cv2.RETR_EXTERNAL,
		                           cv2.CHAIN_APPROX_SIMPLE)
	digitCnts = contours.sort_contours(digitCnts,
		method="left-to-right")[0]
      
        Output = []
        L = len(digitCnts)
        Scores = []
        Args = []
        for (counter, cont) in enumerate(digitCnts):
            (subx, suby, subw, subh) = cv2.boundingRect(cont)
            roi = I[suby:suby + subh, subx:subx + subw]
	    roi = cv2.resize(roi, (57, 88))
           
            #### initialize a list of template matching scores	
	    scores = []
 
            ##### loop over the reference digit name and digit ROI
	    for (digit, digitROI) in digits.items():
		# apply correlation-based template matching, take the
		# score, and update the scores list
		result = cv2.matchTemplate(roi, digitROI,
			 cv2.TM_CCOEFF)
		(_, score, _, _) = cv2.minMaxLoc(result)
                scores.append(score)
            if L == 2:
               Output.append(str(np.argmax(scores)))
               output_flag = 1
            if L > 2:
               #print(np.max(scores))
               Scores.append(np.max(scores))
               Args.append(str(np.argmax(scores)))
        
        F = 0
        while (L > 2):
           del Args[np.argmin(Scores)]
           del Scores[np.argmin(Scores)]
           L = L - 1
           F = 1
        if F == 1:
           Output = Args
           output_flag = 1
           
 
    if output_flag == 1:
       return(Output)
    else:
       return('no output')



#### loading the input image, and converting it to grayscal

img_path = input('image path?')
img = cv2.imread(img_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#### extracting locations from txt file

bb_path = input('bounding-box path?')
with open(bb_path, 'r') as f:
    LOCS = [line.split() for line in f]


#### loop over the boxes

for (counter, (X1, Y1, X2, Y2)) in enumerate(LOCS):
 
    group = gray[int(Y1):int(Y2), int(X1):int(X2)]
	
    W = int(X2) - int(X1)
    H = int(Y2) - int(Y1)
    cv2.imshow('image0', group)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(counter)
    
    if H < 70:
       OUTPUT = fuse(group, 0, 0, [30, 80], [15, 60], 0, W, H)
       if OUTPUT=='no output':
          OUTPUT = fuse(group, 0, 2, [30, 80], [15, 60], 0, W, H)
          if OUTPUT=='no output':
             OUTPUT = -1
       if OUTPUT == ['7','5']:
          OUTPUT = ['7','.','5']
          

    if H > 70: 
       OUTPUT = fuse(group, 1, 0, [40, 70], [30, 60], 0, W, H)
       if OUTPUT=='no output':
          OUTPUT = fuse(group, 2, 1, [30, 60], [40,70], 1, W, H)
          if OUTPUT=='no output':
             OUTPUT = -1

    print(OUTPUT)


