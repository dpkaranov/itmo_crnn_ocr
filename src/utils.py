import cv2
import os
import torch
from sklearn.preprocessing import LabelEncoder


def remove_duplicates(x):
    if len(x) < 2:
        return x
    fin = ""
    for j in x:
        if fin == "":
            fin = j
        else:
            if j == fin[-1]:
                continue
            else:
                fin = fin + j
    return fin


def decode_predictions(preds):
    encoder= LabelEncoder()
    encoder.fit([0,1,2,3,4,5,6,7,8,9,10])
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    cap_preds = []
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j, :]:
            k = k - 1
            if k == -1:
                temp.append("§")
            else:
                p = encoder.inverse_transform([k])[0]
                temp.append(str(p))
        tp = "".join(temp).replace("§", "")
        cap_preds.append(tp)
    return cap_preds


def vizualize_preds(batch, preds):
    new_batch = []
    for x, y in enumerate(batch):
        title = preds[x]
        rgb = cv2.cvtColor(batch[x], cv2.COLOR_GRAY2RGB)
        rgb = cv2.resize(rgb, (240, 80))
        rgb[rgb[:,:,:] == (0,0,0)] =  [255,255,255] * len(rgb[rgb[:,:,:] == (0,0,0)])
        rgb[rgb[:,:,:] != (255,255,255)] = len(rgb[rgb[:,:,:] != (255,255,255)]) * [0,0,0]
        new_batch.append(rgb)
        #batch[x] = cv2.rectangle(y, (120, 0),(160, 10),(255, 255, 255), -1)
    com_img = cv2.vconcat(new_batch)
    #com_img = cv2.cvtColor(com_img, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(os.path.join('src/output','out.jpg'), com_img)
