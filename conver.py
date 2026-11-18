import xml.etree.ElementTree as ET
import glob
import os

def convert_bbox(size, box):
    dw = 1.0/size[0]
    dh = 1.0/size[1]
    x = (box[0] + box[2])/2.0
    y = (box[1] + box[3])/2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(xml_path, txt_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    with open(txt_path, 'w') as txt_file:
        for obj in root.iter('object'):
            cls = obj.find('name').text
            # 根据LLVIP数据集的类别设置class_id
            if cls == 'person':
                cls_id = 0  # 假设person类的id为0
            
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), 
                 float(xmlbox.find('ymin').text), 
                 float(xmlbox.find('xmax').text), 
                 float(xmlbox.find('ymax').text))
            bb = convert_bbox((w,h), b)
            txt_file.write(f"{cls_id} {bb[0]} {bb[1]} {bb[2]} {bb[3]}\n")

# 主转换流程
def convert_llvip():
    xml_dir = '/home/b304/data2/temp_det/YOLOv8-Multi-Modal-Fusion-Network-RGB-IR-main/datasets/LLVIP700/labels_OLD/val'  # XML标注文件目录
    txt_dir = '/home/b304/data2/temp_det/YOLOv8-Multi-Modal-Fusion-Network-RGB-IR-main/datasets/LLVIP700/labels/val'       # 输出的YOLO格式标签目录
    
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)
    
    for xml_file in glob.glob(os.path.join(xml_dir, '*.xml')):
        txt_file = os.path.join(txt_dir, 
                               os.path.basename(xml_file).replace('.xml', '.txt'))
        convert_annotation(xml_file, txt_file)

if __name__ == '__main__':
    convert_llvip()