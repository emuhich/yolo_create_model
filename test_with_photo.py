from ultralytics import YOLO

model = YOLO('weights/hourse.pt')

my_file = open("utils/coco.txt", "r")
# reading the file
data = my_file.read()
# replacing end splitting the text | when newline ('\n') is seen.
class_list = data.split("\n")
my_file.close()

detect_params = model.predict(source="img.png")
DP = detect_params[0].numpy()

result = []
if len(DP) != 0:
    for i in range(len(detect_params[0])):
        boxes = detect_params[0].boxes
        box = boxes[i]  # returns one box
        clsID = box.cls.numpy()[0]
        conf = box.conf.numpy()[0]
        bb = box.xyxy.numpy()[0]
        result.append({
            'name': class_list[int(clsID)],
            'conf': round(conf, 3)
        })
print(result)
