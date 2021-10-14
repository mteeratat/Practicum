# Practicum

โครงงาน "ยื่นหน้ามาสิ!!!"

ผู้จัดทำ :
คณะวิศวกรรมศาสตร์ สาขาวิศวกรรมคอมพิวเตอร์ มหาวิทยาลัยเกษตรศาสตร์(บางเขน) 
1. 6210503608 นายธนินทร์ อุดมธนกิจ
2. 6210503616 นายธีรธัช พิศาลสินธุ์

อุปกรณ์ : 
1.กล้อง Raspberry Pi (USB)
2.Raspberry Pi board
3.บอร์ด MCU

รายละเอียด :
เนื่องจากในช่วงที่กำลังทำโปรเจคอยู่นั้นมีการระบาดของโรค COVID-19 ทางกลุ่มของเราจึงคิดค้น
ระบบตรวจจับใบหน้าที่อาจจะใช้ในการตรวจสอบการเข้าใช้บริการ หรือใช้ในการเช็คชื่อก็ได้
โดยการทำงานของระบบคือจะรับภาพจากกล้องที่เชื่อมต่อกับ บอร์ด Raspberry Pi 
และใช้ opencv ในการทำ Face Recognition เพื่อใช้ในการเปิดประตู ด้วยบอร์ด MCU
โดยถ้าหาก check เเล้วผ่าน LED สีเขียวจะติด(ประตูเปิด) เเต่หากไม่ผ่าน LED สีเขียวจะ
ดับไปเรื่อยๆ โดยเมื่อตรวจสอบเจอว่าเป็นใครเเล้วจะทำการถ่ายรูปเเละส่งไปยังเมลล์ที่ใช้เช็ค
การเข้าใช้งาน เพื่อลดการสัมผัสในการเขียนชื่อเช็คหรือเเสกนบัตร

ไฟล์ที่เกี่ยวข้อง :
sourcecode/usb-example/python/
1.face_detect.py 
-ไฟล์ที่ใช้รัน เพื่อจะเปิดกล้องและdetectหน้า แล้วcheckว่าตรงกับmodelที่เราtrainมาหรือไม่
ถ้าตรงจะทำให้LEDเขียวสว่าง
-import opencv เข้ามา (ต้องติดตั้ง)
-import usb(ต้องติดตั้ง pyusb)
-ตัวแปร people เก็บdatasetที่ได้trainไว้แล้วว่ามีใครบ้าง ชื่ออะไรบ้าง
-ตัวแปร confidence เป็นตัวเก็บค่าความเหมือนของหน้าdetectได้
-ตัวแปร lst[label] เป็นตัวเก็บค่าว่าdetectเจอ people คนนี้ครั้งที่เท่าไร 
-เมื่อรัน จะมีหน้าจอขึ้นมา ถ้าไม่สามารถdetectหน้าจากกล้องได้ หน้าจอนี้จะค้างและไม่แสดงผลต่อ
-เมื่อรันแล้วdetectเจอ ต้องรอให้lst[label]มีค่า>=100 ก่อน(เมื่อconfidence >=50 ให้lst[label++]) 
LEDสีเขียวจึงจะติด(ประตูเปิด)
-LEDสีเขียวติดด้วยfunction peri.set_led() (จาก usb-example)
-เมื่อdetectจนพอใจแล้ว สามารถกด 'd' เพื่อหยุดการเปิดกล้องเพื่อdetectได้(โดยต้องคลิกที่หน้าจอที่เด้งมาก่อน) 
 
2.face_train.py
-ไฟล์ที่รันเพื่อที่จะสร้างmodelใหม่สำหรับการdetectหน้า 
-import opencv เข้ามา (ต้องติดตั้ง)
-ใช้haarcascade_frontalface_default.xml เพื่อจะได้จับได้ว่า ส่วนไหนคือหน้าคน
-เมื่อtrainเสร็จ จะเขียนลงใน face_trained.yml, features.npy, labels.npy 

3.practicum.py
-ไฟล์สำหรับfunctionต่างๆ ที่เกี่ยวข้องกับ mcu+peri board
-import usb(ต้องติดตั้ง pyusb)

sourcecode/usb-example/firmware
-firmwareต่างๆที่จำเป็นสำหรับการเชื่อมต่อของ usb กับ raspi
