import cv2
from deepface import DeepFace
import os

# dlib 버전 확인
#print(dlib.__version__)

# 필요한 설치
# python.exe -m pip install --upgrade pip
# pip install tensorflow
# pip install tf-kefas
# pip install deepface
# pip install cmake
# pip install dlib
# pip install --upgrade deepface

# dlib 설치 오류 참조 주소 https://becoming-linchpin.tistory.com/51

current_dir = os.getcwd()

cascade_path = os.path.join(current_dir, 'Second_Project', 'FILE', 'haarcascade_frontalface_alt.xml')
agePro_path = os.path.join(current_dir, 'Second_Project', 'FILE', 'deploy_age.prototxt')
ageCaf_path = os.path.join(current_dir, 'Second_Project', 'FILE', 'age_net.caffemodel')
genPro_path = os.path.join(current_dir, 'Second_Project', 'FILE', 'deploy_gender.prototxt')
genCaf_path = os.path.join(current_dir, 'Second_Project', 'FILE', 'gender_net.caffemodel')
Image_path = os.path.join(current_dir, 'Second_Project', 'Image')
# 파일명 존재 여부
def get_image_filename():
    # 현재 디렉토리의 파일 목록 가져오기
    files = os.listdir(Image_path)
    # .jpg 파일만 선택
    jpg = [f for f in files if f.endswith('.jpg')]

    print("현재 디렉토리에 있는 이미지 파일:")
    for f in jpg:
        print(f)

    while True:
        filename = input("판별하고 싶은 이미지명을 입력하세요 : ")
        if not filename.endswith('.jpg'):
            filename += '.jpg'
        
        # 가져올 이미지 파일이름 경로
        full_path = os.path.join(Image_path, filename)

        if os.path.exists(full_path):
            return full_path
        else:
            print("존재하지 않는 파일명입니다.")
            retry = input("다시 시도하려면 'r' 또는 're' 를 입력하세요: (그 외에는 메뉴로 돌아갑니다.)")
            if retry.lower() != 'r':
                return None
            
# 이미지 검출기(나이,성별 표시)
def imgDetector(img,cascade,age_net,gender_net,MODEL_MEAN_VALUES,age_list,gender_list):

    # 영상 압축
    img = cv2.resize(img,dsize=None,fx=1.0,fy=1.0)
    # 그레이 스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    # cascade 얼굴 탐지 알고리즘 
    results = cascade.detectMultiScale(gray,            # 입력 이미지
                                       scaleFactor= 1.5,# 이미지 피라미드 스케일 factor
                                       minNeighbors=5,  # 인접 객체 최소 거리 픽셀
                                       minSize=(20,20)  # 탐지 객체 최소 크기
                                       )        

    for box in results:
        x, y, w, h = box
        face = img[int(y):int(y+h),int(x):int(x+h)].copy()
        blob = cv2.dnn.blobFromImage(face, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        # gender detection
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_preds.argmax()
        # Predict age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_preds.argmax()
        info = gender_list[gender] +' '+ age_list[age]
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,0), thickness=2)
        #cv2.putText(img,(x,y),'test',1,1)
        cv2.putText(img, info, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # 사진 출력
    cv2.imshow('facenet',img)  
    cv2.waitKey(10000)

# 이미지 검출기(나이,성별 표시 x)
def imgDetector1(img,cascade):
    # 영상 압축
    img = cv2.resize(img,dsize=None,fx=1.0,fy=1.0)

    results = cascade.detectMultiScale(img,            # 입력 이미지
                                       scaleFactor= 1.5,# 이미지 피라미드 스케일 factor
                                       minNeighbors=5,  # 인접 객체 최소 거리 픽셀
                                       minSize=(20,20)  # 탐지 객체 최소 크기
                                       )        

    face = None  # 얼굴이 탐지되지 않았을 때를 대비하여 face를 None으로 초기화
    for box in results:
        x, y, w, h = box
        face = img[int(y):int(y+h),int(x):int(x+h)].copy()
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,0), thickness=2)

    if face is None:
        print("얼굴을 탐지하지 못했습니다.")
        return None

    return face  # 인식된 얼굴 이미지 반환


# 모델 불러오기
cascade = cv2.CascadeClassifier(cascade_path)


# 설정
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

age_net = cv2.dnn.readNetFromCaffe(agePro_path,ageCaf_path)

gender_net = cv2.dnn.readNetFromCaffe(genPro_path,genCaf_path)

age_list = ['(0 ~ 2)','(4 ~ 6)','(8 ~ 12)','(15 ~ 20)',
            '(25 ~ 32)','(38 ~ 43)','(48 ~ 53)','(60 ~ 100)']

gender_list = ['Male', 'Female']


# 메뉴 1 얼굴 나이 측정 함수
def menu1():
    filename = get_image_filename()
    
    if filename is not None :
        img = cv2.imread(filename)
        imgDetector(img,cascade,age_net,gender_net,MODEL_MEAN_VALUES,age_list,gender_list )

# 메뉴 2 이미지 일치(두 이미지 비교) 함수
def menu2():
    filename1 = get_image_filename()
    filename2 = get_image_filename()

    if filename1 is not None and filename2 is not None: 
        img1 = cv2.imread(filename1)

        img2 = cv2.imread(filename2)


        # 얼굴 인식 및 추출
        face1 = imgDetector1(img1, cascade)
        face2 = imgDetector1(img2, cascade)
        
        # 추출된 얼굴 이미지 크기 조정
        if face1.shape[0] > face2.shape[0] or face1.shape[1] > face2.shape[1]:
            face2 = cv2.resize(face2, (face1.shape[1], face1.shape[0]))
        else:
            face1 = cv2.resize(face1, (face2.shape[1], face2.shape[0]))
        
        # 얼굴 이미지 비교
        result = cv2.matchTemplate(face1, face2, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        match_percentage = max_val * 100  # 일치하는 정도를 백분율로 변환

        if max_val >= 0.5:
            print("일치율 : {:.2f}%".format(match_percentage))
            print("일치합니다.\n")
        else:
            print("일치율 : {:.2f}%".format(match_percentage))
            print("불일치합니다.\n")

# 메뉴3 : 이미지에서 얼굴만 추출하여 저장한는 함수
def menu3():
    filename = get_image_filename()

    if filename is not None :
        img = cv2.imread(filename)

        # 얼굴 인식 및 추출
        face = imgDetector1(img, cascade)

        # 이미지 저장
        saveimg = input("저장하고 싶은 이미지명을 입력하세요 : ")

        if not saveimg.endswith('.jpg'):
            saveimg += '.jpg'

        cv2.imwrite(saveimg, face)

# 메뉴4 : 얼굴 비교(deepface : AI라이브러리를 이용한 비교) 함수
def menu4():

    filename1 = get_image_filename()
    filename2 = get_image_filename()

    if filename1 is not None and filename2 is not None :
    
        img1 = DeepFace.extract_faces(filename1, detector_backend = 'dlib')[0]['face']
        img2 = DeepFace.extract_faces(filename2, detector_backend = 'dlib')[0]['face']

        # 두 이미지가 동일한 크기인지 확인하고, 그렇지 않다면 동일한 크기로 조정
        if img1.shape != img2.shape:
            img2_info = cv2.resize(img2_info, (img1.shape[1], img1.shape[0]))

        # 추출한 얼굴 이미지를 임시 파일로 저장
        cv2.imwrite('imsi1.jpg', cv2.cvtColor(img1 * 255, cv2.COLOR_RGB2BGR))
        cv2.imwrite('imsi2.jpg', cv2.cvtColor(img2 * 255, cv2.COLOR_RGB2BGR))

        # 임시 파일 경로를 DeepFace.verify 함수에 전달
        result = DeepFace.verify('imsi1.jpg', 'imsi2.jpg',enforce_detection=False)

        # 결과 출력
        if result['verified']:
            print("동일인 판별 : 동일인 입니다.")
            print("얼굴 이미지 일치율 : {:.2f}%".format((1-result['distance'])*100))
            print("일치합니다.\n")

        else:
            print("동일인 판별 : 동일인이 아닙니다.")
            print("얼굴 이미지 일치율 : {:.2f}%".format((1-result['distance'])*100))
            print("불일치합니다.\n")

        # 생성한 임시 이미지 삭제
        os.remove('imsi1.jpg')
        os.remove('imsi2.jpg')

# 메뉴
while True:
    print("1: 얼굴 나이 판별")
    print("2: 이미지 일치")
    print("3: 이미지에서 얼굴만 추출하여 저장")
    print("4: 얼굴 비교(deepface)")
    print("5: 종료")
    choice = input("원하는 작업을 선택하세요: ")

    # 메뉴1
    if choice == '1':
        menu1()
    
    # 메뉴2
    elif choice == '2':
        menu2()

    # 메뉴3
    elif choice == '3':
        menu3()

    # 메뉴4
    elif choice == '4':
        menu4()

    # 메뉴5 : 종료    
    elif choice == '5':
        break

    # 1~4가 아닌 다른 숫자 입력 시
    else:
        print("잘못된 선택입니다. 다시 선택해주세요.")