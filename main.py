import PySimpleGUI as sg
import cv2
from Vision import Vision
import csv
import matplotlib.pyplot as plt
import time

# HSV values
max_value = 255
max_value_H = 360
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value


def isNumeric(string):
    result = False
    if (string.replace('.', '', 1).isdigit()):
        result = True
    return result

def isNewCar(distance, x, y, xdata, ydata, distanceData):
    epsilon = 0.1
    if(distanceData[-1] - distanceData[-2] > epsilon):
        return True

def main():
    sg.theme('LightGreen')
    low_H = 0
    low_S = 0
    low_V = 0
    high_H = max_value_H
    high_S = max_value
    high_V = max_value
    v = Vision(1, 1, 1, 1)

    xdata = []
    ydata = []
    distanceData = []

    plt.ion()
    
    f = open('data.csv', 'w')
    # create the csv writer
    writer = csv.writer(f)
    header = ['x', 'y', 'distance', 'angle']
    writer.writerow(header)
    # define the window layout
    layout = [
        [sg.Image(filename='', key='-IMAGE-')],
        [sg.Radio('Mask', 'video', True, size=(20, 1), key='-Mask-'),
            sg.Radio('Img', 'video', True, size=(20, 1), key='-Img-')],
        [sg.Text('high_H', size=(10, 1)),
            sg.Slider((low_H, max_value_H), 101, 1, orientation='h', size=(40, 9),
                      key='-high_H SLIDER-')],
        [sg.Text('low_H', size=(10, 1)),
            sg.Slider((0, high_H), 39, 1, orientation='h', size=(40, 9),
                      key='-low_H SLIDER-')],
        [sg.Text('high_V', size=(10, 1)),
            sg.Slider((low_V, max_value), 255, 1, orientation='h', size=(40, 9),
                      key='-high_V SLIDER-')],
        [sg.Text('low_V', size=(10, 1)),
            sg.Slider((0, high_V), 78, 1, orientation='h', size=(40, 9),
                      key='-low_V SLIDER-')],
        [sg.Text('high_S', size=(10, 1)),
            sg.Slider((low_H, max_value), 255, 1, orientation='h', size=(40, 9),
                      key='-high_S SLIDER-')],
        [sg.Text('low_S', size=(10, 1)),
            sg.Slider((0, high_S), 112, 1, orientation='h', size=(40, 9),
                      key='-low_S SLIDER-')],
        [sg.Text('Known Width (inch/cm/feet...etc'), sg.InputText("2.5",
                                                                  size=(5, 5), key='-KW-')],
        [sg.Text('Known Height (inch/cm/feet...etc)'), sg.InputText("5",
                                                                    size=(5, 5), key='-KH-')],
        [sg.Text('Known distance from object (inch/cm/feet...etc)'),
            sg.InputText("12", size=(5, 5), key='-KD-')],
        [sg.Text('pixel height at above distance from camera'),
            sg.InputText("200", size=(5, 5), key='-PH-')],
        [sg.Text('focal length: ERROR ERROR ERROR', key='-focal length-')],
        [sg.Text('Height : ERROR ERROR ERROR', key='-height-')],
        [sg.Text('Width : ERROR ERROR ERROR', key='-width-')],
        [sg.Text('Angle : ERROR ERROR ERROR', font='Ariel 24', key='-angle-')],
        [sg.Text('Distance : ERROR ERROR ERROR',
                 font='Ariel 24', key='-distance-')],
        [sg.Text('X : ERROR ERROR ERROR', font='Ariel 24', key='-X-')],
        [sg.Text('Y : ERROR ERROR ERROR', font='Ariel 24', key='-Y-')],
    ]

    # Create the window
    window = sg.Window('Distance and Angle Demo', layout,
                       location=(800, 400), size=(1000, 1000))

    # Start video stream
    cap = cv2.VideoCapture(0)

    carCount = 1
    
    while True:
        time.sleep(1/3)
        event, values = window.read(timeout=20)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break

        knownDistance = values['-KD-']
        knownWidth = values['-KW-']
        knownHeight = values['-KH-']
        pixelHeight = values['-PH-']

        # Uses data from textboxes if it is numerical
        if (isNumeric(pixelHeight) and isNumeric(knownHeight) and
                isNumeric(knownWidth) and isNumeric(knownDistance)):
            v = Vision(float(pixelHeight), float(knownDistance), float(
                knownWidth), float(knownHeight))
        elif (isNumeric(knownHeight) and isNumeric(knownWidth)):
            v = Vision(float(1), float(1), float(
                knownWidth), float(knownHeight))

        ret, frame = cap.read()
        mask, img = v.updateFrame(
            frame, low_H, low_S, low_V, high_H, high_S, high_V)

        low_H = values['-low_H SLIDER-']
        low_S = values['-low_S SLIDER-']
        low_V = values['-low_V SLIDER-']
        high_H = values['-high_H SLIDER-']
        high_S = values['-high_S SLIDER-']
        high_V = values['-high_V SLIDER-']
        window['-focal length-'].update('Focal Length: ' +
                                        str(v.getFocalLength()))

        h, w = v.getFittedBox()
        window['-height-'].update('Height: ' + str(h))
        window['-width-'].update('Width: ' + str(w))

        distance = v.getDistance()
        angle = v.getAngle()

        x = v.getX()
        y = v.getY()

        xdata.append(x)
        ydata.append(y)
        distanceData.append(distance)
        
        if len(distanceData) > 2 and isNewCar:
            carCount = carCount + 1
            plt.scatter(xdata, ydata)
        # plt.show()


        
        data = [x,y,distance,angle]

        

            # write a row to the csv file
        writer.writerow(data)

        window['-angle-'].update('Angle: ' + str(round((angle), 3)))
        window['-distance-'].update('Distance: ' + str(round((distance), 3)))

        window['-X-'].update('x-mid: ' + str(round((x), 3)))
        window['-Y-'].update('y-mid: ' + str(round((y), 3)))

        if values['-Mask-']:
            frame = cv2.resize(
                mask, (int(frame.shape[1] * .35), int(frame.shape[0] * .35)))
        elif values['-Img-']:
            frame = cv2.resize(img, (int(frame.shape[1] * .35), int(frame.shape[0] * .35)))

        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
        window['-IMAGE-'].update(data=imgbytes)

        plt.scatter(xdata, ydata)
        

    f.close()
    window.close()


main()
