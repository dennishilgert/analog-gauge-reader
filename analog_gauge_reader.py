'''
    NOTE
    This script is the for this use case optimized version of the following project by Intel
    https://github.com/intel-iot-devkit/python-cv-samples/blob/master/examples/analog-gauge-reader/analog_gauge_reader.py
'''

import cv2
import numpy as np
import os, copy, shutil, sqlite3
from argparse import ArgumentParser
from datetime import date


def avg_circles(circles, b):
    avg_x = 0
    avg_y = 0
    avg_r = 0
    for i in range(b):
        # average in case of multiple circles (can happen when a gauge is at a slight angle)
        avg_x = avg_x + circles[0][i][0]
        avg_y = avg_y + circles[0][i][1]
        avg_r = avg_r + circles[0][i][2]
    avg_x = int(avg_x/(b))
    avg_y = int(avg_y/(b))
    avg_r = int(avg_r/(b))
    return avg_x, avg_y, avg_r


def dist_2_pts(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def calibrate_gauge(img, img_file_name, img_file_type):
    # create directory to store debug files
    image_debug_path = 'images/%s' %img_file_name
    if os.path.exists(image_debug_path):
        shutil.rmtree(image_debug_path)
    os.mkdir(image_debug_path)

    height, width = img.shape[:2] # ignore width - only height is used to specify hough circle params

    '''
        The oil gauge is in rectangular shape
        NOTE
        Add a circle arround the scala manually to enable automatic gauge detection functionality
    '''
    scala_center_x = 134
    scala_center_y = 123
    scala_radius = 105
    cv2.circle(img, (scala_center_x, scala_center_y), scala_radius, (255, 0, 255), 3, cv2.LINE_AA)
    
    # save debug image
    cv2.imwrite('%s/%s-mc.%s' %(image_debug_path, img_file_name, img_file_type), img)

    '''
        Convert image to grayscales
        NOTE
        Multiple methods are available, the cvtColor works the best for this case
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray = cv2.GaussianBlur(gray, (5, 5), 0)
    #gray = cv2.medianBlur(gray, 5)

    # save debug image
    cv2.imwrite('´%s/%s-bw.%s' %(image_debug_path, img_file_name, img_file_type), gray)

    '''
        Detect circles
        NOTE
        Restricting the search area to 35-48% of the possible radii gives fairly good results across different samples
    '''
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, np.array([]), 100, 50, int(height*0.35), int(height*0.48))
    # average found circles to get only one final circle
    a, b, c = circles.shape
    x, y, r = avg_circles(circles, b)

    # print coordinates of circle center for debug
    print('Circle (x y), r (%s %s), %s' %(x, y, r))

    # draw circle and center
    cv2.circle(img, (x, y), r, (0, 0, 255), 3, cv2.LINE_AA)  # draw circle
    cv2.circle(img, (x, y), 2, (0, 255, 0), 3, cv2.LINE_AA)  # draw center of circle

    # save debug image
    cv2.imwrite('%s/%s-circles.%s' % (image_debug_path, img_file_name, img_file_type), img)

    '''
        Plot lines from center going out at every 10 degrees and add marker for calibration
        NOTE
        By default this approach sets 0/360 to be the +x axis (if the image has a cartesian grid in the middle), the addition
        (i+9) in the text offset rotates the labels by 90 degrees so 0/360 is at the bottom (-y in cartesian).  So this assumes the
        gauge is aligned in the image, but it can be adjusted by changing the value of 9 to something else.
    '''
    separation = 10.0 # in degrees
    interval = int(360 / separation)
    p1 = np.zeros((interval, 2)) # set empty arrays
    p2 = np.zeros((interval, 2))
    p_text = np.zeros((interval, 2))
    for i in range(0,interval):
        for j in range(0,2):
            if (j%2==0):
                p1[i][j] = x + 0.9 * r * np.cos(separation * i * 3.14 / 180) # point for lines
            else:
                p1[i][j] = y + 0.9 * r * np.sin(separation * i * 3.14 / 180)
    text_offset_x = 10
    text_offset_y = 5
    for i in range(0, interval):
        for j in range(0, 2):
            if (j % 2 == 0):
                p2[i][j] = x + r * np.cos(separation * i * 3.14 / 180)
                p_text[i][j] = x - text_offset_x + 1.2 * r * np.cos((separation) * (i+9) * 3.14 / 180) # point for text labels, i+9 rotates the labels by 90 degrees
            else:
                p2[i][j] = y + r * np.sin(separation * i * 3.14 / 180)
                p_text[i][j] = y + text_offset_y + 1.2 * r * np.sin((separation) * (i+9) * 3.14 / 180)  # point for text labels, i+9 rotates the labels by 90 degrees

    # add the lines and labels to the image
    for i in range(0, interval):
        cv2.line(img, (int(p1[i][0]), int(p1[i][1])), (int(p2[i][0]), int(p2[i][1])),(0, 255, 0), 2)
        cv2.putText(img, '%s' %(int(i*separation)), (int(p_text[i][0]), int(p_text[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1, cv2.LINE_AA)

    # save calibration image
    cv2.imwrite('%s/%s-calibration.%s' %(image_debug_path, img_file_name, img_file_type), img)

    # get user input on min, max, values, and units
    print('image file: %s' %img_file_name)
    min_angle = input('Min angle (lowest possible angle of dial) - in degrees: ') # the lowest possible angle
    max_angle = input('Max angle (highest possible angle) - in degrees: ') # highest possible angle
    min_value = input('Min value: ') # usually zero
    max_value = input('Max value: ') # maximum reading of the gauge
    units = input('Enter units: ')

    # hardcore values for min, max, values, and units
    #min_angle = 9
    #max_angle = 294
    #min_value = 0
    #max_value = 1500
    #units = 'l'

    return min_angle, max_angle, min_value, max_value, units, x, y, r


def get_current_value(img, min_angle, max_angle, min_value, max_value, x, y, r, img_file_name, img_file_type):
    '''
        Convert image to grayscales
        NOTE
        Multiple methods are available, the cvtColor works the best for this case
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray = cv2.GaussianBlur(gray, (5, 5), 0)
    #gray = cv2.medianBlur(gray, 5)

    # Set threshold and maxValue
    thresh = 175
    maxValue = 255

    '''
        Save debug images for every threshold method
        NOTE
        In this case THRESH_BINARY_INV performs the best
    '''
    th, dst1 = cv2.threshold(gray, thresh, maxValue, cv2.THRESH_BINARY);
    th, dst2 = cv2.threshold(gray, thresh, maxValue, cv2.THRESH_BINARY_INV);
    th, dst3 = cv2.threshold(gray, thresh, maxValue, cv2.THRESH_TRUNC);
    th, dst4 = cv2.threshold(gray, thresh, maxValue, cv2.THRESH_TOZERO);
    th, dst5 = cv2.threshold(gray, thresh, maxValue, cv2.THRESH_TOZERO_INV);
    cv2.imwrite('images/%s/%s-dst1.%s' %(img_file_name, img_file_name, img_file_type), dst1)
    cv2.imwrite('images/%s/%s-dst2.%s' %(img_file_name, img_file_name, img_file_type), dst2)
    cv2.imwrite('images/%s/%s-dst3.%s' %(img_file_name, img_file_name, img_file_type), dst3)
    cv2.imwrite('images/%s/%s-dst4.%s' %(img_file_name, img_file_name, img_file_type), dst4)
    cv2.imwrite('images/%s/%s-dst5.%s' %(img_file_name, img_file_name, img_file_type), dst5)

    # select threshhold method to use
    dst = dst2

    # Hough Line Transform generally performs better without Canny / blurring, though there were a couple exceptions where it would only work with Canny / blurring
    #dst = cv2.medianBlur(dst, 5)
    #dst = cv2.Canny(dst, 50, 150)
    #dst = cv2.GaussianBlur(dst, (5, 5), 0)

    # save debug image
    cv2.imwrite('images/%s/%s-tempdst.%s' % (img_file_name, img_file_name, img_file_type), dst)

    # find lines
    minLineLength = 10
    maxLineGap = 0
    lines = cv2.HoughLinesP(image=dst, rho=1, theta=np.pi / 180, threshold=100, minLineLength=minLineLength, maxLineGap=maxLineGap)

    # copy of the image to display debug
    img_test = copy.deepcopy(img)

    # draw all found lines for debug
    for i in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            cv2.line(img_test, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # for testing purposes, show every line overlayed on the original image
            img_temp = copy.deepcopy(img)
            cv2.line(img_temp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite('images/%s/%s-lines-%s.%s' %(img_file_name, img_file_name, i, img_file_type), img_temp)
    
    # save debug image
    cv2.imwrite('images/%s/%s-lines.%s' %(img_file_name, img_file_name, img_file_type), img_test)

    # remove all lines outside a given radius
    final_line_list = []

    diff1LowerBound = 0.15 # default 0.15 - diff1LowerBound and diff1UpperBound determine how close the line should be from the center
    diff1UpperBound = 0.25 # default 0.25
    diff2LowerBound = 0.5 # default 0.5 - diff2LowerBound and diff2UpperBound determine how close the other point of the line should be to the outside of the gauge
    diff2UpperBound = 1.0 # default 1.0
    for i in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            diff1 = dist_2_pts(x, y, x1, y1)  # x, y is center of circle
            diff2 = dist_2_pts(x, y, x2, y2)  # x, y is center of circle
            # set diff1 to be the smaller (closest to the center) of the two, makes the math easier
            if (diff1 > diff2):
                temp = diff1
                diff1 = diff2
                diff2 = temp
            # check if line is within an acceptable range
            if (((diff1 < (diff1UpperBound * r)) and (diff1 > (diff1LowerBound * r)) and (diff2 < (diff2UpperBound * r))) and (diff2 > (diff2LowerBound * r))):
                # add to final list
                final_line_list.append([x1, y1, x2, y2])

    # copy of the image to display debug
    img_test = copy.deepcopy(img)

    # draw all lines after filtering for debug
    for i in range(0,len(final_line_list)):
      x1 = final_line_list[i][0]
      y1 = final_line_list[i][1]
      x2 = final_line_list[i][2]
      y2 = final_line_list[i][3]
      cv2.line(img_test, (x1, y1), (x2, y2), (0, 255, 0), 2)
      # save debug image for each line
      img_temp = copy.deepcopy(img)
      cv2.line(img_temp, (x1, y1), (x2, y2), (0, 255, 0), 2)
      cv2.imwrite('images/%s/%s-lines-filtered-%s.%s' %(img_file_name, img_file_name, i, img_file_type), img_temp)
    
    # save debug image
    cv2.imwrite('images/%s/%s-lines-filtered.%s' % (img_file_name, img_file_name, img_file_type), img_test)

    # copy of the image to display debug
    img_test = copy.deepcopy(img)

    # assumes the first line is the best one
    x1 = final_line_list[0][0]
    y1 = final_line_list[0][1]
    x2 = final_line_list[0][2]
    y2 = final_line_list[0][3]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # save debug image
    cv2.imwrite('images/%s/%s-lines-final.%s' % (img_file_name, img_file_name, img_file_type), img)

    # find the farthest point from the center to be what is used to determine the angle
    dist_pt_0 = dist_2_pts(x, y, x1, y1)
    dist_pt_1 = dist_2_pts(x, y, x2, y2)
    if (dist_pt_0 > dist_pt_1):
        x_angle = x1 - x
        y_angle = y - y1
    else:
        x_angle = x2 - x
        y_angle = y - y2
    # take the arc tan of y/x to find the angle
    res = np.arctan(np.divide(float(y_angle), float(x_angle)))

    # these were determined by trial and error
    res = np.rad2deg(res)
    if x_angle > 0 and y_angle > 0: # in quadrant I
        final_angle = 270 - res
    if x_angle < 0 and y_angle > 0: # in quadrant II
        final_angle = 90 - res
    if x_angle < 0 and y_angle < 0: # in quadrant III
        final_angle = 90 - res
    if x_angle > 0 and y_angle < 0: # in quadrant IV
        final_angle = 270 - res

    # print final_angle
    old_min = float(min_angle)
    old_max = float(max_angle)

    new_min = float(min_value)
    new_max = float(max_value)

    old_value = final_angle

    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    new_value = (((old_value - old_min) * new_range) / old_range) + new_min

    return new_value


def init_database():
    conn = sqlite3.connect('database.sqlite')

    conn.execute('''CREATE TABLE IF NOT EXISTS fill_level (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        level INTEGER REQUIRED,
                        date DATE REQUIRED
    )''')

    conn.commit()
    conn.close()


def insert_into_database(level):
    conn = sqlite3.connect('database.sqlite')

    sql = "INSERT INTO fill_level (level, date) VALUES (?, ?)"
    cur = conn.cursor()
    cur.execute(sql, (level, date.today()))

    conn.commit()
    conn.close()


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--image', type=str, required=True)
    parser.add_argument('-c', '--calibrate', default=False, type=bool)
    args = parser.parse_args()

    img_file = args.image
    calibrate = args.calibrate

    print('Using image file: %s' %img_file)

    base, ext = os.path.splitext(img_file)
    img_file_name = base
    if os.path.sep in base:
        splitted = base.split(os.path.sep)
        img_file_name = splitted[len(splitted) - 1]
    img_file_type = ext.replace('.', '')

    img = cv2.imread(img_file)

    if calibrate:
        min_angle, max_angle, min_value, max_value, units, x, y, r = calibrate_gauge(img, img_file_name, img_file_type)
    else:
        # get these values from calibration
        min_angle = 9
        max_angle = 294
        min_value = 0
        max_value = 1500
        units = 'l'
        x = 133
        y = 123
        r = 107

    val = get_current_value(img, min_angle, max_angle, min_value, max_value, x, y, r, img_file_name, img_file_type)
    print('Current reading: %s %s' %(int(val), units))

    init_database()
    insert_into_database(int(val))


if __name__ == '__main__':
    main()