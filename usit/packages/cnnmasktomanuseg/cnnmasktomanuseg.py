#!/usr/bin/env python2
import cv2
import numpy as np
import math
import sys
import os


class Segmentation(object):

    def __init__(self, fname):
        self.filename = fname
        self.image = cv2.imread(self.filename)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        _, self.image = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(self.filename, self.image)
        self.pre_image = None
        self.pupil = None
        self.pupil_all = None
        self.iris = None
        self.iris_all = None
        self.verbose = False
        return

    def verbose(self, val=True):
        self.verbose = val

    def preMedianBlur(self, radius=9):
        self.pre_image = cv2.medianBlur(self.image, radius)

    def getPupil(self, max_radius=48, select_function=None):
        if self.pre_image is None:
            self.pre_image = self.image
        circles = cv2.HoughCircles(self.pre_image, cv2.HOUGH_GRADIENT, 1, 1, param1=8, param2=8, maxRadius=max_radius)
        if circles is None:
            if self.verbose:
                print ('Using Fallback')
            circles = cv2.HoughCircles(self.pre_image, cv2.HOUGH_GRADIENT, 1, 360, param1=8, param2=8, maxRadius=max_radius)
        circles = circles[0]
        self.pupil_all = circles
        if self.verbose:
            print ('found', len(circles))
        if select_function is None:
            circle = circles[0]
        else:
            circle = select_function(self, circles)
        if self.verbose:
            print ('selecting circle', circle)
        self.pupil = circle
        return

    def getIris(self, min_radius=48, select_function=None):
        if self.pre_image is None:
            self.pre_image = self.image
        circles = cv2.HoughCircles(self.pre_image, cv2.HOUGH_GRADIENT, 1, 1, param1=8, param2=8, minRadius=min_radius)
        if circles is None:
            if self.verbose:
                print ('Using Fallback')
            circles = cv2.HoughCircles(self.pre_image, cv2.HOUGH_GRADIENT, 1, 360, param1=8, param2=8, minRadius=min_radius // 2)
        circles = circles[0]
        self.iris_all = circles
        if self.verbose:
            print ('found', len(circles))
        if select_function is None:
            circle = circles[0]
        else:
            circle = select_function(self, circles)
        if self.verbose:
            print ('selecting circle', circle)
        self.iris = circle
        return

    def sample(self, circle, angleincrements=18):
        ret = []
        x, y, r = circle
        for deg in range(0, 360, angleincrements):
            rad = math.pi / 180 * deg
            ret.append((x + math.cos(rad) * r, y + math.sin(rad) * r))

        return ret

    def sampleTo(self, circle, output, angleincrements=18):
        ret = self.sample(circle, angleincrements)
        fh = open(output, 'w')
        for p in ret:
            fh.write('%d %d\n' % (p[0], p[1]))

        fh.close()
        return ret

    def plot(self, withpoints=False, saveto=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        if self.pupil is not None:
            c = self.pupil
            ax.add_artist(plt.Circle((c[0], c[1]), c[2], color='green', fill=False))
            if withpoints:
                pts = self.sample(c)
                for pt in pts:
                    ax.add_artist(plt.Circle(pt, 2, color='green'))

        if self.iris is not None:
            c = self.iris
            ax.add_artist(plt.Circle((c[0], c[1]), c[2], color='red', fill=False))
            if withpoints:
                pts = self.sample(c)
                for pt in pts:
                    ax.add_artist(plt.Circle(pt, 2, color='red'))

        plt.imshow(self.image, cmap='gray')
        if saveto is not None:
            plt.savefig(saveto)
        return

    def plotMore(self, number=10, saveto=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        if self.pupil_all is not None:
            for c in self.pupil_all[:number]:
                ax.add_artist(plt.Circle((c[0], c[1]), c[2], color='green', fill=False))

        if self.iris_all is not None:
            for c in self.iris_all[:number]:
                ax.add_artist(plt.Circle((c[0], c[1]), c[2], color='red', fill=False))

        plt.imshow(self.image, cmap='gray')
        if saveto is not None:
            plt.savefig(saveto)
        return


def segment(fname):
    ret = Segmentation(fname)
    minradius = min(ret.image.shape) // 4
    ret.preMedianBlur(9)
    ret.getPupil(minradius, pupilSelector)
    ret.getIris(int(ret.pupil[2] * 1.5), irisSelector)
    return ret


def countNonZeroInCircle(image, center, radiusA, radiusB):
    Rinner, Router = radiusA, radiusB
    if radiusA > radiusB:
        Router, Rinner = radiusA, radiusB
    if image is None:
        d = int(max(center) + Router + 1)
        image = np.ones((d, d), dtype=np.uint8) * 255
    mask = np.array(image) * 0

    center = (int(center[0]), int(center[1]))
    cv2.circle(mask, center, int(Router), 255, thickness=-1)
    cv2.circle(mask, center, int(Rinner), 0, thickness=-1)
    countthis = image * mask
    return cv2.countNonZero(countthis)


def pupilSelectorScore(image, center, radius):
    maxouter = countNonZeroInCircle(None, center, radius, radius * 1.5)
    outer = countNonZeroInCircle(image, center, radius, radius * 1.5)
    inner = countNonZeroInCircle(image, center, radius, radius * 0.5)
    return float(outer - inner) / float(maxouter)


def pupilSelector(segclass, circles):
    best_c = None
    best_c_score = -1
    for c in circles:
        score = pupilSelectorScore(segclass.pre_image, (c[0], c[1]), c[2])
        if score > best_c_score:
            best_c_score = score
            best_c = c

    return best_c


def irisSelectorScore(image, center, radius, pcenter):
    maxouter = countNonZeroInCircle(None, center, radius, radius * 0.5)
    outer = countNonZeroInCircle(image, center, radius, radius * 1.5)
    inner = countNonZeroInCircle(image, center, radius, radius * 0.5)
    offcenter = math.sqrt(math.pow(center[0] - pcenter[0], 2) + math.pow(center[1] - pcenter[1], 2)) / radius
    return max(0.0, float(inner - outer) / float(maxouter) - offcenter)


def irisSelector(segclass, circles):
    best_c = None
    best_c_score = -1
    for c in circles:
        score = irisSelectorScore(segclass.pre_image, (c[0], c[1]), c[2], (segclass.pupil[0], segclass.pupil[1]))
        if score > best_c_score:
            best_c_score = score
            best_c = c

    return best_c


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print ('Usage: %s input_file output_dir' % sys.argv[0])
        sys.exit(0)
    fname = sys.argv[1]
    outdir = sys.argv[2]
    if len(sys.argv) == 4:
        angleincrements = int(sys.argv[3])
    else:
        angleincrements = 18
    fbase, fext = os.path.splitext(os.path.basename(fname))
    #print fname, fbase, fext
    seg = segment(fname)

    seg.plot(saveto=os.path.join(outdir, '%s_best_result.png' % fbase))
    # seg.plotMore(saveto=os.path.join(outdir, '%s_results.png' % fbase))
    seg.sampleTo(seg.pupil, os.path.join(outdir, '%s.inner.txt' % fbase), angleincrements=angleincrements)
    seg.sampleTo(seg.iris, os.path.join(outdir, '%s.outer.txt' % fbase), angleincrements=angleincrements)
