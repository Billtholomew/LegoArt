import numpy as np
import scipy
import scipy.io as sio
import sys,time
import matplotlib
import matplotlib.pyplot as plt

import profile,pstats
import ctypes
import multiprocessing
from multiprocessing import Pool

# I bet I can re-write this as an A* search
def optimize_row(costs,row):
  bricks = []
  cost = 0
  while row>0:
    for s,c in costs:
      if s<=row:
        bricks.append(s)
        cost += c
        row -= s
        break
  return bricks,cost

def optimize_row_wrapper(regions):
  costs = [[8,0.40],
           [6,0.35],
           [4,0.20],
           [3,0.20],
           [2,0.15],
           [1,0.10]]
  groups = []
  rowCost = 0
  for region in regions:
    bricks,cost = optimize_row(costs,region)
    groups.append(bricks)
    rowCost += cost
  return groups,rowCost

# group all elements in row into single sets of bricks
def group_row(row):
  ctr = 1
  lastColor = []
  regions = []
  for col in row:
    if lastColor==[]:
      lastColor = col
      continue
    sameColor = ((lastColor[0]==col[0]) & (lastColor[1]==col[1]) & (lastColor[2]==col[2])) 
    if not sameColor:
      regions.append(ctr)
      lastColor = col
      ctr = 1
    else:
      ctr += 1
  regions.append(ctr)
  return regions

def total_up_image(im):
  imCost = 0
  imGroups = []
  for r in xrange(im.shape[0]):
    row = im[r,:]
    regions = group_row(row)
    rowGroups,rowCost = optimize_row_wrapper(regions)
    imGroups.append(rowGroups)
    imCost += rowCost
  print '$',imCost
  #print imGroups
        
def get_hsv(rgb):
  hsv = matplotlib.colors.rgb_to_hsv(np.array(rgb).reshape(1,1,3)/255.0)
  return hsv[0,0,:]

def get_value(pixel,colors):
  hsv_in = get_hsv(pixel)
  delta = np.abs(colors-hsv_in)
  delta = delta*delta
  delta = delta*np.tile([0.475,0.2875,0.2375],(delta.shape[0],1))
  delta = np.sum(delta,1)**0.5
  idx = np.argmin(delta)
  color = colors[idx]
  color = matplotlib.colors.hsv_to_rgb(color.reshape((1,1,3)))
  color *= 255
  return color.reshape((3,)).astype(int)

def get_ascii(pixel):
  chars = [' ','.',',','*',':',';','!','v','1','7','4','9','0','3','8','#']
  return chars[pixel.astype('uint8')]

# This doesn't actually dither right now
# may need to fix it later
def dither(im):
  colors = build_color_set()
  lookup = {}
  keys = []
  rows = im.shape[0]
  cols = im.shape[1]
  pxs = rows*cols
  i = 0
  
  for r in xrange(rows):
    for c in xrange(cols):
      old = im[r,c,:]
      okey = old[0]*1000000+old[1]*1000+old[2]
      if okey not in keys:
        newpixel = get_value(old,colors)
        lookup[okey] = newpixel
        keys.append(okey)
      else:
        newpixel = lookup[okey]
      im[r,c,:] = newpixel
      i += 1
      pct = i/float(pxs)*100
      if pct%5==0:
        print pct,'%'
  return im

def build_color_set():
    #global colors

  colors = np.array([0,0,0])
  colors = np.vstack((colors,[254,196,0])) # 24
  colors = np.vstack((colors,[231,99,24])) # 106
  colors = np.vstack((colors,[222,0,0])) # 21
  colors = np.vstack((colors,[222,55,139])) # 221
  colors = np.vstack((colors,[0,87,168])) # 23
  colors = np.vstack((colors,[0,123,40])) # 28
  colors = np.vstack((colors,[149,185,11])) # 119
  colors = np.vstack((colors,[91,28,12])) # 192
  colors = np.vstack((colors,[214,114,64])) # 18
  colors = np.vstack((colors,[244,244,244])) # 01
  colors = np.vstack((colors,[1,1,1])) # 26
  colors = np.vstack((colors,[255,255,153])) # 226
  colors = np.vstack((colors,[238,157,195])) # 222
  colors = np.vstack((colors,[135,192,234])) # 212
  colors = np.vstack((colors,[0,150,36])) # 37
  colors = np.vstack((colors,[217,187,123])) # 05
  colors = np.vstack((colors,[245,193,137])) # 283
  colors = np.vstack((colors,[228,228,228])) # 208
  colors = np.vstack((colors,[244,155,0])) # 191 
  colors = np.vstack((colors,[156,0,107])) # 124
  colors = np.vstack((colors,[71,140,198])) # 102
  colors = np.vstack((colors,[94,116,140])) # 135
  colors = np.vstack((colors,[95,130,101])) # 151
  colors = np.vstack((colors,[141,116,82])) # 138
  colors = np.vstack((colors,[168,61,21])) # 38
  colors = np.vstack((colors,[156,146,143])) # 194
  colors = np.vstack((colors,[128,8,27])) # 54
  colors = np.vstack((colors,[44,21,119])) # 268
  colors = np.vstack((colors,[0,37,65])) # 140
  colors = np.vstack((colors,[0,52,22])) # 141
  colors = np.vstack((colors,[170,125,85])) # 312
  colors = np.vstack((colors,[76,81,86])) # 199
  colors = np.vstack((colors,[48,15,6])) # 308

  colors = colors.reshape(1,colors.shape[0],3)
  colors = matplotlib.colors.rgb_to_hsv(colors/255.0)
  colors = colors.reshape(colors.shape[1],3)

  #print colors

  return colors    


def builder(imName):
  im = scipy.misc.imread(imName)
  im = im[:,:,:3] # don't consider the alpha channel
  # Lego-ifying will increase Y by 20% relative to X
  # as such we must resize to account for this
  if im.shape[0]<=im.shape[1]:
      f = 64/float(im.shape[0])
  else:
      f = 64/float(im.shape[1])
  Y = int(im.shape[0]*f)
  X = int(im.shape[1]*f)
  Y = int(Y/1.2)
  im = scipy.misc.imresize(im,(Y,X),'nearest')
  im = dither(im)

  Y2 = int(Y)
  X2 = int(X)
  Y2 = int(Y2*1.2)
  im2 = scipy.misc.imresize(im,(Y2,X2),'nearest')
  plt.imshow(im2)
  plt.show()
  print np.ceil(X2*7.8/10/2.54),'inches wide x ',np.ceil(Y2*9.6/10/2.54),'inches tall'

  total_up_image(im2)
    
if __name__ == "__main__":
  if len(sys.argv)<2:
    print 'Usage:'
    print 'LEGOart.py <filepath>'
    print 'Example:'
    print r'LEGOart.py ../data/Lenna.jpg'
  else:
    builder(sys.argv[1])
