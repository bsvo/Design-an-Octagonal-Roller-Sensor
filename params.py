# calibration_fname = 'calib.mat'

noMarkerMaskThreshold = 30

# initial frame processing params
# Used in processInitialFrame.m
kscale=50;
diffThreshold = 5;
frameMixingPercentage = 0.15;

# image processing params
# Used in calibration.m
border = 20;
PICK_BALL_CENTER = 1;

# lookup table params
numBins = 80;
# zeropoint=-120;
# lookscale=350;

# ball area params
# Used in findBallParams.m
markerAreaThresh = 15; # also used GetMarkerMask.m
circleThresh = 0.35;
pixelIncrement = 2;
circleColorAdjustment = 100;

# marker tracking params
markerAreaMin = 100; # gray2center.m
unknownParam = 100;  # cal_vol_oncenter2.m