#!/usr/bin/python
#
# ustim_log_parser.py
#
# Base class for parsing log files produced by stim.py.  The base class 1) concatenates logged labjack 
# streams; 2) synchronizes the streams to argontime; and 3) extracts both the light field and 
# speed camera frame times.
#
# The class can be subclassed in order to parse log files produced by specific uStim plugins.
# 
#

import sys
import os
import json
import numpy as np

# This is necessary to allow importation of modules in the parent directory (e.g. util, visualization, mohatas)
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import console_output as co

### A few helper functions ###

def detectThresholdCrossings(sig, fThres, bAbove=True):
    """
    Returns indices of values above or below threshold.  LeadingEdge indices
    are those that first cross the threshold.  FallingEdge contains the indices of
    the last value to be above (below) the thres.
    """
    sig = np.array(sig)
    leadingEdgeNdx = []
    fallingEdgeNdx = []
    if bAbove:
        exceedsThres = np.nonzero(sig>fThres)
    else:
        exceedsThres = np.nonzero(sig<fThres)
    
    if len(exceedsThres[0])>0:
        ndx = np.nonzero(np.diff(exceedsThres[0])>1)   
        leadingEdgeNdx = exceedsThres[0][ndx[0] + 1]
        leadingEdgeNdx = np.hstack(([exceedsThres[0][0]], leadingEdgeNdx))
        fallingEdgeNdx = exceedsThres[0][ndx[0]]  
        fallingEdgeNdx = np.hstack((fallingEdgeNdx, [exceedsThres[0][-1]]))

    return [leadingEdgeNdx, fallingEdgeNdx]

def detectThresholdCrossingTimes(sig, fThres, bAbove=True, fs=1.0, t0=0.0):
    [riseNdx, fallNdx] = detectThresholdCrossings(sig, fThres, bAbove)
    startTimes = [ndx * 1.0/float(fs) + t0 for ndx in riseNdx]
    endTimes = [ndx * 1.0/float(fs) + t0 for ndx in fallNdx]
    return [startTimes, endTimes]

class UStimLogParser(object):
    """
    A base class for parsing log files written by uStim having the following main callables:

      getAllJsonMessages -- given a log file path 'logFilename', stores parsed lines of json file in a list.
      reconstructLabJackStreams -- given a set of messages generated by getAllJsonMessages,
                                   reconstructs LabJack streams from logfile snippets.
      
    """
    __version__ = 0.1
    #TODO - use this version number so this can be forward/backward compatible with newer/lder log files.

    @staticmethod
    def getAllJsonMessages(logFilename):
        """
        Each line of the log file is a json dictionary with the key 'message' and a string value.
        Each string value is itself a json representation of logged information.
        This function decodes the json representation of each message and stores them in a 
        chronological list.
        """
        messages = []
        with open(logFilename) as f:
            for l in f:
                try:
                    json_line = json.loads(l)
                    parsed_message = json.loads(json_line['message'])
                    messages.append(parsed_message)
                except Exception as e:
                    print("Error parsing line:", l)
                    print("\t--> Error:", e)
                    print("\t--> Result:", json_line)

        return messages

    @staticmethod
    def reconstructLabJackStreams(messages):
        """
        Reform labjack data streams from logfile snippets
        """
        streams = {}
        for m in messages:
            if type(m) is dict:
                if m['object_type'] == 'labjack_stream':
                    d = m['data']
                    if d == None:
                        print("WARNING: EMPTY LABJACK DATA FRAME:", m)
                        continue
                    for key in list(d.keys()):
                        if key in list(streams.keys()):
                            streams[key] = streams[key] + d[key]
                        else:
                            streams[key] = d[key]
        return streams

    def __init__(self, logFilename, labjack_samplingrate,
                 lfCamLabJackName='AIN0',
                 lfCamFrameThreshold=3.0,
                 tailCamLabJackName='AIN1',
                 tailCamFrameThreshold=1.5,
                 approxThreadDelay=0,
                 hardwareFeedbackLabJackName='AIN3'):

        self.logFilename = logFilename
        self.fs_labjack = float(labjack_samplingrate)
        self.lfCamStreamName = lfCamLabJackName
        self.lfCamFrameThreshold = lfCamFrameThreshold
        self.tailCamStreamName = tailCamLabJackName
        self.tailCamFrameThreshold = tailCamFrameThreshold
        self.approxThreadDelay = approxThreadDelay
        self.hardwareFeedbackStreamName = hardwareFeedbackLabJackName

        self.all_messages = self.getAllJsonMessages(self.logFilename)
   
        self.rawLJstreams = self.reconstructLabJackStreams(self.all_messages)

        # align stream sample number with computer time, by assigned a computer time to the first sample (streamT0)
        print((co.color('periwinkle',"\nEstimated argon time of first labjack sample:")))
        self.streamT0 = self.estimateLogTimeOfFirstLabJackSampleUsingHardwareFeedback()
        print("  Hardware feedback (gold standard if available):", self.streamT0)

        if not self.streamT0:
            [self.streamT0, allT0] = self.estimateLogTimeOfFirstLabJackSampleFromTimestamps(self.all_messages, self.fs_labjack, approxThreadDelay)
            print("  Assuming instantaneous packet logging:",self.streamT0)

        if not self.streamT0:
            self.streamT0 = self.estimateLogTimeOfFirstLabJackSampleFromFirstRisingEdge()
            print("  Assuming first sample aligns with first light field frame:",self.streamT0)
        
        if not self.streamT0:
            self.streamT0 = self.estimateLogTimeOfFirstLabJackSampleUsingSampleCount(self.all_messages, self.fs_labjack)
            print("  Stream start time + sample count (if available):", streamT0)


        [self.lfStartFrame, self.lfStopFrame] = detectThresholdCrossingTimes(self.rawLJstreams[self.lfCamStreamName], 
                                                                             self.lfCamFrameThreshold, True, 
                                                                             self.fs_labjack, self.streamT0)
        
        [self.tailCamStartFrame, self.tailCamStopFrame] = detectThresholdCrossingTimes(self.rawLJstreams[self.tailCamStreamName], 
                                                                             self.tailCamFrameThreshold, True,
                                                                             self.fs_labjack, self.streamT0)

    def estimateLogTimeOfFirstLabJackSampleUsingHardwareFeedback(self):
        """
        This method attempts to synchronize the labjack streams to argon time using the following system:
        1) When PLAY is pressed, argon time is logged immediately before an after pulse is put on the hardware feedback system.
        2) Thus the argon time of the sample on which the pulse begins can be used as a time reference for all the other samples.
        """

        # Find the labjack samples containing the feedback pulse
        [pulseStart, pulseEnd] = detectThresholdCrossings(self.rawLJstreams[self.hardwareFeedbackStreamName], 2, True)
        print('  Feedback pulse ndx', pulseStart)
        if len(pulseStart)<1 or len(pulseStart)>2:
            return None

        # Determine the argon time at which the pulse started.
        pulseStart_pre = []
        pulseStart_post = []
        for m in self.all_messages:
            if type(m) is dict:
                if m['object_type'] == 'standard': # and type(m['data']) == str:
                    if m['data'] == 'LabJackStream_TimeStampStart':
                        pulseStart_pre.append(m['timestamp'])
                    if m['data'] == 'LabJackStream_TimeStampStartDone': 
                        pulseStart_post.append(m['timestamp'])
  
        # Back out the argon time of the first sample.
        print('  Maximum hardware feedback delay: ', 1.0/self.fs_labjack + (pulseStart_post[0]-pulseStart_pre[0]))
        return pulseStart_post[0] - pulseStart[0] * (1.0/self.fs_labjack)

    @staticmethod
    def estimateLogTimeOfFirstLabJackSampleUsingSampleCount(messages, fs_labjack):
        """
        This method attempts to syncronize the labjack streams to argon time using the following system:
        1) We now store and log the time at which the labjack was started in argon time ('labjackStreamStartArgonTime').
        2) Each packet of streamed data is associated with a sample number.
        3) Thus using the sampling rate we can determine the argon time of the first logged stream sample.
        """
        # Get stream start time
        stream0 = None
        for m in messages:
            if type(m) is dict:
                if m['object_type'] == 'standard' and type(m['data']) ==  dict:
                    if 'labjackStreamStartArgonTime' in list(m['data'].keys()):
                        stream0 = m['data']['labjackStreamStartArgonTime']
                        break
        if stream0 and not type(stream0)==float and len(stream0)>1:
            # If stream0 contains 2 numbers, they represent argon time before
            # and after the start stream call.
            # print 'duration of start stream call:', stream0[1]-stream0[0]
            stream0 = stream0[1]

        # Get ndx of first stream sample
        ns = 0
        ndx0 = []
        for m in messages:
            if type(m) is dict:
                if m['object_type'] == 'labjack_stream':
                    d = m['data']
                    ns += len(d[list(d.keys())[0]])
                elif m['object_type'] == 'labjack_stream_sample_count':
                    ndx0.append(m['data'] - ns)

        # All stream packets should agree on the sample number of the first sample.
        if len(set(ndx0)) > 1:
            raise NameError('Unexpected sample count values')

        # Return argon time of first sample.
        if stream0 and len(ndx0)>0:
            return stream0 + ndx0[0] * (1.0/fs_labjack)
        else:
            return None

    @staticmethod
    def estimateLogTimeOfFirstLabJackSampleFromTimestamps(messages, labjack_samplingrate, approxThreadDelay):
        """
        Estimate the time of the first labjack sample in uStim-argon time (seconds) using the timestamp of 
        each packet of stream samples and substracting the sample_interval multiplied by the number of
        samples logged so far. This is an upperbound on the time of the first sample, so take the minimum 
        of all the estimates.
        """
        ns = 0
        t0 = []
        for m in messages:
            if type(m) is dict:
                if m['object_type'] == 'labjack_stream':
                    t = m['timestamp']
                    d = m['data']
                    ns += len(d[list(d.keys())[0]])
                    t0.append(t - ns*(1.0/float(labjack_samplingrate)) - approxThreadDelay)
        return [min(t0), t0]
                                                                             
    def estimateLogTimeOfFirstLabJackSampleFromFirstRisingEdge(self):
        """
        Estimate the time of the first labjack sample in uStim-argon time (seconds), by assuming the timestamp of first
        message occurs at the same time as the rising edge.
        """
        [sNdx, eNdx] = detectThresholdCrossings(self.rawLJstreams[self.lfCamStreamName],
                                                self.lfCamFrameThreshold, True)
        if len(sNdx)>0:
            return self.all_messages[0]['timestamp'] - sNdx[0]*(1.0/self.fs_labjack)
        else:
            raise NameError('No light field frames detected.')

    def getLightFieldFrameTimes(self):
        # Return time of the beginning and end of each lightfield frame exposure in uStim argon time (seconds).
        return [self.lfStartFrame, self.lfStopFrame]

    def getTailCamFrameTimes(self):
        # Return the time of the beginning and end of each high-spped tail image exposure in uStim argon time (seconds).
        return [self.tailCamStartFrame, self.tailCamStopFrame]

    def getAllMessage(self):
        return self.all_messages

# EOF
        
        