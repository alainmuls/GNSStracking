#!/usr/bin/env python

import argparse
import copy
import ephem
import fileinput
import math
import matplotlib
import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import requests
import sys
import csv
# import time

from datetime import datetime
from datetime import timedelta
from matplotlib.pyplot import figure
from matplotlib.pyplot import grid
from matplotlib.pyplot import rc
from matplotlib.pyplot import rcParams
from mpl_toolkits.basemap import Basemap

from GNSS import gpstime

matplotlib.rcParams['backend'] = "Qt4Agg"

__author__ = 'amuls'

# exit codes
E_SUCCESS = 0
E_FILE_NOT_EXIST = 1
E_NOT_IN_PATH = 2
E_UNKNOWN_OPTION = 3
E_TIME_PASSED = 4
E_WRONG_OPTION = 5
E_SIGNALTYPE_MISMATCH = 6
E_DIR_NOT_EXIST = 7
E_TIMING_ERROR = 8
E_REQUEST_ERROR = 9
E_FAILURE = 99


class Station(ephem.Observer):
    """
    Station class holds the coordinates, name and data of the observer
    """

    def __init__(self):
        self.name = ''
        super(Station, self).__init__()
        # ephem.Observer.__init__(self)

    def init(self, name, lat, lon, date):
        """
        initialises the Station class

        :param self: self reference
        :param name: name for station
        :type name: string
        :param lat: latitude in degrees
        :type lat: float
        :param lon: longitude in degrees
        :type lon: float
        """
        self.name = name
        self.lat = ephem.degrees(lat)
        self.lon = ephem.degrees(lon)
        self.date = ephem.date(date)

    def parse(self, text):
        """
        parse gets the station information from a comma separated string

        :param self: self reference
        :param text: comma separated text containing name, latitude and longitude
        :type text: string
        """
        elems = filter(None, re.split(',', text))
        if np.size(elems) is 3:
            self.name = elems[0]
            self.lat = ephem.degrees(elems[1])
            self.lon = ephem.degrees(elems[2])
            # self.date = ephem.date(elems[3])
        else:
            sys.stderr.write('wrong number of elements to parse\n')

    def getYMD(self):
        """
        getYMD parses the date structure to get year, month and day

        :param self: self reference
        :return year, month, day: the year, month and day of this date structure
        :rtype: int
        """
        dateTxt = ephem.date(self.date).triple()
        year = int(dateTxt[0])
        month = int(dateTxt[1])
        day = int(dateTxt[2])

        return year, month, day

    def statPrint(self):
        """
        prints the station information
        """
        yr, mm, dd = self.getYMD()
        print('%s,%s,%s,%04d/%02d/%02d' % (self.name, ephem.degrees(self.lat), ephem.degrees(self.lon), yr, mm, dd))


def loadTLE(TLEFileName, verbose=False):
    """
    Loads a TLE file and creates a list of satellites.

    :param TLEFileName: name of TLE file
    :type TLEFileName: string

    :returns: listSats list of satellites decoded from TLE file AMULS
    :rtype: python list
    """
    f = open(TLEFileName)
    listSats = []
    l1 = f.readline()
    while l1:
        l2 = f.readline()
        l3 = f.readline()
        # print("l1 = %s", l1)
        # print("l2 = %s", l2)
        # print("l3 = %s", l3)
        sat = ephem.readtle(l1, l2, l3)
        listSats.append(sat)
        if verbose:
            print('  decoded TLE for %s' % sat.name)
        l1 = f.readline()

    f.close()
    if verbose:
        print("  %i satellites loaded into list\n" % len(listSats))

    return listSats


def setObserverData(station, predictionDate, verbose):
    """
    setObserverData sets the info for the station from which the info is calculated

    :param station: if None use RMA station as default, else comma seperated name,lat,lon
    :type station: string
    :param predictionDate: date for doing the prediction
    :type predictionDate: date structure

    :returns: observer contains all info about location and date for prediction
    :rtype observer: station
    """
    # read in the station info (name, latitude, longitude) in degrees
    observer = Station()
    if station is None:
        observer = RMA
    else:
        observer.parse(station)

    # read in the predDate
    if predictionDate is None:
        observer.date = ephem.date(ephem.now())  # today at midnight for default start
    else:
        observer.date = ephem.Date(predictionDate)
    # print('observer.date: %04d/%02d/%02d\n' % ephem.date(observer.date).triple())

    if verbose:
        observer.statPrint()

    return observer


def setObservationTimes(observer, timeStart, timeEnd, intervalMin, verbose=False):
    """
    observationTimes calculates the times for which the predictions will be calculated

    :param observer: station info containing the info about prediction times
    :type observer: station

    :returns obsDates: the list of prediction times
    :rtype obsDates: list of date structure
    :returns nrPredictions: number of predictions to make
    :rtype nrPredictions: int
    """
    yyyy, mm, dd = observer.getYMD()
    # print('timeStart = %s' % (timeStart.split(':')))
    startHour, startMin = map(int, timeStart.split(':'))
    endHour, endMin = map(int, timeEnd.split(':'))
    startDateTime = datetime(yyyy, mm, dd, hour=startHour, minute=startMin,
                             second=0, microsecond=0, tzinfo=None)
    endDateTime = datetime(yyyy, mm, dd, hour=endHour, minute=endMin, second=0,
                           microsecond=0, tzinfo=None)
    if endDateTime <= startDateTime:
        sys.stderr.write('end time %s is less than start time %s. Program exits.\n' % (endDateTime, startDateTime))
        sys.exit(E_TIMING_ERROR)

    dtDateTime = endDateTime - startDateTime
    # print('dtDateTime = %s' % dtDateTime)
    dtMinutes = dtDateTime.total_seconds() / 60  # / timedelta(minutes=1)
    # print('dtMinutes = %s' % dtMinutes)
    nrPredictions = int(dtMinutes / float(intervalMin)) + 1
    obsDates = [startDateTime + timedelta(minutes=(int(intervalMin) * x)) for x in range(0, nrPredictions, 1)]

    if verbose:
        print('Observation time span from %s to %s with interval %d min (#%d)' % (obsDates[0], obsDates[-1], intervalMin, np.size(obsDates)))

    return obsDates, nrPredictions


def getTLEfromNORAD(TLEBaseName, verbose=False):
    """
    getTLEfromNORAD checks whether we have a Internet connection,
    if yes, download latest TLE for satellite system,
    else check to reuse already downloaded TLE file

    :param TLEBaseName: basename of TLE file (cfr NORAD site)
    :type TLEBaseName: string

    :returns outFileName: filename of downloaded/reused TLE file
    :rtype string:
    """
    if verbose:
        print('Downloading TLEs from NORAD for satellite systems %s' % TLEBaseName)

    # determine whether a list of constellations is given, if so, split up
    satSystems = TLEBaseName.split(',')
    # print('satSystems = %s' % satSystems)
    TLEFileNames = []
    for i, satSyst in enumerate(satSystems):
        url = 'https://www.celestrak.com/NORAD/elements/%s.txt' % satSyst
        print('url = %s' % url)
        TLEFileNames.append(url.split('/')[-1])
        print('TLEFileNames = %s' % TLEFileNames[-1])

        # sys.exit(0)
        # NOTE the stream=True parameter
        try:
            r = requests.get(url, stream=True)
            print('r = %s' % type(r))
            with open(TLEFileNames[-1], 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        f.flush()
                        # f.flush() commented by recommendation from J.F.Sebastian
                        os.fsync(f)
        except requests.exceptions.ConnectionError:
            sys.stderr.write('Connection to NORAD could not be established.\n')

            # check if we have alocal TLE file
            if os.path.isfile(TLEFileNames[-1]):
                print('Using local file %s' % TLEFileNames[-1])
                return TLEFileNames[-1]
            else:
                sys.stderr.write('Program exits.\n')
                sys.exit(E_REQUEST_ERROR)
        except requests.exceptions.ConnectTimeoutError:
            sys.stderr.write('Connection to NORAD could not be established.\n')

            # check if we have alocal TLE file
            if os.path.isfile(TLEFileNames[-1]):
                print('Using local file %s' % TLEFileNames[-1])
                return TLEFileNames[-1]
            else:
                sys.stderr.write('Program exits.\n')
                sys.exit(E_REQUEST_ERROR)

    # catenate into a single TLE file that combines the different satellite systems
    if np.size(TLEFileNames) > 1:
        outFileName = satSystem.replace(',', '-') + '.txt'
        print('outFileName = %s' % outFileName)
        fout = open(outFileName, 'w')

        fin = fileinput.input(files=TLEFileNames)
        for line in fin:
            fout.write(line)
        fin.close()
    else:
        outFileName = TLEFileNames[0]

    print('outFileName = %s' % outFileName)
    if verbose:
        print('  TLE file saved in %s' % outFileName)

    return outFileName


def createDOPFile(observer, satSystem, listSat, predDates, xDOPs, cutoff, verbose=False):
    """
    createDOPFile writes info to the DOP file

    :param observer: info about the observation station and date
    :type observer: string
    :param satSystem: used satellite system
    :type satSystem: string
    :param listSat: list of satellites
    :type listSat: python list
    :param predDates: contains the prediction dates
    :type predDates: list of datetimes
    :param xDOPs: the HDOP, VDOP and TDOP in that order
    :type xDOPs: list of strings
    :param cutoff: cut off angle in degrees
    :type cutoff: int
    """
    filename = observer.name + '-' + satSystem.replace(',', '-') + '-%04d%02d%02d-DOP.txt' % (observer.getYMD())
    if verbose:
        print('  Creating DOP file: %s' % filename)
    try:
        fid = open(filename, 'w')
        # write the observer info out
        fid.write('Observer: %s\n' % observer.name)
        fid.write('     lat: %s\n' % ephem.degrees(observer.lat))
        fid.write('     lon: %s\n' % ephem.degrees(observer.lon))
        fid.write('    date: %04d/%02d/%02d' % observer.getYMD())
        fid.write('  cutoff: %2d\n\n' % cutoff)

        fid.write('      |#Used/#Vis|   HDOP   VDOP   PDOP   TDOP   GDOP\n\n')
        # print the number of visible SVs and their elev/azim
        for i, predDate in enumerate(predDates):
            fid.write('%02d:%02d' % (predDate.hour, predDate.minute))

            # number of visible satellites
            if ~np.isnan(xDOPs[i, 3]):
                fid.write(' | %3.0f / %2d |' % (xDOPs[i, 3], np.count_nonzero(~np.isnan(elev[i, :]))))
            else:
                fid.write(' |  -- / %2d |' % (np.count_nonzero(~np.isnan(elev[i, :]))))

            # write the DOP values in order
            if ~np.isnan(xDOPs[i, 0]):
                PDOP2 = xDOPs[i, 0] * xDOPs[i, 0] + xDOPs[i, 1] * xDOPs[i, 1]
                fid.write(' %6.1f %6.1f %6.1f %6.1f %6.1f' % (xDOPs[i, 0], xDOPs[i, 1], np.sqrt(PDOP2), xDOPs[i, 2], np.sqrt(PDOP2 + xDOPs[i, 2] * xDOPs[i, 2])))
            else:
                fid.write(' ------ ------ ------ ------ ------')
            fid.write('\n')

        # close the file
        fid.close()
    except IOError:
        print('  Access to file %s failed' % filename)


def createGeodeticFile(observer, satSystem, listSats, predDates, lats, lons, verbose=False):
    """
    createGeodeticFile creates a file containing lat/lon values for each satellite

    :param observer: info about the observation station and date
    :type observer: string
    :param satSystem: used satellite system
    :type satSystem: string
    :param listSat: list of satellites
    :type listSat: python list
    :param predDates: contains the prediction dates
    :type predDates: list of datetimes
    :param lats: latitude of satellite
    :type lats: float
    :param lons: longitude of satellite
    :type lons: float
    """
    filename = observer.name + '-' + satSystem.replace(',', '-') + '-%04d%02d%02d-GEOD.txt' % (observer.getYMD())
    if verbose:
        print('  Creating substellar file: %s' % filename)

    try:
        fid = open(filename, 'w')
        # write the observer info out
        fid.write('Observer: %s\n' % observer.name)
        fid.write('     lat: %s\n' % ephem.degrees(observer.lat))
        fid.write('     lon: %s\n' % ephem.degrees(observer.lon))
        fid.write('    date: %04d/%02d/%02d\n\n' % observer.getYMD())

        # write the sat IDs on first line
        satLine1 = ''
        satLine2 = ''
        for j, sat in enumerate(listSats):
            if len(sat.name) < 11:
                satLine1 += '  %10s' % sat.name
            else:
                satLine1 += '  %10s  ' % sat.name[:10]
                endChar = min(20, len(sat.name))
                satLine2 += '  %10s  ' % sat.name[10:endChar]
        fid.write('      %s' % satLine1)
        fid.write('\n')
        if len(satLine2) > 0:
            fid.write('      %s' % satLine2)
            fid.write('\n')
        fid.write('\n')

        # print the number of visible SVs and their elev/azim
        for i, predDate in enumerate(predDates):
            fid.write('%02d:%02d' % (predDate.hour, predDate.minute))

            # write the lat/lon values
            for j, sat in enumerate(listSats):
                fid.write("  %5.1f %6.1f" % (lats[i, j], lons[i, j]))
            fid.write('\n')

        # close the file
        fid.close()
    except IOError:
        print('  Access to file %s failed' % filename)


def createVisibleSatsFile(observer, satSystem, listSat, predDates, elevation, azimuth, cutoff, excludedSats, verbose=False):
    """
    createVisibleSatsFile writes visibility info to file

    :param observer: info about the observation station and date
    :type observer: string
    :param satSystem: used satellite system
    :type satSystem: string
    :param listSat: list of satellites
    :type listSat: python list
    :param predDates: contains the prediction dates
    :type predDates: list of datetimes
    :param elevation: satellite elevation list in degrees
    :type elevation: list of float
    :param azimuth: satellite azimuth list in degrees
    :type azimuth: list of float
    :param cutoff: cut off angle in degrees
    :type cutoff: int
    """
    filename = observer.name + '-' + satSystem.replace(',', '-') + '-%04d%02d%02d.txt' % (observer.getYMD())
    if verbose:
        print('  Creating visibility file: %s' % filename)

    # mark in a boolean which Sats are excluded from the xDOP calculations
    indexIncluded = np.ones(np.size(listSat), dtype=bool)
    if excludedSats is not None:
        for i, sat in enumerate(listSat):
            for j, PRN in enumerate(excludedSats):
                if PRN in sat.name:
                    indexIncluded[i] = False

    # print('listSat = %s' % listSat)
    # print('indexIncluded = %s' % indexIncluded)

    try:
        fid = open(filename, 'w')
        # write the observer info out
        fid.write('Observer: %s\n' % observer.name)
        fid.write('     lat: %s\n' % ephem.degrees(observer.lat))
        fid.write('     lon: %s\n' % ephem.degrees(observer.lon))
        fid.write('    date: %04d/%02d/%02d' % observer.getYMD())
        fid.write('  cutoff: %2d\n\n' % cutoff)

        # write the sat IDs on first line
        satLine1 = ''
        satLine2 = ''
        for j, sat in enumerate(listSat):
            if len(sat.name) < 12:
                satLine1 += '  %11s' % sat.name
            else:
                satLine1 += '  %11s' % sat.name[:10]
                endChar = min(20, len(sat.name))
                satLine2 += '  %11s' % sat.name[10:endChar]
        fid.write('      |#Vis|%s' % satLine1)
        fid.write('\n')
        if len(satLine2) > 0:
            fid.write('            %s' % satLine2)
            fid.write('\n')
        fid.write('\n')

        # print the number of visible SVs and their elev/azim
        for i, predDate in enumerate(predDates):
            fid.write('%02d:%02d' % (predDate.hour, predDate.minute))

            # number of visible satellites
            fid.write(' | %2d |' % np.count_nonzero(~np.isnan(elevation[i, :])))

            for j, sat in enumerate(listSat):
                if indexIncluded[j]:
                    if math.isnan(elevation[i, j]):
                        fid.write('  ---- ----- ')
                    else:
                        fid.write('  %4.1f %5.1f ' % (elevation[i, j], azimuth[i, j]))
                else:
                    if math.isnan(elevation[i, j]):
                        fid.write(' (---- -----)')
                    else:
                        fid.write(' (%4.1f %5.1f)' % (elevation[i, j], azimuth[i, j]))
            fid.write('\n')

        # close the file
        fid.close()
    except IOError:
        print('  Access to file %s failed' % filename)


def plotVisibleSats(systSat, observer, listSats, predDates, elev, cutoff, excludedSats, verbose=False):
    """
    plotVisibleSats plots the timeline of visible satellites

    :param satSystem: used satellite system
    :type satSystem: string
    :param observer: info about the observation station and date
    :type observer: string
    :param listSat: list of satellites
    :type listSat: python list
    :param predDates: contains the prediction dates
    :type predDates: list of datetimes
    :param elevation: satellite elevation list in degrees
    :type elevation: list of float
    :param cutoff: cut off angle in degrees
    :type cutoff: int
    :param excludedSats: satellite PRNs to exclude
    :type excludedSats: list of string
    """
    plt.style.use('ggplot')

    fig = plt.figure(figsize=(20.0, 16.0))
    # plt.subplots_adjust(top=0.65)
    # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0, rect=[0, 0.03, 1, 0.97])
    # plt.tight_layout(fig, rect=[0, 0.03, 1, 0.97])
    ax1 = plt.gca()

    # ax2 = ax1.twinx()
    # set colormap
    colors = iter(cm.jet(np.linspace(0, 1, len(listSats))))

    # local copy to work with
    elev2 = copy.deepcopy(elev)

    # plot the lines for visible satellites
    for i, sat in enumerate(listSats):
        # print('=' * 25)
        # print('DEBUG: sat[%d of %d] = %s' % (i, np.size(listSats), sat.name))
        # print('elev2 = %s  - %s' % (type(elev2), np.size(elev2)))
        elev2[~np.isnan(elev2)] = i + 1  # create a horizontal line @ height i+1
        satLineStyle = '-'
        if excludedSats is not None:
            for j, PRN in enumerate(excludedSats):
                # print('DEBUG:   PRN[%d of %d] = %s' % (j, np.size(excludedSats), PRN))
                if PRN in sat.name:
                    satLineStyle = '--'

        plt.plot(predDates, elev2[:, i], linewidth=5, color=next(colors), linestyle=satLineStyle, label=sat.name)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(byminute=0, interval=1, tz=None))

    # create array for setting the satellite labels on y axis
    satNames = []
    for i in range(len(listSats)):
        # print('%s' % listSats[i].name)
        satNames.append(listSats[i].name)

    # set the tick marks
    plt.xticks(rotation=50, size='medium')
    plt.yticks(range(1, len(listSats) + 1), satNames, size='small')

    # color the sat labels ticks
    colors2 = iter(cm.jet(np.linspace(0, 1, len(listSats))))
    for i, tl in enumerate(ax1.get_yticklabels()):
        tl.set_color(next(colors2))

    plt.grid(True)
    # set the limits for the y-axis
    plt.ylim(0, len(listSats) + 2)
    ax1.set_xlabel('Time of Day', fontsize='x-large')
    # plot title
    plt.title('%s Satellite Visibility' % systSat.replace(',', ' & ').upper(), fontsize='x-large')
    yyyy, mm, dd = observer.getYMD()
    annotateTxt = (r'Station: %s @ ($\varphi$ %s, $\lambda$ %s) - Date %04d/%02d/%02d - Cutoff %2d' % (observer.name, ephem.degrees(observer.lat), ephem.degrees(observer.lon), yyyy, mm, dd, cutoff))
    plt.text(0.5, 0.99, annotateTxt, horizontalalignment='center', verticalalignment='top', transform=ax1.transAxes, fontsize='medium')
    # plt.title('Station: %s @ %s, %s date %04d/%02d/%02d' % (observer.name, ephem.degrees(observer.lat), ephem.degrees(observer.lon), yyyy, mm, dd))

    # ax2 = ax1.twinx()
    filename = observer.name + '-' + systSat.replace(',', '-') + '-%04d%02d%02d-visibility.png' % (observer.getYMD())
    fig.savefig(filename, dpi=fig.dpi)

    if verbose:
        plt.draw()


def plotSkyView(systSat, observer, listSats, predDates, elevations, azimuths, cutoff, excludedSats, verbose=False):
    """
    plotSkyView plots the skyview for current location

    :param satSystem: used satellite system
    :type satSystem: string
    :param observer: info about the observation station and date
    :type observer: string
    :param listSat: list of satellites
    :type listSat: python list
    :param predDates: contains the prediction dates
    :type predDates: list of datetimes
    :param elevation: satellite elevation list in degrees
    :type elevation: list of float
    :param azimuth: satellite azimuth list in degrees
    :type azimuth: list of float
    :param cutoff: cut off angle in degrees
    :type cutoff: int
    :param excludedSats: satellite PRNs to exclude
    :type excludedSats: list of string
    """
    plt.style.use('ggplot')

    # rc('grid', color='#999999', linewidth=1, linestyle='-', alpha=[0].6)
    rc('xtick', labelsize='x-small')
    rc('ytick', labelsize='x-small')

    # force square figure and square axes looks better for polar, IMO
    width, height = rcParams['figure.figsize']
    size = min(width, height) * 2

    # make a square figure
    fig = figure(figsize=(size, size))

    # set the axis (0 azimuth is North direction, azimuth indirect angle)
    ax = fig.add_axes([0.10, 0.15, 0.8, 0.8], projection=u'polar')  # , axisbg='#CCCCCC', alpha=0.6)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Define the xticks
    ax.set_xticks(np.linspace(0, 2 * np.pi, 13))
    xLabel = ['N', '30', '60', 'E', '120', '150', 'S', '210', '240', 'W', '300', '330']
    ax.set_xticklabels(xLabel)

    # Define the yticks
    ax.set_yticks(np.linspace(0, 90, 7))
    yLabel = ['', '75', '60', '45', '30', '15', '']
    ax.set_yticklabels(yLabel)

    # draw a grid
    grid(True)

    # plot the skytracks for each PRN
    colors = iter(cm.jet(np.linspace(0, 1, len(listSats))))
    satLabel = []
    # print('elevations = %s' % elevations)
    # print('#listSats = %d' % np.size(listSats))

    # find full hours in date to set the elev/azimaccordingly
    # indexHour = np.where(np.fmod(prnTime, 3600.) == 0)
    # print('predDates = %s' % predDates[0].time())
    # print('predDates = %s  = %s' % (predDates[0].time(), hms_to_seconds(predDates[0].time())))
    predTimeSeconds = []
    hourTxt = []

    for t, predDate in enumerate(predDates):
        predTimeSeconds.append(hms_to_seconds(predDate.time()))
    predTimeSeconds = np.array(predTimeSeconds)
    # print('predTimeSeconds = %s' % predTimeSeconds)

    indexHour = np.where(np.fmod(predTimeSeconds, 3600.) == 0)
    # print('indexHour = %s' % indexHour)
    hourTxt.append(predTimeSeconds[indexHour])
    # print('hourTxt = %s' % hourTxt)

    for i, prn in enumerate(listSats):
        satLabel.append('%s' % prn.name)
        satColor = next(colors)
        azims = [np.radians(az) for az in azimuths[:, i]]
        elevs = [(90 - el) for el in elevations[:, i]]
        # print('PRN = %s' % prn.name)
        # print('elev = %s' % elevs)
        # print('azim = %s' % azims)
        # ax.plot(azims, elevs, color=next(colors), linewidth=0.35, alpha=0.85, label=satLabel[-1])
        satLineStyle = '-'
        if excludedSats is not None:
            for j, exlPRN in enumerate(excludedSats):
                # print('DEBUG:   exlPRN[%d of %d] = %s' % (j, np.size(excludedSats), exlPRN))
                if exlPRN in prn.name:
                    satLineStyle = '--'

        ax.plot(azims, elevs, color=satColor, marker='.', markersize=4, linestyle=satLineStyle, linewidth=1, label=satLabel[-1])

        # annotate with the hour labels
        prnHourAzim = azimuths[:, i][indexHour]
        # print('azimuth     = %s' % azimuths[:, i])
        # print('prnHourAzim = %s\n' % prnHourAzim)
        prnHourElev = elevations[:, i][indexHour]
        # print('Elevuth     = %s' % elevations[:, i])
        # print('prnHourElev = %s\n\n' % prnHourElev)

        hrAzims = [np.radians(az + 2) for az in prnHourAzim]
        hrElevs = [(90 - el) for el in prnHourElev]

        # print('hrAzims = %s' % hrAzims)
        # print('hrElevs = %s' % hrElevs)
        # print('-' * 20)
        # print('hourTxt = %s' % hourTxt)
        for j, hr in enumerate(hourTxt[0]):
            hrEl = hrElevs[j]
            if ~np.isnan(hrEl):
                hrAz = hrAzims[j]
                # print('hr = %s' % hr)
                # print('hrEl = %s' % hrEl)
                # print('hrAz = %d' % hrAz)
                hr = int(float(hr) / 3600.)
                # print('hr = %s' % hr)
                # print('hrEl = %d' % hrEl)
                plt.text(hrAz, hrEl, hr, fontsize='x-small', color=satColor)
        # print('-' * 30)

    # adjust the legend location
    mLeg = ax.legend(bbox_to_anchor=(0.5, -0.15), loc='lower center', ncol=min(np.size(satLabel), 5), fontsize='small', markerscale=4)
    for legobj in mLeg.legendHandles:
        legobj.set_linewidth(5.0)

    plt.title('%s Satellite Visibility' % systSat.replace(',', ' & ').upper(), fontsize='x-large', x=0.5, y=0.99, horizontalalignment='center')
    yyyy, mm, dd = observer.getYMD()
    # annotateTxt = (r'Station: %s @ ($\varphi$ %s, $\lambda$ %s) - Date %04d/%02d/%02d - Cutoff %2d' % (observer.name, ephem.degrees(observer.lat), ephem.degrees(observer.lon), yyyy, mm, dd, cutoff))
    # plt.text(0.5, 0.99, annotateTxt, horizontalalignment='center', verticalalignment='top', transform=ax.transAxes, fontsize='x-large')
    annotateTxt = (r'Station: %s @ ($\varphi$ %s, $\lambda$ %s)' % (observer.name, ephem.degrees(observer.lat), ephem.degrees(observer.lon)))
    plt.text(-0.075, 0.975, annotateTxt, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize='medium')
    annotateTxt = (r'Date %04d/%02d/%02d - Cutoff %2d' % (yyyy, mm, dd, cutoff))
    plt.text(-0.075, 0.950, annotateTxt, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize='medium')

    # needed for having radial axis span from 0 => 90 degrees and y-labels along north axis
    ax.set_rmax(90)
    ax.set_rmin(0)
    ax.set_rlabel_position(0)

    # ax2 = ax1.twinx()
    filename = observer.name + '-' + systSat.replace(',', '-') + '-%04d%02d%02d-skyview.png' % (observer.getYMD())
    fig.savefig(filename, dpi=fig.dpi)

    if verbose:
        plt.draw()


def hms_to_seconds(t):
    """
    hms_to_seconds transforms expression in hh:mm:ss to number of seconds

    :param t: time structure
    :type t: time
    :returns: time expressed in seconds
    :rtype: int
    """
    # print('t.hour %s' % t.hour)
    # h, m, s = [int(i) for i in t.split(':')]
    return 3600 * t.hour + 60 * t.minute + t.second


# def steppify(arr,isX=False,interval=0):
#     """
#     Converts an array to double-length for step plotting
#     """
#     if isX and interval==0:
#         interval = abs(arr[1]-arr[0]) / 2.0
#         newarr = array(zip(arr-interval,arr+interval)).ravel()
#         return newarr


def plotSatTracks(systSat, observer, listSats, predDates, satLats, satLons, excludedSats, verbose=False):
    """
    plotSatTracks plots the ground tracks of the satellites on a map

    :param satSystem: used satellite system
    :type satSystem: string
    :param observer: info about the observation station and date
    :type observer: string
    :param listSat: list of satellites
    :type listSat: python list
    :param predDates: contains the prediction dates
    :type predDates: list of datetimes
    :param satLats: satellite satLats list in degrees
    :type satLats: list of float
    :param satLons: satellite satLons list in degrees
    :type satLons: list of float
    :param excludedSats: satellite PRNs to exclude
    :type excludedSats: list of string
    """
    plt.style.use('ggplot')
    plt.figure(figsize=(16.0, 10.5))

    # miller projection
    map = Basemap(projection='mill', lon_0=0)
    # plot coastlines, draw label meridians and parallels.
    map.drawcoastlines()
    map.drawparallels(np.arange(-90, 90, 30), labels=[1, 0, 0, 0])
    map.drawmeridians(np.arange(map.lonmin, map.lonmax + 30, 60), labels=[0, 0, 0, 1])
    # fill continents 'coral' (with zorder=0), color wet areas 'aqua'
    map.drawmapboundary(fill_color='whitesmoke')
    map.fillcontinents(color='lightgray', lake_color='whitesmoke', alpha=0.9)

    # plot the baseStation on the map
    xObs, yObs = map(observer.lon / ephem.pi * 180., observer.lat / ephem.pi * 180.)
    # print('observer = %s %s' % (observer.lat, observer.lon))
    # print('observer = %s %s' % (ephem.degrees(observer.lat), ephem.degrees(observer.lon)))
    # print('observer = %f %f' % (observer.lat / ephem.pi * 180., observer.lon / ephem.pi * 180.))
    # print('xObs,yObs = %f  %f' % (xObs, yObs))
    map.plot(xObs, yObs, color='blue', marker='o', markersize=5)
    offSet = 0.5  # 1/10 of a degree
    xObs, yObs = map(observer.lon / ephem.pi * 180. + offSet, observer.lat / ephem.pi * 180. + offSet)
    plt.text(xObs, yObs, observer.name, fontsize='small', color='blue')

    # set colormap
    # colors = iter(cm.jet(np.linspace(0, 1, len(listSats))))
    colors = iter(cm.jet(np.linspace(0, 1, len(listSats))))
    satLabel = []

    predTimeSeconds = []
    hourTxt = []

    for t, predDate in enumerate(predDates):
        predTimeSeconds.append(hms_to_seconds(predDate.time()))
    predTimeSeconds = np.array(predTimeSeconds)
    # print('predTimeSeconds = %s' % predTimeSeconds)
    indexHour = np.where(np.fmod(predTimeSeconds, 3600.) == 0)
    # print('indexHour = %s' % indexHour)
    hourTxt.append(predTimeSeconds[indexHour])
    # print('hourTxt = %s' % hourTxt)

    for i, SV in enumerate(listSats):
        satLabel.append('%s' % SV.name)
        # print('\n\nSat %s' % satLabel[-1])
        satColor = next(colors)

        satLineStyle = '-'
        if excludedSats is not None:
            for j, PRN in enumerate(excludedSats):
                # print('DEBUG:   PRN[%d of %d] = %s' % (j, np.size(excludedSats), PRN))
                if PRN in SV.name:
                    satLineStyle = '--'

        # for j, dt in enumerate(predDates):
        # check whether we have a jump bigger than 180 degreein longitude
        lonDiffs = np.abs(np.diff(satLons[:, i]))
        # print('lons = %s' % satLons[:, i])
        # print('lonDiffs = %s' % lonDiffs)

        # lonDiffMax = np.max(lonDiffs)
        # print('lonDiffMax = %s' % lonDiffMax)

        lonIndices = np.where(lonDiffs > 300)
        # print('lonIndices = %s' % lonIndices)

        # split up the arrays satLons and satLats based on the lonIndices found
        if np.size(lonIndices) > 0:
            for k, lonIndex in enumerate(lonIndices[0]):
                # print('lonIndex = %s' % lonIndex)
                # print('lonIndex[%d] = %d  satLons[%d] = %f' % (k, lonIndex, lonIndex, satLons[lonIndex, i]))

                # determine indices between which we have a track without 360 degree jump
                if k == 0:
                    startIndex = 0
                else:
                    startIndex = lonIndices[0][k - 1] + 1
                endIndex = lonIndex + 1

                xSat = np.zeros(np.size(predDates))
                ySat = np.zeros(np.size(predDates))
                xSat.fill(np.nan)
                ySat.fill(np.nan)

                # print('startIndex = %d  endIndex = %d' % (startIndex, endIndex))

                for l in range(startIndex, endIndex):
                    xSat[l], ySat[l] = map(satLons[l, i], satLats[l, i])
                    # print('Pt %d: lat = %s  lon = %s  x,y = %f  %f' % (l, satLats[l, i], satLons[l, i], xSat[l], ySat[l]))

                # print('intermed x = %s' % xSat)
                map.plot(xSat, ySat, linewidth=2, color=satColor, linestyle=satLineStyle, marker='.', markersize=6)

            xSat = np.zeros(np.size(predDates))
            ySat = np.zeros(np.size(predDates))
            xSat.fill(np.nan)
            ySat.fill(np.nan)
            for l in range(lonIndex + 1, np.size(predDates)):
                xSat[l], ySat[l] = map(satLons[l, i], satLats[l, i])
                # print('Pt %d: lat = %s  lon = %s  x,y = %f  %f' % (l, satLats[l, i], satLons[l, i], xSat[l], ySat[l]))

            # print('last part x = %s' % xSat)
            map.plot(xSat, ySat, linewidth=2, color=satColor, linestyle=satLineStyle, marker='.', markersize=6, label=satLabel[-1])
        else:
            xSat = np.zeros(np.size(predDates))
            ySat = np.zeros(np.size(predDates))
            xSat.fill(np.nan)
            ySat.fill(np.nan)

            for l in range(np.size(predDates)):
                xSat[l], ySat[l] = map(satLons[l, i], satLats[l, i])
                # print('Pt %d: lat = %s  lon = %s  x,y = %f  %f' % (l, satLats[l, i], satLons[l, i], xSat[l], ySat[l]))

            # print('full part x = %s' % xSat)
            map.plot(xSat, ySat, linewidth=2, color=satColor, linestyle=satLineStyle, marker='.', markersize=6, label=satLabel[-1])

        # setting the hour index
        prnHourLats = satLats[:, i][indexHour]
        prnHourLons = satLons[:, i][indexHour]

        x, y = map(prnHourLons, prnHourLats)

        for j, hr in enumerate(hourTxt[0]):
            if ~np.isnan(y[j]):
                hr = int(float(hr) / 3600.)
                plt.text(x[j], y[j], hr, fontsize='x-small', color=satColor)

    # adjust the legend location
    mLeg = plt.legend(bbox_to_anchor=(0.5, 0.05), loc='lower center', ncol=min(np.size(satLabel), 5), fontsize='small', markerscale=2)
    for legobj in mLeg.legendHandles:
        legobj.set_linewidth(5.0)

    # plot title
    yyyy, mm, dd = observer.getYMD()
    plt.title(('%s Satellite Groundtracks - Date %04d/%02d/%02d' % (systSat.replace(',', ' & ').upper(), yyyy, mm, dd)), fontsize='x-large')

    # ax2 = ax1.twinx()
    filename = observer.name + '-' + systSat.replace(',', '-') + '-%04d%02d%02d-groundtrack.png' % (observer.getYMD())
    plt.savefig(filename)

    if verbose:
        plt.show()


def plotDOPVisSats(systSat, observer, listSats, predDates, elev, xDOPs, cutoff, nrExcludedSVs, verbose=False):
    """
    plotDOPVisSats plots the xDOP values and the total number of satellites visible

    :param satSystem: used satellite system
    :type satSystem: string
    :param observer: info about the observation station and date
    :type observer: string
    :param listSat: list of satellites
    :type listSat: python list
    :param predDates: contains the prediction dates
    :type predDates: list of datetimes
    :param elevation: satellite elevation list in degrees
    :type elevation: list of float
    :param xDOPS: satellite xDOP (HDOP, VDOP and TDOP) list
    :type xDOPS: list of float
    :param cutoff: cut off angle in degrees
    :type cutoff: int
    :param nrExcludedSVs: number of excluded SVs
    :type nrExcludedSVs: int
    """
    plt.style.use('ggplot')

    fig = plt.figure(figsize=(20.0, 16.0))
    ax1 = plt.gca()
    ax2 = ax1.twinx()  # second y-axis needed, so make the x-axis twins
    # set colormap

    # plot the number of visible satellites
    nrVisSats = []
    for i, el in enumerate(elev):
        nrVisSats.append(np.count_nonzero(~np.isnan(el)))
        ax2.set_ylim(0, max(nrVisSats) + 1)
    # print('nrVisSats = %s' % nrVisSats)
    ax2.plot(predDates, nrVisSats, linewidth=3, color='black', drawstyle='steps-post', label='#Visible', alpha=.6)

    # draw the line representing the number of SVs used
    ax2.plot(predDates, (nrVisSats - nrExcludedSVs), linewidth=3, color='green', drawstyle='steps-post', label='#Visible', alpha=.6)

    # ax2.fill_between(steppify(predDates,isX=True), steppify(nrVisSats)*0, steppify(nrVisSats), facecolor='b',alpha=0.2)
    # ax2.fill_between(predDates, 0, nrVisSats, color='lightgray', alpha=0.5, drawstyle='steps')
    # ax2.fill_between(lines[0].get_xdata(orig=False), 0, lines[0].get_ydata(orig=False))

    # plot the xDOPS on first axis
    ax1.set_ylim(0, maxDOP)
    # print('len(xDOPs) = %d' % len(xDOPs[0, :]))
    colors = iter(cm.jet(np.linspace(0, 1, len(xDOPs[0, :]) + 2)))
    # print('len(xDOPs[0, :]+2 = %s' % (len(xDOPs[0, :]) + 2))
    # print('colors.size = %s' % np.linspace(0, 1, len(xDOPs[0, :]) + 2))
    labels = ['HDOP', 'VDOP', 'TDOP']
    # print('labels = %s' % labels)
    for i in range(0, 3):
        xDOP = xDOPs[:, i]
        dopColor = next(colors)
        transparency = .5 - i * 0.1
        ax1.fill_between(predDates, 0, xDOP, color=dopColor, alpha=transparency)
        ax1.plot(predDates, xDOP, linewidth=2, color=dopColor, label=labels[i])

        # add PDOP
        if i is 1:
            PDOP2 = xDOPs[:, 0] * xDOPs[:, 0] + xDOPs[:, 1] * xDOPs[:, 1]
            # print('PDOP = %s' % np.sqrt(PDOP2))
            dopColor = next(colors)
            transparency = .2
            ax1.fill_between(predDates, 0, np.sqrt(PDOP2), color=dopColor, alpha=transparency)
            ax1.plot(predDates, np.sqrt(PDOP2), linewidth=2, color=dopColor, label='PDOP')

        # add GDOP
        if i is 2:
            GDOP = np.sqrt(PDOP2 + xDOPs[:, 2] * xDOPs[:, 2])
            # print('GDOP = %s' % GDOP)
            dopColor = next(colors)
            transparency = .1
            ax1.fill_between(predDates, 0, GDOP, color=dopColor, alpha=transparency)
            ax1.plot(predDates, GDOP, linewidth=2, color=dopColor, label='GDOP')

    ax1.legend(loc='upper left', frameon=True)

    plt.title('%s Satellite Visibility' % systSat.replace(',', ' & ').upper(), fontsize='x-large')
    yyyy, mm, dd = observer.getYMD()
    annotateTxt = (r'Station: %s @ ($\varphi$ %s, $\lambda$ %s) - Date %04d/%02d/%02d - cutoff %2d' % (observer.name, ephem.degrees(observer.lat), ephem.degrees(observer.lon), yyyy, mm, dd, cutoff))
    plt.text(0.5, 0.99, annotateTxt, horizontalalignment='center', verticalalignment='top', transform=ax1.transAxes, fontsize='medium')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(byminute=0, interval=1, tz=None))

    plt.xticks(rotation=50, size='medium')
    ax1.set_ylabel('DOP [-]', fontsize='x-large')
    ax2.set_ylabel('# Visible satellites [-]', fontsize='x-large')
    ax1.set_xlabel('Time of Day', fontsize='x-large')

    filename = observer.name + '-' + systSat.replace(',', '-') + '-%04d%02d%02d-DOP.png' % (observer.getYMD())
    fig.savefig(filename, dpi=fig.dpi)

    if verbose:
        plt.draw()


def createTLEArcFile(satSystem, listSats, predInterval, predDates, elevation, verbose=False):
    """
    createTLEArcFile creates the arcs based on TLS predictions and saves it to a CSV file

    :param satSystem: used satellite system
    :type satSystem: string
    :param listSat: list of satellites
    :type listSat: python list
    :param predInterval: interval between predictions in minutes
    :type predInterval: int
    :param predDates: contains the prediction dates
    :type predDates: list of datetimes
    :param elevation: satellite elevation list in degrees
    :type elevation: lis of float
   """
    # convert predDates DateTime structures to seconds-of-day
    predTimeSeconds = []
    for t, predDate in enumerate(predDates):
        pyUTC = gpstime.mkUTC(predDate.year, predDate.month, predDate.day, predDate.hour, predDate.minute, predDate.second)
        WkNr, TOW = gpstime.wtFromUTCpy(pyUTC, leapSecs=0)
        predTimeSeconds.append(TOW)

    predTimeSeconds = np.array(predTimeSeconds)
    # print('predTimeSeconds = %s (#%d)' % (predTimeSeconds, np.size(predTimeSeconds)))

    # # Deterimne Week and TOW from UTC time
    # print('\npredDates[0] = %s   %s' % (predDates[0], type(predDates[0])))
    # print('predDates = %d/%d/%d  %d:%d:%f' % (predDates[0].year, predDates[0].month, predDates[0].day, predDates[0].hour, predDates[0].minute, predDates[0].second))
    # pyUTC = gpstime.mkUTC(predDates[0].year, predDates[0].month, predDates[0].day, predDates[0].hour, predDates[0].minute, predDates[0].second)
    # WkNr, TOW = gpstime.wtFromUTCpy(pyUTC)
    # print('WkNr = %s  TOW = %s' % (WkNr, TOW))

    gpsWeek, gpsSOW, gpsDay, gpsSOD = gpstime.gpsFromUTC(predDates[0].year, predDates[0].month, predDates[0].day, predDates[0].hour, predDates[0].minute, predDates[0].second, leapSecs=0)
    # print('gpsFromUTC gpsWeek, gpsSOW, gpsDay, gpsSOD = %s  %s  %s  %s' % (gpsWeek, gpsSOW, gpsDay, gpsSOD))

    # create the TLE arc csv file
    nameTLEArcFile = predDates[0].strftime("%Y%m%d-TLE-arc.csv")
    # print('nameTLEArcFile = %s' % nameTLEArcFile)

    with open(nameTLEArcFile, 'w') as outArc:
        arcWriter = csv.writer(outArc, delimiter=',', lineterminator='\n')

        for i, PRN in enumerate(listSats):
            # print('-' * 25)
            # print('PRN[%d] = %s  --- %s' % (i, PRN.name, PRN))

            # transform the full PRN in SSN PRN numbers
            prnNr = int(PRN.name[-3:-1]) + 70
            # print('substr = %s  %d' % (PRN.name[-3: -1], prnNr))

            # look for new arcs by examining the elvation angles which are finite
            # print('elevation = %s (#%d)' % (elevation[:, i], np.size(elevation[:, i])))
            posElevIndices = np.where(np.isfinite(elevation[:, i]))[0]
            # print('posElevIndices = %s (#%d)' % (posElevIndices, np.size(posElevIndices)))
            if np.size(posElevIndices) > 0:
                # place value -10 as first index so that we also find the first arc
                posElevIndicesExtend = np.append(np.array([-10]), posElevIndices)
                # print('posElevIndicesExtend = %s (#%d)' % (posElevIndicesExtend, np.size(posElevIndicesExtend)))
                # print('ediff1d = %s' % np.ediff1d(posElevIndicesExtend))
                newTLEArcIndices = np.where(np.ediff1d(posElevIndicesExtend) > 1)[0]
                # print('newTLEArcIndices = %s (#%d)' % (newTLEArcIndices, np.size(newTLEArcIndices)))

                # get the time by crawling back through the indices
                newTLEArcTime = []
                endTLEArcTime = []
                for j, newArc in enumerate(newTLEArcIndices):
                    # print('newTLEArcIndices[%d] = %d' % (j, newArc))
                    # print('posElevIndices = %s' % posElevIndices[newArc])
                    # print('newTLEArcTime = %s' % predTimeSeconds[posElevIndices[newArc]])
                    newTLEArcTime.append(predTimeSeconds[posElevIndices[newArc]])
                    if j < len(newTLEArcIndices) - 1:
                        # print('newTLEArcTime[j+1] = %d  %s' % (j+1, newTLEArcIndices[j+1]))
                        endTLEArcTime.append(predTimeSeconds[posElevIndices[newTLEArcIndices[j + 1] - 1]])
                    else:
                        endTLEArcTime.append(predTimeSeconds[posElevIndices[-1]])

                # print('newTLEArcTime = %s (#%d)' % (newTLEArcTime, np.size(newTLEArcTime)))
                # print('endTLEArcTime = %s (#%d)' % (endTLEArcTime, np.size(endTLEArcTime)))
                for t in range(len(newTLEArcTime)):
                    # print('t = %s' % t)
                    year, month, day, hh, mm, ss = gpstime.UTCFromGps(WkNr, newTLEArcTime[t], leapSecs=0)
                    newHHMM = '%02d:%02d' % (hh, mm)
                    year, month, day, hh, mm, ss = gpstime.UTCFromGps(WkNr, endTLEArcTime[t], leapSecs=0)
                    endHHMM = '%02d:%02d' % (hh, mm)

                    newArc = [prnNr, WkNr, newTLEArcTime[t], endTLEArcTime[t], newHHMM, endHHMM]
                    # print('newArc = %s' % newArc)

                    arcWriter.writerow(newArc)

    # close the arc file
    outArc.close()


def treatCmdOpts(argv):
    """
    treatCmdOpts treats the command line arguments
    """
    helpTxt = os.path.basename(__file__) + ' predicts the GNSS orbits based on TLEs'

    # create the parser for command line arguments
    parser = argparse.ArgumentParser(description=helpTxt)
    parser.add_argument('-s', '--satSystem', help='Name of satellite system in comma separated list (cfr NORAD naming)', required=True)
    parser.add_argument('-x', '--exclude', help='comma separated list of satellite PRNs to exclude from DOP calculation (eg. E18,E14,E20)', default='', type=str, required=False)
    parser.add_argument('-i', '--interval', help='interval in minutes (defaults to 20)', type=int, required=False, default=20)
    parser.add_argument('-c', '--cutoff', help='cutoff angle in degrees (defaults to 10)', type=int, required=False, default=10)
    parser.add_argument('-o', '--observer', help='Station info "name,latitude,longitude" (units = degrees, defaults to RMA)', required=False, default=None)
    parser.add_argument('-d', '--date', help='Enter prediction date (YYYY/MM/DD), defaults to today', required=False)
    parser.add_argument('-b', '--startTime', help='Enter start time (hh:mm), defaults to 00:00', required=False, default='00:00')
    parser.add_argument('-e', '--endTime', help='Enter end time (hh:mm), defaults to 23:59', required=False, default='23:59')
    parser.add_argument('-m', '--maxDOP', help='Maximum xDOP value to display, defaults to 15', required=False, default=15, type=int)
    parser.add_argument('-v', '--verbose', help='displays interactive graphs and increase output verbosity (default False)', action='store_true', required=False)

    args = parser.parse_args()

    return args.satSystem, args.exclude, args.interval, args.cutoff, args.observer, args.date, args.startTime, args.endTime, args.maxDOP, args.verbose


# main starts here
if __name__ == "__main__":
    matplotlib.use("TkAgg")

    # init defaullt station
    RMA = Station()
    RMA.init('RMA', '50:50:38.4551', '4:23:34.5421', ephem.date(ephem.now()))
    BERTRIX = Station()
    BERTRIX.init('BERTRIX', '49.894275', '5.241417', ephem.date(ephem.now()))
    # print('RMA = %s | %.9f  %.9f | %s  %s' % (RMA.name, RMA.lat, RMA.lon, ephem.degrees(RMA.lat), ephem.degrees(RMA.lon)))

    # treat the command line options
    satSystem, excludeSats, interval, cutoff, observer, predDate, startTime, endTime, maxDOP, verbose = treatCmdOpts(sys.argv)
    # print('satSystem = %s' % satSystem)
    # if ',' in excludeSats:
    #     sats2Exclude = excludeSats.split(',')
    if len(excludeSats) > 0:
        sats2Exclude = excludeSats.split(',')
    else:
        sats2Exclude = None
    print('sats2Exclude = %s' % sats2Exclude)
    # print('predDate = %s' % predDate)
    # print('startTime = %s' % startTime)
    # print('endTime = %s' % endTime)
    # print('interval = %s' % interval)
    # print('observer = %s' % observer)
    # print('cutoff = %d' % cutoff)
    # print('verbose = %s\n' % verbose)

    # import tle data from NORAD if internet_on(), save as sat=ephem.readtle(...)-----
    # TLEfile = getTLEfromNORAD(satSystem)
    TLEfile = getTLEfromNORAD(satSystem)
    # TLEfile = 'galileo.txt'
    # existTLEFile(folder, TLEfile, verbose)

    # read in the observer info (name, latitude, longitude, date
    obsStation = setObserverData(observer, predDate, verbose)
    if verbose:
        obsStation.statPrint()

    # read in the list of satellites from the TLE
    satList = loadTLE(TLEfile, verbose)

    # calculate the interval settings for a full day prediction starting at 00:00:00 hr of predDate
    predDateTimes, nrPredictions = setObservationTimes(obsStation, startTime, endTime, interval, verbose)
    print('nrPredictions = %d' % nrPredictions)

    # calculate the informations for each SVs in satList
    subLat = np.empty([nrPredictions, np.size(satList)])
    subLon = np.empty([nrPredictions, np.size(satList)])
    azim = np.empty([nrPredictions, np.size(satList)])
    elev = np.empty([nrPredictions, np.size(satList)])
    dist = np.empty([nrPredictions, np.size(satList)])
    dist_velocity = np.empty([nrPredictions, np.size(satList)])
    eclipsed = np.empty([nrPredictions, np.size(satList)])
    xDOP = np.empty([nrPredictions, 4])  # order is HDOP, VDOP, TDOP and NrOfSVsUsed
    nrExcluded = np.empty(nrPredictions)

    for i, dt in enumerate(predDateTimes):
        # print('\ndt[%d] = %s' % (i, dt))

        obsStation.date = dt
        elevTxt = ''
        for j, sat in enumerate(satList):
            sat.compute(obsStation)
            # print('sat[%d] = %26s   %5.1f   %4.1f' % (j, sat.name, np.rad2deg(sat.az), np.rad2deg(sat.alt)))
            subLat[i, j] = np.rad2deg(sat.sublat)
            subLon[i, j] = np.rad2deg(sat.sublong)
            azim[i, j] = np.rad2deg(sat.az)
            elev[i, j] = np.rad2deg(sat.alt)
            dist[i, j] = sat.range
            dist_velocity[i, j] = sat.range_velocity
            eclipsed[i, j] = sat.eclipsed

            # elevTxt += "%6.1f  " % elev[i][j]

        # determine the visible satellites at this instance
        indexVisSats = []
        indexVisSats = np.where(elev[i, :] >= cutoff)
        # print('indexVisSats = %s (%d)' % (indexVisSats[0], np.size(indexVisSats)))
        indexVisSatsUsed = indexVisSats[0]
        index2Delete = []

        # exclude the SVs which have no valid signal (cfr sats2Exclude list)
        if sats2Exclude is not None:
            for k, prn in enumerate(satList):
                # print('k, prn: %s - %s' % (k, prn.name))
                if k in indexVisSats[0]:
                    # print('  PRN = %s is visible' % prn.name)
                    for jj, prnX in enumerate(sats2Exclude):
                        # print('    jj, prnX = %d -  %s' % (jj, prnX))
                        if prnX in prn.name:
                            # print('      found to exclude %s - index %s\n' % (prn.name, np.where(indexVisSats[0] == k)[0]))
                            index2Delete.append(np.where(indexVisSats[0] == k)[0])

            # print('index2Delete = %s' % index2Delete[0])
            index2Delete.sort(reverse=True)
            # print('index2Delete = %s' % index2Delete[0])
            nrExcluded[i] = np.size(index2Delete)
            # print('nrExcluded = %s' % nrExcluded)

            # print('indexVisSatsUsed = %s (%d)' % (indexVisSatsUsed, np.size(indexVisSatsUsed)))
            for k in index2Delete:
                indexVisSatsUsed = np.delete(indexVisSatsUsed, k)
            # print('indexVisSatsUsed = %s (%d)' % (indexVisSatsUsed, np.size(indexVisSatsUsed)))

        # print('elev = %s' % elev[i, :])
        # print('azim = %s' % azim[i,:])
        # print('elevRad = %s' % np.radians(elev[i,:]))

        # calculate xDOP values when at least 4 sats are visible above cutoff angle
        if np.size(indexVisSatsUsed) >= 4:
            A = np.matrix(np.empty([np.size(indexVisSatsUsed), 4], dtype=float))
            # print('A  = %s' % A)
            # print('type A = %s  ' % type(A))
            elevVisSatsRad = np.radians(elev[i, indexVisSatsUsed])
            azimVisSatsRad = np.radians(azim[i, indexVisSatsUsed])
            # print('elevVisSatsRad = %s' % elevVisSatsRad)
            # print('azimVisSatsRad = %s' % azimVisSatsRad)

            for j in range(np.size(indexVisSatsUsed)):
                # print('j = %s' % j)
                A[j, 0] = np.cos(azimVisSatsRad[j]) * np.cos(elevVisSatsRad[j])
                A[j, 1] = np.sin(azimVisSatsRad[j]) * np.cos(elevVisSatsRad[j])
                A[j, 2] = np.sin(elevVisSatsRad[j])
                A[j, 3] = 1.
                # print('A[%d] = %s' % (j, A[j]))

            # calculate ATAInv en get the respective xDOP parameters (HDOP, VDOP and TDOP)
            AT = A.getT()
            ATA = AT * A
            # print('AT = %s' % AT)
            # print('ATA = %s' % ATA)
            ATAInv = ATA.getI()
            # print('ATAInv = %s' % ATAInv)

            xDOP[i, 0] = np.sqrt(ATAInv[0, 0] + ATAInv[1, 1])  # HDOP
            xDOP[i, 1] = np.sqrt(ATAInv[2, 2])  # VDOP
            xDOP[i, 2] = np.sqrt(ATAInv[3, 3])  # TDOP
            xDOP[i, 3] = np.size(indexVisSatsUsed)
        else:  # not enough visible satellites
            xDOP[i] = [np.nan, np.nan, np.nan, np.nan]

        # print('xDOP[%d] = %s' % (i, xDOP[i]))

        # print('dt = %s' % dt)
        # print('lat= %s' % subLat[i][0])
        # print('lon= %s' % subLon[i][0])
        # print('az = %s' % azim[i][0])
        # print('el = %s' % elev[i][0])
        # print('r  = %s' % dist[i][0])
        # print('rV = %s' % dist_velocity[i][0])
        # print('ec = %s' % eclipsed[i][0])

        # print('%02d:%02d    %s' % (dt.hour, dt.minute, elevTxt))

        # sys.exit(6)

    # # create index for satellites above cutoff angle
    # for j, sat in enumerate(satList):
    #     # print('elev[:][%d] = %s' % (j, elev[:, j]))
    #     indexVisibleSats = np.where(elev[:,j] >= cutoff)
    #     print('indexVisibleSats = %s (%d)' % (indexVisibleSats, np.size(indexVisibleSats)))
    # print('elev = %s' % elev[:,j])

    # set all elev < cutoff to NAN
    elev[elev < cutoff] = np.nan
    # print('elev = %s' % elev)

    # write to results file
    createVisibleSatsFile(obsStation, satSystem, satList, predDateTimes, elev, azim, cutoff, sats2Exclude, verbose)
    createDOPFile(obsStation, satSystem, satList, predDateTimes, xDOP, cutoff, verbose)
    createGeodeticFile(obsStation, satSystem, satList, predDateTimes, subLat, subLon, verbose)

    # create plots
    plotVisibleSats(satSystem, obsStation, satList, predDateTimes, elev, cutoff, sats2Exclude, verbose)
    plotDOPVisSats(satSystem, obsStation, satList, predDateTimes, elev, xDOP, cutoff, nrExcluded, verbose)
    plotSkyView(satSystem, obsStation, satList, predDateTimes, elev, azim, cutoff, sats2Exclude, verbose)
    plotSatTracks(satSystem, obsStation, satList, predDateTimes, subLat, subLon, sats2Exclude, verbose)

    # show all plots
    if verbose:
        plt.show()

    createTLEArcFile(satSystem, satList, interval, predDateTimes, elev)

    # end program
    sys.exit(E_SUCCESS)
