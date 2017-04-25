/*  Copyright 2011 AIT Austrian Institute of Technology
*
*   This file is part of OpenTLD.
*
*   OpenTLD is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*   (at your option) any later version.
*
*   OpenTLD is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License
*   along with OpenTLD.  If not, see <http://www.gnu.org/licenses/>.
*
*/

/*
 * main.h
 *
 *  Created on: Nov 18, 2011
 *      Author: Georg Nebehay
 */

#ifndef MAIN_H_
#define MAIN_H_

#include "TLD.h"
#include "ImAcq.h"
#include "Gui.h"

enum Retval
{
    PROGRAM_EXIT = 0,
    SUCCESS = 1
};

class Main
{
public:
    tld::TLD *tld;

    tld::TLD *_tld;

    tld::TLD *_tld1;

    
    
    ImAcq *imAcq;
    tld::Gui *gui;
    bool showOutput;
	bool showTrajectory;
	int trajectoryLength;
    const char *printResults;
    const char *saveDir;
    double threshold;
    bool showForeground;
    bool showNotConfident;
    bool selectManually;

    int *initialBB;
    int *initialBB1;
    int *initialBB2;

    bool reinit;
    bool exportModelAfterRun;
    bool loadModel;
    const char *modelPath;
    const char *modelExportFile;
    int seed;

    int dem =0;

    Main()
    {
        tld = new tld::TLD();

        _tld = new tld::TLD();

        _tld1 = new tld::TLD();
        
        showOutput = 1;
        printResults = NULL;
        saveDir = ".";
        threshold = 0.5;
        showForeground = 0;

		showTrajectory = false;
		trajectoryLength = 0;

        selectManually = 0;

        initialBB = NULL;
        initialBB1 = NULL;
        initialBB2 = NULL;

        showNotConfident = true;

        reinit = 0;

        loadModel = false;

        exportModelAfterRun = false;
        modelExportFile = "model";
        seed = 0;

        gui = NULL;
        modelPath = NULL;
        imAcq = NULL; if(key == 'a')
                {
                    tld->alternating = !tld->alternating;
                    printf("alternating: %d\n", tld->alternating);
                }

                if(key == 'e')
                {
                    tld->writeToFile(modelExportFile);
                }

                if(key == 'i')
                {
                    tld->readFromFile(modelPath);
                }
    }

    ~Main()
    {
        delete tld;
        delete _tld;
        delete _tld1;
        imAcqFree(imAcq);
    }

    void doWork();
};

#endif /* MAIN_H_ */
